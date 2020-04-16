#!/usr/bin/env python
#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


__all__ = ['Ch', 'depends_on', 'MatVecMult', 'ChHandle', 'ChLambda']

import sys
import inspect
import scipy.sparse as sp

import numpy as np
import numbers
import weakref
import copy as external_copy
from functools import wraps
from scipy.sparse.linalg.interface import LinearOperator
import collections
from copy import deepcopy
from functools import reduce
from chumpy.utils import *
# import ipdb
import warnings
import pickle

_props_for_dict = weakref.WeakKeyDictionary()
def _props_for(cls):
    if cls not in _props_for_dict:
        _props_for_dict[cls] = set([p[0] for p in inspect.getmembers(cls, lambda x : isinstance(x, property))])
    return _props_for_dict[cls]

_dep_props_for_dict = weakref.WeakKeyDictionary()      
def _dep_props_for(cls):
    if cls not in _dep_props_for_dict:
        _dep_props_for_dict[cls] = [p for p in inspect.getmembers(cls, lambda x : isinstance(x, property)) if hasattr(p[1].fget, 'deps')]
    return _dep_props_for_dict[cls]
    

_kw_conflict_dict = weakref.WeakKeyDictionary()
def _check_kw_conflict(cls): 
    if cls not in _kw_conflict_dict:
        _kw_conflict_dict[cls] = Ch._reserved_kw.intersection(set(cls.terms).union(set(cls.dterms)))
    if _kw_conflict_dict[cls]:
        raise Exception("In class %s, don't use reserved keywords in terms/dterms: %s" % (str(cls), str(kw_conflict),))


class Term(object):
    creation_counter = 0
    def __init__(self, default=None, desc=None, dr=True):
        self.default = default
        self.desc = desc
        self.dr = dr

        # Add a creation_counter, a la Django models, so we can preserve the order in which parameters are defined in the job.
        # http://stackoverflow.com/a/3288801/893113
        self.creation_counter = Term.creation_counter
        Term.creation_counter += 1


class Ch(object):
    terms = []
    dterms = ['x']
    __array_priority__ = 2.0

    _cached_parms = {}
    _setup_terms = {}

    ########################################################
    # Construction

    def __new__(cls, *args, **kwargs):

        if len(args) > 0 and type(args[0]) == type(lambda : 0):
            cls = ChLambda
        
        # Create empty instance
        result = super(Ch, cls).__new__(cls)

        cls.setup_terms()

        object.__setattr__(result, '_dirty_vars', set())
        object.__setattr__(result, '_itr', None)
        object.__setattr__(result, '_parents', weakref.WeakKeyDictionary())
        object.__setattr__(result, '_cache', {'r': None, 'drs': weakref.WeakKeyDictionary()})
        
        # Set up storage that allows @depends_on to work
        #props = [p for p in inspect.getmembers(cls, lambda x : isinstance(x, property)) if hasattr(p[1].fget, 'deps')]
        props = _dep_props_for(cls)
        cpd = {}
        for p in props:
            func_name = p[0] #id(p[1].fget)
            deps = p[1].fget.deps
            cpd[func_name] = {'deps': deps, 'value': None, 'out_of_date': True}
        
        object.__setattr__(result, '_depends_on_deps', cpd)

        if cls != Ch:
            for idx, a in enumerate(args):
                kwargs[cls.term_order[idx]] = a
        elif len(args)>0:
            kwargs['x'] = np.asarray(args[0], np.float64)

        defs = {p.name : deepcopy(p.default) for p in cls.parm_declarations() if p.default is not None}
        defs.update(kwargs)
        result.set(**defs)
        
        return result

    @classmethod
    def parm_declarations(cls):
        if cls.__name__ not in cls._cached_parms:
            parameter_declarations = collections.OrderedDict()
            parameters = inspect.getmembers(cls, lambda x: isinstance(x, Term))
            for name, decl in sorted(parameters, key=lambda x: x[1].creation_counter):
                decl.name = name
                parameter_declarations[name] = decl
            cls._cached_parms[cls.__name__] = parameter_declarations
        return cls._cached_parms[cls.__name__]

    @classmethod
    def setup_terms(cls):
        if id(cls) in cls._setup_terms: return

        if cls == Ch:
            return

        parm_declarations = cls.parm_declarations()

        if cls.dterms is Ch.dterms:
            cls.dterms = []
        elif isinstance(cls.dterms, str):
            cls.dterms = (cls.dterms,)
        if cls.terms is Ch.terms:
            cls.terms = []
        elif isinstance(cls.terms, str):
            cls.terms = (cls.terms,)

        # Must be either new or old style
        len_oldstyle_parms = len(cls.dterms)+len(cls.terms)
        if len(parm_declarations) > 0:
            assert(len_oldstyle_parms==0)
            cls.term_order = [t.name for t in parm_declarations]
            cls.dterms = [t.name for t in parm_declarations if t.dr]
            cls.terms = [t.name for t in parm_declarations if not t.dr]
        else:
            if not hasattr(cls, 'term_order'):
                cls.term_order = list(cls.terms) + list(cls.dterms)

        _check_kw_conflict(cls)
        cls._setup_terms[id(cls)] = True


    ########################################################
    # Identifiers

    @property
    def sid(self):
        """Semantic id."""
        pnames = list(self.terms)+list(self.dterms)
        pnames.sort()
        return (self.__class__, tuple([(k, id(self.__dict__[k])) for k in pnames if k in self.__dict__]))

    
    def reshape(self, *args):
        return reshape(a=self, newshape=args if len(args)>1 else args[0])
    
    def ravel(self):
        return reshape(a=self, newshape=(-1))
    
    def __hash__(self):
        return id(self)
        
    @property
    def ndim(self):
        return self.r.ndim
        
    @property
    def flat(self):
        return self.r.flat
        
    @property
    def dtype(self):
        return self.r.dtype
        
    @property
    def itemsize(self):
        return self.r.itemsize
            

    ########################################################
    # Redundancy removal

    def remove_redundancy(self, cache=None, iterate=True):

        if cache == None:
            cache = {}
            _ = self.r # may result in the creation of extra dterms that we can cull
            
        replacement_occurred = False
        for propname in list(self.dterms):
            prop = self.__dict__[propname]

            if not hasattr(prop, 'dterms'):
                continue
            sid = prop.sid
            if sid not in cache:
                cache[sid] = prop
            elif self.__dict__[propname] is not cache[sid]:
                self.__dict__[propname] = cache[sid]
                replacement_occurred = True
            if prop.remove_redundancy(cache, iterate=False):
                replacement_occurred = True
                
        if not replacement_occurred:
            return False
        else:
            if iterate:
                self.remove_redundancy(cache, iterate=True)
                return False
            else:
                return True
            
                
                
    def print_labeled_residuals(self, print_newline=True, num_decimals=2, where_to_print=None):
        
        if where_to_print is None:
            where_to_print = sys.stderr
        if hasattr(self, 'label'):
            where_to_print.write(('%s: %.' + str(num_decimals) + 'e | ') % (self.label, np.sum(self.r**2)))
        for dterm in self.dterms:
            dt = getattr(self, dterm)            
            if hasattr(dt, 'dterms'):
                dt.print_labeled_residuals(print_newline=False, where_to_print=where_to_print)            
        if print_newline:
            where_to_print.write(('%.' + str(num_decimals) + 'e\n') % (np.sum(self.r**2),))
        
    

    ########################################################
    # Default methods, for when Ch is not subclassed

    def compute_r(self):
        """Default method for objects that just contain a number or ndarray"""
        return self.x
        
    def compute_dr_wrt(self,wrt):
        """Default method for objects that just contain a number or ndarray"""
        if wrt is self: # special base case  
            return sp.eye(self.x.size, self.x.size)
            #return np.array([[1]])
        return None
        
    
    def _compute_dr_wrt_sliced(self, wrt):
        self._call_on_changed()
        
        # if wrt is self:
        #     return np.array([[1]])

        result = self.compute_dr_wrt(wrt)
        if result is not None:
            return result

        # What allows slicing.
        if True:
            inner = wrt
            while issubclass(inner.__class__, Permute):
                inner = inner.a
                if inner is self: 
                    return None
                result = self.compute_dr_wrt(inner)

                if result is not None:
                    break
        
            if result is None:
                return None
        
            wrt._call_on_changed()
        
            try:
                jac = wrt.compute_dr_wrt(inner).T
            except Exception as e:
                import pdb; pdb.set_trace()

            return self._superdot(result, jac)


    @property
    def shape(self):
        return self.r.shape
    
    @property
    def size(self):
        #return self.r.size
        return np.prod(self.shape) # may be cheaper since it doesn't always mean grabbing "r"
    
    def __len__(self):
        return len(self.r)
        
    def minimize(self, *args, **kwargs):
        from chumpy import optimization
        return optimization.minimize(self, *args, **kwargs)
        
    def __array__(self, *args):
        return self.r
    
    ########################################################
    # State management

    def add_dterm(self, dterm_name, dterm):
        self.dterms = list(set(list(self.dterms) + [dterm_name]))
        setattr(self, dterm_name, dterm)
    
    def copy(self):
        return pickle.loads(pickle.dumps(self))  
    
    def __getstate__(self):
        # Have to get rid of WeakKeyDictionaries for serialization
        result = external_copy.copy(self.__dict__)
        del result['_parents']
        del result['_cache']
        return result

    def __setstate__(self, d):
        # Restore unpickleable WeakKeyDictionaries
        d['_parents'] = weakref.WeakKeyDictionary()
        d['_cache'] = {'r': None, 'drs': weakref.WeakKeyDictionary()}
        object.__setattr__(self, '__dict__', d)

        # This restores our unpickleable "_parents" attribute
        for k in set(self.dterms).intersection(set(self.__dict__.keys())):
            setattr(self, k, self.__dict__[k])
       
    def __setattr__(self, name, value, itr=None):
        #print 'SETTING %s' % (name,)

        # Faster path for basic Ch objects. Not necessary for functionality,
        # but improves performance by a small amount.
        if type(self) == Ch:
            if name == 'x':
                self._dirty_vars.add(name)
                self.clear_cache(itr)
            #else:
            #    import warnings
            #    warnings.warn('Trying to set attribute %s on a basic Ch object? Might be a mistake.' % (name,))

            object.__setattr__(self, name, value)
            return

        name_in_dterms = name in self.dterms
        name_in_terms = name in self.terms
        name_in_props = name in _props_for(self.__class__)# [p[0] for p in inspect.getmembers(self.__class__, lambda x : isinstance(x, property))]
        
        if name_in_dterms and not name_in_props and type(self) != Ch:
            if not hasattr(value, 'dterms'):
                value = Ch(value)                
        
            # Make ourselves not the parent of the old value
            if hasattr(self, name):
                term = getattr(self, name)
                if self in term._parents:
                    term._parents[self]['varnames'].remove(name)
                    if len(term._parents[self]['varnames']) == 0:
                        del term._parents[self]
                    
            # Make ourselves parents of the new value
            if self not in value._parents:
                value._parents[self] = {'varnames': set([name])}
            else:
                value._parents[self]['varnames'].add(name)

        if name_in_dterms or name_in_terms:            
            self._dirty_vars.add(name)
            self._invalidate_cacheprop_names([name])

            # If one of our terms has changed, it has the capacity to have
            # changed our result and all our derivatives wrt everything
            self.clear_cache(itr)
            
        object.__setattr__(self, name, value)
          
    def _invalidate_cacheprop_names(self, names):
        nameset = set(names)
        for func_name, v in list(self._depends_on_deps.items()):
            if len(nameset.intersection(v['deps'])) > 0:
                v['out_of_date'] = True
        
        
    def clear_cache(self, itr=None):
        todo = [self]
        done = set([])
        nodes_visited = 0
        while len(todo) > 0:
            nodes_visited += 1
            next = todo.pop()
            if itr is not None and itr==next._itr:
                continue
            if id(next) not in done:
                next._cache['r'] = None
                next._cache['drs'].clear()
                next._itr = itr

                for parent, parent_dict in list(next._parents.items()):
                    object.__setattr__(parent, '_dirty_vars', parent._dirty_vars.union(parent_dict['varnames']))
                    parent._invalidate_cacheprop_names(parent_dict['varnames']) 
                    todo.append(parent)
                done.add(id(next))
        return nodes_visited
            
        
    def replace(self, old, new):
        if (hasattr(old, 'dterms') != hasattr(new, 'dterms')):
            raise Exception('Either "old" and "new" must both be "Ch", or they must both be neither.')
        
        for term_name in [t for t in list(self.dterms)+list(self.terms) if hasattr(self, t)]:
            term = getattr(self, term_name)
            if term is old:
                setattr(self, term_name, new)
            elif hasattr(term, 'dterms'):
                term.replace(old, new)
        return new
  
    
    def set(self, **kwargs):
        # Some dterms may be aliases via @property.
        # We want to set those last, in case they access non-property members
        #props = [p[0] for p in inspect.getmembers(self.__class__, lambda x : isinstance(x, property))]
        props = _props_for(self.__class__)
        kwarg_keys = set(kwargs.keys())
        kwsecond = kwarg_keys.intersection(props)
        kwfirst = kwarg_keys.difference(kwsecond)
        kwall = list(kwfirst) + list(kwsecond)

        # The complexity here comes because we wish to
        # avoid clearing cache redundantly
        if len(kwall) > 0:
            for k in kwall[:-1]:
                self.__setattr__(k, kwargs[k], 9999)
            self.__setattr__(kwall[-1], kwargs[kwall[-1]], None)
            

    def is_dr_wrt(self, wrt):
        if type(self) == Ch:
            return wrt is self
        dterms_we_have = [getattr(self, dterm) for dterm in self.dterms if hasattr(self, dterm)]
        return wrt in dterms_we_have or any([d.is_dr_wrt(wrt) for d in dterms_we_have])
    

    def is_ch_baseclass(self):
        return self.__class__ is Ch
        

    ########################################################
    # Getters for our outputs

    def __getitem__(self, key):        
        shape = self.shape
        tmp = np.arange(np.prod(shape)).reshape(shape).__getitem__(key)
        idxs = tmp.ravel()
        newshape = tmp.shape
        return Select(a=self, idxs=idxs, preferred_shape=newshape)
        
    def __setitem__(self, key, value, itr=None): 

        if hasattr(value, 'dterms'):
            raise Exception("Can't assign a Ch objects as a subset of another.")
        if type(self) == Ch:# self.is_ch_baseclass():
            data = np.atleast_1d(self.x)
            data.__setitem__(key, value)
            self.__setattr__('x', data, itr=itr)
            return
        # elif False: # Interesting but flawed idea
            # parents = [self.__dict__[k] for k in self.dterms]
            # kids = []
            # while len(parents)>0:
            #     p = parents.pop()
            #     if p.is_ch_baseclass():
            #         kids.append(p)
            #     else:
            #         parents += [p.__dict__[k] for k in p.dterms]
            # from optimization import minimize_dogleg
            # minimize_dogleg(obj=self.__getitem__(key) - value, free_variables=kids, show_residuals=False)            
        else:
            inner = self
            while not inner.is_ch_baseclass():
                if issubclass(inner.__class__, Permute):
                    inner = inner.a
                else: 
                    raise Exception("Can't set array that is function of arrays.")

            self = self[key]
            dr = self.dr_wrt(inner)
            dr_rev = dr.T
            #dr_rev = np.linalg.pinv(dr)
            inner_shape = inner.shape

            t1 = self._superdot(dr_rev, np.asarray(value).ravel())
            t2 = self._superdot(dr_rev, self._superdot(dr, inner.x.ravel()))
            if sp.issparse(t1): t1 = t1.toarray()
            if sp.issparse(t2): t2 = t2.toarray()

            inner.x = inner.x + t1.reshape(inner_shape) - t2.reshape(inner_shape)
            #inner.x = inner.x + self._superdot(dr_rev, value.ravel()).reshape(inner_shape) - self._superdot(dr_rev, self._superdot(dr, inner.x.ravel())).reshape(inner_shape)


    def __str__(self):
        return str(self.r)

    def __repr__(self):
        return object.__repr__(self) + '\n' + str(self.r)

    def __float__(self):
        return self.r.__float__()

    def __int__(self):
        return self.r.__int__()

    def on_changed(self, terms):
        pass
        
    @property
    def T(self):
        return transpose(self)
        
    def transpose(self, *axes):
        return transpose(self, *axes)
        
    # def squeeze(self, axis=None):
    #     return ch_ops.squeeze(self, axis)
        
    def mean(self, axis=None):
        return mean(self, axis=axis)

    def sum(self, axis=None):
        return sum(self, axis=axis)

    def _call_on_changed(self):

        if hasattr(self, 'is_valid'):
            validity, msg = self.is_valid()
            assert validity, msg
        if len(self._dirty_vars) > 0:
            self.on_changed(self._dirty_vars)
            object.__setattr__(self, '_dirty_vars', set())

    @property
    def r(self):
        self._call_on_changed()
        if self._cache['r'] is None:
            self._cache['r'] = np.asarray(np.atleast_1d(self.compute_r()), dtype=np.float64, order='C')
            self._cache['rview'] = self._cache['r'].view()
            self._cache['rview'].flags.writeable = False
        
        return self._cache['rview']


    def _superdot(self, lhs, rhs):
        try:
            if lhs is None:
                return None
            if rhs is None:
                return None
            
            if isinstance(lhs, np.ndarray) and lhs.size==1:
                lhs = lhs.ravel()[0]
                
            if isinstance(rhs, np.ndarray) and rhs.size==1:
                rhs = rhs.ravel()[0]
    
            if isinstance(lhs, numbers.Number) or isinstance(rhs, numbers.Number):
                return lhs * rhs

            if isinstance(rhs, LinearOperator):
                return LinearOperator((lhs.shape[0], rhs.shape[1]), lambda x : lhs.dot(rhs.dot(x)))

            if isinstance(lhs, LinearOperator):                
                if sp.issparse(rhs):
                    return LinearOperator((lhs.shape[0], rhs.shape[1]), lambda x : lhs.dot(rhs.dot(x)))
                else:
                    return lhs.dot(rhs)
            
            # TODO: Figure out how/whether to do this.
            #lhs, rhs = utils.convert_inputs_to_sparse_if_possible(lhs, rhs)

            if not sp.issparse(lhs) and sp.issparse(rhs):
                return rhs.T.dot(lhs.T).T
    
            return lhs.dot(rhs)
        except:
            import pdb; pdb.set_trace()
            
    def lmult_wrt(self, lhs, wrt):
        if lhs is None:
            return None
        
        self._call_on_changed()

        drs = []        

        direct_dr = self._compute_dr_wrt_sliced(wrt)

        if direct_dr != None:
            drs.append(self._superdot(lhs, direct_dr))

        for k in set(self.dterms):
            p = self.__dict__[k]

            if hasattr(p, 'dterms') and p is not wrt and p.is_dr_wrt(wrt):
                if not isinstance(p, Ch):
                    print('BROKEN!')
                    import pdb; pdb.set_trace()

                indirect_dr = p.lmult_wrt(self._superdot(lhs, self._compute_dr_wrt_sliced(p)), wrt)
                if indirect_dr is not None:
                    drs.append(indirect_dr)

        if len(drs)==0:
            result = None

        elif len(drs)==1:
            result = drs[0]

        else:
            result = reduce(lambda x, y: x+y, drs)

        return result
        

    def compute_lop(self, wrt, lhs):
        dr = self._compute_dr_wrt_sliced(wrt)
        if dr is None: return None
        return self._superdot(lhs, dr) if not isinstance(lhs, LinearOperator) else lhs.matmat(dr)
        
        
    def lop(self, wrt, lhs):
        self._call_on_changed()
        
        drs = []
        direct_dr = self.compute_lop(wrt, lhs)
        if direct_dr is not None:
            drs.append(direct_dr)

        for k in set(self.dterms):
            p = getattr(self, k) # self.__dict__[k]
            if hasattr(p, 'dterms') and p is not wrt: # and p.is_dr_wrt(wrt):
                lhs_for_child = self.compute_lop(p, lhs)
                if lhs_for_child is not None: # Can be None with ChLambda, _result etc
                    indirect_dr = p.lop(wrt, lhs_for_child)
                    if indirect_dr is not None:
                        drs.append(indirect_dr)

        for k in range(len(drs)):
            if sp.issparse(drs[k]):
                drs[k] = drs[k].toarray()

        if len(drs)==0:
            result = None

        elif len(drs)==1:
            result = drs[0]

        else:
            result = reduce(lambda x, y: x+y, drs)

            
        return result
        
    def compute_rop(self, wrt, rhs):
        dr = self._compute_dr_wrt_sliced(wrt)
        if dr is None: return None
        
        return self._superdot(dr, rhs)

    def dr_wrt(self, wrt, reverse_mode=False):
        self._call_on_changed()
        # ipdb.set_trace()
        drs = []        

        if wrt in self._cache['drs']:
            return self._cache['drs'][wrt]

        direct_dr = self._compute_dr_wrt_sliced(wrt)

        if direct_dr is not None:
            drs.append(direct_dr)                

        propnames = set(_props_for(self.__class__))
        for k in set(self.dterms).intersection(propnames.union(set(self.__dict__.keys()))):

            p = getattr(self, k)

            if hasattr(p, 'dterms') and p is not wrt:

                indirect_dr = None

                if reverse_mode:
                    lhs = self._compute_dr_wrt_sliced(p)
                    if isinstance(lhs, LinearOperator):
                        dr2 = p.dr_wrt(wrt)
                        indirect_dr = lhs.matmat(dr2) if dr2 != None else None
                    else:
                        indirect_dr = p.lmult_wrt(lhs, wrt)
                else: # forward mode
                    dr2 = p.dr_wrt(wrt)
                    if dr2 is not None:
                        # ipdb.set_trace()
                        indirect_dr = self.compute_rop(p, rhs=dr2)

                if indirect_dr is not None:
                    drs.append(indirect_dr)

        if len(drs)==0:
            result = None

        elif len(drs)==1:
            result = drs[0]

        else:
            if not np.any([isinstance(a, LinearOperator) for a in drs]):
                result = reduce(lambda x, y: sp.csc_matrix(x)+sp.csc_matrix(y), drs)
            else:
                result = LinearOperator(drs[0].shape, lambda x : reduce(lambda a, b: a.dot(x)+b.dot(x),drs))

        # TODO: figure out how/whether to do this.
        # if result is not None and not sp.issparse(result):
        #    nonzero = np.count_nonzero(result)
        #    if nonzero > 0 and hasattr(result, 'size') and result.size / nonzero >= 10.0:
        #         #import pdb; pdb.set_trace()
        #         result = sp.csc_matrix(result)
            
            
        if (result is not None) and (not sp.issparse(result)) and (not isinstance(result, LinearOperator)):
            result = np.atleast_2d(result)
            
        # When the number of parents is one, it indicates that
        # caching this is probably not useful because not 
        # more than one parent will likely ask for this same
        # thing again in the same iteration of an optimization.
        #
        # If we *always* filled in the cache, it would require 
        # more memory but would occasionally save a little cpu,
        # on average.
        if len(list(self._parents.keys())) != 1:
            self._cache['drs'][wrt] = result

        if not sp.issparse(result) and (result is not None):
            return np.array(result)
        else:
            return result


    def __call__(self, **kwargs):
        self.set(**kwargs)
        return self.r


    ########################################################
    # Visualization

    def show_tree(self, cachelim=np.inf):
        """Cachelim is in Mb. For any cached jacobians above cachelim, they are also added to the graph. """
        
        import tempfile
        import subprocess
        def string_for(self, my_name):
            if hasattr(self, 'label'):
                my_name = self.label
            my_name = '%s (%s)' % (my_name, str(self.__class__.__name__))
            result = []
            if not hasattr(self, 'dterms'):
                return result
            for dterm in self.dterms:
                if hasattr(self, dterm):
                    dtval = getattr(self, dterm)
                    if hasattr(dtval, 'dterms') or hasattr(dtval, 'terms'):
                        child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                        child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                        src = 'aaa%d' % (id(self))
                        dst = 'aaa%d' % (id(dtval))
                        result += ['%s -> %s;' % (src, dst)]
                        result += ['%s [label="%s"];' % (src, my_name)]
                        result += ['%s [label="%s"];' % (dst, child_label)]
                        result += string_for(getattr(self, dterm), dterm)

            if cachelim != np.inf and hasattr(self, '_cache') and 'drs' in self._cache:
                import pickle as pickle
                for dtval, jac in list(self._cache['drs'].items()):
                    # child_label = getattr(dtval, 'label') if hasattr(dtval, 'label') else dterm
                    # child_label = '%s (%s)' % (child_label, str(dtval.__class__.__name__))
                    src = 'aaa%d' % (id(self))
                    dst = 'aaa%d' % (id(dtval))
                    try:                    
                        sz = sys.getsizeof(pickle.dumps(jac, -1))
                    except: # some are functions
                        sz = 0
                    # colorattr = "#%02x%02x%02x" % (szpct*255, 0, (1-szpct)*255)
                    # print colorattr
                    if sz > (cachelim * 1024 * 1024):
                        result += ['%s -> %s [style=dotted,color="<<<%d>>>"];' % (src, dst, sz)]
                    #
                    # result += ['%s -> %s [style=dotted];' % (src, dst)]
                    # result += ['%s [label="%s"];' % (src, my_name)]
                    # result += ['%s [label="%s"];' % (dst, child_label)]
                    # result += string_for(getattr(self, dterm), dterm)
                    
            return result

        dot_file_contents = 'digraph G {\n%s\n}' % '\n'.join(list(set(string_for(self, 'root'))))
        if cachelim != np.inf:
            import re
            strs = re.findall(r'<<<(\d+)>>>', dot_file_contents, re.DOTALL)
            if len(strs) > 0:
                the_max = np.max(np.array([int(d) for d in strs]))
                for s in strs:
                    szpct = float(s)/the_max
                    sz = float(s)
                    unit = 'b'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'K'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'M'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'G'
                    if sz > 1024.: 
                        sz /= 1024
                        unit = 'T'
                        
                    dot_file_contents = re.sub('<<<%s>>>' % s, '#%02x%02x%02x",label="%d%s' % (szpct*255, 0, (1-szpct)*255, sz, unit), dot_file_contents)

        dot_file = tempfile.NamedTemporaryFile()
        dot_file.write(bytes(dot_file_contents, 'UTF-8'))
        dot_file.flush()
        png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        subprocess.call(['dot', '-Tpng', '-o', png_file.name, dot_file.name])
        import webbrowser
        webbrowser.open('file://' + png_file.name)

    def floor(self):
        return floor(self)
    
    def ceil(self):
        return ceil(self)

    def dot(self, other):
        return dot(self, other)

    def cumsum(self, axis=None):
        return cumsum(a=self, axis=axis)
        
    def min(self, axis=None):
        return amin(a=self, axis=axis)
    
    def max(self, axis=None):
        return amax(a=self, axis=axis)

    ########################################################
    # Operator overloads        

    def __pos__(self): return self    
    def __neg__(self): return negative(self)

    def __add__ (self, other): return add(a=self, b=other)
    def __radd__(self, other): return add(a=other, b=self)

    def __sub__ (self, other): return subtract(a=self, b=other)
    def __rsub__(self, other): return subtract(a=other, b=self)
        
    def __mul__ (self, other): return multiply(a=self, b=other)
    def __rmul__(self, other): return multiply(a=other, b=self)
        
    def __truediv__ (self, other): return divide(x1=self, x2=other)
    def __rtruediv__(self, other): return divide(x1=other, x2=self)

    def __pow__ (self, other): return ch_power(x=self, pow=other)
    def __rpow__(self, other): return ch_power(x=other, pow=self)
        
    def __rand__(self, other): return self.__and__(other)

    def __abs__ (self): return abs(self)
    
    def __gt__(self, other): return greater(self, other)
    def __ge__(self, other): return greater_equal(self, other)
        
    def __lt__(self, other): return less(self, other)
    def __le__(self, other): return less_equal(self, other)
    
    def __ne__(self, other): return not_equal(self, other)
    
    # not added yet because of weak key dict conflicts
    #def __eq__(self, other): return ch_ops.equal(self, other)


Ch._reserved_kw = set(Ch.__dict__.keys())
        

class MatVecMult(Ch):
    terms = 'mtx'
    dterms = 'vec'
    def compute_r(self):
        result = self.mtx.dot(self.vec.r.ravel()).reshape((-1,1)).ravel()
        if len(self.vec.r.shape) > 1 and self.vec.r.shape[1] > 1:
            result = result.reshape((-1,self.vec.r.shape[1]))
        return result

    def compute_dr_wrt(self, wrt):
        if wrt is self.vec:
            return sp.csc_matrix(self.mtx)
        
    
#def depends_on(*dependencies):
#    def _depends_on(func):
#        @wraps(func)
#        def with_caching(self, *args, **kwargs):
#            return func(self, *args, **kwargs)
#        return property(with_caching)
#    return _depends_on
    
        
def depends_on(*dependencies):
    deps = set()
    for dep in dependencies:
        if isinstance(dep, str):
            deps.add(dep)
        else:
            [deps.add(d) for d in dep]
    
    def _depends_on(func):
        want_out = 'out' in inspect.getargspec(func).args
        
        @wraps(func)
        def with_caching(self, *args, **kwargs):
            func_name = func.__name__
            sdf = self._depends_on_deps[func_name]
            if sdf['out_of_date'] == True:
                #tm = time.time()
                if want_out: 
                    kwargs['out'] = sdf['value']
                sdf['value'] = func(self, *args, **kwargs)
                sdf['out_of_date'] = False
                #print 'recomputed %s in %.2e' % (func_name, time.time() - tm)
            return sdf['value']
        with_caching.deps = deps # set(dependencies)
        result = property(with_caching)
        return result
    return _depends_on  



class ChHandle(Ch):
    dterms = ('x',)
    
    def compute_r(self):
        assert(self.x is not self)
        return self.x.r
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return 1
    

class ChLambda(Ch):
    terms = ['lmb', 'initial_args']
    dterms = []
    term_order = ['lmb', 'initial_args']

    def on_changed(self, which):
        for argname in set(which).intersection(set(self.args.keys())):
            self.args[argname].x = getattr(self, argname)
    
    def __init__(self, lmb, initial_args=None):
        args = {argname: ChHandle(x=Ch(idx)) for idx, argname in enumerate(inspect.getargspec(lmb)[0])}
        if initial_args is not None:
            for initial_arg in initial_args:
                if initial_arg in args:
                    args[initial_arg].x = initial_args[initial_arg]        
        result = lmb(**args)
        for argname, arg in list(args.items()):
            if result.is_dr_wrt(arg.x):
                self.add_dterm(argname, arg.x)
            else:
                self.terms.append(argname)
                setattr(self, argname, arg.x)
        self.args = args
        self.add_dterm('_result', result)
        
    def __getstate__(self):
        # Have to get rid of lambda for serialization
        if hasattr(self, 'lmb'):
            self.lmb = None
        return super(self.__class__, self).__getstate__()
        
        
    def compute_r(self):
        return self._result.r
    
    def compute_dr_wrt(self, wrt):
        if wrt is self._result:
            return 1


# Numpy functions
__all__ += ['array', 'amax','amin', 'max', 'min', 'maximum','minimum','nanmax','nanmin',
            'sum', 'exp', 'log', 'mean','std', 'var',
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
            'sqrt', 'square', 'absolute', 'abs', 'clip',
            'power',
            'add', 'divide', 'multiply', 'negative', 'subtract', 'reciprocal',
            'nan_to_num',
            'dot', 'cumsum',
            'floor', 'ceil',
            'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
            'nonzero', 'ascontiguousarray', 'asfarray', 'arange', 'asarray', 'copy',
            'cross',
            'shape', 'sign']


__all__ += ['SumOfSquares',
           'NanDivide', ]
           

# These can be wrapped directly as Ch(routine(*args, **kwargs)),
# so that for example "eye(3)" translates into Ch(np.eye(3))
numpy_array_creation_routines = [
    'empty','empty_like','eye','identity','ones','ones_like','zeros','zeros_like',
    'array',
    'arange','linspace','logspace','meshgrid','mgrid','ogrid',
    'fromfunction', 'fromiter', 'meshgrid', 'tri'
]

wont_implement = ['asanyarray', 'asmatrix', 'frombuffer', 'copy', 'fromfile', 'fromstring', 'loadtxt', 'copyto', 'asmatrix', 'asfortranarray', 'asscalar', 'require']
not_yet_implemented = ['tril', 'triu', 'vander']

__all__ += not_yet_implemented
__all__ += wont_implement
__all__ += numpy_array_creation_routines
        

from copy import copy as copy_copy

__all__ += ['pi', 'set_printoptions']
pi = np.pi
set_printoptions = np.set_printoptions
arange = np.arange

for rtn in ['argmax', 'nanargmax', 'argmin', 'nanargmin']:
    exec('def %s(a, axis=None) : return np.%s(a.r, axis) if hasattr(a, "compute_r") else np.%s(a, axis)' % (rtn, rtn, rtn))
    __all__ += [rtn]
    
for rtn in ['argwhere', 'nonzero', 'flatnonzero']:
    exec('def %s(a) : return np.%s(a.r) if hasattr(a, "compute_r") else np.%s(a)' % (rtn, rtn, rtn))
    __all__ += [rtn]

for rtn in numpy_array_creation_routines:
    exec('def %s(*args, **kwargs) : return Ch(np.%s(*args, **kwargs))' % (rtn, rtn))


class WontImplement(Exception):
    pass

for rtn in wont_implement:
    exec('def %s(*args, **kwargs) : raise WontImplement' % (rtn))
    
for rtn in not_yet_implemented:
    exec('def %s(*args, **kwargs) : raise NotImplementedError' % (rtn))

def asarray(a, dtype=None, order=None):
    assert(dtype is None or dtype is np.float64)
    assert(order is 'C' or order is None)
    if hasattr(a, 'dterms'):
        return a
    return Ch(np.asarray(a, dtype, order))

# Everythign is always c-contiguous
def ascontiguousarray(a, dtype=None): return a

# Everything is always float
asfarray = ascontiguousarray

def copy(self):
    return pickle.loads(pickle.dumps(self))    

def asfortranarray(a, dtype=None): raise WontImplement


class Simpleton(Ch):
    dterms = 'x'    
    def compute_dr_wrt(self, wrt): 
        return None



class Cross(Ch):
    dterms = 'a', 'b'
    terms = 'axisa', 'axisb', 'axisc', 'axis'
    term_order = 'a', 'b', 'axisa', 'axisb', 'axisc', 'axis'
    
    def compute_r(self):
        return np.cross(self.a.r, self.b.r, self.axisa, self.axisb, self.axisc, self.axis)
        
    
    def _load_crossprod_cache(self, h, w):
        if not hasattr(self, '_w'):
            self._w = 0
            self._h = 0

        if h!=self._h or w!=self._w:
            sz = h*w
            rng = np.arange(sz)
            self._JS = np.repeat(rng.reshape((-1,w)), w, axis=0).ravel()
            self._IS = np.repeat(rng, w)
            self._tiled_identity = np.tile(np.eye(w), (h, 1))
            self._h = h
            self._w = w
            
        return self._tiled_identity, self._IS, self._JS, 


    # Could be at least 2x faster, with some work
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return

        sz = self.a.size
        h, w = self.a.shape
        tiled_identity, IS, JS = self._load_crossprod_cache(h, w)
        
        #import time
        #tm = time.time()
        if wrt is self.a:
            rp = np.repeat(-self.b.r, w, axis=0) 
            result = np.cross(
                tiled_identity, 
                rp,
                self.axisa,
                self.axisb, 
                self.axisc, 
                self.axis)

        elif wrt is self.b:
            result = np.cross(
                np.repeat(-self.a.r, w, axis=0),
                tiled_identity,
                self.axisa,
                self.axisb, 
                self.axisc, 
                self.axis)
                
        # rng = np.arange(sz)
        # JS = np.repeat(rng.reshape((-1,w)), w, axis=0).ravel()
        # IS = np.repeat(rng, w)
        data = result.ravel()
        result = sp.csc_matrix((data, (IS,JS)), shape=(self.size, wrt.size))
        #import pdb; pdb.set_trace()
        #print 'B TOOK %es' % (time.time() -tm )
        return result
    
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return Cross(a, b, axisa, axisb, axisc, axis)
    



         

class UnaryElemwise(Ch):
    dterms = 'x'
    
    def compute_r(self):
        return self._r(self.x.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = self._d(self.x.r)
            return sp.diags([result.ravel()], [0]) if len(result)>1 else np.atleast_2d(result)


class nan_to_num(UnaryElemwise):
    _r = lambda self, x : np.nan_to_num(x)
    _d = lambda self, x : np.asarray(np.isfinite(x), np.float64)

class reciprocal(UnaryElemwise):
    _r = np.reciprocal
    _d = lambda self, x : -np.reciprocal(np.square(x))

class square(UnaryElemwise):
    _r = np.square
    _d = lambda self, x : x * 2.

def my_power(a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        return np.nan_to_num(np.power(a, b))

class sqrt(UnaryElemwise):
    _r = np.sqrt
    _d = lambda self, x : .5 * my_power(x, -0.5)

class exp(UnaryElemwise):
    _r = np.exp
    _d = np.exp    

class log(UnaryElemwise):
    _r = np.log
    _d = np.reciprocal

class sin(UnaryElemwise):
    _r = np.sin
    _d = np.cos
    
class arcsin(UnaryElemwise):
    _r = np.arcsin
    _d = lambda self, x : np.reciprocal(np.sqrt(1.-np.square(x)))
    
class cos(UnaryElemwise):
    _r = np.cos
    _d = lambda self, x : -np.sin(x)

class arccos(UnaryElemwise):
    _r = np.arccos
    _d = lambda self, x : -np.reciprocal(np.sqrt(1.-np.square(x)))

class tan(UnaryElemwise):
    _r = np.tan
    _d = lambda self, x : np.reciprocal(np.cos(x)**2.)
    
class arctan(UnaryElemwise):
    _r = np.arctan
    _d = lambda self, x : np.reciprocal(np.square(x)+1.)

# class arctan2(Ch):
#     dterms = 'x', 'y'
#
#     def compute_r(self):
#         return np.arctan2(self.y.r,self.x.r)
#
#     def compute_dr_wrt(self, wrt):
#         if self.x.r != 0:
#             return -self.y/(self.x.r**2 + self.y.r**2)
#         elif self.x.r == 0:
#             return 0

class negative(UnaryElemwise):
    _r = np.negative
    _d = lambda self, x : np.negative(np.ones_like(x))

class absolute(UnaryElemwise):
    _r = np.abs
    _d = lambda self, x : (x>0)*2-1.

abs = absolute

class clip(Ch):
    dterms = 'a'
    terms = 'a_min', 'a_max'
    term_order = 'a', 'a_min', 'a_max'
    
    def compute_r(self):
        return np.clip(self.a.r, self.a_min, self.a_max)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            result = np.asarray((self.r != self.a_min) & (self.r != self.a_max), np.float64)
            return sp.diags([result.ravel()], [0]) if len(result)>1 else np.atleast_2d(result)





def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if (dtype != None or out != None or ddof != 0 or keepdims != False):
        raise NotImplementedException('Unimplemented for non-default dtype, out, ddof, and keepdims.')
    return mean(a**2., axis=axis)
    
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if (dtype != None or out != None or ddof != 0 or keepdims != False):
        raise NotImplementedException('Unimplemented for non-default dtype, out, ddof, and keepdims.')
    return sqrt(var(a, axis=axis))
    

class SumOfSquares(Ch):
    dterms = 'x',

    def compute_r(self):
        return np.sum(self.x.r.ravel()**2.)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return self.x.r.ravel().reshape((1,-1))*2.
    
    
class divide (Ch):
    dterms = 'x1', 'x2'

    def compute_r(self):
        return self.x1.r / self.x2.r

    def compute_dr_wrt(self, wrt):
        
        if (wrt is self.x1) == (wrt is self.x2):
            return None
            
        IS, JS, input_sz, output_sz = _broadcast_setup(self.x1, self.x2, wrt)
        
        x1r, x2r = self.x1.r, self.x2.r
        if wrt is self.x1:
            data = (np.ones_like(x1r) / x2r).ravel()
        else:
            data = (-x1r / (x2r*x2r)).ravel()
            
        return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))


            

class NanDivide(divide):
    dterms = 'x1', 'x2'
    
    def compute_r(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = super(self.__class__, self).compute_r()
        shape = result.shape
        result = result.ravel()
        result[np.isinf(result)] = 0
        result[np.isnan(result)] = 0
        return result.reshape(shape)
        
    def compute_dr_wrt(self, wrt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = super(self.__class__, self).compute_dr_wrt(wrt)        
        if result is not None:
            result = result.copy()
            if sp.issparse(result):
                result.data[np.isinf(result.data)] = 0
                result.data[np.isnan(result.data)] = 0
                return result
            else:
                rr = result.ravel()
                rr[np.isnan(rr)] = 0.
                rr[np.isinf(rr)] = 0.
                return result


def shape(a):
    return a.shape if hasattr(a, 'shape') else np.shape(a)    


_bs_setup_data1 = {}
_bs_setup_data2 = {}
def _broadcast_matrix(a, b, wrt, data):
    global _bs_setup_data1, _bs_setup_data2

    if len(set((a.shape, b.shape))) == 1:
        uid = a.shape
        if uid not in _bs_setup_data1:
            asz = a.size
            IS = np.arange(asz)
            _bs_setup_data1[uid] = sp.csc_matrix((np.empty(asz), (IS, IS)), shape=(asz, asz))
        result = copy_copy(_bs_setup_data1[uid])
        if isinstance(data, np.ndarray):
            result.data = data.ravel()
        else: # assumed scalar
            result.data = np.empty(result.nnz)
            result.data.fill(data)
    else:
        uid = (a.shape, b.shape, wrt is a, wrt is b)
        if uid not in _bs_setup_data2:
            input_sz = wrt.size
            output_sz = np.broadcast(a.r, b.r).size
            a2 = np.arange(a.size).reshape(a.shape) if wrt is a else np.zeros(a.shape)
            b2 = np.arange(b.size).reshape(b.shape) if (wrt is b and wrt is not a) else np.zeros(b.shape)
            IS = np.arange(output_sz)
            JS = np.asarray((np.add(a2,b2)).ravel(), np.uint32)

            _bs_setup_data2[uid] = sp.csc_matrix((np.arange(IS.size), (IS, JS)), shape=(output_sz, input_sz))

        result = copy_copy(_bs_setup_data2[uid])
        if isinstance(data, np.ndarray):
            result.data = data[result.data]
        else: # assumed scalar
            result.data = np.empty(result.nnz)
            result.data.fill(data)

    if np.prod(result.shape) == 1:
        return np.array(data)
    else:
        return result



broadcast_shape_cache = {}
def broadcast_shape(a_shape, b_shape):
    global broadcast_shape_cache

    raise Exception('This function is probably a bad idea, because shape is not cached and overquerying can occur.')

    uid = (a_shape, b_shape)

    if uid not in broadcast_shape_cache:
        la = len(a_shape)
        lb = len(b_shape)
        ln = la if la > lb else lb

        ash = np.ones(ln, dtype=np.uint32)
        bsh = np.ones(ln, dtype=np.uint32)
        ash[-la:] = a_shape
        bsh[-lb:] = b_shape

        our_result = np.max(np.vstack((ash, bsh)), axis=0)

        if False:
            numpy_result = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape
            #print 'aaa' + str(our_result)
            #print 'bbb' + str(numpy_result)
            if not np.array_equal(our_result, numpy_result):
                import pdb; pdb.set_trace()
            assert(np.array_equal(our_result, numpy_result))

        broadcast_shape_cache[uid] = tuple(our_result)
    return broadcast_shape_cache[uid]


def _broadcast_setup(a, b, wrt):
    if len(set((a.shape, b.shape))) == 1:
        asz = a.size
        IS = np.arange(asz)
        return IS, IS, asz, asz
    input_sz = wrt.r.size
    output_sz = np.broadcast(a.r, b.r).size
    a2 = np.arange(a.size).reshape(a.shape) if wrt is a else np.zeros(a.shape)
    b2 = np.arange(b.size).reshape(b.shape) if (wrt is b and wrt is not a) else np.zeros(b.shape)    
    IS = np.arange(output_sz)
    JS = np.asarray((np.add(a2,b2)).ravel(), np.uint32)
    return IS, JS, input_sz, output_sz


  
class add(Ch):
    dterms = 'a', 'b'
        
    def compute_r(self):
        return self.a.r + self.b.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None

        m = 2. if self.a is self.b else 1.
        return _broadcast_matrix(self.a, self.b, wrt, m)


            
            
class subtract(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        return self.a.r - self.b.r

    def compute_dr_wrt(self, wrt):
        if (wrt is self.a) == (wrt is self.b):
            return None

        m = 1. if wrt is self.a else -1.
        return _broadcast_matrix(self.a, self.b, wrt, m)


    
    
    
class ch_power (Ch):
    """Given vector \f$x\f$, computes \f$x^2\f$ and \f$\frac{dx^2}{x}\f$"""
    dterms = 'x', 'pow'

    def compute_r(self):
        return self.safe_power(self.x.r, self.pow.r)

    def compute_dr_wrt(self, wrt):

        if wrt is not self.x and wrt is not self.pow:
            return None
            
        x, pow = self.x.r, self.pow.r
        result = []
        if wrt is self.x:
            result.append(pow * self.safe_power(x, pow-1.))
        if wrt is self.pow:
            result.append(np.log(x) * self.safe_power(x, pow))
            
        data = reduce(lambda x, y : x + y, result).ravel()

        return _broadcast_matrix(self.x, self.pow, wrt, data)

    
    def safe_power(self, x, sigma):
        # This throws a RuntimeWarning sometimes, but then the infs are corrected below
        result = np.power(x, sigma)
        result.ravel()[np.isinf(result.ravel())] = 0
        return result



        

class A_extremum(Ch):
    """Superclass for various min and max subclasses"""
    dterms = 'a'
    terms = 'axis'
    term_order = 'a', 'axis'

    def f(self, axis):    raise NotImplementedError
    def argf(self, axis): raise NotImplementedError

    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
    
    def compute_r(self):        
        return self.f(self.a.r, axis=self.axis)
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:

            mn, stride = self._stride_for_axis(self.axis, self.a.r)
            JS = np.asarray(np.round(mn + stride * self.argf(self.a.r, axis=self.axis)), dtype=np.uint32).ravel()
            IS = np.arange(JS.size)
            data = np.ones(JS.size)
            
            if self.r.size * wrt.r.size == 1:
                return data.ravel()[0]
            return sp.csc_matrix((data, (IS, JS)), shape = (self.r.size, wrt.r.size))
            
    def _stride_for_axis(self,axis, mtx):
        if axis is None:
            mn = np.array([0])
            stride = np.array([1])
        else:    
            # TODO: make this less expensive. Shouldn't need to call
            # np.amin here probably
            idxs = np.arange(mtx.size).reshape(mtx.shape)
            mn = np.amin(idxs, axis=axis)
            stride = np.array(mtx.strides)
            stride //= np.min(stride) # go from bytes to num elements
            stride = stride[axis]
        return mn, stride



#Reordering

__all__ += ['sort', 'tile', 'repeat', 'transpose', 'rollaxis', 'swapaxes', 'reshape', 'Select',
           'atleast_1d', 'atleast_2d', 'atleast_3d', 'squeeze', 'expand_dims', 'fliplr', 'flipud',
           'concatenate', 'vstack', 'hstack', 'dstack', 'ravel', 'diag', 'diagflat', 'roll', 'rot90']

import weakref


# Classes deriving from "Permute" promise to only reorder/reshape
class Permute(Ch):
    pass

def ravel(a, order='C'):
    assert(order=='C')
    if isinstance (a, np.ndarray):
        self = Ch(a)

    return reshape(a=a, newshape=(-1,))

class Reorder(Permute):
    dterms = 'a',

    def on_changed(self, which):
        if not hasattr(self, 'dr_lookup'):
            self.dr_lookup = {}
            
    def compute_r(self):
        return self.reorder(self.a.r)
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            if False:
                from scipy.sparse.linalg.interface import LinearOperator
                return LinearOperator((self.size, wrt.size), lambda x : self.reorder(x.reshape(self.a.shape)).ravel())
            else:
                a = self.a
                asz = a.size
                ashape = a.shape
                key = self.unique_reorder_id()
                if key not in self.dr_lookup or key is None:
                    JS = self.reorder(np.arange(asz).reshape(ashape))
                    IS = np.arange(JS.size)
                    data = np.ones_like(IS)
                    shape = JS.shape
                    self.dr_lookup[key] = sp.csc_matrix((data, (IS, JS.ravel())), shape=(self.r.size, wrt.r.size))
                return self.dr_lookup[key]
                 
class Sort(Reorder):
    dterms = 'a'
    terms = 'axis', 'kind', 'order'
    
    def reorder(self, a): return np.sort(a, self.axis, self.kind, self.order)
    def unique_reorder_id(self): return None

def sort(a, axis=-1, kind='quicksort', order=None):
    return Sort(a=a, axis=axis, kind=kind, order=order)
    
    
class Tile(Reorder):
    dterms = 'a',
    terms = 'reps',
    term_order = 'a', 'reps'
    
    def reorder(self, a): return np.tile(a, self.reps)
    def unique_reorder_id(self): return (self.a.shape, tuple(self.reps))
    
def tile(A, reps):
    return Tile(a=A, reps=reps)
    
    
class Diag(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.diag(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)
    
def diag(v, k=0):
    return Diag(a=v, k=k)
    
class DiagFlat(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.diagflat(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)

def diagflat(v, k=0):
    return DiagFlat(a=v, k=k)
    
    
class Repeat(Reorder):
    dterms = 'a',
    terms = 'repeats', 'axis'
    
    def reorder(self, a): return np.repeat(a, self.repeats, self.axis)
    def unique_reorder_id(self): return (self.repeats, self.axis)

def repeat(a, repeats, axis=None):
    return Repeat(a=a, repeats=repeats, axis=axis)

class transpose(Reorder):        
    dterms = 'a'
    terms = 'axes'
    term_order = 'a', 'axes'

    def reorder(self, a):    return np.require(np.transpose(a, axes=self.axes), requirements='C')        
    def unique_reorder_id(self): return (self.a.shape, None if self.axes is None else tuple(self.axes))
    def on_changed(self, which):
        if not hasattr(self, 'axes'):
            self.axes = None
        super(self.__class__, self).on_changed(which)
    
class rollaxis(Reorder):        
    dterms = 'a'
    terms = 'axis', 'start'
    term_order = 'a', 'axis', 'start'

    def reorder(self, a):    return np.rollaxis(a, axis=self.axis, start=self.start)
    def unique_reorder_id(self): return (self.a.shape, self.axis, self.start)
    def on_changed(self, which):
        if not hasattr(self, 'start'):
            self.start = 0
        super(self.__class__, self).on_changed(which)

class swapaxes(Reorder):        
    dterms = 'a'
    terms = 'axis1', 'axis2'
    term_order = 'a', 'axis1', 'axis2'

    def reorder(self, a):    return np.swapaxes(a, axis1=self.axis1, axis2=self.axis2)
    def unique_reorder_id(self): return (self.a.shape, self.axis1, self.axis2)
    


class Roll(Reorder):
    dterms = 'a',
    terms = 'shift', 'axis'
    term_order = 'a', 'shift', 'axis'
    
    def reorder(self, a): return np.roll(a, self.shift, self.axis)
    def unique_reorder_id(self): return (self.shift, self.axis)
    
def roll(a, shift, axis=None):
    return Roll(a, shift, axis)

class Rot90(Reorder):
    dterms = 'a',
    terms = 'k',
    
    def reorder(self, a): return np.rot90(a, self.k)
    def unique_reorder_id(self): return (self.a.shape, self.k)

def rot90(m, k=1):
    return Rot90(a=m, k=k)

class Reshape(Permute):
    dterms = 'a',
    terms = 'newshape',
    term_order= 'a', 'newshape'

    def compute_r(self):
        return self.a.r.reshape(self.newshape)

    def compute_dr_wrt(self, wrt):
        if wrt is self.a:
            return sp.eye(self.a.size, self.a.size)
        #return self.a.dr_wrt(wrt)

# def reshape(a, newshape):
#     if isinstance(a, Reshape) and a.newshape == newshape:
#         return a
#     return Reshape(a=a, newshape=newshape)
def reshape(a, newshape):
    while isinstance(a, Reshape):
        a = a.a
    return Reshape(a=a, newshape=newshape)

# class And(Ch):
#     dterms = 'x1', 'x2'
#
#     def compute_r(self):
#         if True:
#             needs_work = [self.x1, self.x2]
#             done = []
#             while len(needs_work) > 0:
#                 todo = needs_work.pop()
#                 if isinstance(todo, And):
#                     needs_work += [todo.x1, todo.x2]
#                 else:
#                     done = [todo] + done
#             return np.concatenate([d.r.ravel() for d in done])
#         else:
#             return np.concatenate((self.x1.r.ravel(), self.x2.r.ravel()))
#
#     # This is only here for reverse mode to work.
#     # Most of the time, the overridden dr_wrt is callpath gets used.
#     def compute_dr_wrt(self, wrt):
#
#         if wrt is not self.x1 and wrt is not self.x2:
#             return
#
#         input_len = wrt.r.size
#         x1_len = self.x1.r.size
#         x2_len = self.x2.r.size
#
#         mtxs = []
#         if wrt is self.x1:
#             mtxs.append(sp.spdiags(np.ones(x1_len), 0, x1_len, x1_len))
#         else:
#             mtxs.append(sp.csc_matrix((x1_len, input_len)))
#
#         if wrt is self.x2:
#             mtxs.append(sp.spdiags(np.ones(x2_len), 0, x2_len, x2_len))
#         else:
#             mtxs.append(sp.csc_matrix((x2_len, input_len)))
#
#
#         if any([sp.issparse(mtx) for mtx in mtxs]):
#             result = sp.vstack(mtxs, format='csc')
#         else:
#             result = np.vstack(mtxs)
#
#         return result
#
#     def dr_wrt(self, wrt, want_stacks=False, reverse_mode=False):
#         self._call_on_changed()
#
#         input_len = wrt.r.size
#         x1_len = self.x1.r.size
#         x2_len = self.x2.r.size
#
#         mtxs = []
#         if wrt is self.x1:
#             mtxs.append(sp.spdiags(np.ones(x1_len), 0, x1_len, x1_len))
#         else:
#             if isinstance(self.x1, And):
#                 tmp_mtxs = self.x1.dr_wrt(wrt, want_stacks=True, reverse_mode=reverse_mode)
#                 for mtx in tmp_mtxs:
#                     mtxs.append(mtx)
#             else:
#                 mtxs.append(self.x1.dr_wrt(wrt, reverse_mode=reverse_mode))
#             if mtxs[-1] is None:
#                 mtxs[-1] = sp.csc_matrix((x1_len, input_len))
#
#         if wrt is self.x2:
#             mtxs.append(sp.spdiags(np.ones(x2_len), 0, x2_len, x2_len))
#         else:
#             if isinstance(self.x2, And):
#                 tmp_mtxs = self.x2.dr_wrt(wrt, want_stacks=True, reverse_mode=reverse_mode)
#                 for mtx in tmp_mtxs:
#                     mtxs.append(mtx)
#             else:
#                 mtxs.append(self.x2.dr_wrt(wrt, reverse_mode=reverse_mode))
#             if mtxs[-1] is None:
#                 mtxs[-1] = sp.csc_matrix((x2_len, input_len))
#
#         if want_stacks:
#             return mtxs
#         else:
#             if any([sp.issparse(mtx) for mtx in mtxs]):
#                 result = sp.vstack(mtxs, format='csc')
#             else:
#                 result = np.vstack(mtxs)
#
#         return result
        
class Select(Permute):
    terms = ['idxs', 'preferred_shape']
    dterms = ['a']
    term_order = 'a', 'idxs', 'preferred_shape'

    def compute_r(self):
        result = self.a.r.ravel().take(self.idxs).copy()
        if hasattr(self, 'preferred_shape'):
            return result.reshape(self.preferred_shape)
        else:
            return result

    def compute_dr_wrt(self, obj):
        if obj is self.a:
            if not hasattr(self, '_dr_cached'):
                IS = np.arange(len(self.idxs)).ravel()
                JS = self.idxs.ravel()
                ij = np.vstack((IS.reshape((1,-1)), JS.reshape((1,-1))))
                data = np.ones(len(self.idxs))
                self._dr_cached = sp.csc_matrix((data, ij), shape=(len(self.idxs), len(self.a.r.ravel())))
            return self._dr_cached
        
    def on_changed(self, which):
        if hasattr(self, '_dr_cached'):
            if 'idxs' in which or self.a.r.size != self._dr_cached.shape[1]:
                del self._dr_cached

    

class AtleastNd(Ch):
    dterms = 'x'
    terms = 'ndims'
    
    def compute_r(self):
        xr = self.x.r
        if self.ndims == 1:
            target_shape = np.atleast_1d(xr).shape
        elif self.ndims == 2:
            target_shape = np.atleast_2d(xr).shape
        elif self.ndims == 3:
            target_shape = np.atleast_3d(xr).shape
        else:
            raise Exception('Need ndims to be 1, 2, or 3.')

        return xr.reshape(target_shape)
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return 1

def atleast_nd(ndims, *arys):
    arys = [AtleastNd(x=ary, ndims=ndims) for ary in arys]
    return arys if len(arys) > 1 else arys[0]

def atleast_1d(*arys):
    return atleast_nd(1, *arys)

def atleast_2d(*arys):
    return atleast_nd(2, *arys)

def atleast_3d(*arys):
    return atleast_nd(3, *arys)
    
def squeeze(a, axis=None):
    if isinstance(a, np.ndarray):
        return np.squeeze(a, axis)
    shape = np.squeeze(a.r, axis).shape
    return a.reshape(shape)
    
def expand_dims(a, axis):
    if isinstance(a, np.ndarray):
        return np.expand_dims(a, axis)
    shape = np.expand_dims(a.r, axis).shape
    return a.reshape(shape)
    
def fliplr(m):
    return m[:,::-1]
    
def flipud(m):
    return m[::-1,...]
    
class Concatenate(Ch):

    def on_changed(self, which):
        if not hasattr(self, 'dr_cached'):
            self.dr_cached = weakref.WeakKeyDictionary()
            
    @property
    def our_terms(self):
        if not hasattr(self, '_our_terms'):
            self._our_terms = [getattr(self, s) for s in self.dterms]
        return self._our_terms
    
    def __getstate__(self):
        # Have to get rid of WeakKeyDictionaries for serialization
        if hasattr(self, 'dr_cached'):
            del self.dr_cached
        return super(self.__class__, self).__getstate__()
    
    def compute_r(self):
        return np.concatenate([t.r for t in self.our_terms], axis=self.axis)
                    
    @property
    def everything(self):
        if not hasattr(self, '_everything'):
            self._everything = np.arange(self.r.size).reshape(self.r.shape)
            self._everything = np.swapaxes(self._everything, self.axis, 0)
        return self._everything
    
    def compute_dr_wrt(self, wrt):
        if wrt in self.dr_cached:
            return self.dr_cached[wrt]
        
        if wrt not in self.our_terms:
            return
                        
        _JS = np.arange(wrt.size)
        _data = np.ones(wrt.size)
        
        IS = []
        JS = []
        data = []
        
        offset = 0
        for term in self.our_terms:
            tsz = term.shape[self.axis]
            if term is wrt:
                JS += [_JS]
                data += [_data]
                IS += [np.swapaxes(self.everything[offset:offset+tsz], self.axis, 0).ravel()]
            offset += tsz
        IS   = np.concatenate(IS).ravel()
        JS   = np.concatenate(JS).ravel()
        data = np.concatenate(data)
                
        self.dr_cached[wrt] = sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.size))
        return self.dr_cached[wrt]

def expand_concatenates(mtxs, axis=0):
    mtxs = list(mtxs)
    done = []
    while len(mtxs) > 0:
        mtx = mtxs.pop(0)
        if isinstance(mtx, Concatenate) and mtx.axis == axis:
            mtxs = [getattr(mtx, s) for s in mtx.dterms] + mtxs
        else:
            done.append(mtx)
    return done


def concatenate(mtxs, axis=0):

    mtxs = expand_concatenates(mtxs, axis)

    result = Concatenate()
    result.dterms = []
    for i, mtx in enumerate(mtxs):
        result.dterms.append('m%d' % (i,))
        setattr(result, result.dterms[-1], mtx)
    result.axis = axis
    return result
    
def hstack(mtxs):
    return concatenate(mtxs, axis=1)

def vstack(mtxs):
    return concatenate([atleast_2d(m) for m in mtxs], axis=0)

def dstack(mtxs):
    return concatenate([atleast_3d(m) for m in mtxs], axis=2)


api_not_implemented = ['choice','bytes','shuffle','permutation']

api_wrapped_simple = [
    # simple random data
    'rand','randn','randint','random_integers','random_sample','random','ranf','sample',
    
    # distributions
    'beta','binomial','chisquare','dirichlet','exponential','f','gamma','geometric','gumbel','hypergeometric',
    'laplace','logistic','lognormal','logseries','multinomial','multivariate_normal','negative_binomial',
    'noncentral_chisquare','noncentral_f','normal','pareto','poisson','power','rayleigh','standard_cauchy',
    'standard_exponential','standard_gamma','standard_normal','standard_t','triangular','uniform','vonmises',
    'wald','weibull','zipf']
    
api_wrapped_direct = ['seed', 'get_state', 'set_state']

for rtn in api_wrapped_simple:
    exec('def %s(*args, **kwargs) : return Ch(np.random.%s(*args, **kwargs))' % (rtn, rtn))

for rtn in api_wrapped_direct:
    exec('%s = np.random.%s' % (rtn, rtn))
    
__all__ += api_wrapped_simple + api_wrapped_direct


__all__ += ['inv', 'svd', 'det', 'slogdet', 'pinv', 'lstsq', 'norm']


#Linalg

try:
    asarray = asarray
    import inspect
    exec(''.join(inspect.getsourcelines(np.linalg.tensorinv)[0]))
    __all__.append('tensorinv')
except: pass

def norm(x, ord=None, axis=None):
    if ord is not None or axis is not None:
        raise NotImplementedError("'ord' and 'axis' should be None for now.")

    return sqrt(sum(x**2))

# This version works but derivatives are too slow b/c of nested loop in Svd implementation.
# def lstsq(a, b):
#     u, s, v = Svd(a)
#     x = (v.T / s).dot(u.T.dot(b))
#     residuals = NotImplementedError # sum((a.dot(x) - b)**2, axis=0)
#     rank = NotImplementedError
#     s = NotImplementedError
#     return x, residuals, rank, s

def lstsq(a, b, rcond=-1):
    if rcond != -1:
        raise Exception('non-default rcond not yet implemented')
        
    x = Ch(lambda a, b : pinv(a).dot(b))
    x.a = a
    x.b = b
    residuals = sum(  (x.a.dot(x) - x.b) **2 , axis=0)
    rank = NotImplementedError
    s = NotImplementedError
    
    return x, residuals, rank, s

def Svd(x, full_matrices=0, compute_uv=1):
    
    if full_matrices != 0:
        raise Exception('full_matrices must be 0')
    if compute_uv != 1:
        raise Exception('compute_uv must be 1')
        
    need_transpose = x.shape[0] < x.shape[1]
    
    if need_transpose:
        x = x.T
        
    svd_d = SvdD(x=x)
    svd_v = SvdV(x=x, svd_d=svd_d)
    svd_u = SvdU(x=x, svd_d=svd_d, svd_v=svd_v)

    if need_transpose:
        return svd_v, svd_d, svd_u.T
    else:
        return svd_u, svd_d, svd_v.T

    
class Pinv(Ch):
    dterms = 'mtx'
    
    def on_changed(self, which):
        mtx = self.mtx
        if mtx.shape[1] > mtx.shape[0]:
            result = mtx.T.dot(Inv(mtx.dot(mtx.T)))
        else:
            result = Inv(mtx.T.dot(mtx)).dot(mtx.T)
        self._result = result
        
    def compute_r(self):
        return self._result.r
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.mtx:
            return self._result.dr_wrt(self.mtx)
        
# Couldn't make the SVD version of pinv work yet...
#
# class Pinv(Ch):
#     dterms = 'mtx'
#     
#     def on_changed(self, which):
#         u, s, v = Svd(self.mtx)
#         result = (v.T * (NanDivide(1.,row(s)))).dot(u.T)
#         self.add_dterm('_result', result)
# 
#     def compute_r(self):
#         return self._result.r
#         
#     def compute_dr_wrt(self, wrt):
#         if wrt is self._result:
#             return 1



class LogAbsDet(Ch):
    dterms = 'x'
    
    def on_changed(self, which):
        self.sign, self.slogdet = np.linalg.slogdet(self.x.r)
    
    def compute_r(self):
        return self.slogdet
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:         
            return np.linalg.inv(self.x.r).T.reshape((1,-1))

class SignLogAbsDet(Ch):
    dterms = 'logabsdet',
    
    def compute_r(self):
        _ = self.logabsdet.r
        return self.logabsdet.sign
        
    def compute_dr_wrt(self, wrt):
        return None


class Det(Ch):
    dterms = 'x'

    def compute_r(self):
        return np.linalg.det(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return self.r * np.linalg.inv(self.x.r).T.reshape((1,-1))


class Inv(Ch):
    dterms = 'a'
    
    def compute_r(self):
        return np.linalg.inv(self.a.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a:
            return None
    
        Ainv = self.r

        if Ainv.ndim <= 2:
            return -np.kron(Ainv, Ainv.T)
        else:
            Ainv = np.reshape(Ainv,  (-1, Ainv.shape[-2], Ainv.shape[-1]))
            AinvT = np.rollaxis(Ainv, -1, -2)
            AinvT = np.reshape(AinvT, (-1, AinvT.shape[-2], AinvT.shape[-1]))
            result = np.dstack([-np.kron(Ainv[i], AinvT[i]).T for i in range(Ainv.shape[0])]).T
            result = sp.block_diag(result)

        return result


class SvdD(Ch):
    dterms = 'x'

    @depends_on('x')
    def UDV(self):
        result = np.linalg.svd(self.x.r, full_matrices=False)
        result = [result[0], result[1], result[2].T]
        result[1][np.abs(result[1]) < np.spacing(1)] = 0.
        return result
    
    def compute_r(self):
        return self.UDV[1]
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        
        u, d, v = self.UDV
        shp = self.x.r.shape
        u = u[:shp[0], :shp[1]]
        v = v[:shp[1], :d.size]
        
        result = np.einsum('ik,jk->kij', u, v)
        result = result.reshape((result.shape[0], -1))
        return result
    
    
class SvdV(Ch):
    terms = 'svd_d'
    dterms = 'x'
    
    def compute_r(self):
        return self.svd_d.UDV[2]

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        
        U,_D,V = self.svd_d.UDV
        
        shp = self.svd_d.x.r.shape
        mxsz = max(shp[0], shp[1])
        #mnsz = min(shp[0], shp[1])
        D = np.zeros(mxsz)
        D[:_D.size] = _D

        omega = np.zeros((shp[0], shp[1], shp[1], shp[1]))

        M = shp[0]
        N = shp[1]
        
        assert(M >= N)
        
        for i in range(shp[0]):
            for j in range(shp[1]):
                for k in range(N):
                    for l in range(k+1, N):
                        mtx = np.array([
                            [D[l],D[k]],
                            [D[k],D[l]]])
                    
                        rhs = np.array([U[i,k]*V[j,l], -U[i,l]*V[j,k]])
                        result = np.linalg.solve(mtx, rhs)
                        
                        omega[i,j,k,l] =  result[1]
                        omega[i,j,l,k] = -result[1]
                   
        #print 'v size is %s' % (str(V.shape),)
        #print 'v omega size is %s' % (str(omega.shape),)
        assert(V.shape[1] == omega.shape[2])
        return np.einsum('ak,ijkl->alij', -V, omega).reshape((self.r.size, wrt.r.size))
        
        
class SvdU(Ch):
    dterms = 'x'
    terms = 'svd_d', 'svd_v'

    def compute_r(self):
        return self.svd_d.UDV[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            # return (
            #     self.svd_d.x.dot(self.svd_v)
            #     /
            #     self.svd_d.reshape((1,-1))
            #     ).dr_wrt(self.svd_d.x)
            return (
                NanDivide(
                    self.svd_d.x.dot(self.svd_v),
                    self.svd_d.reshape((1,-1)))
                ).dr_wrt(self.svd_d.x)
    

inv = Inv
svd = Svd
det = Det
pinv = Pinv

def slogdet(*args):
    n = len(args)
    if n == 1:
        r2 = LogAbsDet(x=args[0])
        r1 = SignLogAbsDet(r2)
        return r1, r2
    else:
        r2 = [LogAbsDet(x=arg) for arg in args]
        r1 = [SignLogAbsDet(r) for r in r2]
        r2 = concatenate(r2)
        return r1, r2


#Ch Ops

class mean(Ch):
    dterms = 'x',
    terms  = 'axis',
    term_order = 'x', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
        if not hasattr(self, 'dr_cache'):
            self.dr_cache = {}

    def compute_r(self):
        return np.array(np.mean(self.x.r, axis=self.axis))

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        if self.axis == None:
            return np.ones((len(self.x.r),1))/len(self.x.r)
        else:
            uid = tuple(list(self.x.shape) + [self.axis])
            if uid not in self.dr_cache:
                idxs_presum = np.arange(self.x.size).reshape(self.x.shape)
                idxs_presum = np.rollaxis(idxs_presum, self.axis, 0)
                idxs_postsum = np.arange(self.r.size).reshape(self.r.shape)
                tp = np.ones(idxs_presum.ndim, np.int64)
                tp[0] = idxs_presum.shape[0]
                idxs_postsum = np.tile(idxs_postsum, tp)
                data = np.ones(idxs_postsum.size) / self.x.shape[self.axis]
                result = sp.csc_matrix((data, (idxs_postsum.ravel(), idxs_presum.ravel())), (self.r.size, wrt.size))
                self.dr_cache[uid] = result
            return self.dr_cache[uid]

class sum(Ch):
    dterms = 'x',
    terms  = 'axis',
    term_order = 'x', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
        if not hasattr(self, 'dr_cache'):
            self.dr_cache = {}

    def compute_r(self):
        return np.sum(self.x.r, axis=self.axis)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x:
            return
        if self.axis == None:
            return np.ones((1,len(self.x.r.ravel())))
        else:
            uid = tuple(list(self.x.shape) + [self.axis])
            if uid not in self.dr_cache:
                idxs_presum = np.arange(self.x.size).reshape(self.x.shape)
                idxs_presum = np.rollaxis(idxs_presum, self.axis, 0)
                idxs_postsum = np.arange(self.r.size).reshape(self.r.shape)
                tp = np.ones(idxs_presum.ndim, dtype=np.uint32)
                tp[0] = idxs_presum.shape[0]
                idxs_postsum = np.tile(idxs_postsum, tp)
                data = np.ones(idxs_postsum.size)
                result = sp.csc_matrix((data, (idxs_postsum.ravel(), idxs_presum.ravel())), (self.r.size, wrt.size))
                self.dr_cache[uid] = result
            return self.dr_cache[uid]
            

class floor(Simpleton):
    def compute_r(self): return np.floor(self.x.r)

class ceil(Simpleton):
    def compute_r(self): return np.ceil(self.x.r)

class sign(Simpleton):
    def compute_r(self): return np.sign(self.x.r)

class dot(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        try:
            return self.a.r.dot(self.b.r)
        except:
            import pdb; pdb.set_trace()
    
    def compute_d1(self):
        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        ar = self.a.r.reshape((1,-1)) if len(self.a.r.shape)<2 else self.a.r.reshape((-1, self.a.r.shape[-1]))
        br = self.b.r.reshape((-1,1)) if len(self.b.r.shape)<2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if ar.ndim <= 2:
            return sp.kron(sp.eye(ar.shape[0], ar.shape[0]),br.T)
        else:
            raise NotImplementedError

    def compute_d2(self):

        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        ar = self.a.r.reshape((1,-1)) if len(self.a.r.shape)<2 else self.a.r.reshape((-1, self.a.r.shape[-1]))
        br = self.b.r.reshape((-1,1)) if len(self.b.r.shape)<2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if br.ndim <= 1:
            return self.ar
        elif br.ndim <= 2:
            return sp.kron(ar, sp.eye(br.shape[1],br.shape[1]))
        else:
            raise NotImplementedError
            
    
    def compute_dr_wrt(self, wrt):

        if wrt is self.a and wrt is self.b:
            return self.compute_d1() + self.compute_d2()
        elif wrt is self.a:
            return self.compute_d1()
        elif wrt is self.b:
            return self.compute_d2()
  
class cumsum(Ch):
    dterms = 'a'
    terms = 'axis'
    term_order = 'a', 'axis'
    
    def on_changed(self, which):
        if not hasattr(self, 'axis'):
            self.axis = None
    
    def compute_r(self):
        return np.cumsum(self.a.r, axis=self.axis)
        
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a:
            return None
        
        if self.axis is not None:
            raise NotImplementedError
            
        IS = np.tile(np.arange(self.a.size).reshape((1,-1)), (self.a.size, 1))
        JS = IS.T
        IS = IS.ravel()
        JS = JS.ravel()
        which = IS >= JS
        IS = IS[which]
        JS = JS[which]
        data = np.ones_like(IS)
        result = sp.csc_matrix((data, (IS, JS)), shape=(self.a.size, self.a.size))
        return result


class amax(A_extremum):
    def f(self, *args, **kwargs):    return np.amax(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.argmax(*args, **kwargs)
    
max = amax    

class amin(A_extremum):
    def f(self, *args, **kwargs):    return np.amin(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.argmin(*args, **kwargs)

min = amin

class nanmin(A_extremum):
    def f(self, *args, **kwargs):    return np.nanmin(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.nanargmin(*args, **kwargs)

class nanmax(A_extremum):
    def f(self, *args, **kwargs):    return np.nanmax(*args, **kwargs)
    def argf(self, *args, **kwargs): return np.nanargmax(*args, **kwargs)
    

class Extremum(Ch):
    dterms = 'a','b'
    
    def compute_r(self): return self.f(self.a.r, self.b.r)
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None
                
        IS, JS, input_sz, output_sz = _broadcast_setup(self.a, self.b, wrt)
        if wrt is self.a:
            whichmax = (self.r == self.f(self.a.r, self.b.r-self.f(1,-1))).ravel()
        else:
            whichmax = (self.r == self.f(self.b.r, self.a.r-self.f(1,-1))).ravel()
        IS = IS[whichmax]
        JS = JS[whichmax]
        data = np.ones(JS.size)
        
        return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))

class maximum(Extremum):
    def f(self, a, b): return np.maximum(a, b)
    
class minimum(Extremum):
    def f(self, a, b): return np.minimum(a, b)

 
class multiply(Ch):
    dterms = 'a', 'b'

    def compute_r(self):
        return self.a.r * self.b.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.a and wrt is not self.b:
            return None
        
        a2 = self.a.r if wrt is self.b else np.ones(self.a.shape)
        b2 = self.b.r if (wrt is self.a and wrt is not self.b) else np.ones(self.b.shape)
        data = (a2 * b2).ravel()
        
        if self.a is self.b:
            data *= 2.

        return _broadcast_matrix(self.a, self.b, wrt, data)


       
      
class BinaryElemwiseNoDrv(Ch):
    dterms = 'x1', 'x2'
    
    def compute_r(self):
        return self._f(self.x1.r, self.x2.r)
    
    def compute_dr_wrt(self, wrt):
        return None
    
class greater(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.greater(a,b)

class greater_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.greater_equal(a,b)

class less(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.less(a,b)

class less_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.less_equal(a,b)
    
class equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.equal(a,b)

class not_equal(BinaryElemwiseNoDrv):
    def _f(self, a, b): return np.not_equal(a,b)
    
def nonzero(a):
    if hasattr(a, 'compute_r'):
        a = a.r
    return np.nonzero(a)

try:
    import inspect
    exec(''.join(inspect.getsourcelines(np.tensordot)[0]))
    __all__.append('tensordot')
except: pass


class tst(Ch):
    dterms = ['a', 'b', 'c']
    
    def compute_r(self):
        return self.a.r + self.b.r + self.c.r
        
    def compute_dr_wrt(self, wrt):
        return 1

def main():
    foo = tst
    
    x10 = Ch(10)
    x20 = Ch(20)
    x30 = Ch(30)
    
    tmp = ChLambda(lambda x, y, z: Ch(1) + Ch(2) * Ch(3) + 4)
    print(tmp.dr_wrt(tmp.x))
    import pdb; pdb.set_trace()
    #a(b(c(d(e(f),g),h)))
    
    blah = tst(x10, x20, x30)
    
    print(blah.r)


    print(foo)
    
    import pdb; pdb.set_trace()
    
    # import unittest
    # from test_ch import TestCh
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCh)
    # unittest.TextTestRunner(verbosity=2).run(suite)
        


if __name__ == '__main__':
    main()

