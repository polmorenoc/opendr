#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['minimize']

import time
import math
import sys
import time
import numpy as np
from numpy.linalg import norm

from chumpy import ch, utils
from chumpy.utils import row, col

import scipy.sparse as sp
import scipy.sparse
import scipy.optimize
from scipy.sparse.linalg.interface import LinearOperator
import collections
import chumpy.minimize_ras as min_ras
# import probLineSearch as pls
# import ipdb


def vstack(x):
    x = [a if not isinstance(a, LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.vstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.vstack(x)
def hstack(x):
    x = [a if not isinstance(a, LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.hstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.hstack(x)


# Nelder-Mead
# Powell
# CG
# BFGS
# Newton-CG
# Anneal
# L-BFGS-B
# TNC
# COBYLA
# SLSQP
# dogleg
# trust-ncg

def chFuncProb(fun, grad, var_f, var_df, args):
    def funValues(X):
        f = fun(X, *args)
        df = grad(X, *args)

        return f,df, var_f, var_df
    return funValues



# def probLineSearchMin(x0, fun, grad, args, df_vars, on_step=None, maxnumfuneval=None, verbose=0):

#     def call_cb():
#         if on_step is not None:
#             on_step(fun)

#     f = fun(x0, *args)
#     df = grad(x0, *args)

#     var_f = 0.001*np.ones(f.shape)
#     var_df = df_vars

#     ff = chFuncProb(fun, grad, var_f, var_df, args)

#     # f,df, var_f, var_df = ff(x0)
#     paras = []
#     search_direction = - df   # search direction (not normalized)
#     xt               = x0     # current position
#     alpha0           = 0.01   # initial step size

#     outs = {}
#     outs['counter'] = 1 # counts # function evaluations

#     path            = [x0]
#     function_values = [f]
#     oldf = np.inf
#     while outs['counter'] < maxnumfuneval and np.abs(oldf - f) > 10e-5:
#         oldf = f
#         [outs, alpha0, f, df, xt, var_f, var_df] = pls.probLineSearch(ff, xt, f, df, search_direction, alpha0, 0, outs, paras, var_f, var_df)
#         # alpha0           = 0.05
#         search_direction   = - df     # new search direction

#         if np.sqrt(np.sum(search_direction**2)) < 10e-4:
#             print("Stopping: Approximate Gradient is almost 0")
#             break
#         # print("x " + str(xt))
#         # print("Shape x " + str(xt.shape))
#         # print("df " + str(df))
#         print("Value of function:" + str(f))
#         # print("Shape dfx " + str(df.shape))
#         # print("New alpha " + str(alpha0))

#         path            = path + [xt]
#         function_values = function_values + [f]

#         call_cb()

#         print("COUNTER :::::: " + str(outs['counter']))

#     fun(xt, *args)

#     return xt



def minimize_sgdmom(obj, free_variables, lr=0.01, momentum=0.9, decay=0.9, tol=1e-5, on_step=None, maxiters=None):

    verbose = False

    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])

    num_unique_ids = len(np.unique(np.array([id(freevar) for freevar in free_variables])))
    if num_unique_ids != len(free_variables):
        raise Exception('The "free_variables" param contains duplicate variables.')

    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))

    def call_cb():
        if on_step is not None:
            on_step(obj)

        report_line = ""
        if len(labels) > 0:
            report_line += '%.2e | ' % (np.sum(obj.r**2),)
        for label in sorted(labels.keys()):
            objective = labels[label]
            report_line += '%s: %.2e | ' % (label, np.sum(objective.r**2))
        if len(labels) > 0:
            report_line += '\n'
        sys.stderr.write(report_line)

    call_cb()

    # pif = print-if-verbose.
    # can't use "print" because it's a statement, not a fn
    pif = lambda x: print(x) if verbose else 0

    # optimization parms
    k_max = maxiters

    k = 0

    p = col(obj.x.r)

    tm = time.time()
    pif('computing Jacobian...')
    J = obj.J

    if sp.issparse(J):
        assert(J.nnz > 0)
    pif('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], time.time() - tm))

    if J.shape[1] != p.size:
        import pdb; pdb.set_trace()
    assert(J.shape[1] == p.size)

    tm = time.time()
    pif('updating A and g...')

    stop = False
    dp = np.array([[0]])

    bestParams = p
    bestEval = obj.r
    numWorse = 0

    while (not stop) and (k < k_max):
        k += 1

        pif('beginning iteration %d' % (k,))
        arrJ = J
        if sp.issparse(J):
            arrJ = J.toarray()
        dp = col(lr*np.array(J)) + momentum*dp

        p_new = p - dp

        lr = lr*decay

        obj.x = p_new.ravel()

        if norm(dp) < tol:
            pif('stopping due to small update')
            stop = True

        J = obj.J.copy()

        if bestEval > obj.r:
            numWorse = 0
            bestEval = obj.r.copy()
            bestParams = p.copy()
        else:
            numWorse += 1
            if numWorse >= 10:
                print("Stopping due to increasing evaluation error.")
                stop = True
                obj.x = bestParams.ravel()
                obj.r

        p = col(obj.x.r)

        call_cb()

        if k >= k_max:
            pif('stopping because max number of user-specified iterations (%d) has been met' % (k_max,))

    return obj.free_variables


def gradCheckSimple(fun, var, delta):
    f0 = fun.r
    oldvar = var.r[:].copy()
    var[:] = var.r[:] + delta
    f1 = fun.r
    diff = (f1 - f0)/np.abs(delta)
    var[:] = oldvar.copy()
    return diff

def gradCheck(fun, vars, delta):

    jacs = np.concatenate([fun.dr_wrt(wrt).toarray()[0] for wrt in vars])
    approxjacs = []
    for idx, freevar in enumerate(vars):
        approxjacs = approxjacs + [gradCheckSimple(fun, freevar, delta[idx])]
    approxjacs = np.concatenate(approxjacs)
    check = jacs/approxjacs
    return jacs, approxjacs, check

def scipyGradCheck(fun, x0):

    if isinstance(fun, list) or isinstance(fun, tuple):
        fun = ch.concatenate([f.ravel() for f in fun])
    if isinstance(fun, dict):
        fun = ch.concatenate([f.ravel() for f in list(fun.values())])

    obj = fun
    free_variables = x0

    from chumpy.ch import SumOfSquares

    hessp = None
    hess = None
    if obj.size == 1:
        obj_scalar = obj
    else:
        obj_scalar = SumOfSquares(obj)

        def hessp(vs, p,obj, obj_scalar, free_variables):
            changevars(vs,obj,obj_scalar,free_variables)
            if not hasattr(hessp, 'vs'):
                hessp.vs = vs*0+1e16
            if np.max(np.abs(vs-hessp.vs)) > 0:

                J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                hessp.J = J
                hessp.H = 2. * J.T.dot(J)
                hessp.vs = vs
            return np.array(hessp.H.dot(p)).ravel()
            #return 2*np.array(hessp.J.T.dot(hessp.J.dot(p))).ravel()

        if method.lower() != 'newton-cg':
            def hess(vs, obj, obj_scalar, free_variables):
                changevars(vs,obj,obj_scalar,free_variables)
                if not hasattr(hessp, 'vs'):
                    hessp.vs = vs*0+1e16
                if np.max(np.abs(vs-hessp.vs)) > 0:
                    J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                    hessp.H = 2. * J.T.dot(J)
                return hessp.H

    def changevars(vs, obj, obj_scalar, free_variables):
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size
            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        return changed

    def residuals(vs,obj, obj_scalar, free_variables):
        changevars(vs, obj, obj_scalar, free_variables)
        residuals = obj_scalar.r.ravel()[0]
        return residuals

    def scalar_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(scalar_jacfunc, 'vs'):
            scalar_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-scalar_jacfunc.vs)) == 0:
            return scalar_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)

        if False: # faster, at least on some problems
            result = np.concatenate([np.array(obj_scalar.lop(wrt, np.array([[1]]))).ravel() for wrt in free_variables])
        else:
            jacs = [obj_scalar.dr_wrt(wrt) for wrt in free_variables]
            for idx, jac in enumerate(jacs):
                if sp.issparse(jac):
                    jacs[idx] = jacs[idx].toarray()
            result = np.concatenate([jac.ravel() for jac in jacs])

        scalar_jacfunc.J = result
        scalar_jacfunc.vs = vs
        return np.squeeze(np.asarray(result.ravel()))

    def ns_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(ns_jacfunc, 'vs'):
            ns_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-ns_jacfunc.vs)) == 0:
            return ns_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)
        jacs = [obj.dr_wrt(wrt) for wrt in free_variables]
        result = hstack(jacs)

        ns_jacfunc.J = result
        ns_jacfunc.vs = vs
        return result

    err = scipy.optimize.check_grad(residuals, scalar_jacfunc, np.concatenate([free_variable.r.ravel() for free_variable in free_variables]), obj, obj_scalar, free_variables)
    print("Grad check (Root Sum Sq. of Diff.) error of real and finite difference gradients: " + str(err))

    return err



def minimize(fun, x0, method='dogleg', bounds=None, constraints=(), tol=None, callback=None, options=None):

    if method == 'dogleg':
        if options is None: options = {}
        return _minimize_dogleg(fun, free_variables=x0, on_step=callback, **options)

    maxiter = None
    if options != None:
        maxiter = options['maxiter']

    if isinstance(fun, list) or isinstance(fun, tuple):
        fun = ch.concatenate([f.ravel() for f in fun])
    if isinstance(fun, dict):
        fun = ch.concatenate([f.ravel() for f in list(fun.values())])
    obj = fun
    free_variables = x0

    from chumpy.ch import SumOfSquares

    hessp = None
    hess = None
    if obj.size == 1:
        obj_scalar = obj
    else:
        obj_scalar = SumOfSquares(obj)
    
        def hessp(vs, p,obj, obj_scalar, free_variables):
            changevars(vs,obj,obj_scalar,free_variables)
            if not hasattr(hessp, 'vs'):
                hessp.vs = vs*0+1e16
            if np.max(np.abs(vs-hessp.vs)) > 0:

                J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                hessp.J = J
                hessp.H = 2. * J.T.dot(J)
                hessp.vs = vs
            return np.array(hessp.H.dot(p)).ravel()
            #return 2*np.array(hessp.J.T.dot(hessp.J.dot(p))).ravel()
            
        if method.lower() != 'newton-cg':
            def hess(vs, obj, obj_scalar, free_variables):
                changevars(vs,obj,obj_scalar,free_variables)
                if not hasattr(hessp, 'vs'):
                    hessp.vs = vs*0+1e16
                if np.max(np.abs(vs-hessp.vs)) > 0:
                    J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                    hessp.H = 2. * J.T.dot(J)
                return hessp.H
        
    def changevars(vs, obj, obj_scalar, free_variables):
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size
            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        methods_without_callback = ('anneal', 'powell', 'cobyla', 'slsqp')
        if callback is not None and changed and method.lower() in methods_without_callback:
            callback(None)

        return changed

    def residuals(vs,obj, obj_scalar, free_variables):
        changevars(vs, obj, obj_scalar, free_variables)
        residuals = obj_scalar.r.ravel()[0]
        return residuals

    def scalar_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(scalar_jacfunc, 'vs'):
            scalar_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-scalar_jacfunc.vs)) == 0:
            return scalar_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)

        if False: # faster, at least on some problems
            result = np.concatenate([np.array(obj_scalar.lop(wrt, np.array([[1]]))).ravel() for wrt in free_variables])            
        else:
            jacs = [obj_scalar.dr_wrt(wrt) for wrt in free_variables]
            for idx, jac in enumerate(jacs):
                if sp.issparse(jac):
                    jacs[idx] = jacs[idx].toarray()
            result = np.concatenate([jac.ravel() for jac in jacs])

        scalar_jacfunc.J = result
        scalar_jacfunc.vs = vs
        return np.squeeze(np.asarray(result.ravel()))
        
    def ns_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(ns_jacfunc, 'vs'):
            ns_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-ns_jacfunc.vs)) == 0:
            return ns_jacfunc.J
            
        changevars(vs, obj, obj_scalar, free_variables)
        jacs = [obj.dr_wrt(wrt) for wrt in free_variables]
        result = hstack(jacs)

        ns_jacfunc.J = result
        ns_jacfunc.vs = vs
        return result

    if method == 'minimize':
        x1, fX, i = min_ras.minimize(np.concatenate([free_variable.r.ravel() for free_variable in free_variables]), residuals, scalar_jacfunc, args=(obj, obj_scalar, free_variables), on_step=callback, maxnumfuneval=maxiter)
    elif method == 'SGDMom':
        return minimize_sgdmom(obj=fun, free_variables=x0 , lr=options['lr'], momentum=options['momentum'], decay=options['decay'], on_step=callback, maxiters=maxiter)
    # elif method == 'probLineSearch':
        # x1 = probLineSearchMin(np.concatenate([free_variable.r.ravel() for free_variable in free_variables]), residuals, scalar_jacfunc, args=(obj, obj_scalar, free_variables), df_vars=options['df_vars'], on_step=callback, maxnumfuneval=maxiter)
    else:
        # ipdb.set_trace()
        x1 = scipy.optimize.minimize(
            method=method,
            fun=residuals,
            callback=callback,
            x0=np.concatenate([free_variable.r.ravel() for free_variable in free_variables]),
            jac=scalar_jacfunc,
            hessp=hessp, hess=hess, args=(obj, obj_scalar, free_variables),
            bounds=bounds, constraints=constraints, tol=tol, options=options).x

    changevars(x1, obj, obj_scalar, free_variables)

    return free_variables

_giter = 0
class ChInputsStacked(ch.Ch):
    dterms = 'x', 'obj'
    terms = 'free_variables'

    def compute_r(self):
        return self.obj.r.ravel()
    
    # def compute_dr_wrt(self, wrt):
    #     if wrt is self.x:
    #         return hstack([self.obj.dr_wrt(freevar) for freevar in self.free_variables])
    
    def dr_wrt(self, wrt):
        if wrt is self.x:
            mtxs = []
            for freevar in self.free_variables:
                if isinstance(freevar, ch.Select):
                    new_mtx = self.obj.dr_wrt(freevar.a)
                    try:
                        mtxs.append(new_mtx[:,freevar.idxs])
                    except:
                        mtxs.append(new_mtx.tocsc()[:,freevar.idxs])
                else:
                    mtxs.append(self.obj.dr_wrt(freevar))
            return hstack(mtxs)
            #return hstack([self.obj.dr_wrt(freevar) for freevar in self.free_variables])
    
    def on_changed(self, which):
        global _giter
        _giter += 1
        
        if 'x' in which:
            pos = 0
            for idx, freevar in enumerate(self.free_variables):
                sz = freevar.r.size
                rng = np.arange(pos, pos+sz)
                
                if isinstance(self.free_variables[idx], ch.Select):
                    newv = self.free_variables[idx].a.x.copy()
                    newv.ravel()[self.free_variables[idx].idxs] = self.x.r[rng]
                    self.free_variables[idx].a.__setattr__('x', newv, _giter)
                    #self.free_variables[idx].a.x = newv
                elif isinstance(self.free_variables[idx].x, np.ndarray):
                    #self.free_variables[idx].x = self.x.r[rng].copy().reshape(self.free_variables[idx].x.shape)
                    self.free_variables[idx].__setattr__('x', self.x.r[rng].copy().reshape(self.free_variables[idx].x.shape), _giter)
                else: # a number
                    #self.free_variables[idx].x = self.x.r[rng]
                    self.free_variables[idx].__setattr__('x', self.x.r[rng], _giter)
                #self.free_variables[idx] = self.obj.replace(freevar, Ch(self.x.r[rng].copy()))
                pos += sz
    

    @property
    def J(self):
        result = self.dr_wrt(self.x).copy()
        return np.atleast_2d(result) if not sp.issparse(result) else result
    
    def JT_dot(self, y):
        return self.J.T.dot(y)
    
    def J_dot(self, y):
        return self.J.dot(y)
    
    # Have not observed this to be faster than just using J directly
    def JTJ(self):
        if False:
            return self.J.T.dot(self.J)
        else:
            Js = [self.obj.dr_wrt(freevar) for freevar in self.free_variables]
            zeroArray=[None]*len(Js)
            A = [zeroArray[:] for i in range(len(Js))]
            for y in range(len(Js)):
                for x in range(len(Js)):
                    if y > x:
                        A[y][x] = A[x][y].T
                    else:
                        A[y][x] = Js[y].T.dot(Js[x])
            return vstack([hstack(A[y]) for y in range(len(Js))])

    
_solver_fns = {
    'cg': lambda A, x, M=None : scipy.sparse.linalg.cg(A, x, M=M, tol=1e-10)[0],
    'spsolve': lambda A, x : scipy.sparse.linalg.spsolve(A, x)
}



def _minimize_dogleg(obj, free_variables, on_step=None,
                     maxiter=200, max_fevals=np.inf, sparse_solver='spsolve',
                     disp=False, show_residuals=None, e_1=1e-18, e_2=1e-18, e_3=0., delta_0=None):

    """"Nonlinear optimization using Powell's dogleg method.

    See Lourakis et al, 2005, ICCV '05, "Is Levenberg-Marquardt
    the Most Efficient Optimization for Implementing Bundle
    Adjustment?":
    http://www.ics.forth.gr/cvrl/publications/conferences/0201-P0401-lourakis-levenberg.pdf
    """

    import warnings
    if show_residuals is not None:
        import warnings
        warnings.warn('minimize_dogleg: show_residuals parm is deprecaed, pass a dict instead.')

    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])


    niters = maxiter
    verbose = disp
    num_unique_ids = len(np.unique(np.array([id(freevar) for freevar in free_variables])))
    if num_unique_ids != len(free_variables):
        raise Exception('The "free_variables" param contains duplicate variables.')
        
    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))

    def call_cb():
        if on_step is not None:
            on_step(obj)

        report_line = ""
        if len(labels) > 0:
            report_line += '%.2e | ' % (np.sum(obj.r**2),)
        for label in sorted(labels.keys()):
            objective = labels[label]
            report_line += '%s: %.2e | ' % (label, np.sum(objective.r**2))
        if len(labels) > 0:
            report_line += '\n'
        sys.stderr.write(report_line)

    call_cb()

    # pif = print-if-verbose.
    # can't use "print" because it's a statement, not a fn
    pif = lambda x: print(x) if verbose else 0

    if isinstance(sparse_solver, collections.Callable):
        solve = sparse_solver
    elif isinstance(sparse_solver, str) and sparse_solver in list(_solver_fns.keys()):
        solve = _solver_fns[sparse_solver]
    else:
        raise Exception('sparse_solver argument must be either a string in the set (%s) or have the api of scipy.sparse.linalg.spsolve.' % ''.join(list(_solver_fns.keys()), ' '))

    # optimization parms
    k_max = niters
    fevals = 0

    k = 0
    delta = delta_0
    p = col(obj.x.r) 

    fevals += 1
    
    tm = time.time()
    pif('computing Jacobian...')
    J = obj.J

    if sp.issparse(J):
        assert(J.nnz > 0)
    pif('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], time.time() - tm))
    
    if J.shape[1] != p.size:
        import pdb; pdb.set_trace()
    assert(J.shape[1] == p.size)
    
    tm = time.time()
    pif('updating A and g...')
    A = J.T.dot(J)    
    r = col(obj.r.copy())
    
    g = col(J.T.dot(-r))
    pif('A and g updated in %.2fs' % (time.time() - tm))
    
    stop = norm(g, np.inf) < e_1
    while (not stop) and (k < k_max) and (fevals < max_fevals):
        k += 1
        pif('beginning iteration %d' % (k,))
        d_sd = col((sqnorm(g)) / (sqnorm (J.dot(g))) * g)
        GNcomputed = False

        while True:
            # if the Cauchy point is outside the trust region,
            # take that direction but only to the edge of the trust region
            if delta is not None and norm(d_sd) >= delta:
                pif('PROGRESS: Using stunted cauchy')
                d_dl = np.array(col(delta/norm(d_sd) * d_sd))
            else:
                if not GNcomputed:
                    tm = time.time()
                    if scipy.sparse.issparse(A):
                        A.eliminate_zeros()
                        pif('sparse solve...sparsity infill is %.3f%% (hessian %dx%d), J infill %.3f%%' % (
                            100. * A.nnz / (A.shape[0] * A.shape[1]),
                            A.shape[0],
                            A.shape[1],
                            100. * J.nnz / (J.shape[0] * J.shape[1])))
                            
                        if g.size > 1:             
                            d_gn = col(solve(A, g))
                            if np.any(np.isnan(d_gn)) or np.any(np.isinf(d_gn)):
                                from scipy.sparse.linalg import lsqr
                                d_gn = col(lsqr(A, g)[0])
                        else:
                            d_gn = np.atleast_1d(g.ravel()[0]/A[0,0])
                        pif('sparse solve...done in %.2fs' % (time.time() - tm))
                    else:
                        pif('dense solve...')
                        try:
                            d_gn = col(np.linalg.solve(A, g))
                        except Exception:
                            d_gn = col(np.linalg.lstsq(A, g)[0])
                        pif('dense solve...done in %.2fs' % (time.time() - tm))
                    GNcomputed = True

                # if the gauss-newton solution is within the trust region, use it
                if delta is None or norm(d_gn) <= delta:
                    pif('PROGRESS: Using gauss-newton solution')
                    d_dl = np.array(d_gn)
                    if delta is None:
                        delta = norm(d_gn)

                else: # between cauchy step and gauss-newton step
                    pif('PROGRESS: between cauchy and gauss-newton')

                    # compute beta multiplier
                    delta_sq = delta**2
                    pnow = (
                        (d_gn-d_sd).T.dot(d_gn-d_sd)*delta_sq
                        + d_gn.T.dot(d_sd)**2
                        - sqnorm(d_gn) * (sqnorm(d_sd)))
                    B = delta_sq - sqnorm(d_sd)
                    B /= ((d_gn-d_sd).T.dot(d_sd) + math.sqrt(pnow))

                    # apply step
                    d_dl = np.array(d_sd + float(B) * (d_gn - d_sd))

                    #assert(math.fabs(norm(d_dl) - delta) < 1e-12)
            if norm(d_dl) <= e_2*norm(p):
                pif('stopping because of small step size (norm_dl < %.2e)' % (e_2*norm(p)))
                stop = True
            else:
                p_new = p + d_dl

                tm_residuals = time.time()
                obj.x = p_new
                fevals += 1

                r_trial = obj.r.copy()
                tm_residuals = time.time() - tm

                # rho is the ratio of...
                # (improvement in SSE) / (predicted improvement in SSE)  
                
                # slower
                #rho = norm(e_p)**2 - norm(e_p_trial)**2
                #rho = rho / (L(d_dl*0, e_p, J) - L(d_dl, e_p, J))              
                
                # faster
                sqnorm_ep = sqnorm(r)
                rho = sqnorm_ep - norm(r_trial)**2
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',category=RuntimeWarning)
                    if rho > 0:
                        rho /= predicted_improvement(d_dl, -r, J, sqnorm_ep, A, g)
                    
                improvement_occurred = rho > 0

                # if the objective function improved, update input parameter estimate.
                # Note that the obj.x already has the new parms,
                # and we should not set them again to the same (or we'll bust the cache)                
                if improvement_occurred:
                    p = col(p_new)
                    call_cb()

                    if (sqnorm_ep - norm(r_trial)**2) / sqnorm_ep < e_3:
                        stop = True
                        pif('stopping because improvement < %.1e%%' % (100*e_3,))


                else:  # Put the old parms back
                    obj.x = ch.Ch(p)
                    obj.on_changed('x') # copies from flat vector to free variables

                # if the objective function improved and we're not done,
                # get ready for the next iteration
                if improvement_occurred and not stop:
                    tm_jac = time.time()
                    pif('computing Jacobian...')
                    J = obj.J.copy()
                    tm_jac = time.time() - tm_jac
                    pif('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], tm_jac))

                    pif('Residuals+Jac computed in %.2fs' % (tm_jac + tm_residuals,))

                    tm = time.time()
                    pif('updating A and g...')
                    A = J.T.dot(J)
                    r = col(r_trial)
                    g = col(J.T.dot(-r))
                    pif('A and g updated in %.2fs' % (time.time() - tm))
                    
                    if norm(g, np.inf) < e_1:
                        stop = True
                        pif('stopping because norm(g, np.inf) < %.2e' % (e_1))

                # update our trust region
                delta = updateRadius(rho, delta, d_dl)
                
                if delta <= e_2*norm(p):
                    stop = True
                    pif('stopping because trust region is too small')

            # the following "collect" is very expensive.
            # please contact matt if you find situations where it actually helps things.
            #import gc; gc.collect()
            if stop or improvement_occurred or (fevals >= max_fevals):
                break
        if k >= k_max:
            pif('stopping because max number of user-specified iterations (%d) has been met' % (k_max,))
        elif fevals >= max_fevals:
            pif('stopping because max number of user-specified func evals (%d) has been met' % (max_fevals,))

    return obj.free_variables


def sqnorm(a):
    return norm(a)**2

def updateRadius(rho, delta, d_dl, lb=.05, ub=.9):

    if rho > ub:
        delta = max(delta, 2.5*norm(d_dl))
    elif rho < lb:
        delta *= .25
    return delta


def predicted_improvement(d, e, J, sqnorm_e, JTJ, JTe):
    d = col(d)
    e = col(e)
    aa = .5 * sqnorm_e
    bb = JTe.T.dot(d)
    c1 = .5 * d.T
    c2 = JTJ
    c3 = d
    cc = c1.dot(c2.dot(c3))
    result = 2. * (aa - bb + cc)[0,0]
    return sqnorm_e - result


def main():
    pass


if __name__ == '__main__':
    main()

