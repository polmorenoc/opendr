#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import cv2
import scipy.stats
import OpenGL.GL as GL
from PIL import Image
from chumpy.utils import row, col
from opendr.contexts._constants import *
import bottleneck as bn
import ipdb
import matplotlib.pyplot as plt
import warnings

def nanmean(a, axis):
    # don't call nan_to_num in here, unless you check that
    # occlusion_test.py still works after you do it!
    result = np.nanmean(a, axis=axis)
    return result

def nangradients(arr):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dy = np.expand_dims(arr[:-1,:,:] - arr[1:,:,:], axis=3)
        dx = np.expand_dims(arr[:,:-1,:] - arr[:, 1:, :], axis=3)

        dy = np.concatenate((dy[1:,:,:], dy[:-1,:,:]), axis=3)
        dy = np.nanmean(dy, axis=3)
        dx = np.concatenate((dx[:,1:,:], dx[:,:-1,:]), axis=3)
        dx = np.nanmean(dx, axis=3)

        if arr.shape[2] > 1:
            gy, gx, _ = np.gradient(arr)
        else:
            gy, gx = np.gradient(arr.squeeze())
            gy = np.atleast_3d(gy)
            gx = np.atleast_3d(gx)
        gy[1:-1,:,:] = -dy
        gx[:,1:-1,:] = -dx

    return gy, gx

#Based on Ravi Ramamoorthi and 3D gradient enhancement by Fukai Zhao et al
# def dImage_wrt_2dVerts_bnd_rev(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
#     a = -vn[:,0]/vn[:,2]
#
#     b = -vn[:,]
#
#     c =

    # 0 Compute dxdu

    # 1 For each visible point:
        # Compute the unprojected x
        # take corresponding triangle.
        # Take corresponding normal and barycentric gradient.
        # Use that to compute dr vc wrt normal * dr normal wrt x
    # 2 Idx * dxdu

def boundary_neighborhood(boundary):
    shape = boundary.shape

    notboundary = np.logical_not(boundary)
    horizontall = np.hstack((np.diff(notboundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
    horizontalr = np.hstack((np.diff(boundary.astype(np.int8),axis=1), np.zeros((shape[0],1), dtype=np.int8)))
    verticalt = np.vstack((np.diff(notboundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))
    verticalb = np.vstack((np.diff(boundary.astype(np.int8), axis=0), np.zeros((1,shape[1]), dtype=np.int8)))

    pixr = (horizontalr == 1)
    pixl = (horizontall == 1)
    pixt = (verticalt == 1)
    pixb = (verticalb == 1)

    # plt.imshow((pixrl | pixlr | pixtb | pixbt))

    #Quicker, convolve (FFT) and take mask * etc.

    lidxs_out = np.where(pixl.ravel())[0]
    ridxs_out = np.where(pixr.ravel())[0] + 1
    tidxs_out = np.where(pixt.ravel())[0]
    bidxs_out = np.where(pixb.ravel())[0] + shape[1]
    lidxs_int = np.where(pixl.ravel())[0] + 1
    ridxs_int = np.where(pixr.ravel())[0]
    tidxs_int = np.where(pixt.ravel())[0] + shape[1]
    bidxs_int = np.where(pixb.ravel())[0]

    return lidxs_out, ridxs_out, tidxs_out, bidxs_out, lidxs_int, ridxs_int, tidxs_int, bidxs_int

def dImage_wrt_2dVerts_bnd_new(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
    """Construct a sparse jacobian that relates 2D projected vertex positions
    (in the columns) to pixel values (in the rows). This can be done
    in two steps."""

    n_channels = np.atleast_3d(observed).shape[2]
    shape = visibility.shape

    #Pol:
    #1; Expand visible to those around edges of bounding pixels that are not in visible, keep track of the corresponding visible f index.
    #Add them to IJs.

    background = visibility == 4294967295

    lidxs_int, ridxs_int, tidxs_int, bidxs_int = boundary_neighborhood(bnd_bool)

    visibleidxs = np.zeros(shape).ravel().astype(np.uint32)
    visibleidxs[visible] = np.arange(visible.size)

    #2: Take the data and copy the corresponding dxs and dys to these new pixels.

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())
    JS = np.hstack((JS*2, JS*2+1)).ravel()

    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    if n_channels > 1:
        IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
        JS = np.concatenate([JS for i in range(n_channels)])

    # Step 2: get the data ready, ie the actual values of the derivatives
    # ksize = 1
    # bndf = bnd_bool.astype(np.float64)
    # nbndf = np.logical_not(bnd_bool).astype(np.float64)
    # sobel_normalizer = cv2.Sobel(np.asarray(np.tile(row(np.arange(10)), (10, 1)), np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize)[5,5]
    #
    # bnd_nan = bndf.reshape((observed.shape[0], observed.shape[1], -1)).copy()
    # bndindices = bnd_nan.ravel()>0
    # bnd_nan.ravel()[bnd_nan.ravel()>0] = np.nan
    # bnd_nan += 1
    # obs_nonbnd = np.atleast_3d(observed) * bnd_nan
    #
    # ydiffnb_, xdiffnb = nangradients(obs_nonbnd)
    # ydiffnb, xdiffnb = nangradients(obs_nonbnd)

    observed = np.atleast_3d(observed)

    if observed.shape[2] > 1:
        ydiffbnd, xdiffbnd, _ = np.gradient(observed)
    else:
        ydiffbnd, xdiffbnd = np.gradient(observed.squeeze())
        ydiffbnd = np.atleast_3d(ydiffbnd)
        xdiffbnd = np.atleast_3d(xdiffbnd)

    # ydiffbnd, xdiffbnd = np.gradient(observed.squeeze())
    # ydiffedge = np.vstack(np.diff(observed.squeeze(), axis=0), np.zeros(1,shape[1],n_channels))
    # xdiffedge = np.hstack(np.diff(observed.squeeze(), axis=1), np.zeros(shape[0],1,n_channels))

    # This corrects for a bias imposed boundary differences begin spread over two pixels
    # (by np.gradients or similar) but only counted once (since OpenGL's line
    # drawing spans 1 pixel)
    # xdiffbnd *= 2.0
    # ydiffbnd *= 2.0

    # xdiffnb = -xdiffnb
    # ydiffnb = -ydiffnb

    # ydiffnb *= 0
    # xdiffnb *= 0

    # ipdb.set_trace()

    # xdiffbnd = np.zeros([shape[0],shape[1],n_channels])
    # ydiffbnd = np.zeros([shape[0],shape[1],n_channels])

    xdiffbnd.reshape([shape[0]*shape[1], n_channels])[ridxs_int,:] = observed.reshape([shape[0]*shape[1], n_channels])[ridxs_int,:] - observed.reshape([shape[0]*shape[1], n_channels])[ridxs_int-1,:]

    xdiffbnd.reshape([shape[0]*shape[1], n_channels])[lidxs_int,:] = observed.reshape([shape[0]*shape[1], n_channels])[lidxs_int+1,:] - observed.reshape([shape[0]*shape[1], n_channels])[lidxs_int,:]

    ydiffbnd.reshape([shape[0]*shape[1], n_channels])[bidxs_int,:] = observed.reshape([shape[0]*shape[1], n_channels])[bidxs_int,:] - observed.reshape([shape[0]*shape[1], n_channels])[bidxs_int-shape[1],:]

    ydiffbnd.reshape([shape[0]*shape[1], n_channels])[tidxs_int,:] = observed.reshape([shape[0]*shape[1], n_channels])[tidxs_int+ shape[1],:] - observed.reshape([shape[0]*shape[1], n_channels])[tidxs_int,:]


    # xdiffbnd.reshape([shape[0]*shape[1], n_channels])[ridxs_int,:]=     xdiffbnd.reshape([shape[0]*shape[1], n_channels])[ridxs_int,:]*2
    #
    # xdiffbnd.reshape([shape[0]*shape[1], n_channels])[lidxs_int,:]= xdiffbnd.reshape([shape[0]*shape[1], n_channels])[lidxs_int,:]*2
    #
    # ydiffbnd.reshape([shape[0]*shape[1], n_channels])[bidxs_int,:]=     ydiffbnd.reshape([shape[0]*shape[1], n_channels])[bidxs_int,:]*2
    #
    # ydiffbnd.reshape([shape[0]*shape[1], n_channels])[tidxs_int,:]=     ydiffbnd.reshape([shape[0]*shape[1], n_channels])[tidxs_int,:]*2

    # xdiffbnd.reshape([shape[0]*shape[1], n_channels])[ridxs_int,:] = 0
    #
    # xdiffbnd.reshape([shape[0]*shape[1], n_channels])[lidxs_int,:] = 0
    #
    # ydiffbnd.reshape([shape[0]*shape[1], n_channels])[bidxs_int,:] = 0
    #
    # ydiffbnd.reshape([shape[0]*shape[1], n_channels])[tidxs_int,:] = 0

    xdiffbnd = -xdiffbnd
    ydiffbnd = -ydiffbnd

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(ydiffbnd)
        plt.title('xdiffnb')
        plt.subplot(122)
        plt.imshow(xdiffbnd)
        plt.title('xdiffbnd')
        import pdb; pdb.set_trace()

    # idxs = np.isnan(xdiffnb.ravel())
    # xdiffnb.ravel()[idxs] = xdiffbnd.ravel()[idxs]
    #
    # idxs = np.isnan(ydiffnb.ravel())
    # ydiffnb.ravel()[idxs] = ydiffbnd.ravel()[idxs]

    # idxs = np.isnan(xdiffnb.ravel())
    # xdiffnb.ravel()[idxs] = xdiffbnd.ravel()[idxs]
    # boundxdiff = xdiffnb.ravel().copy()

    # xdiffnb.ravel()[bndindices] = xdiffbnd.ravel()[bndindices]

    # ipdb.set_trace()

    # idxs = np.isnan(ydiffnb.ravel())
    # ydiffnb.ravel()[idxs] = ydiffbnd.ravel()[idxs]
    # boundydiff = ydiffnb.ravel().copy()

    # ydiffnb.ravel()[bndindices] = ydiffbnd.ravel()[bndindices]

    if True: # should be right thing
        xdiff = xdiffbnd
        ydiff = ydiffbnd
    else:  #should be old way
        xdiff = xdiffbnd
        ydiff = ydiffbnd


    # TODO: NORMALIZER IS WRONG HERE
    # xdiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / np.atleast_3d(cv2.Sobel(row(np.arange(obs_nonbnd.shape[1])).astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize))
    # ydiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / np.atleast_3d(cv2.Sobel(col(np.arange(obs_nonbnd.shape[0])).astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=ksize))
    #
    # xdiffnb.ravel()[np.isnan(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isnan(ydiffnb.ravel())] = 0.
    # xdiffnb.ravel()[np.isinf(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isinf(ydiffnb.ravel())] = 0.

    # xdiffnb = np.atleast_3d(xdiffnb)
    # ydiffnb = np.atleast_3d(ydiffnb)
    #
    # xdiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / sobel_normalizer
    # ydiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / sobel_normalizer
    #
    # xdiff = xdiffnb * np.atleast_3d(nbndf)
    # xdiff.ravel()[np.isnan(xdiff.ravel())] = 0
    # xdiff += xdiffbnd*np.atleast_3d(bndf)
    #
    # ydiff = ydiffnb * np.atleast_3d(nbndf)
    # ydiff.ravel()[np.isnan(ydiff.ravel())] = 0
    # ydiff += ydiffbnd*np.atleast_3d(bndf)

    #import pdb; pdb.set_trace()

    #xdiff = xdiffnb
    #ydiff = ydiffnb

    #import pdb; pdb.set_trace()

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = col(barycentric[pys, pxs, 0])
    bc1 = col(barycentric[pys, pxs, 1])
    bc2 = col(barycentric[pys, pxs, 2])
    for k in range(n_channels):
        dxs = xdiff[pys, pxs, k]
        dys = ydiff[pys, pxs, k]
        if f.shape[1] == 3:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
        else:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

    data = np.concatenate(datas)

    ij = np.vstack((IS.ravel(), JS.ravel()))
    #
    # #
    # hJS_idx = np.where((np.arange(2*f.shape[1]*n_channels) + 1 ) % 2)[0].tolist()
    # #
    # # ldata = np.zeros([len(lidxs_int), f.shape[1]*n_channels]).astype(np.float64)
    # # ldata[:, hJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[lidxs_int]).tolist(), hJS_idx]
    # shape3d = [shape[0],shape[1],3]
    # numPixels = shape[0]*shape[1]
    #
    #
    # ldata = np.concatenate([np.concatenate([boundxdiff.reshape(shape3d)[:,:,channel].ravel()[lidxs_int]*bc0[visibleidxs[lidxs_int]].ravel(), boundxdiff.reshape(shape3d)[:,:,channel].ravel()[lidxs_int]*bc1[visibleidxs[lidxs_int]].ravel(), boundxdiff.reshape(shape3d)[:,:,channel].ravel()[lidxs_int]*bc2[visibleidxs[lidxs_int]].ravel()]) for channel in range(n_channels)])
    #
    # ISnew = np.tile(col(lidxs_out), (1, f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[lidxs_int]].ravel())*2
    # # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ijnew = np.vstack((ISnew.ravel(), JSnew.ravel()))
    #
    # # rdata = np.zeros([len(ridxs_int), f.shape[1]*n_channels]).astype(np.float64)
    # # rdata[:, hJS_idx] =  data.reshape([len(visible), f.shape[1]*n_channels])[col(visibleidxs[ridxs_int]), hJS_idx]
    # rdata = np.concatenate([np.concatenate([boundxdiff.reshape(shape3d)[:,:,channel].ravel()[ridxs_int]*bc0[visibleidxs[ridxs_int]].ravel(), boundxdiff.reshape(shape3d)[:,:,channel].ravel()[ridxs_int]*bc1[visibleidxs[ridxs_int]].ravel(), boundxdiff.reshape(shape3d)[:,:,channel].ravel()[ridxs_int]*bc2[visibleidxs[ridxs_int]].ravel()]) for channel in range(n_channels)])
    #
    # ISnew = np.tile(col(ridxs_out), (1, f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[ridxs_int]].ravel())*2
    # # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    #
    # ijnew = np.hstack((ijnew, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # vJS_idx = np.where(np.arange(2*f.shape[1]*n_channels) % 2)[0].tolist()
    #
    # # tdata = np.zeros([len(tidxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # # tdata[:, vJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[tidxs_int]), vJS_idx]
    # tdata = np.concatenate([np.concatenate([boundydiff.reshape(shape3d)[:,:,channel].ravel()[tidxs_int]*bc0[visibleidxs[tidxs_int]].ravel(), boundydiff.reshape(shape3d)[:,:,channel].ravel()[tidxs_int]*bc1[visibleidxs[tidxs_int]].ravel(), boundydiff.reshape(shape3d)[:,:,channel].ravel()[tidxs_int]*bc2[visibleidxs[tidxs_int]].ravel()]) for channel in range(n_channels)])
    #
    # ISnew = np.tile(col(tidxs_out), (1, f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[tidxs_int]].ravel())*2 + 1
    # # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ijnew = np.hstack((ijnew, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # # bdata = np.zeros([len(bidxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # # bdata[:, vJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[bidxs_int]), vJS_idx]
    # bdata = np.concatenate([np.concatenate([boundydiff.reshape(shape3d)[:,:,channel].ravel()[bidxs_int]*bc0[visibleidxs[bidxs_int]].ravel(), boundydiff.reshape(shape3d)[:,:,channel].ravel()[bidxs_int]*bc1[visibleidxs[bidxs_int]].ravel(), boundydiff.reshape(shape3d)[:,:,channel].ravel()[bidxs_int]*bc2[visibleidxs[bidxs_int]].ravel()]) for channel in range(n_channels)])
    #
    # ISnew = np.tile(col(bidxs_out), (1, f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[bidxs_int]].ravel())*2+1
    # # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # # ipdb.set_trace()
    #
    # ijnew = np.hstack((ijnew, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # ij = np.hstack((ij, ijnew))
    #
    # data = np.concatenate([data, ldata.ravel(), rdata.ravel(), tdata.ravel(), bdata.ravel()])


    result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

    # hJS_idx = np.where(np.arange(num_verts*2) % 2)
    # vJS_idx = np.where((np.arange(num_verts*2)+ 1) % 2)

    #Pol: Is there any problem with this idea? Yep, terribly slow!
    # result[col(lIS_new).tolist(), hJS_idx[0].tolist()] = result[col(lIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(rIS_new).tolist(), hJS_idx[0].tolist()] = result[col(rIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(tIS_new).tolist(), hJS_idx[0].tolist()] = result[col(tIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(bIS_new).tolist(), hJS_idx[0].tolist()] = result[col(bIS_new).tolist(), hJS_idx[0].tolist()]

    return result


def dImage_wrt_2dVerts_bnd(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
    """Construct a sparse jacobian that relates 2D projected vertex positions
    (in the columns) to pixel values (in the rows). This can be done
    in two steps."""

    n_channels = np.atleast_3d(observed).shape[2]
    shape = visibility.shape


    #2: Take the data and copy the corresponding dxs and dys to these new pixels.

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())
    JS = np.hstack((JS*2, JS*2+1)).ravel()

    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    if n_channels > 1:
        IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
        JS = np.concatenate([JS for i in range(n_channels)])

    # Step 2: get the data ready, ie the actual values of the derivatives
    ksize = 1
    bndf = bnd_bool.astype(np.float64)
    nbndf = np.logical_not(bnd_bool).astype(np.float64)
    sobel_normalizer = cv2.Sobel(np.asarray(np.tile(row(np.arange(10)), (10, 1)), np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize)[5,5]

    bnd_nan = bndf.reshape((observed.shape[0], observed.shape[1], -1)).copy()
    bnd_nan.ravel()[bnd_nan.ravel()>0] = np.nan
    bnd_nan += 1
    obs_nonbnd = np.atleast_3d(observed) * bnd_nan

    ydiffnb, xdiffnb = nangradients(obs_nonbnd)


    observed = np.atleast_3d(observed)
    
    if observed.shape[2] > 1:
        ydiffbnd, xdiffbnd, _ = np.gradient(observed)        
    else:
        ydiffbnd, xdiffbnd = np.gradient(observed.squeeze())
        ydiffbnd = np.atleast_3d(ydiffbnd)
        xdiffbnd = np.atleast_3d(xdiffbnd)

    # This corrects for a bias imposed boundary differences begin spread over two pixels
    # (by np.gradients or similar) but only counted once (since OpenGL's line
    # drawing spans 1 pixel)
    xdiffbnd *= 2.0
    ydiffbnd *= 2.0

    xdiffnb = -xdiffnb
    ydiffnb = -ydiffnb
    xdiffbnd = -xdiffbnd
    ydiffbnd = -ydiffbnd
    # ydiffnb *= 0
    # xdiffnb *= 0

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(xdiffnb)
        plt.title('xdiffnb')
        plt.subplot(122)
        plt.imshow(xdiffbnd)
        plt.title('xdiffbnd')
        import pdb; pdb.set_trace()

    idxs = np.isnan(xdiffnb.ravel())
    xdiffnb.ravel()[idxs] = xdiffbnd.ravel()[idxs]

    idxs = np.isnan(ydiffnb.ravel())
    ydiffnb.ravel()[idxs] = ydiffbnd.ravel()[idxs]

    if True: # should be right thing
        xdiff = xdiffnb
        ydiff = ydiffnb
    else:  #should be old way
        xdiff = xdiffbnd
        ydiff = ydiffbnd


    # TODO: NORMALIZER IS WRONG HERE
    # xdiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / np.atleast_3d(cv2.Sobel(row(np.arange(obs_nonbnd.shape[1])).astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize))
    # ydiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / np.atleast_3d(cv2.Sobel(col(np.arange(obs_nonbnd.shape[0])).astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=ksize))
    #
    # xdiffnb.ravel()[np.isnan(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isnan(ydiffnb.ravel())] = 0.
    # xdiffnb.ravel()[np.isinf(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isinf(ydiffnb.ravel())] = 0.

    # xdiffnb = np.atleast_3d(xdiffnb)
    # ydiffnb = np.atleast_3d(ydiffnb)
    #
    # xdiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / sobel_normalizer
    # ydiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / sobel_normalizer
    #
    # xdiff = xdiffnb * np.atleast_3d(nbndf)
    # xdiff.ravel()[np.isnan(xdiff.ravel())] = 0
    # xdiff += xdiffbnd*np.atleast_3d(bndf)
    #
    # ydiff = ydiffnb * np.atleast_3d(nbndf)
    # ydiff.ravel()[np.isnan(ydiff.ravel())] = 0
    # ydiff += ydiffbnd*np.atleast_3d(bndf)

    #import pdb; pdb.set_trace()

    #xdiff = xdiffnb
    #ydiff = ydiffnb

    #import pdb; pdb.set_trace()

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = col(barycentric[pys, pxs, 0])
    bc1 = col(barycentric[pys, pxs, 1])
    bc2 = col(barycentric[pys, pxs, 2])
    for k in range(n_channels):
        dxs = xdiff[pys, pxs, k]
        dys = ydiff[pys, pxs, k]
        if f.shape[1] == 3:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
        else:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

    data = np.concatenate(datas)

    ij = np.vstack((IS.ravel(), JS.ravel()))

    #
    hJS_idx = np.where((np.arange(2*f.shape[1]*n_channels) + 1 ) % 2)[0].tolist()
    #
    # ldata = np.zeros([len(lidxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # ldata[:, hJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[lidxs_int]).tolist(), hJS_idx]
    #
    # ISnew = np.tile(col(lidxs_out), (1, 2*f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[lidxs_int]].ravel())
    # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ij = np.hstack((ij, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # rdata = np.zeros([len(ridxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # rdata[:, hJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[ridxs_int]), hJS_idx]
    #
    # ISnew = np.tile(col(ridxs_out), (1, 2*f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[ridxs_int]].ravel())
    # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ij = np.hstack((ij, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # vJS_idx = np.where(np.arange(2*f.shape[1]*n_channels) % 2)[0].tolist()
    #
    # tdata = np.zeros([len(tidxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # tdata[:, vJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[tidxs_int]), vJS_idx]
    #
    # ISnew = np.tile(col(tidxs_out), (1, 2*f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[tidxs_int]].ravel())
    # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ij = np.hstack((ij, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # bdata = np.zeros([len(bidxs_int), 2*f.shape[1]*n_channels]).astype(np.float64)
    # bdata[:, vJS_idx] =  data.reshape([len(visible), 2*f.shape[1]*n_channels])[col(visibleidxs[bidxs_int]), vJS_idx]
    #
    # ISnew = np.tile(col(bidxs_out), (1, 2*f.shape[1])).ravel()
    # JSnew = col(f[visibility.ravel()[bidxs_int]].ravel())
    # JSnew = np.hstack((JSnew*2, JSnew*2+1)).ravel()
    #
    # if n_channels > 1:
    #     ISnew = np.concatenate([ISnew*n_channels+i for i in range(n_channels)])
    #     JSnew = np.concatenate([JSnew for i in range(n_channels)])
    #
    # ij = np.hstack((ij, np.vstack((ISnew.ravel(), JSnew.ravel()))))
    #
    # data = np.concatenate([data, ldata.ravel(), rdata.ravel(), tdata.ravel(), bdata.ravel()])
    #

    result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

    # hJS_idx = np.where(np.arange(num_verts*2) % 2)
    # vJS_idx = np.where((np.arange(num_verts*2)+ 1) % 2)

    #Pol: Is there any problem with this idea? Yep, terribly slow!
    # result[col(lIS_new).tolist(), hJS_idx[0].tolist()] = result[col(lIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(rIS_new).tolist(), hJS_idx[0].tolist()] = result[col(rIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(tIS_new).tolist(), hJS_idx[0].tolist()] = result[col(tIS_new).tolist(), hJS_idx[0].tolist()]
    # result[col(bIS_new).tolist(), hJS_idx[0].tolist()] = result[col(bIS_new).tolist(), hJS_idx[0].tolist()]

    return result


def dImage_wrt_2dVerts(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
    """Construct a sparse jacobian that relates 2D projected vertex positions
    (in the columns) to pixel values (in the rows). This can be done
    in two steps."""

    n_channels = np.atleast_3d(observed).shape[2]
    shape = visibility.shape

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
    JS = f[visibility.ravel()[visible]].reshape((-1,1))
    JS = np.hstack((JS*2, JS*2+1)).ravel()

    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    if n_channels > 1:
        IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
        JS = np.concatenate([JS for i in range(n_channels)])

    # Step 2: get the data ready, ie the actual values of the derivatives
    ksize=1
    sobel_normalizer = cv2.Sobel(np.asarray(np.tile(row(np.arange(10)), (10, 1)), np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize)[5,5]
    xdiff = -cv2.Sobel(observed, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / sobel_normalizer
    ydiff = -cv2.Sobel(observed, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / sobel_normalizer

    xdiff = np.atleast_3d(xdiff)
    ydiff = np.atleast_3d(ydiff)

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = barycentric[pys, pxs, 0].reshape((-1,1))
    bc1 = barycentric[pys, pxs, 1].reshape((-1,1))
    bc2 = barycentric[pys, pxs, 2].reshape((-1,1))
    for k in range(n_channels):
        dxs = xdiff[pys, pxs, k]
        dys = ydiff[pys, pxs, k]
        if f.shape[1] == 3:
            datas.append(np.hstack((dxs.reshape((-1,1))*bc0,dys.reshape((-1,1))*bc0,dxs.reshape((-1,1))*bc1,dys.reshape((-1,1))*bc1,dxs.reshape((-1,1))*bc2,dys.reshape((-1,1))*bc2)).ravel())
        else:
            datas.append(np.hstack((dxs.reshape((-1,1))*bc0,dys.reshape((-1,1))*bc0,dxs.reshape((-1,1))*bc1,dys.reshape((-1,1))*bc1)).ravel())

    data = np.concatenate(datas)

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

    return result

def flow_to(self, v_next, cam_next):
    from chumpy.ch import MatVecMult

    color_image = self.r
    visibility = self.visibility_image
    pxpos = np.zeros_like(self.color_image)
    pxpos[:,:,0] = np.tile(row(np.arange(self.color_image.shape[1])), (self.color_image.shape[0], 1))
    pxpos[:,:,2] = np.tile(col(np.arange(self.color_image.shape[0])), (1, self.color_image.shape[1]))

    visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    num_visible = len(visible)

    barycentric = self.barycentric_image

    # map 3d to 3d
    JS = col(self.f[visibility.ravel()[visible]]).ravel()
    IS = np.tile(col(np.arange(JS.size/3)), (1, 3)).ravel()
    data = barycentric.reshape((-1,3))[visible].ravel()

    # replicate to xyz
    IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    data = np.concatenate((data, data, data))

    verts_to_visible = sp.csc_matrix((data, (IS, JS)), shape=(np.max(IS)+1, self.v.r.size))

    v_old = self.camera.v
    cam_old = self.camera

    if cam_next is None:
        cam_next = self.camera

    self.camera.v = MatVecMult(verts_to_visible, self.v.r)
    r1 = self.camera.r.copy()

    self.camera = cam_next
    self.camera.v = MatVecMult(verts_to_visible, v_next)
    r2 = self.camera.r.copy()

    n_channels = self.camera.shape[1]
    flow = r2 - r1
    flow_im = np.zeros((self.frustum['height'], self.frustum['width'], n_channels)).reshape((-1,n_channels))

    flow_im[visible] = flow
    flow_im = flow_im.reshape((self.frustum['height'], self.frustum['width'], n_channels))

    self.camera = cam_old
    self.camera.v = v_old
    return flow_im


def dr_wrt_bgcolor(visibility, frustum, num_channels):
    invisible = np.nonzero(visibility.ravel() == 4294967295)[0]
    IS = invisible
    JS = np.zeros(len(IS))
    data = np.ones(len(IS))

    # color image, so 3 channels
    IS = np.concatenate([IS*num_channels+k for k in range(num_channels)])
    JS = np.concatenate([JS*num_channels+k for k in range(num_channels)])
    data = np.concatenate([data for i in range(num_channels)])
    # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    # data = np.concatenate((data, data, data))

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(frustum['width']*frustum['height']*num_channels, num_channels))
    return result


def dr_wrt_vc(visible, visibility, f, barycentric, frustum, vc_size, num_channels):

    # Each pixel relies on three verts
    IS = np.tile(col(visible), (1, 3)).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())

    bc = barycentric.reshape((-1,3))
    data = np.asarray(bc[visible,:], order='C').ravel()

    IS = np.concatenate([IS*num_channels+k for k in range(num_channels)])
    JS = np.concatenate([JS*num_channels+k for k in range(num_channels)])
    data = np.concatenate([data for i in range(num_channels)])
    # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    # data = np.concatenate((data, data, data))

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(frustum['width']*frustum['height']*num_channels, vc_size))
    return result




