__author__ = 'pol'

from utils import *
import opendr
import geometry
import numpy as np
from math import radians
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer
from opendr.lighting import SphericalHarmonics
from opendr.lighting import LambertianPointLight
import chumpy as ch


def transformObject(v, vn, chScale, chObjAz, chPosition):

    if chScale.size == 1:
        scaleMat = geometry.Scale(x=chScale[0], y=chScale[0],z=chScale[0])[0:3,0:3]
    elif chScale.size == 2:
        scaleMat = geometry.Scale(x=chScale[0], y=chScale[0], z=chScale[1])[0:3, 0:3]
    else:
        scaleMat = geometry.Scale(x=chScale[0], y=chScale[1], z=chScale[2])[0:3, 0:3]
    chRotAzMat = geometry.RotateZ(a=chObjAz)[0:3,0:3]
    chRotAzMatX = geometry.RotateX(a=0)[0:3,0:3]

    # transformation = scaleMat
    transformation = ch.dot(ch.dot(chRotAzMat,chRotAzMatX), scaleMat)
    invTranspModel = ch.transpose(ch.inv(transformation))

    vtransf = []
    vntransf = []
    for mesh_i, mesh in enumerate(v):
        vtransf = vtransf + [ch.dot(v[mesh_i], transformation) + chPosition]
        vndot = ch.dot(vn[mesh_i], invTranspModel)
        vndot = vndot/ch.sqrt(ch.sum(vndot**2,1))[:,None]
        vntransf = vntransf + [vndot]
    return vtransf, vntransf


def createRenderer(glMode, cameraParams, v, vc, f_list, vn, uv, haveTextures_list, textures_list, frustum, win ):
    renderer = TexturedRenderer()
    renderer.set(glMode=glMode)

    vflat = [item for sublist in v for item in sublist]
    if len(vflat)==1:
        vstack = vflat[0]
    else:
        vstack = ch.vstack(vflat)

    camera, modelRotation, _ = setupCamera(vstack, cameraParams)

    vnflat = [item for sublist in vn for item in sublist]
    vcflat = [item for sublist in vc for item in sublist]

    setupTexturedRenderer(renderer, vstack, vflat, f_list, vcflat, vnflat,  uv, haveTextures_list, textures_list, camera, frustum, win)
    return renderer


def createRenderer(glMode, cameraParams, v, vc, f_list, vn, uv, haveTextures_list, textures_list, frustum, win ):
    renderer = TexturedRenderer()
    renderer.set(glMode=glMode)

    vflat = [item for sublist in v for item in sublist]

    if len(vflat)==1:
        vstack = vflat[0]
    else:

        vstack = ch.vstack(vflat)

    camera, modelRotation, _ = setupCamera(vstack, cameraParams)

    vnflat = [item for sublist in vn for item in sublist]
    vcflat = [item for sublist in vc for item in sublist]

    setupTexturedRenderer(renderer, vstack, vflat, f_list, vcflat, vnflat,  uv, haveTextures_list, textures_list, camera, frustum, win)
    return renderer

#Method from https://github.com/adamlwgriffiths/Pyrr/blob/master/pyrr/geometry.py
def create_cube(scale=(1.0,1.0,1.0), st=False, rgba=np.array([1.,1.,1.,1.]), dtype='float32', type='triangles'):
    """Returns a Cube reading for rendering."""

    shape = [24, 3]
    rgba_offset = 3

    width, height, depth = scale
    # half the dimensions
    width /= 2.0
    height /= 2.0
    depth /= 2.0

    vertices = np.array([
        # front
        # top right
        ( width, height, depth,),
        # top left
        (-width, height, depth,),
        # bottom left
        (-width,-height, depth,),
        # bottom right
        ( width,-height, depth,),

        # right
        # top right
        ( width, height,-depth),
        # top left
        ( width, height, depth),
        # bottom left
        ( width,-height, depth),
        # bottom right
        ( width,-height,-depth),

        # back
        # top right
        (-width, height,-depth),
        # top left
        ( width, height,-depth),
        # bottom left
        ( width,-height,-depth),
        # bottom right
        (-width,-height,-depth),

        # left
        # top right
        (-width, height, depth),
        # top left
        (-width, height,-depth),
        # bottom left
        (-width,-height,-depth),
        # bottom right
        (-width,-height, depth),

        # top
        # top right
        ( width, height,-depth),
        # top left
        (-width, height,-depth),
        # bottom left
        (-width, height, depth),
        # bottom right
        ( width, height, depth),

        # bottom
        # top right
        ( width,-height, depth),
        # top left
        (-width,-height, depth),
        # bottom left
        (-width,-height,-depth),
        # bottom right
        ( width,-height,-depth),
    ], dtype=dtype)

    st_values = None
    rgba_values = None

    if st:
        # default st values
        st_values = np.tile(
            np.array([
                (1.0, 1.0,),
                (0.0, 1.0,),
                (0.0, 0.0,),
                (1.0, 0.0,),
            ], dtype=dtype),
            (6,1,)
        )

        if isinstance(st, bool):
            pass
        elif isinstance(st, (int, float)):
            st_values *= st
        elif isinstance(st, (list, tuple, np.ndarray)):
            st = np.array(st, dtype=dtype)
            if st.shape == (2,2,):
                # min / max
                st_values *= st[1] - st[0]
                st_values += st[0]
            elif st.shape == (4,2,):
                # per face st values specified manually
                st_values[:] = np.tile(st, (6,1,))
            elif st.shape == (6,2,):
                # st values specified manually
                st_values[:] = st
            else:
                raise ValueError('Invalid shape for st')
        else:
            raise ValueError('Invalid value for st')

        shape[-1] += st_values.shape[-1]
        rgba_offset += st_values.shape[-1]

    if len(rgba) > 0:
        # default rgba values
        rgba_values = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype), (24,1,))

        if isinstance(rgba, bool):
            pass
        elif isinstance(rgba, (int, float)):
            # int / float expands to RGBA with all values == value
            rgba_values *= rgba
        elif isinstance(rgba, (list, tuple, np.ndarray)):
            rgba = np.array(rgba, dtype=dtype)

            if rgba.shape == (3,):
                rgba_values = np.tile(rgba, (24,1,))
            elif rgba.shape == (4,):
                rgba_values[:] = np.tile(rgba, (24,1,))
            elif rgba.shape == (4,3,):
                rgba_values = np.tile(rgba, (6,1,))
            elif rgba.shape == (4,4,):
                rgba_values = np.tile(rgba, (6,1,))
            elif rgba.shape == (6,3,):
                rgba_values = np.repeat(rgba, 4, axis=0)
            elif rgba.shape == (6,4,):
                rgba_values = np.repeat(rgba, 4, axis=0)
            elif rgba.shape == (24,3,):
                rgba_values = rgba
            elif rgba.shape == (24,4,):
                rgba_values = rgba
            else:
                raise ValueError('Invalid shape for rgba')
        else:
            raise ValueError('Invalid value for rgba')

        shape[-1] += rgba_values.shape[-1]

    data = np.empty(shape, dtype=dtype)
    data[:,:3] = vertices
    if st_values is not None:
        data[:,3:5] = st_values
    if rgba_values is not None:
        data[:,rgba_offset:] = rgba_values

    if type == 'triangles':
        # counter clockwise
        # top right -> top left -> bottom left
        # top right -> bottom left -> bottom right
        indices = np.tile(np.array([0, 1, 2, 0, 2, 3], dtype='int'), (6,1))
        for face in range(6):
            indices[face] += (face * 4)
        indices.shape = (-1,)


    return data, indices

def getCubeData(scale=(2,2,2), st=False, rgb=np.array([1.0, 1.0, 1.0])):
        dataCube, facesCube = create_cube(scale=(1,1,1), st=False, rgba=np.array([rgb[0], rgb[1], rgb[2], 1.0]), dtype='float32', type='triangles')
        verticesCube = ch.Ch(dataCube[:,0:3])
        UVsCube = ch.Ch(np.zeros([verticesCube.shape[0],2]))

        facesCube = facesCube.reshape([-1,3])
        normalsCube = geometry.chGetNormals(verticesCube, facesCube)
        haveTexturesCube = [[False]]
        texturesListCube = [[None]]
        vColorsCube = ch.Ch(dataCube[:,3:6])

        return verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube



def setupCamera(v, cameraParams):

    chDistMat = geometry.Translate(x=0, y=cameraParams['Zshift'], z=0)

    chRotElMat = geometry.RotateX(a=-cameraParams['chCamEl'])

    chCamModelWorld = ch.dot(chRotElMat, chDistMat)

    flipZYRotation = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 0, 1.0, 0.0],
                                [0.0, -1.0, 0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

    chMVMat = ch.dot(chCamModelWorld, flipZYRotation)

    chInvCam = ch.inv(chMVMat)

    modelRotation = chInvCam[0:3,0:3]

    chRod = opendr.geometry.Rodrigues(rt=modelRotation).reshape(3)
    chTranslation = chInvCam[0:3,3]

    translation, rotation = (chTranslation, chRod)

    camera = ProjectPoints(v=v, rt=rotation, t=translation, f = 1000*cameraParams['chCamFocalLength']*cameraParams['a'], c=cameraParams['c'], k=ch.zeros(5))

    flipXRotation = np.array([[1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0., 0.0],
            [0.0, 0., -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

    camera.openglMat = flipXRotation #Needed to match OpenGL flipped axis.

    return camera, modelRotation, chMVMat


def computeSphericalHarmonics(vn, vc, light_color, components):
    rangeMeshes = range(len(vn))
    A_list = [SphericalHarmonics(vn=vn[mesh],
                       components=components,
                       light_color=light_color) for mesh in rangeMeshes]

    vc_list = [A_list[mesh]*vc[mesh] for mesh in rangeMeshes]
    return vc_list

def computeGlobalAndDirectionalLighting(vn, vc, chLightAzimuth, chLightElevation, chLightIntensity, chGlobalConstant, finalScale=1.):

    # Construct point light source
    rangeMeshes = range(len(vn))
    vc_list = []
    chRotAzMat = geometry.RotateZ(a=chLightAzimuth)[0:3,0:3]
    chRotElMat = geometry.RotateX(a=chLightElevation)[0:3,0:3]
    chLightVector = -ch.dot(chRotAzMat, ch.dot(chRotElMat, np.array([0,0,-1])))
    for mesh in rangeMeshes:
        l1 = ch.maximum(ch.dot(vn[mesh], chLightVector).reshape((-1,1)), 0.)
        vcmesh = (vc[mesh] * (chLightIntensity * l1 + chGlobalConstant) ) * finalScale
        vc_list = vc_list + [vcmesh]
    return vc_list

def computeGlobalAndPointLighting(v, vn, vc, light_pos, globalConstant, light_color):
    # Construct point light source
    rangeMeshes = range(len(vn))
    vc_list = []
    for mesh in rangeMeshes:
        l1 = LambertianPointLight(
            v=v[mesh],
            vn=vn[mesh],
            num_verts=len(v[mesh]),
            light_pos=light_pos,
            vc=vc[mesh],
            light_color=light_color)

        vcmesh = vc[mesh]*(l1 + globalConstant)
        vc_list = vc_list + [vcmesh]
    return vc_list

def setupTexturedRenderer(renderer, vstack, vch, f_list, vc_list, vnch, uv, haveTextures_list, textures_list, camera, frustum, sharedWin=None):
    f = []
    f_listflat = [item for sublist in f_list for item in sublist]
    lenMeshes = 0

    for mesh_i, mesh in enumerate(f_listflat):
        polygonLen = 0
        for polygons in mesh:
            f = f + [polygons + lenMeshes]
            polygonLen += len(polygons)
        lenMeshes += len(vch[mesh_i])
    fstack = np.vstack(f)

    if len(vnch)==1:
        vnstack = vnch[0]
    else:
        vnstack = ch.vstack(vnch)

    if len(vc_list)==1:
        vcstack = vc_list[0]
    else:
        vcstack = ch.vstack(vc_list)

    uvflat = [item for sublist in uv for item in sublist]
    ftstack = np.vstack(uvflat)

    texturesch = []
    textures_listflat = [item for sublist in textures_list for item in sublist]
    for texture_list in textures_listflat:
        if texture_list != None:
            for texture in texture_list:
                if texture != None:
                    texturesch = texturesch + [ch.array(texture)]

    if len(texturesch) == 0:
        texture_stack = ch.Ch([])
    elif len(texturesch) == 1:
        texture_stack = texturesch[0].ravel()
    else:
        texture_stack = ch.concatenate([tex.ravel() for tex in texturesch])

    haveTextures_listflat = [item for sublist in haveTextures_list for item in sublist]

    renderer.set(camera=camera, frustum=frustum, v=vstack, f=fstack, vn=vnstack, vc=vcstack, ft=ftstack, texture_stack=texture_stack, v_list=vch, f_list=f_listflat, vc_list=vc_list, ft_list=uvflat, textures_list=textures_listflat, haveUVs_list=haveTextures_listflat, bgcolor=ch.ones(3), overdraw=True)
    renderer.msaa = True
    renderer.sharedWin = sharedWin


def addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list):
    v.insert(0,vmod)
    f_list.insert(0,fmod_list)
    vc.insert(0,vcmod)
    vn.insert(0,vnmod)
    uv.insert(0,uvmod)
    haveTextures_list.insert(0,haveTexturesmod_list)
    textures_list.insert(0,texturesmod_list)

def addObjectDataLast(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list):
    v.insert(len(v),vmod)
    f_list.insert(len(f_list),fmod_list)
    vc.insert(len(vc),vcmod)
    vn.insert(len(vn),vnmod)
    uv.insert(len(uv),uvmod)
    haveTextures_list.insert(len(haveTextures_list),haveTexturesmod_list)
    textures_list.insert(len(textures_list),texturesmod_list)

def removeObjectData(objIdx, v, f_list, vc, vn, uv, haveTextures_list, textures_list):

    del v[objIdx]
    del f_list[objIdx]
    del vc[objIdx]
    del vn[objIdx]
    del uv[objIdx]
    del haveTextures_list[objIdx]
    del textures_list[objIdx]
