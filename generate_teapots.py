__author__ = 'pol'

import matplotlib.pyplot as plt
import glfw
import generative_models
from utils import *
import OpenGL.GL as GL
from utils import *
import h5py
import cv2
import os
plt.ion()
from OpenGL import contextdata

#__GL_THREADED_OPTIMIZATIONS

#Main script options:

glModes = ['glfw','mesa']
glMode = glModes[0]

np.random.seed(1)

width, height = (128, 128)
numPixels = width*height
shapeIm = [width, height,3]
win = -1
clip_start = 0.01
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

if glMode == 'glfw':
    #Initialize base GLFW context for the Demo and to share context among all renderers.
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS,32)
    glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
    win = glfw.create_window(width, height, "Demo",  None, None)
    glfw.make_context_current(win)

else:
    from OpenGL.raw.osmesa._types import *
    from OpenGL.raw.osmesa import mesa

winShared = None

gtCamElevation = np.pi/3
gtCamHeight = 0.3 #meters

chLightAzimuthGT = ch.Ch([0])
chLightElevationGT = ch.Ch([np.pi/3])
chLightIntensityGT = ch.Ch([1])
chGlobalConstantGT = ch.Ch([0.5])

chCamElGT = ch.Ch([gtCamElevation])
focalLenght = 35 ##milimeters
chCamFocalLengthGT = ch.Ch([35/1000])

#Move camera backwards to match the elevation desired as it looks at origin:
# bottomElev = np.pi/2 - (gtCamElevation + np.arctan(17.5 / focalLenght ))
# ZshiftGT =  ch.Ch(-gtCamHeight * np.tan(bottomElev)) #Move camera backwards to match the elevation desired as it looks at origin.

ZshiftGT =  ch.Ch([-0.5])

# Baackground cube - add to renderer by default.
verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

uvCube = np.zeros([verticesCube.shape[0],2])

chCubePosition = ch.Ch([0, 0, 0])
chCubeScale = ch.Ch([10.0])
chCubeAzimuth = ch.Ch([0])
chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 1) #white cube
v_transf, vn_transf = transformObject([verticesCube], [normalsCube], chCubeScale, chCubeAzimuth, chCubePosition)

v_scene = [v_transf]
f_list_scene = [[[facesCube]]]
vc_scene = [[chCubeVCColors]]
vn_scene = [vn_transf]
uv_scene = [[uvCube]]
haveTextures_list_scene = [haveTexturesCube]
textures_list_scene = [texturesListCube]

#Example object 1: teapot

chPositionGT = ch.Ch([0, 0, 0.])
# chPositionGT = ch.Ch([-0.23, 0.36, 0.])
chScaleGT = ch.Ch([1.0, 1.0, 1.0])
chColorGT = ch.Ch([1.0, 1.0, 1.0])
chAzimuthGT = ch.Ch([np.pi/3])
chVColorsGT = ch.Ch([0.7, 0.0, 0.0])

import shape_model
# %% Load data
#You can get the teapot data from here: https://drive.google.com/open?id=1JO5ZsXHb_KTsjFMFx7rxY0YVAwnM3TMY
filePath = 'data/teapotModel.pkl'
teapotModel = shape_model.loadObject(filePath)
faces = teapotModel['faces']

# %% Sample random shape Params
latentDim = np.shape(teapotModel['ppcaW'])[1]
shapeParams = np.zeros(latentDim)
chShapeParams = ch.Ch(shapeParams.copy())

meshLinearTransform = teapotModel['meshLinearTransform']
W = teapotModel['ppcaW']
b = teapotModel['ppcaB']

chVertices = shape_model.VerticesModel(chShapeParams=chShapeParams, meshLinearTransform=meshLinearTransform, W=W, b=b)
chVertices.init()

chVertices = ch.dot(geometry.RotateZ(-np.pi/2)[0:3, 0:3], chVertices.T).T

smFaces = [[faces]]
smVColors = [chVColorsGT * np.ones(chVertices.shape)]
smUVs = ch.Ch(np.zeros([chVertices.shape[0],2]))
smHaveTextures = [[False]]
smTexturesList = [[None]]

chVertices = chVertices - ch.mean(chVertices, axis=0)

chVertices = chVertices * 0.09
smCenter = ch.array([0, 0, 0.1])

smVertices = [chVertices]
chNormals = shape_model.chGetNormals(chVertices, faces)
smNormals = [chNormals]

center = smCenter
UVs = smUVs
v = smVertices
vn = smNormals
Faces = smFaces
VColors = smVColors
HaveTextures = smHaveTextures
TexturesList = smTexturesList

v_transf, vn_transf = transformObject(v, vn, chScaleGT, chAzimuthGT, chPositionGT)

vc_illuminated = computeGlobalAndDirectionalLighting(vn_transf, VColors, chLightAzimuthGT, chLightElevationGT, chLightIntensityGT, chGlobalConstantGT)

v_scene += [v_transf]
f_list_scene += [smFaces]
vc_scene += [vc_illuminated]
vn_scene += [vn_transf]
uv_scene += [UVs]
haveTextures_list_scene += [HaveTextures]
textures_list_scene += [TexturesList]

#COnfigure lighting
lightParamsGT = {'chLightAzimuth': chLightAzimuthGT, 'chLightElevation': chLightElevationGT, 'chLightIntensity': chLightIntensityGT, 'chGlobalConstant':chGlobalConstantGT}

c0 = width/2  #principal point
c1 = height/2  #principal point
a1 = 3.657  #Aspect ratio / mm to pixels
a2 = 3.657  #Aspect ratio / mm to pixels

cameraParamsGT = {'Zshift':ZshiftGT, 'chCamEl': chCamElGT, 'chCamFocalLength':chCamFocalLengthGT, 'a':np.array([a1,a2]), 'width': width, 'height':height, 'c':np.array([c0, c1])}

#Create renderer object
renderer = createRenderer(glMode, cameraParamsGT, v_scene, vc_scene, f_list_scene, vn_scene, uv_scene, haveTextures_list_scene,
                               textures_list_scene, frustum, None)
# Initialize renderer
renderer.overdraw = True
renderer.nsamples = 8
renderer.msaa = True  #Without anti-aliasing optimization often does not work.
renderer.initGL()
renderer.initGLTexture()
renderer.debug = False
winShared = renderer.win


print("Creating Ground Truth")

trainAzsGT = np.array([])
trainObjAzsGT = np.array([])
trainElevsGT = np.array([])
trainLightAzsGT = np.array([])
trainLightElevsGT = np.array([])
trainLightIntensitiesGT = np.array([])
trainVColorGT = np.array([])
trainIds = np.array([], dtype=np.uint32)
trainAmbientIntensityGT = np.array([])
trainShapeModelCoeffsGT = np.array([]).reshape([0,latentDim])

prefix = 'teapot_dataset'
gtDir = 'groundtruth/' + prefix + '/'
if not os.path.exists(gtDir + 'images/'):
    os.makedirs(gtDir + 'images/')


gtDtype = [('trainIds', trainIds.dtype.name),
           ('trainAzsGT', trainAzsGT.dtype.name),
           ('trainElevsGT', trainElevsGT.dtype.name),
           ('trainLightAzsGT', trainLightAzsGT.dtype.name),
           ('trainLightElevsGT', trainLightElevsGT.dtype.name),
           ('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),
           ('trainAmbientIntensityGT', trainAmbientIntensityGT.dtype),
           ('trainVColorGT', trainVColorGT.dtype.name, (3,) ),
           ('trainShapeModelCoeffsGT', trainShapeModelCoeffsGT.dtype, (latentDim,)),]


groundTruth = np.array([], dtype = gtDtype)
groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'a')

gtDataset = gtDataFile.create_dataset(prefix, data=groundTruth, maxshape=(None,))

np.random.seed(1) #Set a seed for reproducibility.

dataset_size = 10
for train_i in np.arange(dataset_size):
    chAzGTVals = np.mod(np.random.uniform(0, np.pi, 1) - np.pi / 2, 2 * np.pi)
    chElGTVals = np.random.uniform(0.05, np.pi / 2, 1)
    chLightAzGTVals = np.random.uniform(0, 2 * np.pi, 1)
    chLightElGTVals = np.random.uniform(0, np.pi / 2, 1)
    chAmbientIntensityGTVals = np.random.uniform(0.4, 0.6)
    chLightIntensityGTVals = np.random.uniform(0.8, 1.0)
    chVColorsGTVals = np.random.uniform(0.0, 1.0, [1, 3])
    shapeParamsVals = np.random.randn(latentDim)

    chAzimuthGT[:] = chAzGTVals
    chCamElGT[:] = chElGTVals
    chLightAzimuthGT[:] = chLightAzGTVals
    chLightElevationGT[:] = chLightElGTVals
    chLightIntensityGT[:] = chLightIntensityGTVals
    chGlobalConstantGT[:] = chAmbientIntensityGTVals
    chVColorsGT[:] = chVColorsGTVals
    chShapeParams[:] = shapeParamsVals

    image = renderer.r.copy()

    cv2.imwrite(gtDir + 'images/im' + str(train_i) + '.jpeg', 255 * image[:, :, [2, 1, 0]],
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    gtDataset.resize(gtDataset.shape[0] + 1, axis=0)
    gtDataset[-1] = np.array([(train_i,
                               chAzGTVals,
                               chElGTVals,
                               chLightAzGTVals,
                               chLightElGTVals,
                               chLightIntensityGTVals,
                               chAmbientIntensityGTVals,
                               chVColorsGTVals,
                               shapeParamsVals,
                               )], dtype=gtDtype)

    gtDataFile.flush()
    print("Generated " + str(train_i) + " GT instances.")

plt.figure()
plt.title('GT object')
plt.imshow(renderer.r)
plt.show(0.1)


#Clean up.
renderer.makeCurrentContext()
renderer.clear()
contextdata.cleanupContext(contextdata.getContext())
# glfw.destroy_window(renderer.win)
del renderer