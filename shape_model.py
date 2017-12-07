import numpy as np
import pickle
import chumpy as ch
import ipdb
from chumpy import depends_on, Ch
import scipy.sparse as sp

#%% Helper functions
def longToPoints3D(pointsLong):
    nPointsLong = np.size(pointsLong)
    return np.reshape(pointsLong, [nPointsLong/3, 3])


def shapeParamsToVerts(shapeParams, teapotModel):
    landmarksLong = shapeParams.dot(teapotModel['ppcaW'].T) + teapotModel['ppcaB']
    landmarks = longToPoints3D(landmarksLong)
    vertices = teapotModel['meshLinearTransform'].dot(landmarks)
    return vertices

def chShapeParamsToVerts(landmarks, meshLinearTransform):
    vertices = ch.dot(meshLinearTransform,landmarks)
    return vertices

class VerticesModel(Ch):
    terms = 'meshLinearTransform', 'W', 'b'
    dterms = 'chShapeParams'

    def init(self):
        self.jac = self.meshLinearTransform.dot(self.W.reshape([self.meshLinearTransform.shape[1], -1,  len(self.chShapeParams)]).transpose((1,0,2))).reshape([-1,len(self.chShapeParams)])

    def compute_r(self):
        landmarks = np.dot(self.chShapeParams.r,self.W.T) + self.b
        landmarks = landmarks.reshape([-1,3])
        return np.dot(self.meshLinearTransform, landmarks)


    def compute_dr_wrt(self,wrt):
        if self.chShapeParams is wrt:
            # ipdb.set_trace()
            return self.jac
        return None

def chShapeParamsToNormals(N, landmarks, linT):
    T = ch.dot(linT,landmarks)
    invT = []
    nLandmarks = landmarks.r.shape[0]
    for i in range(nLandmarks):
        R = T[4*i:4*i+3,:3].T
        invR = ch.inv(R.T)
        invT = invT + [invR]

    invT = ch.vstack(invT)
    newNormals = ch.dot(N, invT)

    import opendr.geometry
    n = opendr.geometry.NormalizedNx3(newNormals)

    return newNormals

def getT(targetPoints, linT):
    T = linT.dot(targetPoints)
    return T

def shapeParamsToNormals(shapeParams, teapotModel):
    landmarksLong = shapeParams.dot(teapotModel['ppcaW'].T) + teapotModel['ppcaB']
    landmarks = longToPoints3D(landmarksLong)
    T = getT(landmarks, teapotModel['linT'])
    nLandmarks = np.shape(landmarks)[0]
    invT = np.empty([3*nLandmarks, 3])
    for i in range(nLandmarks):
        R = T[4*i:4*i+3,:3].T
        invR = np.linalg.inv(R)
        invT[3*i:3*(i+1),:] = invR
    newNormals = np.array(teapotModel['N'].dot(invT))
    normalize_v3(newNormals)

    return newNormals

def saveObj(vertices, faces, normals, filePath):
    with open(filePath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for n in normals:
            f.write("vn %.4f %.4f %.4f\n" % (n[0], n[1], n[2]))
        for p in faces:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

def loadObject(fileName):
    with open(fileName, 'rb') as inpt:
        return pickle.load(inpt)

def normalize_v3(arr):
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr


def getNormals(vertices, faces):
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    normalize_v3(n)
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    normalize_v3(norm)
    return norm

def chGetNormals(vertices, faces):
    import opendr.geometry
    return opendr.geometry.VertNormals(vertices, faces).reshape((-1,3))
