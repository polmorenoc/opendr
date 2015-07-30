#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['ColoredRenderer', 'TexturedRenderer']

import numpy as np
import cv2
import time
import platform
import scipy.sparse as sp
from copy import deepcopy
from opendr import common
from opendr.topology import get_vertices_per_edge, get_faces_per_edge

# if platform.system()=='Darwin':
#     from opendr.contexts.ctx_mac import OsContext
# else:
#     from opendr.contexts.ctx_mesa import OsContext

import OpenGL.GL as GL
import OpenGL.GL.shaders as shaders
import glfw
from OpenGL.arrays import vbo
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
from chumpy import *
# from opendr.contexts._constants import *
from chumpy.utils import row, col
import time

pixel_center_offset = 0.5


class BaseRenderer(Ch):
    terms = ['f', 'frustum','overdraw', 'win', 'f_list', 'v_list', 'vn_list', 'vc_list']
    dterms = ['camera', 'v']

    def clear(self):
        # ipdb.set_trace()
        try:
            self.win
        except:
            print ("Clearing when not initialized.")
            return

        if not self.win:
            GL.glDeleteProgram(self.colorProgram)
            glfw.terminate()
            self.win = 0

    def __del__(self):
        self.clear()

    def initGL(self):
        try:
            self.frustum
            self.f
            self.v
            self.vc
        except:
            print ("Necessary variables have not been set (frustum, f, v, or vc).")
            return

        # self.clear()

        glfw.init()
        print("Initializing GLFW.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS,32)

        glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
        self.win = glfw.create_window(self.frustum['width'], self.frustum['height'], "test",  None, None)
        glfw.make_context_current(self.win)
        GL.USE_ACCELERATE = True
        GL.glViewport(0, 0, self.frustum['width'], self.frustum['height'])

        #FBO_f
        self.fbo = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo )

        render_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,render_buf)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf)

        z_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, z_buf)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,  GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, z_buf)

        #FBO_f
        self.fbo_ms = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_ms )

        render_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,render_buf)
        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, 8, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf)

        z_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, z_buf)
        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, 8, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, z_buf)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print ("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print ("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)

        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        ############################
        # ENABLE SHADER

        FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
// Interpolated values from the vertex shaders
in vec3 theColor;
// Ouput data
out vec3 color;
void main(){
	color = theColor;
}""", GL.GL_FRAGMENT_SHADER)


        VERTEX_SHADER = shaders.compileShader("""#version 330 core
// Input vertex data, different for all executions of this shader.
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
uniform mat4 MVP;
out vec3 theColor;
// Values that stay constant for the whole mesh.
void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP* vec4(position,1);
	theColor = color;
}""", GL.GL_VERTEX_SHADER)

        self.colorProgram = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)

        shaders.glUseProgram(self.colorProgram)

        # self.colorProgram = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)

        position_location = GL.glGetAttribLocation(self.colorProgram, 'position')
        color_location = GL.glGetAttribLocation(self.colorProgram, 'color')
        # color_location_ub = GL.glGetAttribLocation(self.colorProgram, 'color')
        self.MVP_location = GL.glGetUniformLocation(self.colorProgram, 'MVP')
        #
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        indices = np.array(self.f, dtype=np.uint32)
        self.vbo_indices = vbo.VBO(indices, target=GL.GL_ELEMENT_ARRAY_BUFFER)
        self.vbo_indices_range = vbo.VBO(np.arange(self.f.size, dtype=np.uint32).ravel(), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        self.vbo_indices_dyn = vbo.VBO(indices, target=GL.GL_ELEMENT_ARRAY_BUFFER)

        self.vbo_verts = vbo.VBO(np.array(self.v, dtype=np.float32))
        self.vbo_verts_face = vbo.VBO(self.verts_by_face.astype(np.float32))
        self.vbo_verts_dyn = vbo.VBO(np.array(self.v, dtype=np.float32))

        self.vbo_colors =  vbo.VBO(np.array(self.vc, dtype=np.float32))

        self.vbo_colors_face = vbo.VBO(np.array(self.vc_by_face, dtype=np.float32))

        self.vao_static = GL.GLuint(0)

        GL.glGenVertexArrays(1, self.vao_static)
        GL.glBindVertexArray(self.vao_static)

        self.vbo_indices.bind()
        self.vbo_verts.bind()
        GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_colors.bind()
        GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_static_face = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_static_face)
        GL.glBindVertexArray(self.vao_static_face)

        #Can arrays be empty?
        # ipdb.set_trace()
        self.vbo_indices_range.bind()
        self.vbo_verts_face.bind()
        GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_colors_face.bind()
        GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_dyn = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_dyn)
        GL.glBindVertexArray(self.vao_dyn)

        #Can arrays be empty?
        # ipdb.set_trace()

        self.vbo_indices_dyn.bind()
        self.vbo_verts_dyn.bind()
        GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_colors.bind()
        GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_dyn_ub = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_dyn_ub)
        GL.glBindVertexArray(self.vao_dyn_ub)

        self.vbo_indices_dyn.bind()
        self.vbo_verts_dyn.bind()
        GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self.vbo_colors_ub = vbo.VBO(np.array(np.array(self.vc, dtype=np.uint8)))
        self.vbo_colors_ub.bind()
        GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(color_location, 3, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 0, None)

        print('glValidateProgram: ' + str(GL.glValidateProgram(self.colorProgram)))
        print('glGetProgramInfoLog ' + str(GL.glGetProgramInfoLog(self.colorProgram)))
        print('GL_MAX_VERTEX_ATTRIBS: ' + str(GL.glGetInteger(GL.GL_MAX_VERTEX_ATTRIBS)))

        print (GL.glGetError())

    @depends_on('f') # not v: specifically, it depends only on the number of vertices, not on the values in v
    def primitives_per_edge(self):
        v=self.v.r.reshape((-1,3))
        f=self.f
        vpe = get_vertices_per_edge(v, f)
        fpe = get_faces_per_edge(v, f, vpe)
        return fpe, vpe

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        return self.draw_barycentric_image(self.boundarybool_image if self.overdraw else None)

    @depends_on(terms+dterms)
    def boundaryid_image(self):
        self._call_on_changed()
        return self.draw_boundaryid_image( self.v.r, self.f, self.vpe, self.fpe, self.camera)

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()
        return self.draw_visibility_image(self.v.r, self.f, self.boundarybool_image if self.overdraw else None)

    @depends_on(terms+dterms)
    def boundarybool_image(self):
        self._call_on_changed()
        boundaryid_image = self.boundaryid_image
        return np.asarray(boundaryid_image != 4294967295, np.uint32).reshape(boundaryid_image.shape)

    @property
    def shape(self):
        raise NotImplementedError('Should be implemented in inherited class.')

    # @v.setter
    # def v(self, newval):
    #     self.camera.v = newval

    @property
    def vpe(self):
        return self.primitives_per_edge[1]

    @depends_on('f', 'v')
    def verts_by_face(self):
        verts_by_face = self.v.reshape((-1,3))[self.f.ravel()]
        return np.asarray(verts_by_face, dtype=np.float64, order='C')

    @depends_on('f', 'v')
    def vc_by_face(self):
        return np.asarray(np.tile(np.eye(3)[:self.f.shape[1], :], (self.verts_by_face.shape[0]/self.f.shape[1], 1)), dtype=np.float64, order='C')


    @depends_on('f', 'v')
    def tn(self):
        from opendr.geometry import TriNormals
        return TriNormals(self.v, self.f).r.reshape((-1,3))

    @property
    def fpe(self):
        return self.primitives_per_edge[0]

    def _setup_camera(self, cx, cy, fx, fy, w, h, near, far, view_matrix, k):
        k = np.asarray(k)
        #Make Projection matrix.
        self.projectionMatrix = np.array([[fx/cx, 0,0,0],   [0, fy/cy, 0,0],    [0,0, -(near + far)/(far - near), -2*near*far/(far-near)],   [0,0, -1, 0]], dtype=np.float32)
        # self.projectionMatrix = np.array([[fx/w, 0,0,0], [0, fy/cy, 0,0], [0,0, -(near + far)/(far - near), -2*near*far/(far-near)], [0,0,-1,1]], dtype=np.float64)


    def draw_colored_verts(self, vc):

        GL.glUseProgram(self.colorProgram)

        GL.glEnable(GL.GL_CULL_FACE)

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if vc.shape[1] != 3:
            #Pol: ??
            vc = np.vstack((vc[:,0], vc[:,1%vc.shape[1]], vc[:,2%vc.shape[1]])).T.copy()
        assert(vc.shape[1]==3)

        GL.glBindVertexArray(self.vao_static)


        self.vbo_colors.set_array(vc.astype(np.float32))
        self.vbo_colors.bind()

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glDrawElements(GL.GL_TRIANGLES, len(self.vbo_indices)*3, GL.GL_UNSIGNED_INT, None)

        GL.glDisable(GL.GL_CULL_FACE)


    def draw_noncolored_verts(self, v, f):
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        shaders.glUseProgram(self.colorProgram)
        GL.glBindVertexArray(self.vao_static)
        self.vbo_colors.set_array(np.zeros_like(v.reshape((-1,3))[f.ravel()], dtype=np.float32, order='C'))
        self.vbo_color.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.vbo_indices)*3, GL.GL_UNSIGNED_INT, None)


    def draw_edge_visibility(self, v, e, f, hidden_wireframe=True):
        """Assumes camera is set up correctly in gl context."""
        shaders.glUseProgram(self.colorProgram)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        ec = np.arange(1, len(e)+1)
        ec = np.tile(ec.reshape((-1,1)), (1, 3))
        ec[:, 0] = ec[:, 0] & 255
        ec[:, 1] = (ec[:, 1] >> 8 ) & 255
        ec[:, 2] = (ec[:, 2] >> 16 ) & 255
        ec = np.asarray(ec, dtype=np.uint8)

        # ipdb.set_trace()
        self.draw_colored_primitives(self.vao_dyn_ub, v, e, ec)

        if hidden_wireframe:
            # GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            #Pol change it to a smaller number to avoid double edges in my teapot.
            GL.glPolygonOffset(1.0, 1.0)
            # delta = -0.0
            # self.projectionMatrix[2,2] += delta
            # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            self.draw_colored_primitives(self.vao_dyn_ub, v, f, fc=np.zeros(f.shape).astype(np.uint8))
            # self.draw_colored_primitives(self.vaoub, v, e, np.zeros_like(ec).astype(np.uint8))
            # self.projectionMatrix[2,2] -= delta
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)


        raw = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))

        raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1

        return raw


    # this assumes that fc is either "by faces" or "verts by face", not "by verts"
    def draw_colored_primitives(self, vao, v, f, fc=None):
        GL.glUseProgram(self.colorProgram)

        # gl.EnableClientState(GL_VERTEX_ARRAY)
        verts_by_face = np.asarray(v.reshape((-1,3))[f.ravel()], dtype=np.float64, order='C')
        # gl.VertexPointer(verts_by_face)
        GL.glBindVertexArray(vao)

        self.vbo_verts_dyn.set_array(verts_by_face.astype(np.float32))
        self.vbo_verts_dyn.bind()

        if fc is not None:
            # gl.EnableClientState(GL_COLOR_ARRAY)
            if fc.size == verts_by_face.size:
                vc_by_face = fc
            else:
                vc_by_face = np.repeat(fc, f.shape[1], axis=0)

            if vc_by_face.size != verts_by_face.size:
                raise Exception('fc must have either rows=(#rows in faces) or rows=(# elements in faces)')

            if isinstance(fc[0,0], np.float32) or isinstance(fc[0,0], np.float64):
                vc_by_face = np.asarray(vc_by_face, dtype=np.float32, order='C')
                self.vbo_colors.set_array(vc_by_face)
                self.vbo_colors.bind()

            elif isinstance(fc[0,0], np.uint8):

                vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
                self.vbo_colors_ub.set_array(vc_by_face)
                self.vbo_colors_ub.bind()
            else:
                raise Exception('Unknown color type for fc')

        else:
            self.vbo_colors.set_array(np.zeros_like(verts_by_face, dtype=np.float32))
            self.vbo_colors.bind()


        if f.shape[1]==2:
            primtype = GL.GL_LINES
        else:
            primtype = GL.GL_TRIANGLES


        self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
        self.vbo_indices_dyn.bind()

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        # ipdb.set_trace()
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)

        #Pol: FIX THIS (UNCOMMENT)
        if primtype == GL.GL_LINES:
            f = np.fliplr(f).copy()
            verts_by_edge = v.reshape((-1,3))[f.ravel()]
            verts_by_edge = np.asarray(verts_by_edge, dtype=np.float32, order='C')

            self.vbo_verts_dyn.set_array(verts_by_edge)
            self.vbo_verts_dyn.bind()

            self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
            self.vbo_indices_dyn.bind()

            GL.glDrawElements(GL.GL_LINES, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)

    #Pol: Not used?
    # def draw_boundary_images(self, v, f, vpe, fpe, camera):
    #     """Assumes camera is set up correctly, and that glf has any texmapping on necessary."""
    #
    #     GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
    #     GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    #
    #     # Figure out which edges are on pairs of differently visible triangles
    #
    #     campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
    #     rays_to_verts = v.reshape((-1,3)) - row(campos)
    #     rays_to_faces = rays_to_verts[f[:,0]] + rays_to_verts[f[:,1]] + rays_to_verts[f[:,2]]
    #     dps = np.sum(rays_to_faces * self.tn, axis=1)
    #     dps = dps[fpe[:,0]] * dps[fpe[:,1]]
    #     silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)
    #     non_silhouette_edges = np.nonzero(dps>0)[0]
    #     lines_e = vpe[silhouette_edges]
    #     lines_v = v
    #
    #     visibility = self.draw_edge_visibility(lines_v, lines_e, f, hidden_wireframe=True)
    #
    #     shape = visibility.shape
    #     visibility = visibility.ravel()
    #     visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    #     visibility[visible] = silhouette_edges[visibility[visible]]
    #     result = visibility.reshape(shape)
    #     return result

    def compute_vpe_boundary_idxs(self, v, f, camera, fpe):

        # Figure out which edges are on pairs of differently visible triangles

        #ray = cv2.Rodrigues(camera.rt.r)[0].T[:,2]
        campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
        rays_to_verts = v.reshape((-1,3)) - row(campos)
        rays_to_faces = rays_to_verts.take(f[:,0],axis=0) +rays_to_verts.take(f[:,1],axis=0) +rays_to_verts.take(f[:,2],axis=0)
        # rays_to_faces = np.sum(rays_to_verts.take(f[:,:],axis=0), axis=1)
        faces_invisible = np.sum(rays_to_faces * self.tn, axis=1)

        dps = faces_invisible.take(fpe[:,0]) * faces_invisible.take(fpe[:,1])
        # dps = faces_invisible0 * faces_invisible1
        # idxs = (dps<=0) & (faces_invisible.take(fpe[:,0]) + faces_invisible.take(fpe[:,1]) > 0.0)
        silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)

        return silhouette_edges, faces_invisible < 0

    def draw_boundaryid_image(self, v, f, vpe, fpe, camera):

        GL.glUseProgram(self.colorProgram)

        if False:
            visibility = self.draw_edge_visibility(v, vpe, f, hidden_wireframe=True)
            return visibility

        if True:
        #try:
            view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);

            silhouette_edges, faces_facing_camera = self.compute_vpe_boundary_idxs(v, f, camera, fpe)

            lines_e = vpe[silhouette_edges]
            lines_v = v

            if len(lines_e)==0:
                return np.ones((self.frustum['height'], self.frustum['width'])).astype(np.int32) * 4294967295

            visibility = self.draw_edge_visibility(lines_v, lines_e, f, hidden_wireframe=True)
            # plt.imsave("opendr_boundary_edge_visibility.png", visibility)

            shape = visibility.shape
            visibility = visibility.ravel()
            visible = np.nonzero(visibility.ravel() != 4294967295)[0]
            visibility[visible] = silhouette_edges.take(visibility.take(visible))

            # plt.imsave("opendr_boundary_edge_visibility_result.png", visibility.reshape(shape))
            return visibility.reshape(shape)

    def draw_visibility_image(self, v, f, boundarybool_image=None):
        v = np.asarray(v)
        # gl.Disable(GL_TEXTURE_2D)
        # gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
        glfw.make_context_current(self.win)
        shaders.glUseProgram(self.colorProgram)

        result = self.draw_visibility_image_internal(v, f)
        if boundarybool_image is None:
            return result

        rr = result.ravel()
        faces_to_draw = np.unique(rr[rr != 4294967295])
        if len(faces_to_draw)==0:
            result = np.ones((self.frustum['height'], self.frustum['width'])).astype(np.uint32)*4294967295
            return result
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        result2 = self.draw_visibility_image_internal(v, f[faces_to_draw])
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        bbi = boundarybool_image

        result2 = result2.ravel()
        idxs = result2 != 4294967295
        result2[idxs] = faces_to_draw[result2[idxs]]

        if False:
            result2[result2==4294967295] = 0
            import matplotlib.pyplot as plt
            result2 = result2.reshape(result.shape[:2])
            plt.figure()
            plt.subplot(121)
            plt.imshow(result.squeeze())
            plt.subplot(122)
            plt.imshow(result2.squeeze())

        result2 = result2.reshape(result.shape[:2])

        return result2 * bbi + result * (1 - bbi)


    def draw_visibility_image_internal(self, v, f):
        """Assumes camera is set up correctly in gl context."""
        glfw.make_context_current(self.win)
        GL.glUseProgram(self.colorProgram)

        #Attach FBO
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        fc = np.arange(1, len(f)+1)
        fc = np.tile(fc.reshape((-1,1)), (1, 3))
        fc[:, 0] = fc[:, 0] & 255
        fc[:, 1] = (fc[:, 1] >> 8 ) & 255
        fc[:, 2] = (fc[:, 2] >> 16 ) & 255
        fc = np.asarray(fc, dtype=np.uint8)

        self.draw_colored_primitives(self.vao_dyn_ub,  v, f, fc)

        #Read image.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        raw = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))
        # plt.imsave("draw_edge_visibility_internal_raw1.png", raw)

        return raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1

    def draw_barycentric_image(self, boundarybool_image=None):
        GL.glDisable(GL.GL_CULL_FACE)

        without_overdraw = self.draw_barycentric_image_internal()
        if boundarybool_image is None:
            plt.imsave("barycentric_image_without_overdraw.png", without_overdraw)
            return without_overdraw

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        overdraw = self.draw_barycentric_image_internal()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        bbi = np.atleast_3d(boundarybool_image)
        return bbi * overdraw + (1. - bbi) * without_overdraw


    def draw_barycentric_image_internal(self):

        glfw.make_context_current(self.win)
        GL.glUseProgram(self.colorProgram)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glBindVertexArray(self.vao_static_face)


        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        GL.glDrawElements(GL.GL_TRIANGLES if self.f.shape[1]==3 else GL.GL_LINES, len(self.vbo_indices_range), GL.GL_UNSIGNED_INT, None)

        #Read image.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        # return np.array(im.transpose(Image.FLIP_TOP_BOTTOM), np.float64)/255.0
        return np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0


    def setup_camera(self, camera, frustum):
        self._setup_camera(camera.c.r[0], camera.c.r[1],
                  camera.f.r[0], camera.f.r[1],
                  frustum['width'], frustum['height'],
                  frustum['near'], frustum['far'],
                  camera.view_matrix,
                  camera.k.r)

    # # May end up using this, maybe not
    # def get_inbetween_boundaries(self):
    #     camera = self.camera
    #     frustum = self.frustum
    #     w = frustum['width']
    #     h = frustum['height']
    #     far = frustum['far']
    #     near = frustum['near']
    #
    #     self.glb.Viewport(0, 0, w-1, h)
    #     self._setup_camera(self.fbo,
    #                   camera.c.r[0]-.5, camera.c.r[1],
    #                   camera.f.r[0], camera.f.r[1],
    #                   w-1, h,
    #                   near, far,
    #                   camera.view_matrix, camera.k)
    #     bnd_x = self.draw_boundaryid_image(self.fbo, self.v.r, self.f, self.vpe, self.fpe, self.camera)[:,:-1]
    #
    #     self.glb.Viewport(0, 0, w, h-1)
    #     self._setup_camera(self.fbo,
    #                   camera.c.r[0], camera.c.r[1]-.5,
    #                   camera.f.r[0], camera.f.r[1],
    #                   w, h-1,
    #                   near, far,
    #                   camera.view_matrix, camera.k)
    #     bnd_y = self.draw_boundaryid_image(self.fbo_b, self.v.r, self.f, self.vpe, self.fpe, self.camera)[:-1,:]
    #
    #     # Put things back to normal
    #     self.glb.Viewport(0, 0, w, h)
    #     self.setup_camera(self.fbo_b, camera, frustum)
    #     return bnd_x, bnd_y


class ColoredRenderer(BaseRenderer):
    terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
    dterms = 'vc', 'camera', 'bgcolor' , 'v'

    @depends_on('vc')
    def num_channels(self):
        if hasattr(self, 'vc'):
            return self.vc.shape[1]
        return 3

    @property
    def shape(self):
        if not hasattr(self, 'num_channels'):
            self.num_channels = 3
        if self.num_channels > 1:
            return (self.frustum['height'], self.frustum['width'], self.num_channels)
        else:
            return (self.frustum['height'], self.frustum['width'])

    def compute_r(self):
        return self.color_image # .reshape((self.frustum['height'], self.frustum['width'], -1)).squeeze()

    def compute_dr_wrt(self, wrt):
        if wrt is not self.camera and wrt is not self.vc and wrt is not self.bgcolor:
            return None

        visibility = self.visibility_image

        shape = visibility.shape
        color = self.color_image

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image

        if wrt is self.camera:
            if self.overdraw:
                return common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
            else:
                return common.dImage_wrt_2dVerts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)

        elif wrt is self.vc:
            return common.dr_wrt_vc(visible, visibility, self.f, barycentric, self.frustum, self.vc.size, num_channels=self.num_channels)

        elif wrt is self.bgcolor:
            return common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

    def on_changed(self, which):
        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.initGL()

        if 'frustum' in which or 'camera' in which:
            self.setup_camera(self.camera, self.frustum)
            # setup_camera(self.glf, self.camera, self.frustum)

        if not hasattr(self, 'num_channels'):
            self.num_channels = 3

        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5]*self.num_channels))
            which.add('bgcolor')

        if not hasattr(self, 'overdraw'):
            self.overdraw = True

        if 'v' or 'f' in which:
            self.vbo_verts_face.set_array(self.verts_by_face.astype(np.float32))
            self.vbo_verts.bind()
            self.vbo_colors_face.set_array(self.vc_by_face.astype(np.float32))
            self.vbo_colors_face.bind()

        if 'v' in which:
            self.vbo_verts.set_array(self.v.r.astype(np.float32))
            self.vbo_verts.bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()


    def flow_to(self, v_next, cam_next=None):
        return common.flow_to(self, v_next, cam_next)

    def filter_for_triangles(self, which_triangles):
        cim = self.color_image
        vim = self.visibility_image+1
        arr = np.zeros(len(self.f)+1)
        arr[which_triangles+1] = 1

        relevant_pixels = arr[vim.ravel()]
        cim2 = cim.copy() * np.atleast_3d(relevant_pixels.reshape(vim.shape))
        relevant_pixels = np.nonzero(arr[vim.ravel()])[0]
        xs = relevant_pixels % vim.shape[1]
        ys = relevant_pixels / vim.shape[1]
        return cim2[np.min(ys):np.max(ys), np.min(xs):np.max(xs), :]


    # @depends_on('f', 'camera', 'vc')
    # def boundarycolor_image(self):
    #
    #     try:
    #         return self.draw_boundarycolor_image(with_vertex_colors=True)
    #     except:
    #         import pdb; pdb.set_trace()

    def draw_color_image(self):
        glfw.make_context_current(self.win)
        self._call_on_changed()
        try:

            GL.glEnable(GL.GL_MULTISAMPLE)
            if hasattr(self, 'bgcolor'):
                GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)
            # use face colors if given
            # FIXME: this won't work for 2 channels
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


            self.draw_colored_verts(self.vc.r)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'], GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

            result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0

            if hasattr(self, 'background_image'):
                bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1,1,self.num_channels)).squeeze()
                fg_px = 1 - bg_px
                result = bg_px * self.background_image + fg_px * result

            # plt.imsave("opendr_draw_color_image.png", result)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glDisable(GL.GL_MULTISAMPLE)
            GL.glClearColor(0.,0.,0., 1.)

            return result
        except:
            import pdb; pdb.set_trace()

    @depends_on(dterms+terms)
    def color_image(self):

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image()

        #Pol: why do we add the lines edges in the final render?
        return no_overdraw

        if not self.overdraw:
            return no_overdraw

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        overdraw = self.draw_color_image()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # return overdraw * np.atleast_3d(self.boundarybool_image)

        boundarybool_image = self.boundarybool_image
        if self.num_channels > 1:
            boundarybool_image = np.atleast_3d(boundarybool_image)

        return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')

    #Pol: Not used?
    # @depends_on('f', 'frustum', 'camera')
    # def boundary_images(self):
    #     self._call_on_changed()
    #     return self.draw_boundary_images(self.v.r, self.f, self.vpe, self.fpe, self.camera)

    # Pol: Commented this out because seems it's not being used.
    # @depends_on(terms+dterms)
    # def boundarycolor_image(self):
    #     self._call_on_changed()
    #     try:
    #         GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
    #         GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    #
    #         colors = self.vc.r.reshape((-1,3))[self.vpe.ravel()]
    #
    #         self.draw_colored_primitives(self.vao, self.v.r.reshape((-1,3)), self.vpe, colors)
    #
    #         return np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0
    #     except:
    #         import pdb; pdb.set_trace()


class TexturedRenderer(ColoredRenderer):
    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'


    def initGLTexture(self):
        print("Initializing Texture OpenGL.")

        FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
// Interpolated values from the vertex shaders
in vec3 theColor;
in vec2 UV;
uniform sampler2D myTextureSampler;
// Ouput data
out vec3 color;
void main(){
	color = theColor * texture2D( myTextureSampler, UV ).rgb;
}""", GL.GL_FRAGMENT_SHADER)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
// Input vertex data, different for all executions of this shader.
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout(location = 2) in vec2 vertexUV;
uniform mat4 MVP;
out vec3 theColor;
out vec2 UV;
// Values that stay constant for the whole mesh.
void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP* vec4(position,1);
	theColor = color;
	UV = vertexUV;
}""", GL.GL_VERTEX_SHADER)

        self.colorTextureProgram = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)

        #Define the other VAO/VBOs and shaders.
        #Text VAO and bind color, vertex indices AND uvbuffer:

        self.vbo_indices_mesh_list = []
        self.vbo_colors_mesh = []
        self.vbo_verts_mesh = []
        self.vao_tex_mesh_list = []
        self.vbo_uvs_mesh = []

        position_location = GL.glGetAttribLocation(self.colorTextureProgram, 'position')
        color_location = GL.glGetAttribLocation(self.colorTextureProgram, 'color')
        uvs_location = GL.glGetAttribLocation(self.colorTextureProgram, 'vertexUV')
        # color_location_ub = GL.glGetAttribLocation(self.colorProgram, 'color')
        self.MVP_texture_location = GL.glGetUniformLocation(self.colorTextureProgram, 'MVP')

        # ipdb.set_trace()

        self.textureID_mesh_list = []

        for mesh in range(len(self.f_list)):

            vbo_verts = vbo.VBO(np.array(self.v_list[mesh]).astype(np.float32))
            vbo_colors = vbo.VBO(np.array(self.vc_list[mesh]).astype(np.float32))
            vbo_uvs = vbo.VBO(np.array(self.ft_list[mesh]).astype(np.float32))


            self.vbo_colors_mesh = self.vbo_colors_mesh + [vbo_colors]
            self.vbo_verts_mesh = self.vbo_verts_mesh + [vbo_verts]
            self.vbo_uvs_mesh = self.vbo_uvs_mesh + [vbo_uvs]

            vaos_mesh = []
            vbo_indices_mesh = []
            textureIDs_mesh = []
            for polygons in range(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                vbo_indices = vbo.VBO(np.array(self.f_list[mesh][polygons]).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
                vbo_indices.bind()
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                vbo_colors.bind()
                GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                if self.haveUVs_list[mesh][polygons]:
                    vbo_uvs.bind()
                    GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
                    GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                #Textures:

                texture = None
                if self.haveUVs_list[mesh][polygons]:
                    texture = GL.GLuint(0)
                    # ipdb.set_trace()
                    GL.glGenTextures( 1, texture )
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
                    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
                    # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT)
                    # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_MIRRORED_REPEAT)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                    #Send texture.
                    #Pol: Check if textures are float or uint from Blender import.
                    # ipdb.set_trace()
                    image = np.array(np.flipud((self.textures_list[mesh][polygons]*255.0)), order='C', dtype=np.uint8)
                    GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB8, image.shape[1], image.shape[0])
                    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())
                textureIDs_mesh = textureIDs_mesh + [texture]
                vbo_indices_mesh = vbo_indices_mesh + [vbo_indices]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID  = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")


    def __del__(self):
        self.release_textures()

    @property
    def shape(self):
        return (self.frustum['height'], self.frustum['width'], 3)

    @property
    def num_channels(self):
        return 3

    def release_textures(self):
        if hasattr(self, 'textureID_mesh_list'):
            if self.textureID_mesh_list != []:
                for texture_mesh in self.textureID_mesh_list:
                    if texture_mesh != None and texture_mesh != []:
                        for texture in texture_mesh:
                            GL.glDeleteTextures(1, texture)

        self.textureID_mesh_list = []

    def compute_r(self):
        return self.color_image # .reshape((self.frustum['height'], self.frustum['width'], -1)).squeeze()

    @depends_on(dterms+terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        #Pol: why do we add the lines edges in the final render?
        return no_overdraw

    def draw_texcoord_image(self, v, f, ft, boundarybool_image=None):
        # gl = glf
        # gl.Disable(GL_TEXTURE_2D)
        # gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
        glfw.make_context_current(self.win)
        shaders.glUseProgram(self.colorProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # want vtc: texture-coordinates per vertex (not per element in vc)
        colors = ft

        #use the third channel to identify the corresponding textures.
        color3 = np.vstack([np.ones([self.ft_list[mesh].shape[0],1])*mesh for mesh in range(len(self.ft_list))]).astype(np.float32) / len(self.ft_list)

        colors = np.asarray(np.hstack((colors, color3)), np.float64, order='C')
        self.draw_colored_primitives(self.vao_dyn, v, f, colors)

        #Pol: Why do we need this?
        if boundarybool_image is not None:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            self.draw_colored_primitives(self.vao_dyn, v, f, colors)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3)[:,:,:3].astype(np.float64))/255.0

        result[:,:,1] = 1. - result[:,:,1]
        return result

    def compute_dr_wrt(self, wrt):
        result = super(TexturedRenderer, self).compute_dr_wrt(wrt)

        if wrt is self.vc:
            cim = self.draw_color_image(with_vertex_colors=False).ravel()
            cim = sp.spdiags(row(cim), [0], cim.size, cim.size)
            result = cim.dot(result)
        elif wrt is self.texture_stack:
            ipdb.set_trace()
            IS = np.nonzero(self.visibility_image.ravel() != 4294967295)[0]
            texcoords, texidx = self.texcoord_image_quantized
            vis_texidx = texidx.ravel()[IS]
            vis_texcoords = texcoords.ravel()[IS]
            JS = vis_texcoords *  np.tile(col(vis_texidx), [1,2]).ravel()

            clr_im = self.draw_color_image(with_vertex_colors=True, with_texture_on=False)

            if False:
                cv2.imshow('clr_im', clr_im)
                # cv2.imshow('texmap', self.texture_image.r)
                cv2.waitKey(1)

            r = clr_im[:,:,0].ravel()[IS]
            g = clr_im[:,:,1].ravel()[IS]
            b = clr_im[:,:,2].ravel()[IS]
            data = np.concatenate((r,g,b))

            IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
            JS = np.concatenate((JS*3, JS*3+1, JS*3+2))


            return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))

        return result

    def on_changed(self, which):
        super(self.__class__, self).on_changed(which)

        # have to redo if frustum changes, b/c frustum triggers new context
        if 'frustum' in which:
            self.initGLTexture()

        if 'texture_stack' in which:
            # gl = self.glf
            # texture_data = np.array(self.texture_image*255., dtype='uint8', order='C')
            tmp = np.zeros(2, dtype=np.uint32)

            # self.release_textures()
            #
            # for mesh in range(len(self.f_list)):
            #     textureIDs = []
            #     for polygons in range(len(self.f_list[mesh])):
            #         texture = None
            #         if self.haveUVs_list[mesh][polygons]:
            #             texture = GL.GLuint(0)
            #             GL.glGenTextures( 1, texture )
            #             GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
            #             GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            #             GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            #             GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
            #             GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
            #             GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            #             GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            #             GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            #             #Send texture.
            #             #Pol: Check if textures are float or uint from Blender import.
            #             image = (self.textures_list[mesh][polygons]*255.0).astype(np.uint8)
            #             GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, image.shape[1], image.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)
            #         textureIDs = textureIDs + [texture]
            #     self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs]

            # gl.GenTextures(1, tmp) # TODO: free after done
            # self.textureID = tmp[0]

            # Pol: fix this on modern OpenGL:

            # self.textureID = GL.glGenTextures(1)
            # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureID)
            # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            #
            # GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL.GL_BGR_EXT, texture_data.ravel())
            # gl.TexImage2Dub(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL.GL_BGR, texture_data.ravel())
            # #gl.Hint(GL_GENERATE_MIPMAP_HINT, GL_NICEST) # must be GL_FASTEST, GL_NICEST or GL_DONT_CARE
            # gl.GenerateMipmap(GL.GL_TEXTURE_2D)



    @depends_on('ft', 'textures')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = self.ft
        # Pol: careful with this:
        data[:,1] = 1.0 - 1.0*data[:,1]
        return data

    # Depends on 'f' because vpe/fpe depend on f
    # Pol: Check that depends on works on other attributes that depend_on x, if x changes.
    @depends_on( 'ft', 'f')
    def wireframe_tex_coords(self):
        print("wireframe_tex_coords is being computed!")
        vvt = np.zeros((self.v.r.size/3,2), dtype=np.float64, order='C')
        vvt[self.f.flatten()] = self.mesh_tex_coords
        edata = np.zeros((self.vpe.size,2), dtype=np.float64, order='C')
        edata = vvt[self.vpe.ravel()]
        return edata



    # TODO: can this not be inherited from base? turning off texture mapping in that instead?
    @depends_on(dterms+terms)
    def boundaryid_image(self):
        self._call_on_changed()

        # self.texture_mapping_off()
        glfw.make_context_current(self.win)
        GL.glUseProgram(self.colorProgram)

        result = self.draw_boundaryid_image(self.v.r, self.f, self.vpe, self.fpe, self.camera)

        GL.glUseProgram(self.colorTextureProgram)
        # self.texture_mapping_on(with_vertex_colors=True)

        return result

    def draw_color_image(self, with_vertex_colors=True, with_texture_on=True):
        glfw.make_context_current(self.win)
        self._call_on_changed()

        GL.glEnable(GL.GL_MULTISAMPLE)

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)
        for mesh in range(len(self.f_list)):

            vbo_color = self.vbo_colors_mesh[mesh]
            vc = self.vc_list[mesh]
            colors = None

            if with_vertex_colors:
                colors = vc.r.astype(np.float32)
            else:
                #Only texture.
                colors = np.ones_like(vc).astype(np.float32)

            #Pol: Make a static zero vbo_color to make it more efficient?
            vbo_color.set_array(colors)

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                vbo_color.bind()

                if self.f.shape[1]==2:
                    primtype = GL.GL_LINES
                else:
                    primtype = GL.GL_TRIANGLES


                if with_texture_on and self.haveUVs_list[mesh][polygons]:
                    GL.glUseProgram(self.colorTextureProgram)
                    texture = self.textureID_mesh_list[mesh][polygons]
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                    GL.glUniform1i(self.textureID, 0)

                    #Pol: do I need to do this everytime? I think not... Or maybe yes (according to Sceneviewer code).

                    GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
                else:
                    GL.glUseProgram(self.colorProgram)
                    GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

                # ipdb.set_trace()
                GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'], GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0

        if hasattr(self, 'background_image'):
            bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1,1,3))
            fg_px = 1 - bg_px
            result = bg_px * self.background_image + fg_px * result

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0.,0.,0., 1.)
        return result

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image_quantized(self):
        texcoord_image = self.texcoord_image[:,:2].copy()
        texcoord_image[:,:,0] *= self.texture_image.shape[1]-1
        texcoord_image[:,:,1] *= self.texture_image.shape[0]-1
        texture_idx = self.texcoord_image[:,2]*len(self.ft_list).astype(np.uint32)
        texcoord_image = np.round(texcoord_image)
        texcoord_image = texcoord_image[:,:,0] + texcoord_image[:,:,1]*self.texture_image.shape[1]
        return texcoord_image, texture_idx

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image(self):
        return self.draw_texcoord_image(self.v.r, self.f, self.ft, self.boundarybool_image if self.overdraw else None)

    # Pol: Commented this out because seems it's not being used.
    # @depends_on(terms+dterms)
    # def boundarycolor_image(self):
    #     self._call_on_changed()
    #     try:
    #         colors = self.vc.r.reshape((-1,3))[self.vpe.ravel()]
    #         GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    #         self.texture_mapping_on(with_vertex_colors=False if colors is None else True)
    #
    #         #Pol: nope.
    #         glTexCoordPointerf(2,0, self.wireframe_tex_coords.ravel())
    #
    #         self.draw_colored_primitives(self.v.r.reshape((-1,3)), self.vpe, colors)
    #
    #         self.texture_mapping_off()
    #
    #         return np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3)[:,:,:2].astype(np.float64))/255.0
    #     except:
    #         import pdb; pdb.set_trace()



# class DepthRenderer(BaseRenderer):
#     terms = 'f', 'frustum', 'background_image','overdraw'
#     dterms = 'camera', 'v'
#
#
#     @property
#     def shape(self):
#         return (self.frustum['height'], self.frustum['width'])
#
#     def compute_r(self):
#         tmp = self.camera.r
#         return self.depth_image.reshape((self.frustum['height'], self.frustum['width']))
#
#     def compute_dr_wrt(self, wrt):
#
#         if wrt is not self.camera and wrt is not self.v:
#             return None
#
#         visibility = self.visibility_image
#         visible = np.nonzero(visibility.ravel() != 4294967295)[0]
#         barycentric = self.barycentric_image
#         if wrt is self.camera:
#             shape = visibility.shape
#             depth = self.depth_image
#
#             if self.overdraw:
#                 result1 = common.dImage_wrt_2dVerts_bnd(depth, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
#             else:
#                 result1 = common.dImage_wrt_2dVerts(depth, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
#
#             # result1 = common.dImage_wrt_2dVerts(depth, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
#
#             return result1
#
#         elif wrt is self.v:
#
#             IS = np.tile(col(visible), (1, 9)).ravel()
#             JS = col(self.f[visibility.ravel()[visible]].ravel())
#             JS = np.hstack((JS*3, JS*3+1, JS*3+2)).ravel()
#
#             # FIXME: there should be a faster way to get the camera axis.
#             # But it should be carefully tested with distortion present!
#             pts = np.array([
#                 [self.camera.c.r[0], self.camera.c.r[1], 2],
#                 [self.camera.c.r[0], self.camera.c.r[1], 1]
#             ])
#             pts = self.camera.unproject_points(pts)
#             cam_axis = pts[0,:] - pts[1,:]
#
#             if True: # use barycentric coordinates (correct way)
#                 w = visibility.shape[1]
#                 pxs = np.asarray(visible % w, np.int32)
#                 pys = np.asarray(np.floor(np.floor(visible) / w), np.int32)
#                 bc0 = col(barycentric[pys, pxs, 0])
#                 bc1 = col(barycentric[pys, pxs, 1])
#                 bc2 = col(barycentric[pys, pxs, 2])
#                 bc = np.hstack((bc0,bc0,bc0,bc1,bc1,bc1,bc2,bc2,bc2)).ravel()
#             else: # each vert contributes equally (an approximation)
#                 bc = 1. / 3.
#
#             data = np.tile(row(cam_axis), (IS.size/3,1)).ravel() * bc
#             result2 = sp.csc_matrix((data, (IS, JS)), shape=(self.frustum['height']*self.frustum['width'], self.v.r.size))
#             return result2
#
#
#     def on_changed(self, which):
#
#         if 'frustum' in which:
#             w = self.frustum['width']
#             h = self.frustum['height']
#             self.initGL()
#             # self.glf = OsContext(w, h, typ=GL_FLOAT)
#             # self.glf.Viewport(0, 0, w, h)
#             # self.glb = OsContext(w, h, typ=GL_UNSIGNED_BYTE)
#             # self.glb.Viewport(0, 0, w, h)
#
#
#         if 'frustum' in which or 'camera' in which:
#             # setup_camera(self.camera, self.frustum)
#             self.setup_camera(self.camera, self.frustum)
#
#
#         # if 'v' in which or 'f' in which or 'frustum' in which or 'camera' in which:
#         #     self.initGL()
#
#         if not hasattr(self, 'overdraw'):
#             self.overdraw = True
#
#         assert(self.v is self.camera.v)
#
#
#     #Pol: need to create proper FBO for depth and see what gl.getDepth does.
#     @depends_on(dterms+terms)
#     def depth_image(self):
#         self._call_on_changed()
#
#         fbo = self.fbo_z
#
#         GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
#
#         GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
#
#         GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
#         self.draw_noncolored_verts(fbo, self.camera.v.r, self.f)
#
#         GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
#
#         GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
#         pixels = GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
#         im = Image.frombuffer("RGB", (self.frustum['width'], self.frustum['height']), pixels, "raw", "RGB", 0, 0)
#         result =  np.array(im.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float64)/255.0
#
#
#         if self.overdraw:
#             GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
#             self.draw_noncolored_verts(fbo, self.camera.v.r, self.f)
#
#             GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
#             GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
#             pixels = GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_FLOAT, GL.GL_FLOAT)
#             overdraw =  np.array(pixels, dtype=np.float64)
#
#             GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
#             boundarybool_image = self.boundarybool_image
#             result = overdraw*boundarybool_image + result*(1-boundarybool_image)
#
#         if hasattr(self, 'background_image'):
#             if False: # has problems at boundaries, not sure why yet
#                 bg_px = self.visibility_image == 4294967295
#                 fg_px = 1 - bg_px
#                 result = bg_px * self.background_image + fg_px * result
#             else:
#                 tmp = np.concatenate((np.atleast_3d(result), np.atleast_3d(self.background_image)), axis=2)
#                 result = np.min(tmp, axis=2)
#
#         return result
#
#     #Pol: what does glb.getDepthCloud do and what is depth_image?
#     def getDepthMesh(self, depth_image=None):
#         self._call_on_changed() # make everything is up-to-date
#         v = self.glb.getDepthCloud(depth_image)
#         w = self.frustum['width']
#         h = self.frustum['height']
#         idxs = np.arange(w*h).reshape((h, w))
#
#         # v0 is upperleft, v1 is upper right, v2 is lowerleft, v3 is lowerright
#         v0 = col(idxs[:-1,:-1])
#         v1 = col(idxs[:-1,1:])
#         v2 = col(idxs[1:,:-1])
#         v3 = col(idxs[1:,1:])
#
#         f = np.hstack((v0, v1, v2, v1, v3, v2)).reshape((-1,3))
#         return v, f


    
    
#
# vs_source = """
# #version 120
#
# uniform float k1, k2, k3, k4, k5, k6;
# uniform float p1, p2;
# uniform float cx, cy, fx, fy;
#
# void main()
# {
#     vec4 p0 = gl_ModelViewMatrix * gl_Vertex;
#
#     float xp = p0[0] / p0[2];
#     float yp = -p0[1] / p0[2];
#
#     float r2 = xp*xp + yp*yp;
#     float r4 = r2 * r2;
#     float r6 = r4 * r2;
#
#     float m = (1.0 + k1*r2 + k2*r4 + k3*r6) / (1.0 + k4*r2 + k5*r4 + k6*r6);
#     //p0[1] = -p0[1];
#     p0[0] = xp * m + 2.*p1*xp*yp + p2*(r2+2*xp*xp);
#     p0[1] = yp * m + p1*(r2+2*yp*yp) + 2.*p2*xp*yp;
#     //p0[1] = -p0[1];
#     p0[1] = -p0[1];
#
#     gl_Position = gl_ProjectionMatrix * p0;
#     //gl_Position = vec4(p0[0]*fx+cx, p0[1]*fy+cy, p0[2], p0[3]);
#     //gl_Position[0] = p0[0]*fx+cx;
#     //gl_Position[0] = p0[0];
#     //gl_Position[0] = gl_Position[0] + 100;
#
#     //----------------------------
#
#
#     gl_FrontColor = gl_Color;
#     gl_BackColor = gl_Color;
#
#     //texture_coordinate = vec2(gl_MultiTexCoord0);
#     gl_TexCoord[0] = gl_MultiTexCoord0;
# }
# """
#
# vs_source = """
# #version 120
#
# uniform float k1, k2, k3, k4, k5, k6;
# uniform float p1, p2;
#
# void main()
# {
#     vec4 p0 = gl_ModelViewMatrix * gl_Vertex;
#     p0 = p0 / p0[3];
#
#     float xp = -p0[0] / p0[2];
#     float yp = p0[1] / p0[2];
#
#     float r2 = xp*xp + yp*yp;
#     float r4 = r2 * r2;
#     float r6 = r4 * r2;
#
#     float m = (1.0 + k1*r2 + k2*r4 + k3*r6) / (1.0 + k4*r2 + k5*r4 + k6*r6);
#
#     float xpp = m*xp + 2.*p1*xp*yp + p2*(r2+2*xp*xp);
#     float ypp = m*yp + p1*(r2+2*yp*yp) + 2.*p2*xp*yp;
#
#     p0[0] = -xpp * p0[2];
#     p0[1] = ypp * p0[2];
#     gl_Position = gl_ProjectionMatrix * p0;
#
#     //----------------------------
#
#     gl_FrontColor = gl_Color;
#     gl_BackColor = gl_Color;
#
#     //texture_coordinate = vec2(gl_MultiTexCoord0);
#     gl_TexCoord[0] = gl_MultiTexCoord0;
# }
# """
#
#


# class BoundaryRenderer(BaseRenderer):
#     terms = 'f', 'frustum', 'num_channels'
#     dterms = 'camera',
#
#     @property
#     def shape(self):
#         return (self.frustum['height'], self.frustum['width'], self.num_channels)
#
#     def compute_r(self):
#         tmp = self.camera.r
#         return self.color_image
#
#     def compute_dr_wrt(self, wrt):
#         if wrt is not self.camera:
#             return None
#
#         visibility = self.boundaryid_image
#         shape = visibility.shape
#
#         visible = np.nonzero(visibility.ravel() != 4294967295)[0]
#         num_visible = len(visible)
#
#         barycentric = self.barycentric_image
#
#         return common.dImage_wrt_2dVerts(self.color_image, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.vpe)
#
#     def on_changed(self, which):
#         if 'frustum' in which:
#             w = self.frustum['width']
#             h = self.frustum['height']
#             self.initGL()
#             # self.glf = OsContext(w, h, typ=GL_FLOAT)
#             # self.glf.Viewport(0, 0, w, h)
#             # self.glb = OsContext(w, h, typ=GL_UNSIGNED_BYTE)
#             # self.glb.Viewport(0, 0, w, h)
#
#         if 'frustum' in which or 'camera' in which:
#             self.setup_camera(self.camera, self.frustum)
#             # setup_camera(self.glf, self.camera, self.frustum)
#
#         if not hasattr(self, 'overdraw'):
#             self.overdraw = True
#
#     @depends_on(terms+dterms)
#     def color_image(self):
#         self._call_on_changed()
#         result = self.boundarybool_image.astype(np.float64)
#         return np.dstack([result for i in range(self.num_channels)])


def main():
    pass

if __name__ == '__main__':
    main()


