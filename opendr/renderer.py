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
    def makeCurrentContext(self):
        if self.glMode == 'glfw':
            import glfw
            glfw.make_context_current(self.win)
        else:
            from OpenGL import arrays
            from OpenGL.raw.osmesa import mesa
            mesa.OSMesaMakeCurrent(self.ctx, GL.GLuint(self.mesap), GL.GL_UNSIGNED_BYTE, self.frustum['width'], self.frustum['height'])
    def clear(self):

        try:
            self.win
        except:
            print ("Clearing when not initialized.")
            return

        if self.win:

            print ("Clearing base renderer.")
            self.makeCurrentContext()
            self.vbo_indices.set_array(np.array([]))
            self.vbo_indices.bind()
            self.vbo_indices.unbind()
            self.vbo_indices.delete()
            self.vbo_indices_range.set_array(np.array([]))
            self.vbo_indices_range.bind()
            self.vbo_indices_range.unbind()
            self.vbo_indices_range.delete()
            self.vbo_indices_dyn.set_array(np.array([]))
            self.vbo_indices_dyn.bind()
            self.vbo_indices_dyn.unbind()
            self.vbo_indices_dyn.delete()
            self.vbo_verts.set_array(np.array([]))
            self.vbo_verts.bind()
            self.vbo_verts.unbind()
            self.vbo_verts.delete()
            self.vbo_verts_face.set_array(np.array([]))
            self.vbo_verts_face.bind()
            self.vbo_verts_face.unbind()
            self.vbo_verts_face.delete()
            self.vbo_verts_dyn.set_array(np.array([]))
            self.vbo_verts_dyn.bind()
            self.vbo_verts_dyn.unbind()
            self.vbo_verts_dyn.delete()
            self.vbo_colors.set_array(np.array([]))
            self.vbo_colors.bind()
            self.vbo_colors.unbind()
            self.vbo_colors.delete()
            self.vbo_colors_face.set_array(np.array([]))
            self.vbo_colors_face.bind()
            self.vbo_colors_face.unbind()
            self.vbo_colors_face.delete()

            GL.glDeleteVertexArrays(1, [self.vao_static])
            GL.glDeleteVertexArrays(1, [self.vao_static_face])
            GL.glDeleteVertexArrays(1, [self.vao_dyn])
            GL.glDeleteVertexArrays(1, [self.vao_dyn_ub])

            GL.glDeleteRenderbuffers(1, [int(self.render_buf)])
            GL.glDeleteRenderbuffers(1, [int(self.z_buf)])
            GL.glDeleteRenderbuffers(1, [int(self.render_buf_ms)])
            GL.glDeleteRenderbuffers(1, [int(self.z_buf_ms)])

            GL.glDeleteFramebuffers(1, [int(self.fbo)])
            GL.glDeleteFramebuffers(1, [int(self.fbo_ms)])
            GL.glDeleteFramebuffers(1, [int(self.fbo_noms)])

            GL.glDeleteProgram(self.colorProgram)

    def initGL(self):
        try:
            self.frustum
            self.f
            self.v
            self.vc
            self.glMode
        except:
            print ("Necessary variables have not been set (frustum, f, v, or vc).")
            return

        if self.glMode == 'glfw':
            import glfw
            glfw.init()
            print("Initializing GLFW.")

            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.DEPTH_BITS,32)

            glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
            self.win = glfw.create_window(self.frustum['width'], self.frustum['height'], "test",  None, self.sharedWin)
            glfw.make_context_current(self.win)

        else: #Mesa
            from OpenGL import arrays
            from OpenGL.raw.osmesa import mesa

            self.ctx = mesa.OSMesaCreateContext(GL.GL_RGBA, None)
            self.buf = arrays.GLubyteArray.zeros((self.frustum['height'], self.frustum['width'], 3))
            self.mesap = arrays.ArrayDatatype.dataPointer(self.buf)
            assert(mesa.OSMesaMakeCurrent(self.ctx, GL.GLuint(self.mesap), GL.GL_UNSIGNED_BYTE, self.frustum['width'], self.frustum['height']))

        GL.USE_ACCELERATE = True

        GL.glViewport(0, 0, self.frustum['width'], self.frustum['height'])

        #FBO_f
        self.fbo = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        self.render_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.render_buf)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.render_buf)


        self.z_buf = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.z_buf)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,  GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.z_buf)

        #FBO_f
        if self.msaa and self.glMode == 'glfw':
            try:
                self.nsamples
            except:
                self.nsamples = 8
            try:
                self.overdraw
            except:
                self.overdraw = True

            self.fbo_ms = GL.glGenFramebuffers(1)

            GL.glDepthMask(GL.GL_TRUE)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_ms )

            self.render_buf_ms = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.render_buf_ms)

            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
            GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.render_buf_ms)

            self.z_buf_ms = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.z_buf_ms)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.z_buf_ms)

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glDisable(GL.GL_CULL_FACE)

            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

            print ("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
            assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)

        self.fbo_noms = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_noms )

        self.render_buf_noms = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.render_buf_noms)

        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER,0, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.render_buf_noms)

        self.z_buf_noms = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.z_buf_noms)
        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER,0 , GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.z_buf_noms)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print ("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # GL.glClear(GL.GL_DEPTH_BUFFER_BIT)


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
        # glGenBuffers(1, &vboID);
        # glBindBuffer(GL_VERTEX_ARRAY, vboID);
        # glBufferData(GL_VERTEX_ARRAY, 3 * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
        # glBindBuffer(GL_VERTEX_ARRAY, NULL);


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

        self.initialized = True

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
        ipdb.set_trace()
        return np.asarray(np.tile(np.eye(3)[:self.f.shape[1], :], (self.verts_by_face.shape[0]/self.f.shape[1], 1)), dtype=np.float64, order='C')


    @depends_on('f', 'v', 'vn')
    def tn(self):
        from opendr.geometry import TriNormals
        # return TriNormals(self.v, self.f).r.reshape((-1,3))

        tn = np.mean(self.vn.r[self.f.ravel()].reshape([-1, 3, 3]), 1)
        return tn

    @property
    def fpe(self):
        return self.primitives_per_edge[0]

    @depends_on(terms+dterms)
    def boundary_neighborhood(self):
        return common.boundary_neighborhood(self.boundarybool_image)


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
        if self.msaa:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_noms)

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        shaders.glUseProgram(self.colorProgram)
        GL.glBindVertexArray(self.vao_static)
        self.vbo_colors.set_array(np.zeros_like(v.reshape((-1,3))[f.ravel()], dtype=np.float32, order='C'))
        self.vbo_color.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.vbo_indices)*3, GL.GL_UNSIGNED_INT, None)


    def draw_edge_visibility(self, v, e, f, hidden_wireframe=True):
        """Assumes camera is set up correctly in gl context."""
        shaders.glUseProgram(self.colorProgram)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1, 1)
        self.draw_colored_verts(np.zeros_like(self.vc.r))
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        ec = np.arange(1, len(e)+1)
        ec = np.tile(ec.reshape((-1,1)), (1, 3))
        ec[:, 0] = ec[:, 0] & 255
        ec[:, 1] = (ec[:, 1] >> 8 ) & 255
        ec[:, 2] = (ec[:, 2] >> 16 ) & 255
        ec = np.asarray(ec, dtype=np.uint8)

        self.draw_colored_primitives(self.vao_dyn_ub, v, e, ec)

        # if hidden_wireframe:
        #     GL.glEnable(GL.GL_DEPTH_TEST)
        #     GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        #     #Pol change it to a smaller number to avoid double edges in my teapot.
        #     GL.glPolygonOffset(10.0, 1.0)
        #     # delta = -0.0
        #     # self.projectionMatrix[2,2] += delta
        #     # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #     self.draw_colored_primitives(self.vao_dyn_ub, v, f, fc=np.zeros(f.shape).astype(np.uint8))
        #     # self.draw_colored_primitives(self.vaoub, v, e, np.zeros_like(ec).astype(np.uint8))
        #     # self.projectionMatrix[2,2] -= delta
        #     GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

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
        silhouette_edges = np.asarray(np.nonzero(dps<=1e-5)[0], np.uint32)

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
        # gl.DisableClientState(GL_TEXTURE_COORD_ARR
        shaders.glUseProgram(self.colorProgram)
        self.makeCurrentContext()

        result = self.draw_visibility_image_internal(v, f)
        if boundarybool_image is None:
            return result
        #
        # #Pol, remove all the rest as seems unnecessary?
        # return result

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

        #Pol: changed to be outside the 'if'
        result2[result2 == 4294967295] = 0

        if False:

            import matplotlib.pyplot as plt
            result2 = result2.reshape(result.shape[:2])
            plt.figure()
            plt.subplot(121)
            plt.imshow(result.squeeze())
            plt.subplot(122)
            plt.imshow(result2.squeeze())
            plt.show()

        result2 = result2.reshape(result.shape[:2])

        return result2 * bbi + result * (1 - bbi)

    def draw_visibility_image_internal(self, v, f):
        """Assumes camera is set up correctly in"""
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
            return without_overdraw

        # return without_overdraw

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        overdraw = self.draw_barycentric_image_internal()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        bbi = np.atleast_3d(boundarybool_image)
        return bbi * overdraw + (1. - bbi) * without_overdraw


    def draw_barycentric_image_internal(self):
        GL.glUseProgram(self.colorProgram)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glBindVertexArray(self.vao_static_face)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glDrawElements(GL.GL_TRIANGLES if self.f.shape[1]==3 else GL.GL_LINES, len(self.vbo_indices_range), GL.GL_UNSIGNED_INT, None)

        #Read image.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        # return np.array(im.transpose(Image.FLIP_TOP_BOTTOM), np.float64)/255.0
        return np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0

    def setup_camera(self, camera):

        near = 0.01
        far = 10
        fx = camera.f.r[0]
        fy = camera.f.r[1]
        cx = camera.c.r[0]
        cy = camera.c.r[1]
        self.projectionMatrix = np.array([[fx / cx, 0, 0, 0], [0, fy / cy, 0, 0], [0, 0, -(near + far) / (far - near), -2 * near * far / (far - near)], [0, 0, -1, 0]], dtype=np.float32)
        # self.projectionMatrix = np.array([[camera.f.r[0], 0, camera.c.r[0], 0], [0, camera.f.r[1], camera.c.r[1], 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.float32, order='F')

    def setup_camera_old(self, camera, frustum):
        self._setup_camera(camera.c.r[0], camera.c.r[1],
                  camera.f.r[0], camera.f.r[1],
                  frustum['width'], frustum['height'],
                  frustum['near'], frustum['far'],
                  camera.view_matrix,
                  camera.k.r)


class ColoredRenderer(BaseRenderer):
    terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
    dterms = 'vc', 'camera', 'bgcolor' , 'v'

    @depends_on('vc')
    def num_channels(self):
        if hasattr(self, 'vc'):
            return self.vc.shape[1]
        return 3

    def clear(self):
        print ("Clearing color renderer.")
        super().clear()

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
                # return common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
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

        if 'frustum' in which or 'camera' in which:
            self.setup_camera(self.camera)
            # setup_camera(self.glf, self.camera, self.frustum)

        if not hasattr(self, 'num_channels'):
            self.num_channels = 3

        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5]*self.num_channels))
            which.add('bgcolor')

        if not hasattr(self, 'overdraw'):
            self.overdraw = True

        if 'v' or 'f' in which:
            self.vbo_verts_face.set_array(np.array(self.verts_by_face).astype(np.float32))
            self.vbo_verts_face.bind()
            self.vbo_colors_face.set_array(np.array(self.vc_by_face).astype(np.float32))
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


    def draw_color_image(self):
        self.makeCurrentContext()
        self._call_on_changed()
        try:

            GL.glEnable(GL.GL_MULTISAMPLE)
            if hasattr(self, 'bgcolor'):
                GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)
            # use face colors if given
            # FIXME: this won't work for 2 channels
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            if self.msaa:
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
            else:
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_noms)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


            self.draw_colored_verts(self.vc.r)
            if self.msaa:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
            else:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_noms)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'], GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

            result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0
            # plt.imsave("opendr_draw_color_image.png", result)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glDisable(GL.GL_MULTISAMPLE)
            GL.glClearColor(0.,0.,0., 1.)

            if hasattr(self, 'background_image'):
                bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1,1,self.num_channels)).squeeze()
                fg_px = 1 - bg_px
                result = bg_px * self.background_image + fg_px * result

            return result
        except:
            import pdb; pdb.set_trace()

    @depends_on(dterms+terms)
    def color_image(self):

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image()

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


class TexturedRenderer(ColoredRenderer):
    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    # def __init__(self):
        # try:
        #     self.overdraw
        # except:
        #     self.overdraw = True
        #
        # try:
        #     self.nsamples
        # except:
        #     self.nsamples = 8

    def clear(self):
        try:
            print ("Clearing textured renderer.")
            [vbo.set_array([]) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.set_array([]) for vbo in self.vbo_colors_mesh]
            [vbo.bind() for vbo in self.vbo_colors_mesh]
            [vbo.delete() for vbo in self.vbo_colors_mesh]
            [vbo.unbind() for vbo in self.vbo_colors_mesh]
            [vbo.delete() for vbo in self.vbo_verts_mesh]
            [vbo.set_array([]) for vbo in self.vbo_uvs_mesh]
            [vbo.bind() for vbo in self.vbo_uvs_mesh]
            [vbo.unbind() for vbo in self.vbo_uvs_mesh]
            [vbo.delete() for vbo in self.vbo_uvs_mesh]
            [GL.glDeleteVertexArrays(1, [vao.value]) for sublist in self.vao_tex_mesh_list for vao in sublist]

            self.release_textures()

            if self.glMode == 'glfw':
                import glfw
                glfw.make_context_current(self.win)

            GL.glDeleteProgram(self.colorTextureProgram)

            super().clear()
            GL.glFlush()
            GL.glFinish()
        except:
            print("Program had not been initialized")

    def initGLTexture(self):
        print("Initializing Texture OpenGL.")

        FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
        // Interpolated values from the vertex shaders
        //#extension GL_EXT_shader_image_load_store : enable 
        in vec3 theColor;
        in vec2 UV;
        uniform sampler2D myTextureSampler;
        // Ouput data
        out vec3 color;
        void main(){
            color = theColor * texture2D( myTextureSampler, UV).rgb;
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

        position_location = GL.glGetAttribLocation(self.colorTextureProgram, 'position')
        color_location = GL.glGetAttribLocation(self.colorTextureProgram, 'color')
        uvs_location = GL.glGetAttribLocation(self.colorTextureProgram, 'vertexUV')
        # color_location_ub = GL.glGetAttribLocation(self.colorProgram, 'color')
        self.MVP_texture_location = GL.glGetUniformLocation(self.colorTextureProgram, 'MVP')

        self.vbo_indices_mesh_list = []
        self.vbo_colors_mesh = []
        self.vbo_verts_mesh = []
        self.vao_tex_mesh_list = []
        self.vbo_uvs_mesh = []
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

                    GL.glGenTextures( 1, texture )
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
                    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)

                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

                    image = np.array(np.flipud((self.textures_list[mesh][polygons])), order='C', dtype=np.float32)
                    GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB32F, image.shape[1], image.shape[0])
                    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)
                    # GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())
                textureIDs_mesh = textureIDs_mesh + [texture]
                vbo_indices_mesh = vbo_indices_mesh + [vbo_indices]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID  = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")

    # def __del__(self):
    #     pass
    #     # self.release_textures()

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
                    if texture_mesh != []:
                        for texture in texture_mesh:
                            if texture != None:
                                GL.glDeleteTextures(1, [texture.value])

        self.textureID_mesh_list = []

    def compute_r(self):
        return self.color_image # .reshape((self.frustum['height'], self.frustum['width'], -1)).squeeze()


    @depends_on(dterms+terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        if not self.overdraw:
            return no_overdraw


        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(1.)
        overdraw = self.draw_color_image()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # return overdraw * np.atleast_3d(self.boundarybool_image)

        boundarybool_image = self.boundarybool_image
        if self.num_channels > 1:
            boundarybool_image = np.atleast_3d(boundarybool_image)

        return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')


    def image_mesh_bool(self, meshes):
        self.makeCurrentContext()
        self._call_on_changed()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0.,0.,0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)
        for mesh in meshes:
            self.draw_index(mesh)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))[:,:,0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result!=0

    @depends_on(dterms+terms)
    def indices_image(self):
        self._call_on_changed()
        self.makeCurrentContext()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0.,0.,0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)

        for index in range(len(self.f_list)):
            self.draw_index(index)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))[:,:,0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result

    def draw_index(self, index):

        mesh = index

        vbo_color = self.vbo_colors_mesh[mesh]
        vc = self.vc_list[mesh]
        colors = np.array(np.ones_like(vc)*(index+1)/255.0, dtype=np.float32)

        #Pol: Make a static zero vbo_color to make it more efficient?
        vbo_color.set_array(colors)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for polygons in np.arange(len(self.f_list[mesh])):

            vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
            vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

            GL.glBindVertexArray(vao_mesh)
            vbo_color.bind()

            if self.f.shape[1]==2:
                primtype = GL.GL_LINES
            else:
                primtype = GL.GL_TRIANGLES

            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

            GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)



    def draw_texcoord_image(self, v, f, ft, boundarybool_image=None):

        # gl = glf
        # gl.Disable(GL_TEXTURE_2D)
        # gl.DisableClientState(GL_TEXTURE_COORD_ARR
        self.makeCurrentContext()
        shaders.glUseProgram(self.colorProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # want vtc: texture-coordinates per vertex (not per element in vc)
        colors = ft

        #use the third channel to identify the corresponding textures.
        color3 = np.vstack([np.ones([self.ft_list[mesh].shape[0],1])*mesh for mesh in range(len(self.ft_list))]).astype(np.float32) / len(self.ft_list)

        colors = np.asarray(np.hstack((colors, color3)), np.float64, order='C')
        self.draw_colored_primitives(self.vao_dyn, v, f, colors)

        #Why do we need this?
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
        result = super().compute_dr_wrt(wrt)

        if wrt is self.vc:
            cim = self.draw_color_image(with_vertex_colors=False).ravel()
            cim = sp.spdiags(row(cim), [0], cim.size, cim.size)
            result = cim.dot(result)
        elif wrt is self.texture_stack:
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
        super().on_changed(which)

        # have to redo if frustum changes, b/c frustum triggers new # context
        # if 'frustum' in  which:

        if 'v' in which:
            for mesh in range(len(self.f_list)):
                self.vbo_verts_mesh[mesh].set_array(np.array(self.v_list[mesh]).astype(np.float32))
                self.vbo_colors_mesh[mesh].set_array(np.array(self.vc_list[mesh]).astype(np.float32))
                self.vbo_verts_mesh[mesh].bind()
                self.vbo_colors_mesh[mesh].bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()


        if 'texture_stack' in which:
            # gl = self.glf
            # texture_data = np.array(self.texture_image*255., dtype='uint8', order='C')

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
            if self.initialized:
                textureCoordIdx = 0
                for mesh in range(len(self.f_list)):
                    for polygons in range(len(self.f_list[mesh])):
                        texture = None

                        if self.haveUVs_list[mesh][polygons]:
                            texture = self.textureID_mesh_list[mesh][polygons]

                            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

                            #Update the OpenGL textures with all the textures. (Inefficient as many might not have changed).
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)
                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size].reshape(image.shape)

                            textureCoordIdx = textureCoordIdx + image.size
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)

                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                                               image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())


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
        edata = vvt[self.ma.ravel()]
        return edata

    # TODO: can this not be inherited from base? turning off texture mapping in that instead?
    @depends_on(dterms+terms)
    def boundaryid_image(self):
        self._call_on_changed()

        # self.texture_mapping_of
        self.makeCurrentContext()
        GL.glUseProgram(self.colorProgram)

        result = self.draw_boundaryid_image(self.v.r, self.f, self.vpe, self.fpe, self.camera)

        GL.glUseProgram(self.colorTextureProgram)
        # self.texture_mapping_on(with_vertex_colors=True)

        return result

    def draw_color_image(self, with_vertex_colors=True, with_texture_on=True):
        self.makeCurrentContext()
        self._call_on_changed()

        GL.glEnable(GL.GL_MULTISAMPLE)

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.msaa:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_noms)

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
                    else:
                        GL.glUseProgram(self.colorProgram)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

        if self.msaa:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_noms)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'], GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.float64))/255.0

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0.,0.,0., 1.)

        if hasattr(self, 'background_image'):
            bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1,1,3))
            fg_px = 1 - bg_px
            result = bg_px * self.background_image + fg_px * result

        return result

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image_quantized(self):

        texcoord_image = self.texcoord_image[:,:, :2].copy()
        #Temprary:
        self.texture_image = self.textures_list[0][0].r.copy()
        texcoord_image[:,:,0] *= self.texture_image.shape[1]-1
        texcoord_image[:,:,1] *= self.texture_image.shape[0]-1
        texture_idx = (self.texcoord_image[:,:,2]*len(self.ft_list)).astype(np.uint32)
        texcoord_image = np.round(texcoord_image)
        texcoord_image = texcoord_image[:,:,0] + texcoord_image[:,:,1]*self.texture_image.shape[1]

        return texcoord_image, texture_idx

    def checkBufferNum(self):
       GL.glGenBuffers(1)

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image(self):
        return self.draw_texcoord_image(self.v.r, self.f, self.ft, self.boundarybool_image if self.overdraw else None)


class SQErrorRenderer(TexturedRenderer):

    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list' , 'imageGT'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    def __init__(self):
        super().__init__()

    def clear(self):
        super().clear()

    def initGL_SQErrorRenderer(self):
        self.initGLTexture()

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 colorIn;
        layout(location = 2) in vec2 vertexUV;
        //layout(location = 3) in vec3 face;
        //layout(location = 4) in vec3 edge_v1;
        //layout(location = 5) in vec3 edge_v2;
        uniform mat4 MVP;
        out vec3 theColor;
        out vec3 pos;
        //out vec3 face_out;
        out vec2 UV;
        //out vec3 face;
        //out vec4 edge_v1;
        //out vec4 edge_v2;
        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            vec4 pos4 =  MVP * vec4(position,1);
            pos =  pos4.xyz;
            theColor = colorIn;
            UV = vertexUV;
            //vec4 face_out_4 = MVP* vec4(face,1);
            //face_out = face_out_4.xyz;
        }""", GL.GL_VERTEX_SHADER)

        ERRORS_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 
            //#extension GL_EXT_shader_image_load_store : enable
            //#extension GL_EXT_shader_image_load_store : enable 
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
            in vec3 theColor;
            in vec2 UV;
            //in vec3 face_out;
            in vec3 pos;
                        
            layout(location = 3) uniform sampler2D myTextureSampler;
        
            //in vec3 edge_v1;
            //in vec3 edge_v2;
                        
            //layout(location = 4) uniform sampler2D edges;
            //layout(location = 4) uniform sampler2D imageGT;
            //layout(location = 5) uniform sampler2D primitive_id;
            
            //readonly uniform layout(binding=1, size4x32) image2D imageGT;

            uniform float ww;
            uniform float wh;
    
            // Ouput data
            layout(location = 0) out vec3 color; 
            layout(location = 1) out vec2 sample_pos;
            //layout(location = 2) out vec3 sample_face;
            
            //layout(location = 1) out vec3 render_nocolor;
            //layout(location = 2) out vec3 render_notexture;
            //layout(location = 3) out vec3 E;
            //layout(location = 4) out vec3 dEdx;
            //layout(location = 5) out vec3 dEdy;

            //out int gl_SampleMask[];
            //const int all_sample_mask = 0xffff;

            void main(){
                vec3 finalColor = theColor * texture2D( myTextureSampler, UV).rgb;
                color = finalColor.rgb;
                //color = vec2(10.0,50.0) + 255.0 * theColor.xy * texture2D( myTextureSampler, UV).xy; 
                
                //sample_pos = gl_FragCoord.xy;
                sample_pos = pos.xy;
                //vec3(ww,wh,1);
                
                //sample_face = face_out;
  
                //render_nocolor = texture2D( myTextureSampler, UV).rgb;
                //render_notexture = theColor;
                                           
                //ivec2 coord = ivec2(gl_FragCoord.xy);
                //vec3 imgColor = texture2D(imageGT, gl_FragCoord.xy/vec2(ww,wh)).rgb;
                                           
                //vec2 edge_x1 = edge_v1.xy;
                //vec2 edge_x2 = edge_v2.xy;
                
                //bool boolx = (1.0-dx) > 0.1;
                //bool booly = (1.0-dy) > 0.1;
                //int x = int(boolx && booly);
                //gl_SampleMask[0] ^= (-x ^ gl_SampleMask[0]) & (1 << gl_SampleID);

                //vec3 dfdx = -dFdxFine(theColor)/dFdxFine(1.0-gl_SamplePosition.x);
                //vec3 dfdy = -dFdyFine(theColor)/dFdyFine(1.0-gl_SamplePosition.y);

                //vec3 Res = imgColor - theColor;
                //E =  pow(Res,vec3(2.0,2.0,2.0));

                //dEdx = -2.0*Res;
                //dEdx = -dFdxFine(E)/(1.0-dx);

                //dEdy = -2.0*Res*dfdy/(1.0-dy);
                //dEdy = -dFdyFine(E)/(1.0-dy);
                //dEdy = -2.0*Res;
            }""", GL.GL_FRAGMENT_SHADER)

        self.errorTextureProgram = shaders.compileProgram(VERTEX_SHADER, ERRORS_FRAGMENT_SHADER)

        FETCH_VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        void main() {}
        """, GL.GL_VERTEX_SHADER)

        FETCH_GEOMETRY_SHADER = shaders.compileShader("""#version 330 core
        layout(points) in;
        layout(triangle_strip, max_vertices = 4) out;
         
        const vec2 data[4] = vec2[]
        (
          vec2(-1.0,  1.0),
          vec2(-1.0, -1.0),
          vec2( 1.0,  1.0),
          vec2( 1.0, -1.0)
        );
         
        void main() {
          for (int i = 0; i < 4; ++i) {
            gl_Position = vec4( data[i], 0.0, 1.0 );
            EmitVertex();
          }
          EndPrimitive();
        }""", GL.GL_GEOMETRY_SHADER)

        
        FETCH_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 
            //#extension GL_EXT_shader_image_load_store : enable
            //#extension GL_EXT_shader_image_load_store : enable 
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
  
            layout(location = 2) uniform sampler2DMS colors;
            layout(location = 3) uniform sampler2DMS sample_positions;
            //layout(location = 4) uniform sampler2DMS sample_faces;

            uniform float ww;
            uniform float wh;
            uniform int sample;

            // Ouput data
            layout(location = 0) out vec3 colorFetchOut;
            layout(location = 1) out vec2 sample_pos;
            //layout(location = 2) out vec3 sample_face;

            //out int gl_SampleMask[];
            const int all_sample_mask = 0xffff;

            void main(){
                ivec2 texcoord = ivec2(gl_FragCoord.xy);
                //colorFetchOut = vec3(1,0,0);
                colorFetchOut = texelFetch(colors, texcoord, sample).xyz;
                sample_pos = texelFetch(sample_positions, texcoord, sample).xy;        
                //sample_face = texelFetch(sample_faces, texcoord, sample).rgb;              

            }""", GL.GL_FRAGMENT_SHADER)

        GL.glClampColor(GL.GL_CLAMP_READ_COLOR, False)

        # GL.glClampColor(GL.GL_CLAMP_VERTEX_COLOR, False)
        # GL.glClampColor(GL.GL_CLAMP_FRAGMENT_COLOR, False)

        self.fetchSamplesProgram = shaders.compileProgram(FETCH_VERTEX_SHADER, FETCH_GEOMETRY_SHADER, FETCH_FRAGMENT_SHADER)

        self.textureGT = GL.GLuint(0)

        # GL.glActiveTexture(GL.GL_TEXTURE1)
        # GL.glGenTextures(1, self.textureGT)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        # self.textureGTLoc = GL.glGetUniformLocation(self.errorTextureProgram, "imageGT")
        # GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
        # #
        # try:
        #     if self.imageGT.r is not None and self.imageGT.r.size != 0: #if GT image is defined.
        #         image = np.array(np.flipud((self.imageGT.r)), order='C', dtype=np.float32)
        #         GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB32F, image.shape[1], image.shape[0])
        #         GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)
        # except:
        #     pass

        GL.glActiveTexture(GL.GL_TEXTURE0)

        whitePixel = np.ones([1,1,3])
        self.whitePixelTextureID = GL.GLuint(0)
        GL.glGenTextures( 1, self.whitePixelTextureID )
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.whitePixelTextureID)
        image = np.array(np.flipud((whitePixel)), order='C', dtype=np.float32)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB32F, image.shape[1], image.shape[0])
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

        self.fbo_ms_errors = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_ms_errors)

        self.texture_errors_render = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RGB8, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)

        self.texture_errors_sample_position = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)

        # self.texture_errors_sample_face = GL.glGenTextures(1)
        # GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_face)
        # GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RGB, self.frustum['width'], self.frustum['height'], False)
        # # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        # GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_face, 0)
        #
        # self.render_buf_errors_dedx = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buf_errors_dedx)
        # GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_RGB32F, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buf_errors_dedx)
        #
        # self.render_buf_errors_dedy = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buf_errors_dedy)
        # GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_RGB32F, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT5, GL.GL_RENDERBUFFER, self.render_buf_errors_dedy)


        self.z_buf_ms_errors = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.z_buf_ms_errors)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D_MULTISAMPLE, self.z_buf_ms_errors, 0)

        # self.z_buf_ms_errors = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.z_buf_ms_errors)
        # GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.z_buf_ms_errors)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        # GL.glDisable(GL.GL_CULL_FACE)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.fbo_sample_fetch = GL.glGenFramebuffers(1)

        GL.glDepthMask(GL.GL_TRUE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_sample_fetch)

        self.render_buffer_fetch_sample_render = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_render)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_render)

        self.render_buffer_fetch_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_position)

        # self.render_buffer_fetch_sample_face = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        # GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        #
        # self.render_buf_errors_dedx = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buf_errors_dedx)
        # GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_RGB32F, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buf_errors_dedx)
        #
        # self.render_buf_errors_dedy = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buf_errors_dedy)
        # GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.nsamples, GL.GL_RGB32F, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT5, GL.GL_RENDERBUFFER, self.render_buf_errors_dedy)

        self.z_buf_samples_errors = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.z_buf_samples_errors)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.z_buf_samples_errors)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        #FBO_f
        self.fbo_errors_nonms = GL.glGenFramebuffers(1)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_errors_nonms)

        render_buf_errors_render = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_render)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf_errors_render)

        # render_buf_errors_nocolor = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_nocolor)
        # GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_nocolor)
        #
        # render_buf_errors_notexture = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_notexture)
        # GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_notexture)

        render_buf_errors_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_sample_position)

        # render_buf_errors_sample_face = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        # GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        #
        # render_buf_errors_dedx = GL.glGenRenderbuffers(1)
        # GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_dedx)
        # GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA, self.frustum['width'], self.frustum['height'])
        # GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, render_buf_errors_dedx)
        #
        z_buf_samples_errors = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, z_buf_samples_errors)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, z_buf_samples_errors)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        print("FRAMEBUFFER ERR: " + str(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)))
        assert (GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.textureObjLoc  = GL.glGetUniformLocation(self.errorTextureProgram, "myTextureSampler")

        #Add background cube:
        position_location = GL.glGetAttribLocation(self.errorTextureProgram, 'position')
        color_location = GL.glGetAttribLocation(self.errorTextureProgram, 'colorIn')
        uvs_location = GL.glGetAttribLocation(self.errorTextureProgram, 'vertexUV')

        self.vbo_verts_cube= vbo.VBO(np.array(self.v_bgCube).astype(np.float32))
        self.vbo_colors_cube= vbo.VBO(np.array(self.vc_bgCube).astype(np.float32))
        self.vbo_uvs_cube = vbo.VBO(np.array(self.ft_bgCube).astype(np.float32))
        self.vao_bgCube = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_bgCube)

        GL.glBindVertexArray(self.vao_bgCube)
        self.vbo_f_bgCube = vbo.VBO(np.array(self.f_bgCube).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        self.vbo_f_bgCube.bind()
        self.vbo_verts_cube.bind()
        GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        self.vbo_colors_cube.bind()
        GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        self.vbo_uvs_cube.bind()
        GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
        GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_quad = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_quad)
        GL.glBindVertexArray(self.vao_quad)

    def render_errors(self):

        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        self.makeCurrentContext()

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)

        GL.glUseProgram(self.errorTextureProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1]
        GL.glDrawBuffers(2, drawingBuffers)

        # GL.glActiveTexture(GL.GL_TEXTURE1)
        # # GL.glBindImageTexture(1,self.textureGT, 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RGBA8)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        # self.textureGTLoc = GL.glGetUniformLocation(self.errorTextureProgram, "imageGT")
        # GL.glUniform1i(self.textureGTLoc, 1)

        wwLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            vbo_color = self.vbo_colors_mesh[mesh]
            vc = self.vc_list[mesh]

            colors = vc.r.astype(np.float32)

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

                assert(primtype == GL.GL_TRIANGLES)

                # GL.glUseProgram(self.errorTextureProgram)
                if self.haveUVs_list[mesh][polygons]:
                    texture =  self.textureID_mesh_list[mesh][polygons]
                    self.vbo_uvs_mesh[mesh].bind()
                else:
                    texture = self.whitePixelTextureID
                    self.vbo_uvs_cube.bind()

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glUniform1i(self.textureObjLoc, 0)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

        # # #Background cube:
        GL.glBindVertexArray(self.vao_bgCube)
        self.vbo_f_bgCube.bind()
        texture = self.whitePixelTextureID
        self.vbo_uvs_cube.bind()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glUniform1i(self.textureObjLoc, 0)
        GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

        GL.glDrawElements(primtype, len(self.vbo_f_bgCube)*self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)


        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # result_blit = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        result_blit2 = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        result_blit_pos = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))


        GL.glUseProgram(self.fetchSamplesProgram)
        # GL.glDisable(GL.GL_MULTISAMPLE)

        self.colorsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "colors")
        self.sample_positionsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_positions")
        # self.sample_facesLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_faces")

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # GL.glActiveTexture(GL.GL_TEXTURE2)
        # GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_face)
        # GL.glUniform1i(self.sample_facesLoc, 2)

        wwLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        self.renders = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],3])
        self.renders_sample_pos = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],2])
        self.renders_faces = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 3])

        GL.glDisable(GL.GL_DEPTH_TEST)

        for sample in np.arange(self.nsamples):

            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_sample_fetch)
            drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1]
            GL.glDrawBuffers(2, drawingBuffers)

            sampleLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'sample')
            GL.glUniform1i(sampleLoc, sample)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
            GL.glUniform1i(self.colorsLoc, 0)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
            GL.glUniform1i(self.sample_positionsLoc, 1)


            GL.glBindVertexArray(self.vao_quad)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

            # GL.glBindVertexArray(self.vao_bgCube)
            # # self.vbo_f_bgCube.bind()
            # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
            #
            # GL.glDrawElements(primtype, len(self.vbo_f_bgCube) * self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_sample_fetch)

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

            self.renders[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:2].astype(np.float64))
            self.renders_sample_pos[sample] = result

            # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            # result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
            # self.renders_faces[sample] = result

        self.render = np.mean(self.renders,0)

        ipdb.set_trace()

        GL.glBindVertexArray(0)

        GL.glClearColor(0.,0.,0., 1.)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)


    def compute_dr_wrt(self, wrt):
        #Base:
        # if wrt is not self.camera and wrt is not self.vc and wrt is not self.bgcolor or wrt is not self.texture_stack:
        #     return None

        #Color:
        visibility = self.visibility_image

        shape = visibility.shape
        color = self.color_image

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image

        if wrt is self.camera:
            dEdx = self.render_dedx
            dEdy = self.render_dedy

            # dEdxtilde = 2*(self.imageGT.r - self.render_image)*np.gradient(self.render_image)[0]
            # dEdytilde = 2 * (self.imageGT.r - self.render_image) * np.gradient(self.render_image)[1]

            # z = self.dErrors_wrt_2dVerts(color, dEdxtilde, dEdytilde, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'],self.v.r.size / 3, self.f)

            return self.dErrors_wrt_2dVerts(color, dEdx, dEdy, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)

        elif wrt is self.bgcolor:
            return 2. * (self.imageGT.r - self.render_image).ravel() * common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

        elif wrt is self.vc:
            result = common.dr_wrt_vc(visible, visibility, self.f, barycentric, self.frustum, self.vc.size, num_channels=self.num_channels)
            cim = -2. * (self.imageGT.r - self.render_image).ravel() * self.renderWithoutColor.ravel()
            cim = sp.spdiags(row(cim), [0], cim.size, cim.size)

            return  cim.dot(result)

        elif wrt is self.texture_stack:
            IS = np.nonzero(self.visibility_image.ravel() != 4294967295)[0]
            texcoords, texidx = self.texcoord_image_quantized
            vis_texidx = texidx.ravel()[IS]
            vis_texcoords = texcoords.ravel()[IS]
            JS = vis_texcoords *  np.tile(col(vis_texidx), [1,2]).ravel()

            clr_im = -2. * (self.imageGT.r - self.render_image) * self.renderWithoutTexture

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

        return None

    def on_changed(self, which):

        super().on_changed(which)

        if 'imageGT' in which:

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
            image = np.array(np.flipud((self.imageGT.r)), order='C', dtype=np.float32)
            # GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA, image.shape[1], image.shape[0])
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

        if 'v' or 'f' or 'vc' or 'ft' or 'camera' or 'texture_stack' or 'imageGT' in which:
            self.render_errors()

    def compute_r(self):
        return self.render_sqerrors

    @depends_on(dterms+terms)
    def render_sqerrors(self):
        self._call_on_changed()

        return self.errors


    @depends_on(dterms+terms)
    def renderWithoutColor(self):
        self._call_on_changed()

        return self.render_nocolor

    @depends_on(dterms+terms)
    def renderWithoutTexture(self):
        self._call_on_changed()

        return self.render_notexture

    @depends_on(dterms+terms)
    def render_dedx(self):
        self._call_on_changed()

        return self.dEdx

    @depends_on(dterms+terms)
    def render_dedy(self):
        self._call_on_changed()

        return self.dEdy

    @depends_on(dterms+terms)
    def render_image(self):
        self._call_on_changed()

        return self.render

    def dErrors_wrt_2dVerts(self, observed, dEdx, dEdy, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""

        # xdiff = dEdx
        # ydiff = dEdy
        nVisF = len(visibility.ravel()[visible])
        projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])
        visTriVC = self.vc.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 3])

        p1 = projVertices[:, 0, :]
        p2 = projVertices[:, 1, :]
        p3 = projVertices[:, 2, :]

        u1 = projVertices[:,0,0]
        v1 = projVertices[:,0,1]
        u2 = projVertices[:,1,0]
        v2 = projVertices[:,1,1]
        u3 = projVertices[:,2,0]
        v3 = projVertices[:,2,1]

        D = np.linalg.det(np.concatenate([(p3-p1).reshape([nVisF, 1, 2]), (p1-p2).reshape([nVisF, 1, 2])], axis=1))
        dBar1dx = (v2 -v3)/D
        dBar2dx = (v3 - v1)/D
        dBar3dx = (v1 - v2)/D

        dBar1dy = (u3 - u2)/D
        dBar2dy = (u1 - u3)/D
        dBar3dy = (u2 - u1)/D

        dEVis = dEdx.reshape([-1,3])[visible]

        visTriVC = -dEVis[:,None] * visTriVC

        # xdiff = (visTriVC[:,0,:] * dBar1dx[:,None] + visTriVC[:,1,:]* dBar2dx[:,None] + visTriVC[:,2,:]* dBar3dx[:,None])
        # ydiff = (visTriVC[:, 0, :] * dBar1dy[:, None] + visTriVC[:, 1, :] * dBar2dy[:, None] + visTriVC[:, 2, :] * dBar3dy[:, None])

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

        datas = []

        # The data is weighted according to barycentric coordinates
        # bc0 = col(barycentric[pys, pxs, 0])
        # bc1 = col(barycentric[pys, pxs, 1])
        # bc2 = col(barycentric[pys, pxs, 2])
        # for k in range(n_channels):
            # dxs = xdiff[pys, pxs, k]
            # dys = ydiff[pys, pxs, k]
            # if f.shape[1] == 3:
            # datas.append(np.hstack((col(visTriVC[:,0,:] * dBar1dx[:,None]),col(visTriVC[:, 0, :] * dBar1dy[:, None]), col(visTriVC[:,1,:]* dBar2dx[:,None]),col(visTriVC[:, 1, :] * dBar2dy[:, None]),col(visTriVC[:,2,:]* dBar3dx[:,None]),col(visTriVC[:, 2, :] * dBar3dy[:, None]))).ravel())
                # datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
            # else:
                # datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())
                # datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())
        data = np.concatenate(((visTriVC[:,0,:] * dBar1dx[:,None])[:,:,None],(visTriVC[:, 0, :] * dBar1dy[:, None])[:,:,None], (visTriVC[:,1,:]* dBar2dx[:,None])[:,:,None], (visTriVC[:, 1, :] * dBar2dy[:, None])[:,:,None],(visTriVC[:,2,:]* dBar3dx[:,None])[:,:,None],(visTriVC[:, 2, :] * dBar3dy[:, None])[:,:,None]),axis=2).swapaxes(0,1).ravel()
        # data = np.concatenate(datas)

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

        return result


def main():
    pass

if __name__ == '__main__':
    main()


