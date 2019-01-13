#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['ColoredRenderer', 'TexturedRenderer']

import numpy as np
import pdb
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
# import pdb
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
            # print ("Clearing when not initialized.")
            return

        if self.win:
            try:
                # print ("Clearing base renderer.")

                GL.glDeleteProgram(self.colorProgram)

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
                self.vbo_colors_ub.set_array(np.array([]))
                self.vbo_colors_ub.bind()
                self.vbo_colors_ub.unbind()
                self.vbo_colors_ub.delete()
                self.vbo_colors.set_array(np.array([]))
                self.vbo_colors.bind()
                self.vbo_colors.unbind()
                self.vbo_colors.delete()
                self.vbo_colors_face.set_array(np.array([]))
                self.vbo_colors_face.bind()
                self.vbo_colors_face.unbind()
                self.vbo_colors_face.delete()

                GL.glDeleteVertexArrays(1, [self.vao_static.value])
                GL.glDeleteVertexArrays(1, [self.vao_static_face.value])
                GL.glDeleteVertexArrays(1, [self.vao_dyn.value])
                GL.glDeleteVertexArrays(1, [self.vao_dyn_ub.value])

                GL.glDeleteRenderbuffers(1, [int(self.render_buf)])
                GL.glDeleteRenderbuffers(1, [int(self.z_buf)])
                if self.msaa:
                    GL.glDeleteRenderbuffers(1, [int(self.render_buf_ms)])
                    GL.glDeleteRenderbuffers(1, [int(self.z_buf_ms)])

                GL.glDeleteFramebuffers(1, [int(self.fbo)])
                GL.glDeleteFramebuffers(1, [int(self.fbo_noms)])
                if self.msaa:
                    GL.glDeleteFramebuffers(1, [int(self.fbo_ms)])

                # print("Finished clearning base renderer")

            except:
                pdb.set_trace()

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
            try:
                self.sharedWin
            except:
                self.sharedWin = None
            self.ctx = mesa.OSMesaCreateContext(GL.GL_RGBA, self.sharedWin)
            self.win = self.ctx
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

        self.line_width = 1.

        #FBO_f
        # if self.msaa and self.glMode == 'glfw':
        if self.msaa:
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


        self.vao_static = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_static)
        GL.glBindVertexArray(self.vao_static)

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

        FRAGMENT_SHADER_NOPERSP = shaders.compileShader("""#version 330 core
        // Interpolated values from the vertex shaders
        in vec3 theColor;
        //noperspective in vec3 theColor;
        // Ouput data
        out vec3 color;
        void main(){
            color = color.xyz;
        }""", GL.GL_FRAGMENT_SHADER)

        VERTEX_SHADER_NOPERSP = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;
        uniform mat4 MVP;
        out vec3 theColor;
        //noperspective out vec3 theColor;
        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            theColor = color;
        }""", GL.GL_VERTEX_SHADER)

        self.colorProgram_noperspective = shaders.compileProgram(VERTEX_SHADER_NOPERSP,FRAGMENT_SHADER_NOPERSP)

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

    @depends_on(terms+dterms)
    def boundarybool_image_aa(self):
        self._call_on_changed()
        boundaryid_image = self.boundaryid_image_aa
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
        return np.asarray(np.tile(np.eye(3)[:self.f.shape[1], :], (self.verts_by_face.shape[0]//self.f.shape[1], 1)), dtype=np.float64, order='C')


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

        GL.glDisable(GL.GL_CULL_FACE)

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
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1, 1)
        self.draw_colored_verts(np.zeros_like(self.vc.r))
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        ec = np.arange(1, len(e)+1)
        ec = np.tile(ec.reshape((-1,1)), (1, 3))
        ec[:, 0] = ec[:, 0] & 255
        ec[:, 1] = (ec[:, 1] >> 8 ) & 255
        ec[:, 2] = (ec[:, 2] >> 16 ) & 255
        ec = np.asarray(ec, dtype=np.uint8)

        # GL.glDepthFunc(GL.GL_GREATER)

        # GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
        # GL.glPolygonOffset(-10000.0, -10000.0)
        # GL.glDepthMask(GL.GL_FALSE)
        # self.projectionMatrix[2, 2] += 0.0000001

        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        self.draw_colored_primitives(self.vao_dyn_ub, v, e, ec)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthFunc(GL.GL_LESS)

        # self.projectionMatrix[2, 2] -= 0.0000001
        # GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
        # GL.glDepthMask(GL.GL_TRUE)

        # if hidden_wireframe:
        #     GL.glEnable(GL.GL_DEPTH_TEST)
        #     GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        #     #Pol change it to a smaller number to avoid double edges in my teapot.
        #     GL.glPolygonOffset(1.0, 1.0)
        #     GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #     # self.draw_colored_primitives(self.vao_dyn_ub, v, f, fc=np.zeros(f.shape).astype(np.uint8))
        #     self.draw_colored_verts(np.zeros_like(self.vc.r))
        #     # self.draw_colored_primitives(self.vaoub, v, e, np.zeros_like(ec).astype(np.uint8))
        #     # self.projectionMatrix[2,2] -= delta
        #     GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        raw = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))

        raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        return raw

    def draw_edge_visibility_aa(self, v, e, f, hidden_wireframe=True):
        """Assumes camera is set up correctly in gl context."""
        shaders.glUseProgram(self.colorProgram)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1, 1)
        self.draw_colored_verts(np.zeros_like(self.vc.r))
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        ec = np.arange(1, len(e)+1)
        ec = np.tile(ec.reshape((-1,1)), (1, 3))
        ec[:, 0] = ec[:, 0] & 255
        ec[:, 1] = (ec[:, 1] >> 8 ) & 255
        ec[:, 2] = (ec[:, 2] >> 16 ) & 255
        ec = np.asarray(ec, dtype=np.uint8)
        ec = np.ones_like(ec, dtype=np.uint8)*255

        # GL.glDepthFunc(GL.GL_GREATER)

        # GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
        # GL.glPolygonOffset(-10000.0, -10000.0)
        # GL.glDepthMask(GL.GL_FALSE)
        # self.projectionMatrix[2, 2] += 0.0000001

        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        # GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(1)
        self.draw_colored_primitives(self.vao_dyn_ub, v, e, ec)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(self.line_width)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glDisable(GL.GL_LINE_SMOOTH)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthFunc(GL.GL_LESS)

        # self.projectionMatrix[2, 2] -= 0.0000001
        # GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
        # GL.glDepthMask(GL.GL_TRUE)

        # if hidden_wireframe:
        #     GL.glEnable(GL.GL_DEPTH_TEST)
        #     GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        #     #Pol change it to a smaller number to avoid double edges in my teapot.
        #     GL.glPolygonOffset(1.0, 1.0)
        #     GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #     # self.draw_colored_primitives(self.vao_dyn_ub, v, f, fc=np.zeros(f.shape).astype(np.uint8))
        #     self.draw_colored_verts(np.zeros_like(self.vc.r))
        #     # self.draw_colored_primitives(self.vaoub, v, e, np.zeros_like(ec).astype(np.uint8))
        #     # self.projectionMatrix[2,2] -= delta
        #     GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        raw = np.flipud(np.frombuffer(GL.glReadPixels( 0,0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'],self.frustum['height'],3).astype(np.uint32))

        raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 

        plt.imsave('raw.png',raw)
        import ipdb; ipdb.set_trace()

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

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

        # if primtype == GL.GL_LINES:
        #     GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        # else:
        #     GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)

        #Pol: FIX THIS (UNCOMMENT)
        if primtype == GL.GL_LINES:
            # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

            f = np.fliplr(f).copy()
            verts_by_edge = v.reshape((-1,3))[f.ravel()]
            verts_by_edge = np.asarray(verts_by_edge, dtype=np.float32, order='C')

            self.vbo_verts_dyn.set_array(verts_by_edge)
            self.vbo_verts_dyn.bind()

            self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
            self.vbo_indices_dyn.bind()

            # GL.glDrawElements(GL.GL_LINES, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)
            # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

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
        silhouette_edges = np.asarray(np.nonzero(dps<=0.)[0], np.uint32)

        self.vis_silhouette_face = np.c_[faces_invisible.take(fpe[:, 0])[dps <= 0.], faces_invisible.take(fpe[:, 1])[dps <= 0.]] < 0

        # silhouette_edges = np.asarray(np.nonzero(dps<=1e-5)[0], np.uint32)

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

            # self.faces_facing_camera = faces_facing_camera
            self.silhouette_edges = silhouette_edges

            lines_e = vpe[silhouette_edges]
            self.lines_e = lines_e
            lines_v = v

            if len(lines_e)==0:
                return np.ones((self.frustum['height'], self.frustum['width'])).astype(np.int32) * 4294967295

            # fpe = fpe[np.any(np.in1d(fpe, np.unique(self.visibility_image[self.visibility_image != 4294967295])).reshape([-1, 2]), 1)]

            visibility = self.draw_edge_visibility(lines_v, lines_e, f, hidden_wireframe=True)
            visibility_edge = visibility.copy()
            # plt.imsave("opendr_boundary_edge_visibility.png", visibility)

            shape = visibility.shape
            visibility = visibility.ravel()
            visible = np.nonzero(visibility.ravel() != 4294967295)[0]

            visibility[visible] = silhouette_edges.take(visibility.take(visible))

            self.frontFacingEdgeFaces = np.zeros([visibility_edge.shape[0],visibility_edge.shape[1], 2]).astype(np.int32).reshape([-1,2])

            self.frontFacingEdgeFaces[visible] = self.vis_silhouette_face[visibility_edge.ravel().take(visible)]

            # plt.imsave("opendr_boundary_edge_visibility_result.png", visibility.reshape(shape))
            return visibility.reshape(shape)


    def draw_boundaryid_image_aa(self, v, f, vpe, fpe, camera):

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

            # self.faces_facing_camera = faces_facing_camera
            self.silhouette_edges = silhouette_edges

            lines_e = vpe[silhouette_edges]
            self.lines_e = lines_e
            lines_v = v

            if len(lines_e)==0:
                return np.ones((self.frustum['height'], self.frustum['width'])).astype(np.int32) * 4294967295

            # fpe = fpe[np.any(np.in1d(fpe, np.unique(self.visibility_image[self.visibility_image != 4294967295])).reshape([-1, 2]), 1)]

            visibility = self.draw_edge_visibility_aa(lines_v, lines_e, f, hidden_wireframe=True)
            visibility_edge = visibility.copy()
            # plt.imsave("opendr_boundary_edge_visibility.png", visibility)

            shape = visibility.shape
            visibility = visibility.ravel()
            visible = np.nonzero(visibility.ravel() != 4294967295)[0]

            visibility[visible] = silhouette_edges.take(visibility.take(visible))

            self.frontFacingEdgeFaces = np.zeros([visibility_edge.shape[0],visibility_edge.shape[1], 2]).astype(np.int32).reshape([-1,2])

            self.frontFacingEdgeFaces[visible] = self.vis_silhouette_face[visibility_edge.ravel().take(visible)]

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

        rr = result.ravel()
        faces_to_draw = np.unique(rr[rr != 4294967295])
        if len(faces_to_draw)==0:
            result = np.ones((self.frustum['height'], self.frustum['width'])).astype(np.uint32)*4294967295
            return result
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        result2 = self.draw_visibility_image_internal(v, f)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        bbi = boundarybool_image

        # result2 = result2.ravel()
        # idxs = result2 != 4294967295
        # result2[idxs] = faces_to_draw[result2[idxs]]

        if False:
            import matplotlib.pyplot as plt
            result2 = result2.reshape(result.shape[:2])
            plt.figure()
            plt.subplot(121)
            plt.imshow(result.squeeze())
            plt.subplot(122)
            plt.imshow(result2.squeeze())
            plt.show()
            pdb.set_trace()

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
        # print ("Clearing color renderer.")
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
            GL.glFlush()
            GL.glFinish()
            # print ("Clearing textured renderer.")
            # for msh in self.vbo_indices_mesh_list:
            #     for vbo in msh:
            #         vbo.set_array([])
            [vbo.set_array(np.array([])) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.set_array(np.array([])) for vbo in self.vbo_colors_mesh]
            [vbo.bind() for vbo in self.vbo_colors_mesh]
            [vbo.delete() for vbo in self.vbo_colors_mesh]
            [vbo.unbind() for vbo in self.vbo_colors_mesh]
            [vbo.delete() for vbo in self.vbo_verts_mesh]
            [vbo.set_array(np.array([])) for vbo in self.vbo_uvs_mesh]
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
        except:
            pdb.set_trace()
            print("Program had not been initialized")

    def initGLTexture(self):
        print("Initializing Texture OpenGL.")

        GL.glLineWidth(1.)

        FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
        // Interpolated values from the vertex shaders
        //#extension GL_EXT_shader_image_load_store : enable 
        in vec3 theColor;
        in vec2 UV;
        uniform sampler2D myTextureSampler;
        // Ouput data
        out vec3 color;
        void main(){
            color = theColor * texture( myTextureSampler, UV).rgb;
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

        if not self.overdraw or self.msaa:
            return no_overdraw

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

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
        colors = np.array(np.ones_like(vc)*(index)/255.0, dtype=np.float32)

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

                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size+textureCoordIdx].reshape(image.shape)

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


class AnalyticRenderer(ColoredRenderer):

    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list' , 'imageGT'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    def __init__(self):
        super().__init__()


    def clear(self):
        try:
            GL.glFlush()
            GL.glFinish()
            # print ("Clearing textured renderer.")
            # for msh in self.vbo_indices_mesh_list:
            #     for vbo in msh:
            #         vbo.set_array([])
            [vbo.set_array(np.array([])) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_colors_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_verts_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_uvs_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_face_ids_list for vbo in sublist]

            [GL.glDeleteVertexArrays(1, [vao.value]) for sublist in self.vao_tex_mesh_list for vao in sublist]

            self.release_textures()

            if self.glMode == 'glfw':
                import glfw
                glfw.make_context_current(self.win)

            GL.glDeleteProgram(self.colorTextureProgram)

            super().clear()
        except:
            import pdb
            pdb.set_trace()
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
            color = theColor * texture( myTextureSampler, UV).rgb;
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

        # GL.glEnable(GL.GL_LINE_SMOOTH)
        # GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(2.)

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_indices_mesh = []
            vbo_face_ids_mesh = []
            vbo_colors_mesh = []
            vbo_vertices_mesh = []
            vbo_uvs_mesh = []
            textureIDs_mesh = []
            for polygons in range(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_verts = vbo.VBO(np.array(verts_by_face).astype(np.float32))
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_colors = vbo.VBO(np.array(colors_by_face).astype(np.float32))
                uvs_by_face = np.asarray(self.ft_list[mesh].reshape((-1, 2))[f.ravel()], dtype=np.float32, order='C')
                vbo_uvs = vbo.VBO(np.array(uvs_by_face).astype(np.float32))

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
                vbo_colors_mesh = vbo_colors_mesh + [vbo_colors]
                vbo_vertices_mesh = vbo_vertices_mesh + [vbo_verts]
                vbo_uvs_mesh = vbo_uvs_mesh + [vbo_uvs]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

            self.vbo_colors_mesh = self.vbo_colors_mesh + [vbo_colors_mesh]
            self.vbo_verts_mesh = self.vbo_verts_mesh + [vbo_vertices_mesh]
            self.vbo_uvs_mesh = self.vbo_uvs_mesh + [vbo_uvs_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID  = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")


    def initGL_AnalyticRenderer(self):
        self.initGLTexture()

        self.updateRender = True
        self.updateDerivatives = True

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 colorIn;
        layout(location = 2) in vec2 vertexUV;
        layout(location = 3) in uint face_id;
        layout(location = 4) in vec3 barycentric;

        uniform mat4 MVP;
        out vec3 theColor;
        out vec4 pos;
        flat out uint face_out;
        out vec3 barycentric_vert_out;
        out vec2 UV;
        
        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            pos =  MVP * vec4(position,1);
            //pos =  pos4.xyz;
            theColor = colorIn;
            UV = vertexUV;
            face_out = face_id;
            barycentric_vert_out = barycentric;
            
        }""", GL.GL_VERTEX_SHADER)

        ERRORS_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 

            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
            in vec3 theColor;
            in vec2 UV;
            flat in uint face_out;
            in vec4 pos;
            in vec3 barycentric_vert_out;
                        
            layout(location = 3) uniform sampler2D myTextureSampler;

            uniform float ww;
            uniform float wh;
    
            // Ouput data
            layout(location = 0) out vec3 color; 
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 barycentric1;
            layout(location = 4) out vec2 barycentric2;
            
            void main(){
                vec3 finalColor = theColor * texture( myTextureSampler, UV).rgb;
                color = finalColor.rgb;
                                
                sample_pos = ((0.5*pos.xy/pos.w) + 0.5)*vec2(ww,wh);
                sample_face = face_out;
                barycentric1 = barycentric_vert_out.xy;
                barycentric2 = vec2(barycentric_vert_out.z, 0.);
                
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
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable
  
            layout(location = 2) uniform sampler2DMS colors;
            layout(location = 3) uniform sampler2DMS sample_positions;
            layout(location = 4) uniform usampler2DMS sample_faces;
            layout(location = 5) uniform sampler2DMS sample_barycentric_coords1;
            layout(location = 6) uniform sampler2DMS sample_barycentric_coords2;

            uniform float ww;
            uniform float wh;
            uniform int sample;

            // Ouput data
            layout(location = 0) out vec3 colorFetchOut;
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 sample_barycentric1;
            layout(location = 4) out vec2 sample_barycentric2;

            //out int gl_SampleMask[];
            const int all_sample_mask = 0xffff;

            void main(){
                ivec2 texcoord = ivec2(gl_FragCoord.xy);
                colorFetchOut = texelFetch(colors, texcoord, sample).xyz;
                sample_pos = texelFetch(sample_positions, texcoord, sample).xy;        
                sample_face = texelFetch(sample_faces, texcoord, sample).r;
                sample_barycentric1 = texelFetch(sample_barycentric_coords1, texcoord, sample).xy;
                sample_barycentric2 = texelFetch(sample_barycentric_coords2, texcoord, sample).xy;


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

        # GL.glGenTextures(1, self.textureEdges)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureEdges)
        # GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)

        # GL.glActiveTexture(GL.GL_TEXTURE0)

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

        self.texture_errors_sample_faces = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_R32UI, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces, 0)
        #

        self.texture_errors_sample_barycentric1 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1, 0)

        self.texture_errors_sample_barycentric2 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2, 0)


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

        self.render_buffer_fetch_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        #
        self.render_buffer_fetch_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)

        self.render_buffer_fetch_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)

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

        render_buf_errors_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_sample_position)

        render_buf_errors_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        #

        render_buf_errors_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)

        render_buf_errors_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
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
        face_ids_location = GL.glGetAttribLocation(self.errorTextureProgram, 'face_id')
        barycentric_location = GL.glGetAttribLocation(self.errorTextureProgram, 'barycentric')

        # self.vbo_verts_cube= vbo.VBO(np.array(self.v_bgCube).astype(np.float32))
        # self.vbo_colors_cube= vbo.VBO(np.array(self.vc_bgCube).astype(np.float32))
        # self.vbo_uvs_cube = vbo.VBO(np.array(self.ft_bgCube).astype(np.float32))
        # self.vao_bgCube = GL.GLuint(0)
        # GL.glGenVertexArrays(1, self.vao_bgCube)
        # 
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube = vbo.VBO(np.array(self.f_bgCube).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        # self.vbo_f_bgCube.bind()
        # self.vbo_verts_cube.bind()
        # GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_colors_cube.bind()
        # GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_uvs_cube.bind()
        # GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # 
        # f = self.f_bgCube
        # fc = np.tile(np.arange(len(self.f), len(self.f) + len(f))[:, None], [1, 3]).ravel()
        # # fc[:, 0] = fc[:, 0] & 255
        # # fc[:, 1] = (fc[:, 1] >> 8) & 255
        # # fc[:, 2] = (fc[:, 2] >> 16) & 255
        # fc = np.asarray(fc, dtype=np.uint32)
        # vbo_face_ids_cube = vbo.VBO(fc)
        # vbo_face_ids_cube.bind()
        # GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
        # GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)
        # 
        # #Barycentric cube:
        # f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
        # vbo_barycentric_cube = vbo.VBO(f_barycentric)
        # vbo_barycentric_cube.bind()
        # GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
        # GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_quad = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_quad)
        GL.glBindVertexArray(self.vao_quad)


        #Bind VAO

        self.vbo_face_ids_list = []
        self.vbo_barycentric_list = []
        self.vao_errors_mesh_list = []
        flen = 1

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_face_ids_mesh = []
            vbo_barycentric_mesh = []
            for polygons in np.arange(len(self.f_list[mesh])):

                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]
                vbo_f.bind()
                vbo_verts = self.vbo_verts_mesh[mesh][polygons]
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_colors = self.vbo_colors_mesh[mesh][polygons]
                vbo_colors.bind()

                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_uvs = self.vbo_uvs_mesh[mesh][polygons]
                vbo_uvs.bind()
                GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                f = self.f_list[mesh][polygons]

                fc = np.tile(np.arange(flen, flen + len(f))[:,None], [1,3]).ravel()
                # fc[:, 0] = fc[:, 0] & 255
                # fc[:, 1] = (fc[:, 1] >> 8) & 255
                # fc[:, 2] = (fc[:, 2] >> 16) & 255
                fc = np.asarray(fc, dtype=np.uint32)
                vbo_face_ids = vbo.VBO(fc)
                vbo_face_ids.bind()
                GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
                GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)

                f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
                vbo_barycentric = vbo.VBO(f_barycentric)
                vbo_barycentric.bind()
                GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                flen += len(f)

                vaos_mesh += [vao]

                vbo_face_ids_mesh += [vbo_face_ids]
                vbo_barycentric_mesh += [vbo_face_ids]

                GL.glBindVertexArray(0)

            self.vbo_face_ids_list += [vbo_face_ids_mesh]
            self.vbo_barycentric_list += [vbo_barycentric_mesh]
            self.vao_errors_mesh_list += [vaos_mesh]

    def render_image_buffers(self):

        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        self.makeCurrentContext()

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)

        GL.glUseProgram(self.errorTextureProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        # GL.glClearBufferiv(GL.GL_COLOR​, 0​, 0)
        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        wwLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_errors_mesh_list[mesh][polygons]

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                # vbo_color.bind()

                f = self.f_list[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                self.vbo_colors_mesh[mesh][polygons].bind()

                if self.f.shape[1]==2:
                    primtype = GL.GL_LINES
                else:
                    primtype = GL.GL_TRIANGLES

                assert(primtype == GL.GL_TRIANGLES)

                # GL.glUseProgram(self.errorTextureProgram)
                if self.haveUVs_list[mesh][polygons]:
                    texture =  self.textureID_mesh_list[mesh][polygons]
                else:
                    texture = self.whitePixelTextureID

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glUniform1i(self.textureObjLoc, 0)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawArrays(primtype, 0, len(vbo_f)*vbo_f.data.shape[1])

        # # #Background cube:
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube.bind()
        # texture = self.whitePixelTextureID
        # self.vbo_uvs_cube.bind()
        #
        # GL.glActiveTexture(GL.GL_TEXTURE0)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # GL.glUniform1i(self.textureObjLoc, 0)
        # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
        #
        # GL.glDrawElements(primtype, len(self.vbo_f_bgCube)*self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

        # self.draw_visibility_image_ms(self.v, self.f)

        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # # result_blit = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        # result_blit2 = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # result_blit_pos = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

        GL.glUseProgram(self.fetchSamplesProgram)
        # GL.glDisable(GL.GL_MULTISAMPLE)

        self.colorsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "colors")
        self.sample_positionsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_positions")
        self.sample_facesLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_faces")
        self.sample_barycentric1Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords1")
        self.sample_barycentric2Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords2")

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
        self.renders_faces = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height']]).astype(np.uint32)
        self.renders_sample_barycentric1 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_sample_barycentric2 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],1])
        self.renders_sample_barycentric = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],3])

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_sample_fetch)
        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3,
                          GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for sample in np.arange(self.nsamples):

            sampleLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'sample')
            GL.glUniform1i(sampleLoc, sample)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
            GL.glUniform1i(self.colorsLoc, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
            GL.glUniform1i(self.sample_positionsLoc, 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
            GL.glUniform1i(self.sample_facesLoc, 2)

            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
            GL.glUniform1i(self.sample_barycentric1Loc, 3)

            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
            GL.glUniform1i(self.sample_barycentric2Loc, 4)

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

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT), np.uint32).reshape(self.frustum['height'], self.frustum['height'])[:,:].astype(np.uint32))
            self.renders_faces[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:2].astype(np.float64))
            self.renders_sample_barycentric1[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:1].astype(np.float64))
            self.renders_sample_barycentric2[sample] = result

            self.renders_sample_barycentric[sample] = np.concatenate([self.renders_sample_barycentric1[sample], self.renders_sample_barycentric2[sample][:,:,0:1]], 2)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            # result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
            # self.renders_faces[sample] = result


        GL.glBindVertexArray(0)

        GL.glClearColor(0.,0.,0., 1.)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)

        ##Finally return image and derivatives

        self.render_resolved = np.mean(self.renders, 0)

        self.updateRender = True
        self.updateDerivatives_verts = True
        self.updateDerivatives_vc = True


    def draw_visibility_image_ms(self, v, f):
        """Assumes camera is set up correctly in"""
        GL.glUseProgram(self.visibilityProgram_ms)

        v = np.asarray(v)

        self.draw_visibility_image_ms(v, f)

        #Attach FBO
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        fc = np.arange(1, len(f)+1)
        fc = np.tile(fc.reshape((-1,1)), (1, 3))
        fc[:, 0] = fc[:, 0] & 255
        fc[:, 1] = (fc[:, 1] >> 8 ) & 255
        fc[:, 2] = (fc[:, 2] >> 16 ) & 255
        fc = np.asarray(fc, dtype=np.uint8)

        self.draw_colored_primitives_ms(self.vao_dyn_ub,  v, f, fc)


    # this assumes that fc is either "by faces" or "verts by face", not "by verts"
    def draw_colored_primitives_ms(self, vao, v, f, fc=None):

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

            vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
            self.vbo_colors_ub.set_array(vc_by_face)
            self.vbo_colors_ub.bind()

        primtype = GL.GL_TRIANGLES

        self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
        self.vbo_indices_dyn.bind()

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(1, drawingBuffers)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)
        GL.glEnable(GL.GL_DEPTH_TEST)


    def compute_dr_wrt(self, wrt):

        visibility = self.visibility_image

        if wrt is self.camera:
            derivatives_verts = self.get_derivatives_verts()

            return derivatives_verts

        elif wrt is self.vc:

            derivatives_vc = self.get_derivatives_vc()

            return derivatives_vc

        # Not working atm.:
        elif wrt is self.bgcolor:
            return 2. * (self.imageGT.r - self.render_image).ravel() * common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

        #Not working atm.:
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

    def compute_r(self):
        return self.render()

    @depends_on(dterms+terms)
    def renderWithoutColor(self):
        self._call_on_changed()

        return self.render_nocolor

    @depends_on(dterms+terms)
    def renderWithoutTexture(self):
        self._call_on_changed()

        return self.render_notexture

    # @depends_on(dterms+terms)
    def render(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        barycentric = self.barycentric_image

        if self.updateRender:
            render = self.compute_image(visible, visibility, self.f)
            self.render_result = render
            self.updateRender = False
        return self.render_result

    def get_derivatives_verts(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_verts:
            if self.updateRender:
                self.render()
            derivatives_verts = self.compute_derivatives_verts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
            self.derivatives_verts = derivatives_verts
            self.updateDerivatives_verts = False
        return self.derivatives_verts

    def get_derivatives_vc(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_vc:
            if self.updateRender:
                self.render()
            derivatives_vc = self.compute_derivatives_vc(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
            self.derivatives_vc = derivatives_vc
            self.updateDerivatives_vc = False
        return self.derivatives_vc

    # # @depends_on(dterms+terms)
    # def image_and_derivatives(self):
    #     # self._call_on_changed()
    #     visibility = self.visibility_image
    #
    #     color = self.render_resolved
    #
    #     visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    #     num_visible = len(visible)
    #
    #     barycentric = self.barycentric_image
    #
    #     if self.updateRender:
    #         render, derivatives = self.compute_image_and_derivatives(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
    #         self.render = render
    #         self.derivatives = derivatives
    #         self.updateRender = False
    #
    #     return self.render, self.derivatives
    #

    def barycentricDerivatives(self, vertices, faces, verts):
        import chumpy as ch

        vertices = np.concatenate([vertices, np.ones([vertices.size // 3, 1])], axis=1)
        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])

        viewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])

        # # Check with autodiff:
        #
        # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        # # negYMat = ch.array([[1,0,self.camera.c.r[0]],[0,-1,self.camera.c.r[1]],[0,0,1]])
        # verts_hom_ch = ch.Ch(verts_hom)
        # camMtx = ch.Ch(np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])])
        # projVerts = (camMtx.dot(view_mtx)).dot(verts_hom_ch.T).T[:, :3].reshape([-1, 3])
        # viewVerts = ch.Ch(np.array(projVerts))
        # projVerts = projVerts[:, :2] / projVerts[:, 2:3]
        #
        # chViewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        # p0 = ch.Ch(viewVerticesNonBnd[:, 0, :])
        # chp0 = p0
        #
        # p1 = ch.Ch(viewVerticesNonBnd[:, 1, :])
        # chp1 = p1
        #
        # p2 = ch.Ch(viewVerticesNonBnd[:, 2, :])
        # chp2 = p2
        #
        # # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        # nt = ch.cross(p1 - p0, p2 - p0)
        # chnt = nt
        # A = 0.5 * ch.sqrt(ch.sum(nt ** 2, axis=1))
        # chnt_norm = nt / ch.sqrt(ch.sum(nt ** 2, axis=1))[:, None]
        # # nt = nt / A
        #
        # chb0part2 = ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1)
        # chb0 = 0.5 * ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1) / A
        # chb1part2 = ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1)
        # chb1 = 0.5 * ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1) / A
        # chb2part2 = ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1)
        # chb2 = 0.5 * ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1) / A
        #
        # drb0p0 = chb0.dr_wrt(p0)
        # drb0p1 = chb0.dr_wrt(p1)
        # drb0p2 = chb0.dr_wrt(p2)
        #
        # drb1p0 = chb1.dr_wrt(p0)
        # drb1p1 = chb1.dr_wrt(p1)
        # drb1p2 = chb1.dr_wrt(p2)
        #
        # drb2p0 = chb2.dr_wrt(p0)
        # drb2p1 = chb2.dr_wrt(p1)
        # drb2p2 = chb2.dr_wrt(p2)
        #
        # rows = np.tile(np.arange(drb0p0.shape[0])[None, :], [3, 1]).T.ravel()
        # cols = np.arange(drb0p0.shape[0] * 3)
        #
        # drb0p0 = np.array(drb0p0[rows, cols]).reshape([-1, 3])
        # drb0p1 = np.array(drb0p1[rows, cols]).reshape([-1, 3])
        # drb0p2 = np.array(drb0p2[rows, cols]).reshape([-1, 3])
        # drb1p0 = np.array(drb1p0[rows, cols]).reshape([-1, 3])
        # drb1p1 = np.array(drb1p1[rows, cols]).reshape([-1, 3])
        # drb1p2 = np.array(drb1p2[rows, cols]).reshape([-1, 3])
        # drb2p0 = np.array(drb2p0[rows, cols]).reshape([-1, 3])
        # drb2p1 = np.array(drb2p1[rows, cols]).reshape([-1, 3])
        # drb2p2 = np.array(drb2p2[rows, cols]).reshape([-1, 3])
        #
        # chdp0 = np.concatenate([drb0p0[:, None, :], drb1p0[:, None, :], drb2p0[:, None, :]], axis=1)
        # chdp1 = np.concatenate([drb0p1[:, None, :], drb1p1[:, None, :], drb2p1[:, None, :]], axis=1)
        # chdp2 = np.concatenate([drb0p2[:, None, :], drb1p2[:, None, :], drb2p2[:, None, :]], axis=1)
        # #
        # # dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)
        # # dp = dp[None, :]

        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])
        viewVerts = projVerts
        projVerts = projVerts[:, :2] / projVerts[:, 2:3]

        # viewVerticesNonBnd = negYMat.dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        p0 = viewVerticesNonBnd[:, 0, :]
        p1 = viewVerticesNonBnd[:, 1, :]
        p2 = viewVerticesNonBnd[:, 2, :]

        p0_proj = p0[:,0:2]/p0[:,2:3]
        p1_proj = p1[:,0:2]/p1[:,2:3]
        p2_proj = p2[:,0:2]/p2[:,2:3]

        # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        nt = np.cross(p1 - p0, p2 - p0)
        nt_norm = nt / np.linalg.norm(nt, axis=1)[:, None]

        # a = -nt_norm[:, 0] / nt_norm[:, 2]
        # b = -nt_norm[:, 1] / nt_norm[:, 2]
        # c = np.sum(nt_norm * p0, 1) / nt_norm[:, 2]

        cam_f = 1

        u = p0[:, 0]/p0[:, 2]
        v = p0[:, 1]/p0[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p0[:, 2][:,None], np.zeros([len(p0),1]), (-p0[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p0),1]), p0[:, 2][:,None], (-p0[:,1]/v**2)[:,None]]

        dxdp_0 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p1[:, 0]/p1[:, 2]
        v = p1[:, 1]/p1[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p1[:, 2][:,None], np.zeros([len(p1),1]), (-p1[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p1),1]), p1[:, 2][:,None], (-p1[:,1]/v**2)[:,None]]

        dxdp_1 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p2[:, 0]/p2[:, 2]
        v = p2[:, 1]/p2[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p2[:, 2][:,None], np.zeros([len(p2),1]), (-p2[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p2),1]), p2[:, 2][:,None], (-p2[:,1]/v**2)[:,None]]

        dxdp_2 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        # x = u * c / (cam_f - a * u - b * v)
        # y = v*c/(cam_f - a*u - b*v)
        # z = c*cam_f/(cam_f - a*u - b*v)

        A = 0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0),axis=1)
        nt_mag = A*2
        # nt = nt / A
        # db1 = 0.5*np.cross(nt_norm, p2-p1)/A[:, None]
        # db2 = 0.5*np.cross(nt_norm, p0-p2)/A[:, None]
        # db3_2 = 0.5*np.cross(nt_norm, p1-p0)/A[:, None]
        # db3 = - db1 - db2
        p = viewVerts

        pre1 = -1/(nt_mag[:,None]**2) * nt_norm

        ident = np.identity(3)
        ident = np.tile(ident[None,:],[len(p2),1,1])

        dntdp0 = np.cross((p2-p0)[:,None,:], -ident) + np.cross(-ident, (p1-p0)[:,None,:])
        dntdp1 = np.cross((p2-p0)[:,None,:],ident)
        dntdp2 = np.cross(ident,(p1-p0)[:,None,:])

        #Pol check this!:
        dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]
        # dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]

        dntnormdp0 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp0)
        dntnormdp1 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp1)
        dntnormdp2 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp2)

        dpart1p0 = np.einsum('ij,ijk->ik', pre1, dntdp0)
        dpart1p1 = np.einsum('ij,ijk->ik', pre1, dntdp1)
        dpart1p2 = np.einsum('ij,ijk->ik', pre1, dntdp2)

        b0 = np.sum(np.cross(nt_norm, p2 - p1) * (p - p1), axis=1)[:,None]

        db0part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident,nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1),-ident)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0.swapaxes(1,2)), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident, nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(p2-p1,nt_norm[:, :]),-ident)
        db0part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :],-ident), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1), -ident)
        db0part2p2 = np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p1)

        db0dp0wrtpart1 = dpart1p0*b0
        db0dp1wrtpart1 = dpart1p1*b0
        db0dp2wrtpart1 = dpart1p2*b0

        db0dp0wrtpart2 = 1./(nt_mag[:,None])*db0part2p0
        db0dp1wrtpart2 = 1./(nt_mag[:,None])*db0part2p1
        db0dp2wrtpart2 = 1./(nt_mag[:,None])*db0part2p2

        db0dp0wrt = db0dp0wrtpart1 +  db0dp0wrtpart2
        db0dp1wrt = db0dp1wrtpart1 +  db0dp1wrtpart2
        db0dp2wrt = db0dp2wrtpart1 +  db0dp2wrtpart2

        ######
        b1 = np.sum(np.cross(nt_norm, p0 - p2) * (p - p2), axis=1)[:, None]

        db1part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p2)
        db1part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2)
        db1part2p2 = np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], -ident), p - p2) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p0-p2), -ident)

        db1dp0wrtpart1 = dpart1p0*b1
        db1dp1wrtpart1 = dpart1p1*b1
        db1dp2wrtpart1 = dpart1p2*b1

        db1dp0wrtpart2 = 1./(nt_mag[:,None])*db1part2p0
        db1dp1wrtpart2 = 1./(nt_mag[:,None])*db1part2p1
        db1dp2wrtpart2 = 1./(nt_mag[:,None])*db1part2p2

        db1dp0wrt = db1dp0wrtpart1 + db1dp0wrtpart2
        db1dp1wrt = db1dp1wrtpart1 +  db1dp1wrtpart2
        db1dp2wrt = db1dp2wrtpart1 +  db1dp2wrtpart2

        ######
        b2 = np.sum(np.cross(nt_norm, p1 - p0) * (p - p0), axis=1)[:, None]

        db2part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1, 2),(p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], -ident), p - p0) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p1 - p0), -ident)
        db2part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1, 2),(p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p0)
        db2part2p2 =  np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0)

        db2dp0wrtpart1 = dpart1p0*b2
        db2dp1wrtpart1 = dpart1p1*b2
        db2dp2wrtpart1 = dpart1p2*b2

        db2dp0wrtpart2 = 1./(nt_mag[:,None])*db2part2p0
        db2dp1wrtpart2 = 1./(nt_mag[:,None])*db2part2p1
        db2dp2wrtpart2 = 1./(nt_mag[:,None])*db2part2p2

        db2dp0wrt = db2dp0wrtpart1 + db2dp0wrtpart2
        db2dp1wrt = db2dp1wrtpart1 +  db2dp1wrtpart2
        db2dp2wrt = db2dp2wrtpart1 +  db2dp2wrtpart2

        dp0 = np.concatenate([db0dp0wrt[:, None, :], db1dp0wrt[:, None, :], db2dp0wrt[:, None, :]], axis=1)
        dp1 = np.concatenate([db0dp1wrt[:, None, :], db1dp1wrt[:, None, :], db2dp1wrt[:, None, :]], axis=1)
        dp2 = np.concatenate([db0dp2wrt[:, None, :], db1dp2wrt[:, None, :], db2dp2wrt[:, None, :]], axis=1)
        #
        dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)

        #If dealing with degenerate triangles, ignore that gradient.

        # dp[nt_mag<=1e-15] = 0

        dp = dp[None, :]

        nFaces = len(faces)
        # visTriVC = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc[vc > 1] = 1
        vc[vc < 0] = 0

        visTriVC = vc

        dxdp = np.concatenate([dxdp_0[:,None,:],dxdp_1[:,None,:],dxdp_2[:,None,:]], axis=1)

        dxdp = dxdp[None, :, None]
        # dbvc = np.sum(dp * visTriVC, 2)

        # dbvc = dp * visTriVC * t_area[None, :, None, None, None]
        dbvc = dp * visTriVC

        didp = np.sum(dbvc[:, :, :, :, :, None] * dxdp, 4).sum(2)

        #output should be shape: VC x Ninput x Tri Points x UV

        # drb0p0 # db0dp0wrt
        # drb0p1 # db0dp1wrt
        # drb0p2 # db0dp2wrt
        # drb1p0 # db1dp0wrt
        # drb1p1 # db1dp1wrt
        # drb1p2 # db1dp2wrt
        # drb2p0 # db2dp0wrt
        # drb2p1 # db2dp1wrt
        # drb2p2 # db2dp2wrt
        #

        return didp

    def compute_image(self, visible, visibility, f):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""

        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        # xdiff = dEdx
        # ydiff = dEdy

        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        nsamples = self.nsamples


        if np.any(boundaryImage):
            boundaryFaces = visibility[(boundaryImage) & (visibility != 4294967295)]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

            edgeFaces= np.tile(self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][None, :, :], [8, 1, 1])

            edgeSampled = np.any((edgeFaces[:,:, 0]== sampleFaces) | (edgeFaces[:,:, 1]== sampleFaces),0)

            facesInsideBnd = projFacesBndTiled == sampleFaces
            wrongBnd = ~edgeSampled
            # wrongBnd = np.all(facesInsideBnd, 0)
            whereBnd = np.where(boundaryImage.ravel())[0]
            # boundaryImage.ravel()[whereBnd[wrongBnd]] = False

        if np.any(boundaryImage):

            sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 2])

            # sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:,(zerosIm*boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])
            sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

            boundaryFaces = visibility[(boundaryImage)&(visibility !=4294967295 )]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])
            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            frontFacing = self.frontFacingEdgeFaces[(zerosIm * boundaryImage).ravel().astype(np.bool)].astype(np.bool)
            frontFacingEdgeFaces = self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][frontFacing]

            vertsPerFaceProjBnd = self.camera.r[f[frontFacingEdgeFaces.ravel()].ravel()].reshape([1, -1, 2])
            vertsPerFaceProjBnd = np.tile(vertsPerFaceProjBnd, [self.nsamples, 1,1])
            vertsPerFaceProjBnd = vertsPerFaceProjBnd.reshape([-1,3,2])[facesOutsideBnd.ravel()]

            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_edge = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_edge[t_area_bnd_edge > 1] = 1

            # if self.debug:
            #     import pdb; pdb.set_trace()

            faces = f[sampleFaces[facesOutsideBnd]].ravel()

            vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_outside = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_outside[t_area_bnd_outside > 1] = 1

            faces = f[sampleFaces[facesInsideBnd]].ravel()
            vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_inside = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_inside[t_area_bnd_inside > 1] = 1

            #Trick to cap to 1 while keeping gradients.

            p1 = vertsProjBndSamplesOutside[:,0,:]
            p2 = vertsProjBndSamplesOutside[:,1,:]

            p = sampleV[facesOutsideBnd]

            l = (p2 - p1)
            linedist = np.sqrt((np.sum(l**2,axis=1)))[:,None]
            self.linedist = linedist

            lnorm = l/linedist
            self.lnorm = lnorm

            v1 = p - p1
            self.v1 = v1
            d = v1[:,0]* lnorm[:,0] + v1[:,1]* lnorm[:,1]
            self.d = d
            intersectPoint = p1 + d[:,None] * lnorm
            self.intersectPoint = intersectPoint

            v2 = p - p2
            self.v2 = v2
            l12 = (p1 - p2)
            linedist12 = np.sqrt((np.sum(l12**2,axis=1)))[:,None]
            lnorm12 = l12/linedist12
            d2 = v2[:,0]* lnorm12[:,0] + v2[:,1]* lnorm12[:,1]

            nonIntersect = (d2 < 0) | (d<0)
            self.nonIntersect = nonIntersect

            argminDistNonIntersect = np.argmin(np.c_[d[nonIntersect], d2[nonIntersect]], 1)
            self.argminDistNonIntersect = argminDistNonIntersect

            intersectPoint[nonIntersect] = vertsProjBndSamplesOutside[nonIntersect][np.arange(nonIntersect.sum()), argminDistNonIntersect]

            lineToPoint = (p - intersectPoint)

            n=lineToPoint

            dist = np.sqrt((np.sum(lineToPoint ** 2, axis=1)))[:, None]

            n_norm = lineToPoint /dist

            self.n_norm = n_norm

            self.dist = dist

            d_final = dist.squeeze()

            # max_nx_ny = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))

            # d_final = d_final/max_nx_ny
            # d_final = d_final

            verticesBnd = self.v.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2 , 3])
            verticesBndSamples = np.tile(verticesBnd[None,:,:],[self.nsamples,1,1, 1])
            verticesBndOutside = verticesBndSamples[facesOutsideBnd]

            vc = self.vc.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2 , 3])
            vc[vc > 1] = 1
            vc[vc < 0] = 0

            vcBnd = vc
            vcBndSamples = np.tile(vcBnd[None,:,:],[self.nsamples,1,1,1])
            vcBndOutside = vcBndSamples[facesOutsideBnd]

            invViewMtx =  np.linalg.inv(np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])])
            #
            camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
            # invCamMtx = np.r_[np.c_[np.linalg.inv(self.camera.camera_mtx), np.array([0,0,0])], np.array([[0, 0, 0, 1]])]

            view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]

            verticesBndOutside = np.concatenate([verticesBndOutside.reshape([-1,3]), np.ones([verticesBndOutside.size//3, 1])], axis=1)

            projVerticesBndOutside = (camMtx.dot(view_mtx)).dot(verticesBndOutside.T).T[:,:3].reshape([-1,2,3])
            projVerticesBndDir = projVerticesBndOutside[:,1,:] - projVerticesBndOutside[:,0,:]
            projVerticesBndDir = projVerticesBndDir/np.sqrt((np.sum(projVerticesBndDir ** 2, 1)))[:, None]

            dproj = (intersectPoint[:,0]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,0]) / (projVerticesBndDir[:,0] - projVerticesBndDir[:,2]*intersectPoint[:,0])
            # Code to check computation that dproj == dprojy
            # dproj_y = (intersectPoint[:,1]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,1]) / (projVerticesBndDir[:,1] - projVerticesBndDir[:,2]*intersectPoint[:,1])

            projPoint = projVerticesBndOutside[:,0,:][:,: ] + dproj[:,None]*projVerticesBndDir[:,:]

            projPointVec4 = np.concatenate([projPoint, np.ones([projPoint.shape[0],1])], axis=1)
            viewPointIntersect = (invViewMtx.dot(np.linalg.inv(camMtx)).dot(projPointVec4.T.reshape([4,-1])).reshape([4,-1])).T[:,:3]

            barycentricVertsDistIntesect = np.linalg.norm(viewPointIntersect - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,0,:], axis=1)
            barycentricVertsDistIntesect2 = np.linalg.norm(viewPointIntersect - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,1,:], axis=1)
            # Code to check barycentricVertsDistIntesect + barycentricVertsDistIntesect2 = barycentricVertsDistEdge
            barycentricVertsDistEdge = np.linalg.norm(verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,0,:] - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,1,:], axis=1)

            nonIntersect = np.abs(barycentricVertsDistIntesect + barycentricVertsDistIntesect2 - barycentricVertsDistEdge) > 1e-4
            argminDistNonIntersect = np.argmin(np.c_[barycentricVertsDistIntesect[nonIntersect], barycentricVertsDistIntesect2[nonIntersect]],1)

            barycentricVertsIntersect = barycentricVertsDistIntesect2 / (barycentricVertsDistIntesect + barycentricVertsDistIntesect2)

            barycentricVertsIntersect[nonIntersect] = np.array(argminDistNonIntersect == 0).astype(np.float64)
            self.barycentricVertsIntersect = barycentricVertsIntersect

            self.viewPointIntersect = viewPointIntersect
            self.viewPointIntersect[nonIntersect] = verticesBndOutside.reshape([-1, 2, 4])[nonIntersect, :, 0:3][np.arange(nonIntersect.sum()), argminDistNonIntersect, :]

            vcEdges1 = barycentricVertsIntersect[:, None] * vcBndOutside.reshape([-1, 2, 3])[:, 0, :]
            self.barycentricVertsIntersect = barycentricVertsIntersect
            vcEdges2 = (1-barycentricVertsIntersect[:,None]) * vcBndOutside.reshape([-1,2,3])[:,1,:]

            #Color:
            colorVertsEdge =  vcEdges1 + vcEdges2

            #Point IN edge barycentric

            d_finalNP = np.minimum(d_final.copy(),1.)
            self.d_final_outside = d_finalNP

            self.t_area_bnd_outside =  t_area_bnd_outside
            self.t_area_bnd_edge =  t_area_bnd_edge
            self.t_area_bnd_inside = t_area_bnd_inside
            areaWeights = np.zeros([nsamples, nBndFaces])
            areaWeights[facesOutsideBnd] = (1-d_finalNP)*t_area_bnd_edge + d_finalNP *t_area_bnd_outside
            areaWeights[facesInsideBnd] = t_area_bnd_inside
            areaWeightsTotal = areaWeights.sum(0)
            # areaWeightsTotal[areaWeightsTotal < 1] = 1
            self.areaWeightsTotal = areaWeightsTotal

            finalColorBndOutside = np.zeros([self.nsamples, boundaryFaces.size, 3])
            finalColorBndOutside_edge = np.zeros([self.nsamples, boundaryFaces.size, 3])
            finalColorBndInside = np.zeros([self.nsamples, boundaryFaces.size, 3])


            sampleColorsOutside = sampleColors[facesOutsideBnd]
            self.sampleColorsOutside = sampleColors.copy()

            finalColorBndOutside[facesOutsideBnd] = sampleColorsOutside
            finalColorBndOutside[facesOutsideBnd] = sampleColorsOutside / self.nsamples
            self.finalColorBndOutside_for_dr = finalColorBndOutside.copy()
            # finalColorBndOutside[facesOutsideBnd] *= d_finalNP[:,  None] * t_area_bnd_outside[:,  None]
            finalColorBndOutside[facesOutsideBnd] *= d_finalNP[:,  None]

            finalColorBndOutside_edge[facesOutsideBnd] = colorVertsEdge
            finalColorBndOutside_edge[facesOutsideBnd] = colorVertsEdge/ self.nsamples
            self.finalColorBndOutside_edge_for_dr = finalColorBndOutside_edge.copy()
            # finalColorBndOutside_edge[facesOutsideBnd] *= (1 - d_finalNP[:, None]) * t_area_bnd_edge[:,  None]
            finalColorBndOutside_edge[facesOutsideBnd] *= (1 - d_finalNP[:, None])

            sampleColorsInside = sampleColors[facesInsideBnd]
            self.sampleColorsInside = sampleColorsInside.copy()
            # finalColorBndInside[facesInsideBnd] = sampleColorsInside * self.t_area_bnd_inside[:,  None]

            finalColorBndInside[facesInsideBnd] = sampleColorsInside / self.nsamples

            # finalColorBnd = finalColorBndOutside + finalColorBndOutside_edge + finalColorBndInside
            finalColorBnd = finalColorBndOutside + finalColorBndOutside_edge + finalColorBndInside

            # finalColorBnd /= areaWeightsTotal[None, :, None]

            bndColorsImage = np.zeros_like(self.render_resolved)
            bndColorsImage[(zerosIm * boundaryImage), :] = np.sum(finalColorBnd, axis=0)

            # bndColorsImage1 = np.zeros_like(self.render_resolved)
            # bndColorsImage1[(zerosIm * boundaryImage), :] = np.sum(self.finalColorBndOutside_for_dr, axis=0)
            #
            # bndColorsImage2 = np.zeros_like(self.render_resolved)
            # bndColorsImage2[(zerosIm * boundaryImage), :] = np.sum(self.finalColorBndOutside_edge_for_dr, axis=0)
            #
            # bndColorsImage3 = np.zeros_like(self.render_resolved)
            # bndColorsImage3[(zerosIm * boundaryImage), :] = np.sum(finalColorBndInside, axis=0)

            finalColorImageBnd = bndColorsImage

        if np.any(boundaryImage):
            finalColor = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * finalColorImageBnd
            # finalColor1 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage1
            # finalColor2 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage2
            # finalColor3 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage3

        else:
            finalColor = self.color_image

        finalColor[finalColor>1] = 1
        finalColor[finalColor<0] = 0

        return finalColor

    def compute_derivatives_verts(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size
        n_norm = self.n_norm
        dist = self.dist
        linedist = self.linedist
        d = self.d
        v1 = self.v1
        lnorm = self.lnorm

        finalColorBndOutside_for_dr = self.finalColorBndOutside_for_dr
        finalColorBndOutside_edge_for_dr = self.finalColorBndOutside_edge_for_dr
        d_final_outside = self.d_final_outside

        barycentricVertsIntersect = self.barycentricVertsIntersect

        # xdiff = dEdx
        # ydiff = dEdy

        nVisF = len(visibility.ravel()[visible])
        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility!=4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples
        sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
            [nsamples, -1, 2])

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])

        sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage)&(visibility !=4294967295 )]

        if np.any(boundaryImage):

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            p1 = vertsProjBndSamplesOutside[:, 0, :]
            p2 = vertsProjBndSamplesOutside[:, 1, :]
            p = sampleV[facesOutsideBnd]

            #Computing gradients:
            #A multisampled pixel color is given by: w R + (1-w) R' thus:
            #1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            #2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            #3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            #4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            #5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            #6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            #7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            #for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            #Finally, stack data, and IJ of nonbnd with bnd on both dwrt_v and dwrt_vc.

            ######## 1 derivatives samples outside wrt v 1: (dw * (bar*vc) - dw (bar'*vc') )/ nsamples for face sample

            # #Chumpy autodiff code to check derivatives here:
            # chEdgeVerts = ch.Ch(vertsProjBndSamplesOutside)
            #
            # chEdgeVerts1 = chEdgeVerts[:,0,:]
            # chEdgeVerts2 = chEdgeVerts[:,1,:]
            #
            # chSampleVerts = ch.Ch(sampleV[facesOutsideBnd])
            # # c1 = (chEdgeVerts1 - chSampleVerts)
            # # c2 = (chEdgeVerts2 - chSampleVerts)
            # # n = (chEdgeVerts2 - chEdgeVerts1)
            #
            # #Code to check computation of distance below
            # # d2 = ch.abs(c1[:,:,0]*c2[:,:,1] - c1[:,:,1]*c2[:,:,0]) / ch.sqrt((ch.sum(n**2,2)))
            # # # np_mat = ch.dot(ch.array([[0,-1],[1,0]]), n)
            # # np_mat2 = -ch.concatenate([-n[:,:,1][:,:,None], n[:,:,0][:,:,None]],2)
            # # np_vec2 = np_mat2 / ch.sqrt((ch.sum(np_mat2**2,2)))[:,:,None]
            # # d2 =  d2 / ch.maximum(ch.abs(np_vec2[:,:,0]),ch.abs(np_vec2[:,:,1]))
            #
            # chl = (chEdgeVerts2 - chEdgeVerts1)
            # chlinedist = ch.sqrt((ch.sum(chl**2,axis=1)))[:,None]
            # chlnorm = chl/chlinedist
            #
            # chv1 = chSampleVerts - chEdgeVerts1
            # chd = chv1[:,0]* chlnorm[:,0] + chv1[:,1]* chlnorm[:,1]
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            # # intersectPointDist1 = intersectPoint - chEdgeVerts1
            # # intersectPointDist2 = intersectPoint - chEdgeVerts2
            # # Code to check computation of distances below:
            # # lengthIntersectToPoint1 = np.linalg.norm(intersectPointDist1.r,axis=1)
            # # lengthIntersectToPoint2 = np.linalg.norm(intersectPointDist2.r,axis=1)
            #
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            #
            # chlineToPoint = (chSampleVerts - chintersectPoint)
            # chn_norm = chlineToPoint / ch.sqrt((ch.sum(chlineToPoint ** 2, axis=1)))[:, None]
            #
            # chdist = chlineToPoint[:,0]*chn_norm[:,0] + chlineToPoint[:,1]*chn_norm[:,1]
            #
            # d_final_ch = chdist / ch.maximum(ch.abs(chn_norm[:, 0]), ch.abs(chn_norm[:, 1]))
            #
            # d_final_outside = d_final_ch.ravel()
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts1)
            # rows = np.tile(np.arange(d_final_outside.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_outside.shape[0] * 2)
            #
            # dwdv_r_v1 = np.array(dwdv[rows, cols]).reshape([-1, 2])
            #
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts2)
            # rows = np.tile(np.arange(d_final_ch.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_ch.shape[0] * 2)
            #
            # dwdv_r_v2 = np.array(dwdv[rows, cols]).reshape([-1, 2])

            nonIntersect = self.nonIntersect
            argminDistNonIntersect = self.argminDistNonIntersect

            max_dx_dy = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))
            # d_final_np = dist / max_dx_dy
            d_final_np = dist

            ident = np.identity(2)
            ident = np.tile(ident[None, :], [len(p2), 1, 1])

            dlnorm = (ident - np.einsum('ij,ik->ijk', lnorm, lnorm)) / linedist[:,  None]
            dl_normdp1 = np.einsum('ijk,ikl->ijl', dlnorm, -ident)
            dl_normdp2 = np.einsum('ijk,ikl->ijl', dlnorm, ident)

            dv1dp1 = -ident
            dv1dp2 = 0

            dddp1 = np.einsum('ijk,ij->ik', dv1dp1, lnorm) + np.einsum('ij,ijl->il', v1, dl_normdp1)
            dddp2 = 0 + np.einsum('ij,ijl->il', v1, dl_normdp2)

            dipdp1 = ident + (dddp1[:,None,:]*lnorm[:,:,None]) + d[:,None,None]*dl_normdp1
            dipdp2 = (dddp2[:,None,:]*lnorm[:,:,None]) + d[:,None,None]*dl_normdp2

            dndp1 = -dipdp1
            dndp2 = -dipdp2

            dn_norm = (ident - np.einsum('ij,ik->ijk', n_norm, n_norm)) / dist[:,None]

            dn_normdp1 = np.einsum('ijk,ikl->ijl', dn_norm, dndp1)
            dn_normdp2 = np.einsum('ijk,ikl->ijl', dn_norm, dndp2)

            ddistdp1 = np.einsum('ij,ijl->il', n_norm, dndp1)
            ddistdp2 = np.einsum('ij,ijl->il', n_norm, dndp2)

            argmax_nx_ny = np.argmax(np.abs(n_norm),axis=1)
            dmax_nx_ny_p1 = np.sign(n_norm)[np.arange(len(n_norm)),argmax_nx_ny][:,None]*dn_normdp1[np.arange(len(dn_normdp1)),argmax_nx_ny]
            dmax_nx_ny_p2 = np.sign(n_norm)[np.arange(len(n_norm)),argmax_nx_ny][:,None]*dn_normdp2[np.arange(len(dn_normdp2)),argmax_nx_ny]

            # dd_final_dp1 = -1./max_dx_dy[:,None]**2 * dmax_nx_ny_p1 * dist + 1./max_dx_dy[:,None] *  ddistdp1
            # dd_final_dp2 = -1./max_dx_dy[:,None]**2 * dmax_nx_ny_p2 * dist + 1./max_dx_dy[:,None] *  ddistdp2

            dd_final_dp1 = ddistdp1
            dd_final_dp2 = ddistdp2

            #For those non intersecting points straight to the edge:

            v1 = self.v1[nonIntersect][argminDistNonIntersect==0]
            v1_norm = v1/np.sqrt((np.sum(v1**2,axis=1)))[:,None]

            dd_final_dp1_nonintersect = -v1_norm

            v2 = self.v2[nonIntersect][argminDistNonIntersect==1]
            v2_norm = v2/np.sqrt((np.sum(v2**2,axis=1)))[:,None]
            dd_final_dp2_nonintersect = -v2_norm

            dd_final_dp1[nonIntersect][argminDistNonIntersect == 0] = dd_final_dp1_nonintersect
            dd_final_dp1[nonIntersect][argminDistNonIntersect == 1] = 0
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 1] = dd_final_dp2_nonintersect
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 0] = 0

            dImage_wrt_outside_v1 = finalColorBndOutside_for_dr[facesOutsideBnd][:,:,None]*dd_final_dp1[:,None,:] - dd_final_dp1[:,None,:]*finalColorBndOutside_edge_for_dr[facesOutsideBnd][:,:,None]
            dImage_wrt_outside_v2 =  finalColorBndOutside_for_dr[facesOutsideBnd][:,:,None]*dd_final_dp2[:,None,:] - dd_final_dp2[:,None,:]*finalColorBndOutside_edge_for_dr[facesOutsideBnd][:,:,None]

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2*2)).ravel()
            # faces = f[sampleFaces[facesOutsideBnd]].ravel()
            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1, -1, 2]), [self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data1 = dImage_wrt_outside_v1.transpose([1,0,2])
            data2 = dImage_wrt_outside_v2.transpose([1,0,2])

            data = np.concatenate([data1[:,:,None,:], data2[:,:,None,:]], 2)

            data = data.ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd_outside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ######## 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            ######## 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample

            verticesBnd = self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3])

            sampleBarycentricBar = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([-1, 3, 1])
            verts = np.sum(self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3, 3]) * sampleBarycentricBar, axis=1)

            dImage_wrt_bar_v = self.barycentricDerivatives(verticesBnd, f[sampleFaces.ravel()], verts).swapaxes(0,1)

            dImage_wrt_bar_v[facesOutsideBnd.ravel()] = dImage_wrt_bar_v[facesOutsideBnd.ravel()] * d_final_outside[:,None,None, None] * self.t_area_bnd_outside[:, None, None, None]
            dImage_wrt_bar_v[facesInsideBnd.ravel()] = dImage_wrt_bar_v[facesInsideBnd.ravel()] * self.t_area_bnd_inside[:, None, None, None]

            # dImage_wrt_bar_v /= np.tile(areaWeightsTotal[None,:], [self.nsamples,1]).ravel()[:, None,None, None]
            dImage_wrt_bar_v /= self.nsamples

            ### Derivatives wrt V: 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
            faces = f[sampleFaces[facesOutsideBnd]].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            # data = np.tile(dImage_wrt_bar_v[facesOutsideBnd.ravel()][None,:],[3,1,1,1]).ravel()
            data = np.transpose(dImage_wrt_bar_v[facesOutsideBnd.ravel()],[1,0,2,3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_outside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ### Derivatives wrt V: 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesInsideBnd]
            IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
            faces = f[sampleFaces[facesInsideBnd]].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v[facesInsideBnd.ravel()], [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_inside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ####### 3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)

            frontFacing = self.frontFacingEdgeFaces[(zerosIm * boundaryImage).ravel().astype(np.bool)].astype(np.bool)
            frontFacingEdgeFaces = self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][frontFacing]

            verticesBnd = self.v.r[f[frontFacingEdgeFaces.ravel()].ravel()].reshape([1, -1, 3])
            verticesBnd = np.tile(verticesBnd, [self.nsamples, 1,1])
            verticesBnd = verticesBnd.reshape([-1,3,3])[facesOutsideBnd.ravel()].reshape([-1,3])

            verts = self.viewPointIntersect

            fFrontEdge = np.tile(f[frontFacingEdgeFaces][None,:], [self.nsamples, 1, 1]).reshape([-1,3])[facesOutsideBnd.ravel()]

            dImage_wrt_bar_v_edge = self.barycentricDerivatives(verticesBnd, fFrontEdge, verts).swapaxes(0, 1)

            dImage_wrt_bar_v_edge = dImage_wrt_bar_v_edge * (1-d_final_outside[:,None,None, None]) * self.t_area_bnd_edge[:, None, None, None]

            # dImage_wrt_bar_v_edge /= np.tile(self.areaWeightsTotal[None,:], [self.nsamples,1])[facesOutsideBnd][:, None, None,None]

            dImage_wrt_bar_v_edge /= self.nsamples

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 3 * 2)).ravel()
            # faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = f[frontFacingEdgeFaces]
            faces = np.tile(faces.reshape([1, -1, 3]), [self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()
            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v_edge, [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_outside_edge = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        vertsPerFaceProjBnd = self.camera.r[f[nonBoundaryFaces].ravel()].reshape([-1,3,2])
        nv = len(vertsPerFaceProjBnd)

        p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
        p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
        p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
        t_area_nonbnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)
        t_area_nonbnd[t_area_nonbnd> 1] = 1

        bc = barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))

        verts = np.sum(self.v.r[f[nonBoundaryFaces.ravel()].ravel()].reshape([-1, 3, 3]) * bc[:, :,None], axis=1)


        didp = self.barycentricDerivatives(verticesNonBnd, f[nonBoundaryFaces.ravel()], verts)

        didp = didp * t_area_nonbnd[None,:,None, None]

        n_channels = np.atleast_3d(observed).shape[2]
        shape = visibility.shape

        ####### 2: Take the data and copy the corresponding dxs and dys to these new pixels.

        ### Derivatives wrt V:
        # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
        pixels = np.where(((~boundaryImage)&(visibility !=4294967295 )).ravel())[0]
        IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
        JS = col(f[nonBoundaryFaces].ravel())
        JS = np.hstack((JS*2, JS*2+1)).ravel()

        if n_channels > 1:
            IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        # data = np.concatenate(((visTriVC[:,0,:] * dBar1dx[:,None])[:,:,None],(visTriVC[:, 0, :] * dBar1dy[:, None])[:,:,None], (visTriVC[:,1,:]* dBar2dx[:,None])[:,:,None], (visTriVC[:, 1, :] * dBar2dy[:, None])[:,:,None],(visTriVC[:,2,:]* dBar3dx[:,None])[:,:,None],(visTriVC[:, 2, :] * dBar3dy[:, None])[:,:,None]),axis=2).swapaxes(0,1).ravel()
        data = didp.ravel()

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result_wrt_verts_nonbnd = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))
        # result_wrt_verts_nonbnd.sum_duplicates()

        if np.any(boundaryImage):

            result_wrt_verts =  result_wrt_verts_bnd_outside + result_wrt_verts_bar_outside + result_wrt_verts_bar_inside + result_wrt_verts_bar_outside_edge + result_wrt_verts_nonbnd
            # result_wrt_verts = result_wrt_verts_bnd_outside

        else:
            result_wrt_verts = result_wrt_verts_nonbnd

        return result_wrt_verts


    def compute_derivatives_vc(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        d_final_outside = self.d_final_outside

        barycentricVertsIntersect = self.barycentricVertsIntersect


        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility!=4294967295)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage)&(visibility !=4294967295 )]

        if np.any(boundaryImage):

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            #Computing gradients:
            #A multisampled pixel color is given by: w R + (1-w) R' thus:
            #1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            #2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            #3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            #4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            #5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            #6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            #7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            #for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            ####### 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            dImage_wrt_outside_vc_outside = d_final_outside[:,None] * sampleBarycentric[facesOutsideBnd] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 3)).ravel()

            faces = f[sampleFaces[facesOutsideBnd]].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_outside.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

            result_wrt_vc_bnd_outside = result
            # result_wrt_vc_bnd_outside.sum_duplicates()

            ######## 5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge
            dImage_wrt_outside_vc_edge = (1-d_final_outside[:, None]) * np.c_[barycentricVertsIntersect, 1-barycentricVertsIntersect] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2)).ravel()
            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1,-1,2]),[self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_edge.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result_wrt_vc_bnd_outside_edge = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))
            # result_wrt_vc_bnd_outside_edge.sum_duplicates()

            ######## 7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample
            dImage_wrt_outside_vc_inside = sampleBarycentric[facesInsideBnd] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesInsideBnd]
            IS = np.tile(col(pixels), (1, 3)).ravel()
            faces = f[sampleFaces[facesInsideBnd]].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_inside.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result_wrt_vc_bnd_inside = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))
            # result_wrt_vc_bnd_inside.sum_duplicates()


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        # barySample = self.renders_sample_barycentric[0].reshape([-1,3])[(~boundaryImage)&(visibility !=4294967295 ).ravel().astype(np.bool), :]

        bc = barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))
        # barySample[barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))]

        ### Derivatives wrt VC:

        # Each pixel relies on three verts
        pixels = np.where(((~boundaryImage)&(visibility !=4294967295 )).ravel())[0]
        IS = np.tile(col(pixels), (1, 3)).ravel()
        JS = col(f[nonBoundaryFaces].ravel())

        bc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
        # bc = barySample.reshape((-1, 3))

        data = np.asarray(bc, order='C').ravel()

        IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
        JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
        data = np.concatenate([data for i in range(num_channels)])
        # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        # data = np.concatenate((data, data, data))

        ij = np.vstack((IS.ravel(), JS.ravel()))
        result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

        result_wrt_vc_nonbnd = result
        # result_wrt_vc_nonbnd.sum_duplicates()

        if np.any(boundaryImage):
            # result_wrt_verts = result_wrt_verts_bar_outside_edge

            # result_wrt_verts = result_wrt_verts_nonbnd
            result_wrt_vc = result_wrt_vc_bnd_outside + result_wrt_vc_bnd_outside_edge + result_wrt_vc_bnd_inside + result_wrt_vc_nonbnd
            # result_wrt_vc = sp.csc_matrix((width * height * num_channels, vc_size))
        else:
            # result_wrt_verts = sp.csc_matrix((image_width*image_height*n_channels, num_verts*2))
            result_wrt_vc = result_wrt_vc_nonbnd
            # result_wrt_vc = sp.csc_matrix((width * height * num_channels, vc_size))
        return result_wrt_vc


    def on_changed(self, which):
        super().on_changed(which)

        if 'v' or 'camera' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_verts_mesh[mesh][polygons].set_array(verts_by_face.astype(np.float32))
                    self.vbo_verts_mesh[mesh][polygons].bind()

        if 'vc' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                    self.vbo_colors_mesh[mesh][polygons].bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()
            flen = 1
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]

                    # fc = np.arange(flen, flen + len(f))
                    fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()

                    # fc[:, 0] = fc[:, 0] & 255
                    # fc[:, 1] = (fc[:, 1] >> 8) & 255
                    # fc[:, 2] = (fc[:, 2] >> 16) & 255
                    fc = np.asarray(fc, dtype=np.uint32)
                    self.vbo_face_ids_list[mesh][polygons].set_array(fc)
                    self.vbo_face_ids_list[mesh][polygons].bind()

                    flen += len(f)

                    self.vbo_indices_mesh_list[mesh][polygons].set_array(np.array(self.f_list[mesh][polygons]).astype(np.uint32))
                    self.vbo_indices_mesh_list[mesh][polygons].bind()

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
                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size+textureCoordIdx].reshape(image.shape)

                            textureCoordIdx = textureCoordIdx + image.size
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)

                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                                               image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())



        if 'v' or 'f' or 'vc' or 'ft' or 'camera' or 'texture_stack' in which:
            self.render_image_buffers()


    def release_textures(self):
        if hasattr(self, 'textureID_mesh_list'):
            if self.textureID_mesh_list != []:
                for texture_mesh in self.textureID_mesh_list:
                    if texture_mesh != []:
                        for texture in texture_mesh:
                            if texture != None:
                                GL.glDeleteTextures(1, [texture.value])

        self.textureID_mesh_list = []

    @depends_on(dterms+terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        return no_overdraw

        # if not self.overdraw:
        #     return no_overdraw
        #
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        # overdraw = self.draw_color_image()
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #
        # # return overdraw * np.atleast_3d(self.boundarybool_image)
        #
        # boundarybool_image = self.boundarybool_image
        # if self.num_channels > 1:
        #     boundarybool_image = np.atleast_3d(boundarybool_image)
        #
        # return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')


    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_barycentric_image(self.boundarybool_image if self.overdraw else None)

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()
        #Overload method to call without overdraw.
        return self.draw_visibility_image(self.v.r, self.f, self.boundarybool_image if self.overdraw else None)

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

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        vc = self.vc_list[mesh]

        for polygons in np.arange(len(self.f_list[mesh])):
            vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
            GL.glBindVertexArray(vao_mesh)
            f = self.f_list[mesh][polygons]
            vbo_color = self.vbo_colors_mesh[mesh][polygons]
            colors_by_face = np.asarray(vc.reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
            colors = np.array(np.ones_like(colors_by_face) * (index) / 255.0, dtype=np.float32)

            # Pol: Make a static zero vbo_color to make it more efficient?
            vbo_color.set_array(colors)

            vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

            vbo_color.bind()

            if self.f.shape[1]==2:
                primtype = GL.GL_LINES
            else:
                primtype = GL.GL_TRIANGLES

            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

            GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])


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

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_color = self.vbo_colors_mesh[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vc = colors_by_face

                if with_vertex_colors:
                    colors = vc.astype(np.float32)
                else:
                    # Only texture.
                    colors = np.ones_like(vc).astype(np.float32)

                # Pol: Make a static zero vbo_color to make it more efficient?
                vbo_color.set_array(colors)
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

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])
                # GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

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


class AnalyticRendererOpenDR(ColoredRenderer):

    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list' , 'imageGT'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    def __init__(self):
        super().__init__()


    def clear(self):
        try:
            GL.glFlush()
            GL.glFinish()
            # print ("Clearing textured renderer.")
            # for msh in self.vbo_indices_mesh_list:
            #     for vbo in msh:
            #         vbo.set_array([])
            [vbo.set_array(np.array([])) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_colors_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_verts_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_uvs_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_face_ids_list for vbo in sublist]

            [GL.glDeleteVertexArrays(1, [vao.value]) for sublist in self.vao_tex_mesh_list for vao in sublist]

            self.release_textures()

            if self.glMode == 'glfw':
                import glfw
                glfw.make_context_current(self.win)

            GL.glDeleteProgram(self.colorTextureProgram)

            super().clear()
        except:
            import pdb
            pdb.set_trace()
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
            color = theColor * texture( myTextureSampler, UV).rgb;
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

        # GL.glEnable(GL.GL_LINE_SMOOTH)
        # GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(2.)

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_indices_mesh = []
            vbo_face_ids_mesh = []
            vbo_colors_mesh = []
            vbo_vertices_mesh = []
            vbo_uvs_mesh = []
            textureIDs_mesh = []
            for polygons in range(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_verts = vbo.VBO(np.array(verts_by_face).astype(np.float32))
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_colors = vbo.VBO(np.array(colors_by_face).astype(np.float32))
                uvs_by_face = np.asarray(self.ft_list[mesh].reshape((-1, 2))[f.ravel()], dtype=np.float32, order='C')
                vbo_uvs = vbo.VBO(np.array(uvs_by_face).astype(np.float32))

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
                vbo_colors_mesh = vbo_colors_mesh + [vbo_colors]
                vbo_vertices_mesh = vbo_vertices_mesh + [vbo_verts]
                vbo_uvs_mesh = vbo_uvs_mesh + [vbo_uvs]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

            self.vbo_colors_mesh = self.vbo_colors_mesh + [vbo_colors_mesh]
            self.vbo_verts_mesh = self.vbo_verts_mesh + [vbo_vertices_mesh]
            self.vbo_uvs_mesh = self.vbo_uvs_mesh + [vbo_uvs_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID  = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")


    def initGL_AnalyticRenderer(self):
        self.initGLTexture()

        self.updateRender = True
        self.updateDerivatives = True

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 colorIn;
        layout(location = 2) in vec2 vertexUV;
        layout(location = 3) in uint face_id;
        layout(location = 4) in vec3 barycentric;

        uniform mat4 MVP;
        out vec3 theColor;
        out vec4 pos;
        flat out uint face_out;
        out vec3 barycentric_vert_out;
        out vec2 UV;
        
        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            pos =  MVP * vec4(position,1);
            //pos =  pos4.xyz;
            theColor = colorIn;
            UV = vertexUV;
            face_out = face_id;
            barycentric_vert_out = barycentric;
            
        }""", GL.GL_VERTEX_SHADER)

        ERRORS_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 

            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
            in vec3 theColor;
            in vec2 UV;
            flat in uint face_out;
            in vec4 pos;
            in vec3 barycentric_vert_out;
                        
            layout(location = 3) uniform sampler2D myTextureSampler;

            uniform float ww;
            uniform float wh;
    
            // Ouput data
            layout(location = 0) out vec3 color; 
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 barycentric1;
            layout(location = 4) out vec2 barycentric2;
            
            void main(){
                vec3 finalColor = theColor * texture( myTextureSampler, UV).rgb;
                color = finalColor.rgb;
                                
                sample_pos = ((0.5*pos.xy/pos.w) + 0.5)*vec2(ww,wh);
                sample_face = face_out;
                barycentric1 = barycentric_vert_out.xy;
                barycentric2 = vec2(barycentric_vert_out.z, 0.);
                
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
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable
  
            layout(location = 2) uniform sampler2DMS colors;
            layout(location = 3) uniform sampler2DMS sample_positions;
            layout(location = 4) uniform usampler2DMS sample_faces;
            layout(location = 5) uniform sampler2DMS sample_barycentric_coords1;
            layout(location = 6) uniform sampler2DMS sample_barycentric_coords2;

            uniform float ww;
            uniform float wh;
            uniform int sample;

            // Ouput data
            layout(location = 0) out vec3 colorFetchOut;
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 sample_barycentric1;
            layout(location = 4) out vec2 sample_barycentric2;

            //out int gl_SampleMask[];
            const int all_sample_mask = 0xffff;

            void main(){
                ivec2 texcoord = ivec2(gl_FragCoord.xy);
                colorFetchOut = texelFetch(colors, texcoord, sample).xyz;
                sample_pos = texelFetch(sample_positions, texcoord, sample).xy;        
                sample_face = texelFetch(sample_faces, texcoord, sample).r;
                sample_barycentric1 = texelFetch(sample_barycentric_coords1, texcoord, sample).xy;
                sample_barycentric2 = texelFetch(sample_barycentric_coords2, texcoord, sample).xy;


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

        # GL.glGenTextures(1, self.textureEdges)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureEdges)
        # GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)

        # GL.glActiveTexture(GL.GL_TEXTURE0)

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

        self.texture_errors_sample_faces = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_R32UI, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces, 0)
        #

        self.texture_errors_sample_barycentric1 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1, 0)

        self.texture_errors_sample_barycentric2 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2, 0)


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

        self.render_buffer_fetch_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        #
        self.render_buffer_fetch_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)

        self.render_buffer_fetch_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)

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

        render_buf_errors_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_sample_position)

        render_buf_errors_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        #

        render_buf_errors_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)

        render_buf_errors_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
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
        face_ids_location = GL.glGetAttribLocation(self.errorTextureProgram, 'face_id')
        barycentric_location = GL.glGetAttribLocation(self.errorTextureProgram, 'barycentric')

        # self.vbo_verts_cube= vbo.VBO(np.array(self.v_bgCube).astype(np.float32))
        # self.vbo_colors_cube= vbo.VBO(np.array(self.vc_bgCube).astype(np.float32))
        # self.vbo_uvs_cube = vbo.VBO(np.array(self.ft_bgCube).astype(np.float32))
        # self.vao_bgCube = GL.GLuint(0)
        # GL.glGenVertexArrays(1, self.vao_bgCube)
        #
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube = vbo.VBO(np.array(self.f_bgCube).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        # self.vbo_f_bgCube.bind()
        # self.vbo_verts_cube.bind()
        # GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_colors_cube.bind()
        # GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_uvs_cube.bind()
        # GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        #
        # f = self.f_bgCube
        # fc = np.tile(np.arange(len(self.f), len(self.f) + len(f))[:, None], [1, 3]).ravel()
        # # fc[:, 0] = fc[:, 0] & 255
        # # fc[:, 1] = (fc[:, 1] >> 8) & 255
        # # fc[:, 2] = (fc[:, 2] >> 16) & 255
        # fc = np.asarray(fc, dtype=np.uint32)
        # vbo_face_ids_cube = vbo.VBO(fc)
        # vbo_face_ids_cube.bind()
        # GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
        # GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)
        #
        # #Barycentric cube:
        # f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
        # vbo_barycentric_cube = vbo.VBO(f_barycentric)
        # vbo_barycentric_cube.bind()
        # GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
        # GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_quad = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_quad)
        GL.glBindVertexArray(self.vao_quad)


        #Bind VAO

        self.vbo_face_ids_list = []
        self.vbo_barycentric_list = []
        self.vao_errors_mesh_list = []
        flen = 1

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_face_ids_mesh = []
            vbo_barycentric_mesh = []
            for polygons in np.arange(len(self.f_list[mesh])):

                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]
                vbo_f.bind()
                vbo_verts = self.vbo_verts_mesh[mesh][polygons]
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_colors = self.vbo_colors_mesh[mesh][polygons]
                vbo_colors.bind()

                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_uvs = self.vbo_uvs_mesh[mesh][polygons]
                vbo_uvs.bind()
                GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                f = self.f_list[mesh][polygons]

                fc = np.tile(np.arange(flen, flen + len(f))[:,None], [1,3]).ravel()
                # fc[:, 0] = fc[:, 0] & 255
                # fc[:, 1] = (fc[:, 1] >> 8) & 255
                # fc[:, 2] = (fc[:, 2] >> 16) & 255
                fc = np.asarray(fc, dtype=np.uint32)
                vbo_face_ids = vbo.VBO(fc)
                vbo_face_ids.bind()
                GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
                GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)

                f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
                vbo_barycentric = vbo.VBO(f_barycentric)
                vbo_barycentric.bind()
                GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                flen += len(f)

                vaos_mesh += [vao]

                vbo_face_ids_mesh += [vbo_face_ids]
                vbo_barycentric_mesh += [vbo_face_ids]

                GL.glBindVertexArray(0)

            self.vbo_face_ids_list += [vbo_face_ids_mesh]
            self.vbo_barycentric_list += [vbo_barycentric_mesh]
            self.vao_errors_mesh_list += [vaos_mesh]

    def render_image_buffers(self):

        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        self.makeCurrentContext()

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1%self.num_channels], self.bgcolor.r[2%self.num_channels], 1.)

        GL.glUseProgram(self.errorTextureProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        # GL.glClearBufferiv(GL.GL_COLOR​, 0​, 0)
        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        wwLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_errors_mesh_list[mesh][polygons]

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                # vbo_color.bind()

                f = self.f_list[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                self.vbo_colors_mesh[mesh][polygons].bind()

                if self.f.shape[1]==2:
                    primtype = GL.GL_LINES
                else:
                    primtype = GL.GL_TRIANGLES

                assert(primtype == GL.GL_TRIANGLES)

                # GL.glUseProgram(self.errorTextureProgram)
                if self.haveUVs_list[mesh][polygons]:
                    texture =  self.textureID_mesh_list[mesh][polygons]
                else:
                    texture = self.whitePixelTextureID

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glUniform1i(self.textureObjLoc, 0)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawArrays(primtype, 0, len(vbo_f)*vbo_f.data.shape[1])

        # # #Background cube:
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube.bind()
        # texture = self.whitePixelTextureID
        # self.vbo_uvs_cube.bind()
        #
        # GL.glActiveTexture(GL.GL_TEXTURE0)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # GL.glUniform1i(self.textureObjLoc, 0)
        # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
        #
        # GL.glDrawElements(primtype, len(self.vbo_f_bgCube)*self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

        # self.draw_visibility_image_ms(self.v, self.f)

        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # # result_blit = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        # result_blit2 = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # result_blit_pos = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

        GL.glUseProgram(self.fetchSamplesProgram)
        # GL.glDisable(GL.GL_MULTISAMPLE)

        self.colorsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "colors")
        self.sample_positionsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_positions")
        self.sample_facesLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_faces")
        self.sample_barycentric1Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords1")
        self.sample_barycentric2Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords2")

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
        self.renders_faces = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height']]).astype(np.uint32)
        self.renders_sample_barycentric1 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_sample_barycentric2 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],1])
        self.renders_sample_barycentric = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'],3])

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_sample_fetch)
        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3,
                          GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for sample in np.arange(self.nsamples):

            sampleLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'sample')
            GL.glUniform1i(sampleLoc, sample)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
            GL.glUniform1i(self.colorsLoc, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
            GL.glUniform1i(self.sample_positionsLoc, 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
            GL.glUniform1i(self.sample_facesLoc, 2)

            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
            GL.glUniform1i(self.sample_barycentric1Loc, 3)

            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
            GL.glUniform1i(self.sample_barycentric2Loc, 4)

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

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT), np.uint32).reshape(self.frustum['height'], self.frustum['height'])[:,:].astype(np.uint32))
            self.renders_faces[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:2].astype(np.float64))
            self.renders_sample_barycentric1[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
            result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:1].astype(np.float64))
            self.renders_sample_barycentric2[sample] = result

            self.renders_sample_barycentric[sample] = np.concatenate([self.renders_sample_barycentric1[sample], self.renders_sample_barycentric2[sample][:,:,0:1]], 2)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            # result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
            # self.renders_faces[sample] = result


        GL.glBindVertexArray(0)

        GL.glClearColor(0.,0.,0., 1.)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)

        ##Finally return image and derivatives

        self.render_resolved = np.mean(self.renders, 0)

        self.updateRender = True
        self.updateDerivatives_verts = True
        self.updateDerivatives_vc = True


    def draw_visibility_image_ms(self, v, f):
        """Assumes camera is set up correctly in"""
        GL.glUseProgram(self.visibilityProgram_ms)

        v = np.asarray(v)

        self.draw_visibility_image_ms(v, f)

        #Attach FBO
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        fc = np.arange(1, len(f)+1)
        fc = np.tile(fc.reshape((-1,1)), (1, 3))
        fc[:, 0] = fc[:, 0] & 255
        fc[:, 1] = (fc[:, 1] >> 8 ) & 255
        fc[:, 2] = (fc[:, 2] >> 16 ) & 255
        fc = np.asarray(fc, dtype=np.uint8)

        self.draw_colored_primitives_ms(self.vao_dyn_ub,  v, f, fc)


    # this assumes that fc is either "by faces" or "verts by face", not "by verts"
    def draw_colored_primitives_ms(self, vao, v, f, fc=None):

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

            vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
            self.vbo_colors_ub.set_array(vc_by_face)
            self.vbo_colors_ub.bind()

        primtype = GL.GL_TRIANGLES

        self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
        self.vbo_indices_dyn.bind()

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(1, drawingBuffers)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)
        GL.glEnable(GL.GL_DEPTH_TEST)


    def compute_dr_wrt(self, wrt):

        visibility = self.visibility_image

        if wrt is self.camera:
            derivatives_verts = self.get_derivatives_verts()

            return derivatives_verts

        elif wrt is self.vc:

            derivatives_vc = self.get_derivatives_vc()

            return derivatives_vc

        # Not working atm.:
        elif wrt is self.bgcolor:
            return 2. * (self.imageGT.r - self.render_image).ravel() * common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

        #Not working atm.:
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

    def compute_r(self):
        return self.render()

    @depends_on(dterms+terms)
    def renderWithoutColor(self):
        self._call_on_changed()

        return self.render_nocolor

    @depends_on(dterms+terms)
    def renderWithoutTexture(self):
        self._call_on_changed()

        return self.render_notexture

    # @depends_on(dterms+terms)
    def render(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        barycentric = self.barycentric_image

        if self.updateRender:
            render = self.compute_image(visible, visibility, self.f)
            self.render_result = render
            self.updateRender = False
        return self.render_result

    def get_derivatives_verts(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_verts:
            if self.updateRender:
                self.render()
            if self.overdraw:
                # return common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
                derivatives_verts = common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)

            else:
                derivatives_verts = common.dImage_wrt_2dVerts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
            self.derivatives_verts = derivatives_verts
            self.updateDerivatives_verts = False
        return self.derivatives_verts

    def get_derivatives_vc(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_vc:
            if self.updateRender:
                self.render()
            derivatives_vc = self.compute_derivatives_vc(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
            self.derivatives_vc = derivatives_vc
            self.updateDerivatives_vc = False
        return self.derivatives_vc

    # # @depends_on(dterms+terms)
    # def image_and_derivatives(self):
    #     # self._call_on_changed()
    #     visibility = self.visibility_image
    #
    #     color = self.render_resolved
    #
    #     visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    #     num_visible = len(visible)
    #
    #     barycentric = self.barycentric_image
    #
    #     if self.updateRender:
    #         render, derivatives = self.compute_image_and_derivatives(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
    #         self.render = render
    #         self.derivatives = derivatives
    #         self.updateRender = False
    #
    #     return self.render, self.derivatives
    #

    def barycentricDerivatives(self, vertices, faces, verts):
        import chumpy as ch

        vertices = np.concatenate([vertices, np.ones([vertices.size // 3, 1])], axis=1)
        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])

        viewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])

        # # Check with autodiff:
        #
        # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        # # negYMat = ch.array([[1,0,self.camera.c.r[0]],[0,-1,self.camera.c.r[1]],[0,0,1]])
        # verts_hom_ch = ch.Ch(verts_hom)
        # camMtx = ch.Ch(np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])])
        # projVerts = (camMtx.dot(view_mtx)).dot(verts_hom_ch.T).T[:, :3].reshape([-1, 3])
        # viewVerts = ch.Ch(np.array(projVerts))
        # projVerts = projVerts[:, :2] / projVerts[:, 2:3]
        #
        # chViewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        # p0 = ch.Ch(viewVerticesNonBnd[:, 0, :])
        # chp0 = p0
        #
        # p1 = ch.Ch(viewVerticesNonBnd[:, 1, :])
        # chp1 = p1
        #
        # p2 = ch.Ch(viewVerticesNonBnd[:, 2, :])
        # chp2 = p2
        #
        # # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        # nt = ch.cross(p1 - p0, p2 - p0)
        # chnt = nt
        # A = 0.5 * ch.sqrt(ch.sum(nt ** 2, axis=1))
        # chnt_norm = nt / ch.sqrt(ch.sum(nt ** 2, axis=1))[:, None]
        # # nt = nt / A
        #
        # chb0part2 = ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1)
        # chb0 = 0.5 * ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1) / A
        # chb1part2 = ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1)
        # chb1 = 0.5 * ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1) / A
        # chb2part2 = ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1)
        # chb2 = 0.5 * ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1) / A
        #
        # drb0p0 = chb0.dr_wrt(p0)
        # drb0p1 = chb0.dr_wrt(p1)
        # drb0p2 = chb0.dr_wrt(p2)
        #
        # drb1p0 = chb1.dr_wrt(p0)
        # drb1p1 = chb1.dr_wrt(p1)
        # drb1p2 = chb1.dr_wrt(p2)
        #
        # drb2p0 = chb2.dr_wrt(p0)
        # drb2p1 = chb2.dr_wrt(p1)
        # drb2p2 = chb2.dr_wrt(p2)
        #
        # rows = np.tile(np.arange(drb0p0.shape[0])[None, :], [3, 1]).T.ravel()
        # cols = np.arange(drb0p0.shape[0] * 3)
        #
        # drb0p0 = np.array(drb0p0[rows, cols]).reshape([-1, 3])
        # drb0p1 = np.array(drb0p1[rows, cols]).reshape([-1, 3])
        # drb0p2 = np.array(drb0p2[rows, cols]).reshape([-1, 3])
        # drb1p0 = np.array(drb1p0[rows, cols]).reshape([-1, 3])
        # drb1p1 = np.array(drb1p1[rows, cols]).reshape([-1, 3])
        # drb1p2 = np.array(drb1p2[rows, cols]).reshape([-1, 3])
        # drb2p0 = np.array(drb2p0[rows, cols]).reshape([-1, 3])
        # drb2p1 = np.array(drb2p1[rows, cols]).reshape([-1, 3])
        # drb2p2 = np.array(drb2p2[rows, cols]).reshape([-1, 3])
        #
        # chdp0 = np.concatenate([drb0p0[:, None, :], drb1p0[:, None, :], drb2p0[:, None, :]], axis=1)
        # chdp1 = np.concatenate([drb0p1[:, None, :], drb1p1[:, None, :], drb2p1[:, None, :]], axis=1)
        # chdp2 = np.concatenate([drb0p2[:, None, :], drb1p2[:, None, :], drb2p2[:, None, :]], axis=1)
        # #
        # # dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)
        # # dp = dp[None, :]

        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])
        viewVerts = projVerts
        projVerts = projVerts[:, :2] / projVerts[:, 2:3]

        # viewVerticesNonBnd = negYMat.dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        p0 = viewVerticesNonBnd[:, 0, :]
        p1 = viewVerticesNonBnd[:, 1, :]
        p2 = viewVerticesNonBnd[:, 2, :]

        p0_proj = p0[:,0:2]/p0[:,2:3]
        p1_proj = p1[:,0:2]/p1[:,2:3]
        p2_proj = p2[:,0:2]/p2[:,2:3]

        # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        nt = np.cross(p1 - p0, p2 - p0)
        nt_norm = nt / np.linalg.norm(nt, axis=1)[:, None]

        # a = -nt_norm[:, 0] / nt_norm[:, 2]
        # b = -nt_norm[:, 1] / nt_norm[:, 2]
        # c = np.sum(nt_norm * p0, 1) / nt_norm[:, 2]

        cam_f = 1

        u = p0[:, 0]/p0[:, 2]
        v = p0[:, 1]/p0[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p0[:, 2][:,None], np.zeros([len(p0),1]), (-p0[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p0),1]), p0[:, 2][:,None], (-p0[:,1]/v**2)[:,None]]

        dxdp_0 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p1[:, 0]/p1[:, 2]
        v = p1[:, 1]/p1[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p1[:, 2][:,None], np.zeros([len(p1),1]), (-p1[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p1),1]), p1[:, 2][:,None], (-p1[:,1]/v**2)[:,None]]

        dxdp_1 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p2[:, 0]/p2[:, 2]
        v = p2[:, 1]/p2[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p2[:, 2][:,None], np.zeros([len(p2),1]), (-p2[:,0]/u**2)[:,None]]
        xv = np.c_[np.zeros([len(p2),1]), p2[:, 2][:,None], (-p2[:,1]/v**2)[:,None]]

        dxdp_2 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        # x = u * c / (cam_f - a * u - b * v)
        # y = v*c/(cam_f - a*u - b*v)
        # z = c*cam_f/(cam_f - a*u - b*v)

        A = 0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0),axis=1)
        nt_mag = A*2
        # nt = nt / A
        # db1 = 0.5*np.cross(nt_norm, p2-p1)/A[:, None]
        # db2 = 0.5*np.cross(nt_norm, p0-p2)/A[:, None]
        # db3_2 = 0.5*np.cross(nt_norm, p1-p0)/A[:, None]
        # db3 = - db1 - db2
        p = viewVerts

        pre1 = -1/(nt_mag[:,None]**2) * nt_norm

        ident = np.identity(3)
        ident = np.tile(ident[None,:],[len(p2),1,1])

        dntdp0 = np.cross((p2-p0)[:,None,:], -ident) + np.cross(-ident, (p1-p0)[:,None,:])
        dntdp1 = np.cross((p2-p0)[:,None,:],ident)
        dntdp2 = np.cross(ident,(p1-p0)[:,None,:])

        #Pol check this!:
        dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]
        # dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]

        dntnormdp0 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp0)
        dntnormdp1 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp1)
        dntnormdp2 = np.einsum('ijk,ikl->ijl',dntnorm, dntdp2)

        dpart1p0 = np.einsum('ij,ijk->ik', pre1, dntdp0)
        dpart1p1 = np.einsum('ij,ijk->ik', pre1, dntdp1)
        dpart1p2 = np.einsum('ij,ijk->ik', pre1, dntdp2)

        b0 = np.sum(np.cross(nt_norm, p2 - p1) * (p - p1), axis=1)[:,None]

        db0part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident,nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1),-ident)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0.swapaxes(1,2)), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident, nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(p2-p1,nt_norm[:, :]),-ident)
        db0part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :],-ident), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1), -ident)
        db0part2p2 = np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1,2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p1)

        db0dp0wrtpart1 = dpart1p0*b0
        db0dp1wrtpart1 = dpart1p1*b0
        db0dp2wrtpart1 = dpart1p2*b0

        db0dp0wrtpart2 = 1./(nt_mag[:,None])*db0part2p0
        db0dp1wrtpart2 = 1./(nt_mag[:,None])*db0part2p1
        db0dp2wrtpart2 = 1./(nt_mag[:,None])*db0part2p2

        db0dp0wrt = db0dp0wrtpart1 +  db0dp0wrtpart2
        db0dp1wrt = db0dp1wrtpart1 +  db0dp1wrtpart2
        db0dp2wrt = db0dp2wrtpart1 +  db0dp2wrtpart2

        ######
        b1 = np.sum(np.cross(nt_norm, p0 - p2) * (p - p2), axis=1)[:, None]

        db1part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p2)
        db1part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2)
        db1part2p2 = np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1, 2),(p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], -ident), p - p2) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p0-p2), -ident)

        db1dp0wrtpart1 = dpart1p0*b1
        db1dp1wrtpart1 = dpart1p1*b1
        db1dp2wrtpart1 = dpart1p2*b1

        db1dp0wrtpart2 = 1./(nt_mag[:,None])*db1part2p0
        db1dp1wrtpart2 = 1./(nt_mag[:,None])*db1part2p1
        db1dp2wrtpart2 = 1./(nt_mag[:,None])*db1part2p2

        db1dp0wrt = db1dp0wrtpart1 + db1dp0wrtpart2
        db1dp1wrt = db1dp1wrtpart1 +  db1dp1wrtpart2
        db1dp2wrt = db1dp2wrtpart1 +  db1dp2wrtpart2

        ######
        b2 = np.sum(np.cross(nt_norm, p1 - p0) * (p - p0), axis=1)[:, None]

        db2part2p0 = np.einsum('ikj,ij->ik',np.cross(dntnormdp0.swapaxes(1, 2),(p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], -ident), p - p0) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p1 - p0), -ident)
        db2part2p1 = np.einsum('ikj,ij->ik',np.cross(dntnormdp1.swapaxes(1, 2),(p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(nt_norm[:, None, :], ident), p - p0)
        db2part2p2 =  np.einsum('ikj,ij->ik',np.cross(dntnormdp2.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0)

        db2dp0wrtpart1 = dpart1p0*b2
        db2dp1wrtpart1 = dpart1p1*b2
        db2dp2wrtpart1 = dpart1p2*b2

        db2dp0wrtpart2 = 1./(nt_mag[:,None])*db2part2p0
        db2dp1wrtpart2 = 1./(nt_mag[:,None])*db2part2p1
        db2dp2wrtpart2 = 1./(nt_mag[:,None])*db2part2p2

        db2dp0wrt = db2dp0wrtpart1 + db2dp0wrtpart2
        db2dp1wrt = db2dp1wrtpart1 +  db2dp1wrtpart2
        db2dp2wrt = db2dp2wrtpart1 +  db2dp2wrtpart2

        dp0 = np.concatenate([db0dp0wrt[:, None, :], db1dp0wrt[:, None, :], db2dp0wrt[:, None, :]], axis=1)
        dp1 = np.concatenate([db0dp1wrt[:, None, :], db1dp1wrt[:, None, :], db2dp1wrt[:, None, :]], axis=1)
        dp2 = np.concatenate([db0dp2wrt[:, None, :], db1dp2wrt[:, None, :], db2dp2wrt[:, None, :]], axis=1)
        #
        dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)

        #If dealing with degenerate triangles, ignore that gradient.

        # dp[nt_mag<=1e-15] = 0

        dp = dp[None, :]

        nFaces = len(faces)
        # visTriVC = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc[vc > 1] = 1
        vc[vc < 0] = 0

        visTriVC = vc

        dxdp = np.concatenate([dxdp_0[:,None,:],dxdp_1[:,None,:],dxdp_2[:,None,:]], axis=1)

        dxdp = dxdp[None, :, None]
        # dbvc = np.sum(dp * visTriVC, 2)

        # dbvc = dp * visTriVC * t_area[None, :, None, None, None]
        dbvc = dp * visTriVC

        didp = np.sum(dbvc[:, :, :, :, :, None] * dxdp, 4).sum(2)

        #output should be shape: VC x Ninput x Tri Points x UV

        # drb0p0 # db0dp0wrt
        # drb0p1 # db0dp1wrt
        # drb0p2 # db0dp2wrt
        # drb1p0 # db1dp0wrt
        # drb1p1 # db1dp1wrt
        # drb1p2 # db1dp2wrt
        # drb2p0 # db2dp0wrt
        # drb2p1 # db2dp1wrt
        # drb2p2 # db2dp2wrt
        #

        return didp

    def compute_image(self, visible, visibility, f):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""

        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        # xdiff = dEdx
        # ydiff = dEdy

        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        nsamples = self.nsamples


        if np.any(boundaryImage):
            boundaryFaces = visibility[(boundaryImage) & (visibility != 4294967295)]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

            edgeFaces= np.tile(self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][None, :, :], [8, 1, 1])

            edgeSampled = np.any((edgeFaces[:,:, 0]== sampleFaces) | (edgeFaces[:,:, 1]== sampleFaces),0)

            facesInsideBnd = projFacesBndTiled == sampleFaces
            wrongBnd = ~edgeSampled
            # wrongBnd = np.all(facesInsideBnd, 0)
            whereBnd = np.where(boundaryImage.ravel())[0]
            # boundaryImage.ravel()[whereBnd[wrongBnd]] = False

        if np.any(boundaryImage):

            sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 2])

            # sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:,(zerosIm*boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])
            sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

            boundaryFaces = visibility[(boundaryImage)&(visibility !=4294967295 )]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])
            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            frontFacing = self.frontFacingEdgeFaces[(zerosIm * boundaryImage).ravel().astype(np.bool)].astype(np.bool)
            frontFacingEdgeFaces = self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][frontFacing]

            vertsPerFaceProjBnd = self.camera.r[f[frontFacingEdgeFaces.ravel()].ravel()].reshape([1, -1, 2])
            vertsPerFaceProjBnd = np.tile(vertsPerFaceProjBnd, [self.nsamples, 1,1])
            vertsPerFaceProjBnd = vertsPerFaceProjBnd.reshape([-1,3,2])[facesOutsideBnd.ravel()]

            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_edge = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_edge[t_area_bnd_edge > 1] = 1

            # if self.debug:
            #     import pdb; pdb.set_trace()

            faces = f[sampleFaces[facesOutsideBnd]].ravel()

            vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_outside = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_outside[t_area_bnd_outside > 1] = 1

            faces = f[sampleFaces[facesInsideBnd]].ravel()
            vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:,0,:], np.ones([nv,1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:,1,:], np.ones([nv,1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:,2,:], np.ones([nv,1])]
            t_area_bnd_inside = np.abs(np.linalg.det(np.concatenate([p0_proj[:,None], p1_proj[:,None], p2_proj[:,None]], axis=1))*0.5)
            t_area_bnd_inside[t_area_bnd_inside > 1] = 1

            #Trick to cap to 1 while keeping gradients.

            p1 = vertsProjBndSamplesOutside[:,0,:]
            p2 = vertsProjBndSamplesOutside[:,1,:]

            p = sampleV[facesOutsideBnd]

            l = (p2 - p1)
            linedist = np.sqrt((np.sum(l**2,axis=1)))[:,None]
            self.linedist = linedist

            lnorm = l/linedist
            self.lnorm = lnorm

            v1 = p - p1
            self.v1 = v1
            d = v1[:,0]* lnorm[:,0] + v1[:,1]* lnorm[:,1]
            self.d = d
            intersectPoint = p1 + d[:,None] * lnorm
            self.intersectPoint = intersectPoint

            v2 = p - p2
            self.v2 = v2
            l12 = (p1 - p2)
            linedist12 = np.sqrt((np.sum(l12**2,axis=1)))[:,None]
            lnorm12 = l12/linedist12
            d2 = v2[:,0]* lnorm12[:,0] + v2[:,1]* lnorm12[:,1]

            nonIntersect = (d2 < 0) | (d<0)
            self.nonIntersect = nonIntersect

            argminDistNonIntersect = np.argmin(np.c_[d[nonIntersect], d2[nonIntersect]], 1)
            self.argminDistNonIntersect = argminDistNonIntersect

            intersectPoint[nonIntersect] = vertsProjBndSamplesOutside[nonIntersect][np.arange(nonIntersect.sum()), argminDistNonIntersect]

            lineToPoint = (p - intersectPoint)

            n=lineToPoint

            dist = np.sqrt((np.sum(lineToPoint ** 2, axis=1)))[:, None]

            n_norm = lineToPoint /dist

            self.n_norm = n_norm

            self.dist = dist

            d_final = dist.squeeze()

            # max_nx_ny = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))

            # d_final = d_final/max_nx_ny
            # d_final = d_final

            verticesBnd = self.v.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2 , 3])
            verticesBndSamples = np.tile(verticesBnd[None,:,:],[self.nsamples,1,1, 1])
            verticesBndOutside = verticesBndSamples[facesOutsideBnd]

            vc = self.vc.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2 , 3])
            vc[vc > 1] = 1
            vc[vc < 0] = 0

            vcBnd = vc
            vcBndSamples = np.tile(vcBnd[None,:,:],[self.nsamples,1,1,1])
            vcBndOutside = vcBndSamples[facesOutsideBnd]

            invViewMtx =  np.linalg.inv(np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])])
            #
            camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
            # invCamMtx = np.r_[np.c_[np.linalg.inv(self.camera.camera_mtx), np.array([0,0,0])], np.array([[0, 0, 0, 1]])]

            view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]

            verticesBndOutside = np.concatenate([verticesBndOutside.reshape([-1,3]), np.ones([verticesBndOutside.size//3, 1])], axis=1)

            projVerticesBndOutside = (camMtx.dot(view_mtx)).dot(verticesBndOutside.T).T[:,:3].reshape([-1,2,3])
            projVerticesBndDir = projVerticesBndOutside[:,1,:] - projVerticesBndOutside[:,0,:]
            projVerticesBndDir = projVerticesBndDir/np.sqrt((np.sum(projVerticesBndDir ** 2, 1)))[:, None]

            dproj = (intersectPoint[:,0]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,0]) / (projVerticesBndDir[:,0] - projVerticesBndDir[:,2]*intersectPoint[:,0])
            # Code to check computation that dproj == dprojy
            # dproj_y = (intersectPoint[:,1]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,1]) / (projVerticesBndDir[:,1] - projVerticesBndDir[:,2]*intersectPoint[:,1])

            projPoint = projVerticesBndOutside[:,0,:][:,: ] + dproj[:,None]*projVerticesBndDir[:,:]

            projPointVec4 = np.concatenate([projPoint, np.ones([projPoint.shape[0],1])], axis=1)
            viewPointIntersect = (invViewMtx.dot(np.linalg.inv(camMtx)).dot(projPointVec4.T.reshape([4,-1])).reshape([4,-1])).T[:,:3]

            barycentricVertsDistIntesect = np.linalg.norm(viewPointIntersect - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,0,:], axis=1)
            barycentricVertsDistIntesect2 = np.linalg.norm(viewPointIntersect - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,1,:], axis=1)
            # Code to check barycentricVertsDistIntesect + barycentricVertsDistIntesect2 = barycentricVertsDistEdge
            barycentricVertsDistEdge = np.linalg.norm(verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,0,:] - verticesBndOutside[:,0:3].reshape([-1, 2, 3])[:,1,:], axis=1)

            nonIntersect = np.abs(barycentricVertsDistIntesect + barycentricVertsDistIntesect2 - barycentricVertsDistEdge) > 1e-4
            argminDistNonIntersect = np.argmin(np.c_[barycentricVertsDistIntesect[nonIntersect], barycentricVertsDistIntesect2[nonIntersect]],1)

            barycentricVertsIntersect = barycentricVertsDistIntesect2 / (barycentricVertsDistIntesect + barycentricVertsDistIntesect2)

            barycentricVertsIntersect[nonIntersect] = np.array(argminDistNonIntersect == 0).astype(np.float64)
            self.barycentricVertsIntersect = barycentricVertsIntersect

            self.viewPointIntersect = viewPointIntersect
            self.viewPointIntersect[nonIntersect] = verticesBndOutside.reshape([-1, 2, 4])[nonIntersect, :, 0:3][np.arange(nonIntersect.sum()), argminDistNonIntersect, :]

            vcEdges1 = barycentricVertsIntersect[:, None] * vcBndOutside.reshape([-1, 2, 3])[:, 0, :]
            self.barycentricVertsIntersect = barycentricVertsIntersect
            vcEdges2 = (1-barycentricVertsIntersect[:,None]) * vcBndOutside.reshape([-1,2,3])[:,1,:]

            #Color:
            colorVertsEdge =  vcEdges1 + vcEdges2

            #Point IN edge barycentric

            d_finalNP = np.minimum(d_final.copy(),1.)
            self.d_final_outside = d_finalNP

            self.t_area_bnd_outside =  t_area_bnd_outside
            self.t_area_bnd_edge =  t_area_bnd_edge
            self.t_area_bnd_inside = t_area_bnd_inside
            areaWeights = np.zeros([nsamples, nBndFaces])
            areaWeights[facesOutsideBnd] = (1-d_finalNP)*t_area_bnd_edge + d_finalNP *t_area_bnd_outside
            areaWeights[facesInsideBnd] = t_area_bnd_inside
            areaWeightsTotal = areaWeights.sum(0)
            # areaWeightsTotal[areaWeightsTotal < 1] = 1
            self.areaWeightsTotal = areaWeightsTotal

            finalColorBndOutside = np.zeros([self.nsamples, boundaryFaces.size, 3])
            finalColorBndOutside_edge = np.zeros([self.nsamples, boundaryFaces.size, 3])
            finalColorBndInside = np.zeros([self.nsamples, boundaryFaces.size, 3])


            sampleColorsOutside = sampleColors[facesOutsideBnd]
            self.sampleColorsOutside = sampleColors.copy()

            finalColorBndOutside[facesOutsideBnd] = sampleColorsOutside
            finalColorBndOutside[facesOutsideBnd] = sampleColorsOutside / self.nsamples
            self.finalColorBndOutside_for_dr = finalColorBndOutside.copy()
            # finalColorBndOutside[facesOutsideBnd] *= d_finalNP[:,  None] * t_area_bnd_outside[:,  None]
            finalColorBndOutside[facesOutsideBnd] *= d_finalNP[:,  None]

            finalColorBndOutside_edge[facesOutsideBnd] = colorVertsEdge
            finalColorBndOutside_edge[facesOutsideBnd] = colorVertsEdge/ self.nsamples
            self.finalColorBndOutside_edge_for_dr = finalColorBndOutside_edge.copy()
            # finalColorBndOutside_edge[facesOutsideBnd] *= (1 - d_finalNP[:, None]) * t_area_bnd_edge[:,  None]
            finalColorBndOutside_edge[facesOutsideBnd] *= (1 - d_finalNP[:, None])

            sampleColorsInside = sampleColors[facesInsideBnd]
            self.sampleColorsInside = sampleColorsInside.copy()
            # finalColorBndInside[facesInsideBnd] = sampleColorsInside * self.t_area_bnd_inside[:,  None]

            finalColorBndInside[facesInsideBnd] = sampleColorsInside / self.nsamples

            # finalColorBnd = finalColorBndOutside + finalColorBndOutside_edge + finalColorBndInside
            finalColorBnd = finalColorBndOutside + finalColorBndOutside_edge + finalColorBndInside

            # finalColorBnd /= areaWeightsTotal[None, :, None]

            bndColorsImage = np.zeros_like(self.render_resolved)
            bndColorsImage[(zerosIm * boundaryImage), :] = np.sum(finalColorBnd, axis=0)

            # bndColorsImage1 = np.zeros_like(self.render_resolved)
            # bndColorsImage1[(zerosIm * boundaryImage), :] = np.sum(self.finalColorBndOutside_for_dr, axis=0)
            #
            # bndColorsImage2 = np.zeros_like(self.render_resolved)
            # bndColorsImage2[(zerosIm * boundaryImage), :] = np.sum(self.finalColorBndOutside_edge_for_dr, axis=0)
            #
            # bndColorsImage3 = np.zeros_like(self.render_resolved)
            # bndColorsImage3[(zerosIm * boundaryImage), :] = np.sum(finalColorBndInside, axis=0)

            finalColorImageBnd = bndColorsImage

        if np.any(boundaryImage):
            finalColor = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * finalColorImageBnd
            # finalColor1 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage1
            # finalColor2 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage2
            # finalColor3 = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * bndColorsImage3

        else:
            finalColor = self.color_image

        finalColor[finalColor>1] = 1
        finalColor[finalColor<0] = 0

        return finalColor

    def compute_derivatives_verts(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size
        n_norm = self.n_norm
        dist = self.dist
        linedist = self.linedist
        d = self.d
        v1 = self.v1
        lnorm = self.lnorm

        finalColorBndOutside_for_dr = self.finalColorBndOutside_for_dr
        finalColorBndOutside_edge_for_dr = self.finalColorBndOutside_edge_for_dr
        d_final_outside = self.d_final_outside

        barycentricVertsIntersect = self.barycentricVertsIntersect

        # xdiff = dEdx
        # ydiff = dEdy

        nVisF = len(visibility.ravel()[visible])
        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility!=4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples
        sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
            [nsamples, -1, 2])

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])

        sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage)&(visibility !=4294967295 )]

        if np.any(boundaryImage):

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            p1 = vertsProjBndSamplesOutside[:, 0, :]
            p2 = vertsProjBndSamplesOutside[:, 1, :]
            p = sampleV[facesOutsideBnd]

            #Computing gradients:
            #A multisampled pixel color is given by: w R + (1-w) R' thus:
            #1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            #2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            #3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            #4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            #5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            #6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            #7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            #for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            #Finally, stack data, and IJ of nonbnd with bnd on both dwrt_v and dwrt_vc.

            ######## 1 derivatives samples outside wrt v 1: (dw * (bar*vc) - dw (bar'*vc') )/ nsamples for face sample

            # #Chumpy autodiff code to check derivatives here:
            # chEdgeVerts = ch.Ch(vertsProjBndSamplesOutside)
            #
            # chEdgeVerts1 = chEdgeVerts[:,0,:]
            # chEdgeVerts2 = chEdgeVerts[:,1,:]
            #
            # chSampleVerts = ch.Ch(sampleV[facesOutsideBnd])
            # # c1 = (chEdgeVerts1 - chSampleVerts)
            # # c2 = (chEdgeVerts2 - chSampleVerts)
            # # n = (chEdgeVerts2 - chEdgeVerts1)
            #
            # #Code to check computation of distance below
            # # d2 = ch.abs(c1[:,:,0]*c2[:,:,1] - c1[:,:,1]*c2[:,:,0]) / ch.sqrt((ch.sum(n**2,2)))
            # # # np_mat = ch.dot(ch.array([[0,-1],[1,0]]), n)
            # # np_mat2 = -ch.concatenate([-n[:,:,1][:,:,None], n[:,:,0][:,:,None]],2)
            # # np_vec2 = np_mat2 / ch.sqrt((ch.sum(np_mat2**2,2)))[:,:,None]
            # # d2 =  d2 / ch.maximum(ch.abs(np_vec2[:,:,0]),ch.abs(np_vec2[:,:,1]))
            #
            # chl = (chEdgeVerts2 - chEdgeVerts1)
            # chlinedist = ch.sqrt((ch.sum(chl**2,axis=1)))[:,None]
            # chlnorm = chl/chlinedist
            #
            # chv1 = chSampleVerts - chEdgeVerts1
            # chd = chv1[:,0]* chlnorm[:,0] + chv1[:,1]* chlnorm[:,1]
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            # # intersectPointDist1 = intersectPoint - chEdgeVerts1
            # # intersectPointDist2 = intersectPoint - chEdgeVerts2
            # # Code to check computation of distances below:
            # # lengthIntersectToPoint1 = np.linalg.norm(intersectPointDist1.r,axis=1)
            # # lengthIntersectToPoint2 = np.linalg.norm(intersectPointDist2.r,axis=1)
            #
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            #
            # chlineToPoint = (chSampleVerts - chintersectPoint)
            # chn_norm = chlineToPoint / ch.sqrt((ch.sum(chlineToPoint ** 2, axis=1)))[:, None]
            #
            # chdist = chlineToPoint[:,0]*chn_norm[:,0] + chlineToPoint[:,1]*chn_norm[:,1]
            #
            # d_final_ch = chdist / ch.maximum(ch.abs(chn_norm[:, 0]), ch.abs(chn_norm[:, 1]))
            #
            # d_final_outside = d_final_ch.ravel()
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts1)
            # rows = np.tile(np.arange(d_final_outside.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_outside.shape[0] * 2)
            #
            # dwdv_r_v1 = np.array(dwdv[rows, cols]).reshape([-1, 2])
            #
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts2)
            # rows = np.tile(np.arange(d_final_ch.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_ch.shape[0] * 2)
            #
            # dwdv_r_v2 = np.array(dwdv[rows, cols]).reshape([-1, 2])

            nonIntersect = self.nonIntersect
            argminDistNonIntersect = self.argminDistNonIntersect

            max_dx_dy = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))
            # d_final_np = dist / max_dx_dy
            d_final_np = dist

            ident = np.identity(2)
            ident = np.tile(ident[None, :], [len(p2), 1, 1])

            dlnorm = (ident - np.einsum('ij,ik->ijk', lnorm, lnorm)) / linedist[:,  None]
            dl_normdp1 = np.einsum('ijk,ikl->ijl', dlnorm, -ident)
            dl_normdp2 = np.einsum('ijk,ikl->ijl', dlnorm, ident)

            dv1dp1 = -ident
            dv1dp2 = 0

            dddp1 = np.einsum('ijk,ij->ik', dv1dp1, lnorm) + np.einsum('ij,ijl->il', v1, dl_normdp1)
            dddp2 = 0 + np.einsum('ij,ijl->il', v1, dl_normdp2)

            dipdp1 = ident + (dddp1[:,None,:]*lnorm[:,:,None]) + d[:,None,None]*dl_normdp1
            dipdp2 = (dddp2[:,None,:]*lnorm[:,:,None]) + d[:,None,None]*dl_normdp2

            dndp1 = -dipdp1
            dndp2 = -dipdp2

            dn_norm = (ident - np.einsum('ij,ik->ijk', n_norm, n_norm)) / dist[:,None]

            dn_normdp1 = np.einsum('ijk,ikl->ijl', dn_norm, dndp1)
            dn_normdp2 = np.einsum('ijk,ikl->ijl', dn_norm, dndp2)

            ddistdp1 = np.einsum('ij,ijl->il', n_norm, dndp1)
            ddistdp2 = np.einsum('ij,ijl->il', n_norm, dndp2)

            argmax_nx_ny = np.argmax(np.abs(n_norm),axis=1)
            dmax_nx_ny_p1 = np.sign(n_norm)[np.arange(len(n_norm)),argmax_nx_ny][:,None]*dn_normdp1[np.arange(len(dn_normdp1)),argmax_nx_ny]
            dmax_nx_ny_p2 = np.sign(n_norm)[np.arange(len(n_norm)),argmax_nx_ny][:,None]*dn_normdp2[np.arange(len(dn_normdp2)),argmax_nx_ny]

            # dd_final_dp1 = -1./max_dx_dy[:,None]**2 * dmax_nx_ny_p1 * dist + 1./max_dx_dy[:,None] *  ddistdp1
            # dd_final_dp2 = -1./max_dx_dy[:,None]**2 * dmax_nx_ny_p2 * dist + 1./max_dx_dy[:,None] *  ddistdp2

            dd_final_dp1 = ddistdp1
            dd_final_dp2 = ddistdp2

            #For those non intersecting points straight to the edge:

            v1 = self.v1[nonIntersect][argminDistNonIntersect==0]
            v1_norm = v1/np.sqrt((np.sum(v1**2,axis=1)))[:,None]

            dd_final_dp1_nonintersect = -v1_norm

            v2 = self.v2[nonIntersect][argminDistNonIntersect==1]
            v2_norm = v2/np.sqrt((np.sum(v2**2,axis=1)))[:,None]
            dd_final_dp2_nonintersect = -v2_norm

            dd_final_dp1[nonIntersect][argminDistNonIntersect == 0] = dd_final_dp1_nonintersect
            dd_final_dp1[nonIntersect][argminDistNonIntersect == 1] = 0
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 1] = dd_final_dp2_nonintersect
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 0] = 0

            dImage_wrt_outside_v1 = finalColorBndOutside_for_dr[facesOutsideBnd][:,:,None]*dd_final_dp1[:,None,:] - dd_final_dp1[:,None,:]*finalColorBndOutside_edge_for_dr[facesOutsideBnd][:,:,None]
            dImage_wrt_outside_v2 =  finalColorBndOutside_for_dr[facesOutsideBnd][:,:,None]*dd_final_dp2[:,None,:] - dd_final_dp2[:,None,:]*finalColorBndOutside_edge_for_dr[facesOutsideBnd][:,:,None]

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2*2)).ravel()
            # faces = f[sampleFaces[facesOutsideBnd]].ravel()
            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1, -1, 2]), [self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data1 = dImage_wrt_outside_v1.transpose([1,0,2])
            data2 = dImage_wrt_outside_v2.transpose([1,0,2])

            data = np.concatenate([data1[:,:,None,:], data2[:,:,None,:]], 2)

            data = data.ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd_outside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ######## 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            ######## 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample

            verticesBnd = self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3])

            sampleBarycentricBar = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([-1, 3, 1])
            verts = np.sum(self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3, 3]) * sampleBarycentricBar, axis=1)

            dImage_wrt_bar_v = self.barycentricDerivatives(verticesBnd, f[sampleFaces.ravel()], verts).swapaxes(0,1)

            dImage_wrt_bar_v[facesOutsideBnd.ravel()] = dImage_wrt_bar_v[facesOutsideBnd.ravel()] * d_final_outside[:,None,None, None] * self.t_area_bnd_outside[:, None, None, None]
            dImage_wrt_bar_v[facesInsideBnd.ravel()] = dImage_wrt_bar_v[facesInsideBnd.ravel()] * self.t_area_bnd_inside[:, None, None, None]

            # dImage_wrt_bar_v /= np.tile(areaWeightsTotal[None,:], [self.nsamples,1]).ravel()[:, None,None, None]
            dImage_wrt_bar_v /= self.nsamples

            ### Derivatives wrt V: 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
            faces = f[sampleFaces[facesOutsideBnd]].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            # data = np.tile(dImage_wrt_bar_v[facesOutsideBnd.ravel()][None,:],[3,1,1,1]).ravel()
            data = np.transpose(dImage_wrt_bar_v[facesOutsideBnd.ravel()],[1,0,2,3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_outside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ### Derivatives wrt V: 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesInsideBnd]
            IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
            faces = f[sampleFaces[facesInsideBnd]].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v[facesInsideBnd.ravel()], [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_inside = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

            ####### 3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)

            frontFacing = self.frontFacingEdgeFaces[(zerosIm * boundaryImage).ravel().astype(np.bool)].astype(np.bool)
            frontFacingEdgeFaces = self.fpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]][frontFacing]

            verticesBnd = self.v.r[f[frontFacingEdgeFaces.ravel()].ravel()].reshape([1, -1, 3])
            verticesBnd = np.tile(verticesBnd, [self.nsamples, 1,1])
            verticesBnd = verticesBnd.reshape([-1,3,3])[facesOutsideBnd.ravel()].reshape([-1,3])

            verts = self.viewPointIntersect

            fFrontEdge = np.tile(f[frontFacingEdgeFaces][None,:], [self.nsamples, 1, 1]).reshape([-1,3])[facesOutsideBnd.ravel()]

            dImage_wrt_bar_v_edge = self.barycentricDerivatives(verticesBnd, fFrontEdge, verts).swapaxes(0, 1)

            dImage_wrt_bar_v_edge = dImage_wrt_bar_v_edge * (1-d_final_outside[:,None,None, None]) * self.t_area_bnd_edge[:, None, None, None]

            # dImage_wrt_bar_v_edge /= np.tile(self.areaWeightsTotal[None,:], [self.nsamples,1])[facesOutsideBnd][:, None, None,None]

            dImage_wrt_bar_v_edge /= self.nsamples

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 3 * 2)).ravel()
            # faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = f[frontFacingEdgeFaces]
            faces = np.tile(faces.reshape([1, -1, 3]), [self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)
            JS = np.hstack((JS*2, JS*2+1)).ravel()
            if n_channels > 1:
                IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v_edge, [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bar_outside_edge = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        vertsPerFaceProjBnd = self.camera.r[f[nonBoundaryFaces].ravel()].reshape([-1,3,2])
        nv = len(vertsPerFaceProjBnd)

        p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
        p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
        p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
        t_area_nonbnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)
        t_area_nonbnd[t_area_nonbnd> 1] = 1

        bc = barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))

        verts = np.sum(self.v.r[f[nonBoundaryFaces.ravel()].ravel()].reshape([-1, 3, 3]) * bc[:, :,None], axis=1)


        didp = self.barycentricDerivatives(verticesNonBnd, f[nonBoundaryFaces.ravel()], verts)

        didp = didp * t_area_nonbnd[None,:,None, None]

        n_channels = np.atleast_3d(observed).shape[2]
        shape = visibility.shape

        ####### 2: Take the data and copy the corresponding dxs and dys to these new pixels.

        ### Derivatives wrt V:
        # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
        pixels = np.where(((~boundaryImage)&(visibility !=4294967295 )).ravel())[0]
        IS = np.tile(col(pixels), (1, 2*f.shape[1])).ravel()
        JS = col(f[nonBoundaryFaces].ravel())
        JS = np.hstack((JS*2, JS*2+1)).ravel()

        if n_channels > 1:
            IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        # data = np.concatenate(((visTriVC[:,0,:] * dBar1dx[:,None])[:,:,None],(visTriVC[:, 0, :] * dBar1dy[:, None])[:,:,None], (visTriVC[:,1,:]* dBar2dx[:,None])[:,:,None], (visTriVC[:, 1, :] * dBar2dy[:, None])[:,:,None],(visTriVC[:,2,:]* dBar3dx[:,None])[:,:,None],(visTriVC[:, 2, :] * dBar3dy[:, None])[:,:,None]),axis=2).swapaxes(0,1).ravel()
        data = didp.ravel()

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result_wrt_verts_nonbnd = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))
        # result_wrt_verts_nonbnd.sum_duplicates()

        if np.any(boundaryImage):

            result_wrt_verts =  result_wrt_verts_bnd_outside + result_wrt_verts_bar_outside + result_wrt_verts_bar_inside + result_wrt_verts_bar_outside_edge + result_wrt_verts_nonbnd
            # result_wrt_verts = result_wrt_verts_bnd_outside

        else:
            result_wrt_verts = result_wrt_verts_nonbnd

        return result_wrt_verts


    def compute_derivatives_vc(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        d_final_outside = self.d_final_outside

        barycentricVertsIntersect = self.barycentricVertsIntersect


        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility!=4294967295)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage)&(visibility !=4294967295 )]

        if np.any(boundaryImage):

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)
            projFacesBndTiled = np.tile(boundaryFaces[None, :], [self.nsamples, 1])

            facesInsideBnd = projFacesBndTiled == sampleFaces
            facesOutsideBnd = ~facesInsideBnd

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1,1,1])
            vertsProjBndSamplesOutside = vertsProjBndSamples[facesOutsideBnd]

            #Computing gradients:
            #A multisampled pixel color is given by: w R + (1-w) R' thus:
            #1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            #2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            #3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            #4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            #5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            #6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            #7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            #for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            ####### 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            dImage_wrt_outside_vc_outside = d_final_outside[:,None] * sampleBarycentric[facesOutsideBnd] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 3)).ravel()

            faces = f[sampleFaces[facesOutsideBnd]].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_outside.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

            result_wrt_vc_bnd_outside = result
            # result_wrt_vc_bnd_outside.sum_duplicates()

            ######## 5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge
            dImage_wrt_outside_vc_edge = (1-d_final_outside[:, None]) * np.c_[barycentricVertsIntersect, 1-barycentricVertsIntersect] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesOutsideBnd]
            IS = np.tile(col(pixels), (1, 2)).ravel()
            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1,-1,2]),[self.nsamples, 1, 1])[facesOutsideBnd].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_edge.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result_wrt_vc_bnd_outside_edge = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))
            # result_wrt_vc_bnd_outside_edge.sum_duplicates()

            ######## 7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample
            dImage_wrt_outside_vc_inside = sampleBarycentric[facesInsideBnd] / self.nsamples

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None,:], [self.nsamples, 1])[facesInsideBnd]
            IS = np.tile(col(pixels), (1, 3)).ravel()
            faces = f[sampleFaces[facesInsideBnd]].ravel()
            JS = col(faces)

            data = dImage_wrt_outside_vc_inside.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
            data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result_wrt_vc_bnd_inside = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))
            # result_wrt_vc_bnd_inside.sum_duplicates()


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        # barySample = self.renders_sample_barycentric[0].reshape([-1,3])[(~boundaryImage)&(visibility !=4294967295 ).ravel().astype(np.bool), :]

        bc = barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))
        # barySample[barycentric[((~boundaryImage)&(visibility !=4294967295 ))].reshape((-1, 3))]

        ### Derivatives wrt VC:

        # Each pixel relies on three verts
        pixels = np.where(((~boundaryImage)&(visibility !=4294967295 )).ravel())[0]
        IS = np.tile(col(pixels), (1, 3)).ravel()
        JS = col(f[nonBoundaryFaces].ravel())

        bc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
        # bc = barySample.reshape((-1, 3))

        data = np.asarray(bc, order='C').ravel()

        IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
        JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])
        data = np.concatenate([data for i in range(num_channels)])
        # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        # data = np.concatenate((data, data, data))

        ij = np.vstack((IS.ravel(), JS.ravel()))
        result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

        result_wrt_vc_nonbnd = result
        # result_wrt_vc_nonbnd.sum_duplicates()

        if np.any(boundaryImage):
            # result_wrt_verts = result_wrt_verts_bar_outside_edge

            # result_wrt_verts = result_wrt_verts_nonbnd
            result_wrt_vc = result_wrt_vc_bnd_outside + result_wrt_vc_bnd_outside_edge + result_wrt_vc_bnd_inside + result_wrt_vc_nonbnd
            # result_wrt_vc = sp.csc_matrix((width * height * num_channels, vc_size))
        else:
            # result_wrt_verts = sp.csc_matrix((image_width*image_height*n_channels, num_verts*2))
            result_wrt_vc = result_wrt_vc_nonbnd
            # result_wrt_vc = sp.csc_matrix((width * height * num_channels, vc_size))
        return result_wrt_vc


    def on_changed(self, which):
        super().on_changed(which)

        if 'v' or 'camera' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_verts_mesh[mesh][polygons].set_array(verts_by_face.astype(np.float32))
                    self.vbo_verts_mesh[mesh][polygons].bind()

        if 'vc' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                    self.vbo_colors_mesh[mesh][polygons].bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()
            flen = 1
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]

                    # fc = np.arange(flen, flen + len(f))
                    fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()

                    # fc[:, 0] = fc[:, 0] & 255
                    # fc[:, 1] = (fc[:, 1] >> 8) & 255
                    # fc[:, 2] = (fc[:, 2] >> 16) & 255
                    fc = np.asarray(fc, dtype=np.uint32)
                    self.vbo_face_ids_list[mesh][polygons].set_array(fc)
                    self.vbo_face_ids_list[mesh][polygons].bind()

                    flen += len(f)

                    self.vbo_indices_mesh_list[mesh][polygons].set_array(np.array(self.f_list[mesh][polygons]).astype(np.uint32))
                    self.vbo_indices_mesh_list[mesh][polygons].bind()

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
                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size+textureCoordIdx].reshape(image.shape)

                            textureCoordIdx = textureCoordIdx + image.size
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)

                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                                               image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())



        if 'v' or 'f' or 'vc' or 'ft' or 'camera' or 'texture_stack' in which:
            self.render_image_buffers()


    def release_textures(self):
        if hasattr(self, 'textureID_mesh_list'):
            if self.textureID_mesh_list != []:
                for texture_mesh in self.textureID_mesh_list:
                    if texture_mesh != []:
                        for texture in texture_mesh:
                            if texture != None:
                                GL.glDeleteTextures(1, [texture.value])

        self.textureID_mesh_list = []

    @depends_on(dterms+terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        return no_overdraw

        # if not self.overdraw:
        #     return no_overdraw
        #
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        # overdraw = self.draw_color_image()
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #
        # # return overdraw * np.atleast_3d(self.boundarybool_image)
        #
        # boundarybool_image = self.boundarybool_image
        # if self.num_channels > 1:
        #     boundarybool_image = np.atleast_3d(boundarybool_image)
        #
        # return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')


    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_barycentric_image(self.boundarybool_image if self.overdraw else None)

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()
        #Overload method to call without overdraw.
        return self.draw_visibility_image(self.v.r, self.f, self.boundarybool_image if self.overdraw else None)

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

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))),np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        vc = self.vc_list[mesh]

        for polygons in np.arange(len(self.f_list[mesh])):
            vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
            GL.glBindVertexArray(vao_mesh)
            f = self.f_list[mesh][polygons]
            vbo_color = self.vbo_colors_mesh[mesh][polygons]
            colors_by_face = np.asarray(vc.reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
            colors = np.array(np.ones_like(colors_by_face) * (index) / 255.0, dtype=np.float32)

            # Pol: Make a static zero vbo_color to make it more efficient?
            vbo_color.set_array(colors)

            vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

            vbo_color.bind()

            if self.f.shape[1]==2:
                primtype = GL.GL_LINES
            else:
                primtype = GL.GL_TRIANGLES

            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

            GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])


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

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_color = self.vbo_colors_mesh[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vc = colors_by_face

                if with_vertex_colors:
                    colors = vc.astype(np.float32)
                else:
                    # Only texture.
                    colors = np.ones_like(vc).astype(np.float32)

                # Pol: Make a static zero vbo_color to make it more efficient?
                vbo_color.set_array(colors)
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

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])
                # GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

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




class ResidualRenderer(ColoredRenderer):
    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list', 'imageGT'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    def __init__(self):
        super().__init__()

    def clear(self):
        try:
            GL.glFlush()
            GL.glFinish()
            # print ("Clearing textured renderer.")
            # for msh in self.vbo_indices_mesh_list:
            #     for vbo in msh:
            #         vbo.set_array([])
            [vbo.set_array(np.array([])) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_colors_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_verts_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_uvs_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_face_ids_list for vbo in sublist]

            [GL.glDeleteVertexArrays(1, [vao.value]) for sublist in self.vao_tex_mesh_list for vao in sublist]

            self.release_textures()

            if self.glMode == 'glfw':
                import glfw
                glfw.make_context_current(self.win)

            GL.glDeleteProgram(self.colorTextureProgram)

            super().clear()
        except:
            import pdb
            pdb.set_trace()
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
            color = theColor * texture( myTextureSampler, UV).rgb;
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

        self.colorTextureProgram = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

        # Define the other VAO/VBOs and shaders.
        # Text VAO and bind color, vertex indices AND uvbuffer:

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

        # GL.glEnable(GL.GL_LINE_SMOOTH)
        # GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(2.)

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_indices_mesh = []
            vbo_face_ids_mesh = []
            vbo_colors_mesh = []
            vbo_vertices_mesh = []
            vbo_uvs_mesh = []
            textureIDs_mesh = []
            for polygons in range(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_verts = vbo.VBO(np.array(verts_by_face).astype(np.float32))
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_colors = vbo.VBO(np.array(colors_by_face).astype(np.float32))
                uvs_by_face = np.asarray(self.ft_list[mesh].reshape((-1, 2))[f.ravel()], dtype=np.float32, order='C')
                vbo_uvs = vbo.VBO(np.array(uvs_by_face).astype(np.float32))

                vbo_indices = vbo.VBO(np.array(self.f_list[mesh][polygons]).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
                vbo_indices.bind()
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                vbo_colors.bind()
                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                if self.haveUVs_list[mesh][polygons]:
                    vbo_uvs.bind()

                    GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                    GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                # Textures:
                texture = None
                if self.haveUVs_list[mesh][polygons]:
                    texture = GL.GLuint(0)

                    GL.glGenTextures(1, texture)
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
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
                vbo_colors_mesh = vbo_colors_mesh + [vbo_colors]
                vbo_vertices_mesh = vbo_vertices_mesh + [vbo_verts]
                vbo_uvs_mesh = vbo_uvs_mesh + [vbo_uvs]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

            self.vbo_colors_mesh = self.vbo_colors_mesh + [vbo_colors_mesh]
            self.vbo_verts_mesh = self.vbo_verts_mesh + [vbo_vertices_mesh]
            self.vbo_uvs_mesh = self.vbo_uvs_mesh + [vbo_uvs_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")

    def initGL_AnalyticRenderer(self):
        self.initGLTexture()

        self.updateRender = True
        self.updateDerivatives = True

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 colorIn;
        layout(location = 2) in vec2 vertexUV;
        layout(location = 3) in uint face_id;
        layout(location = 4) in vec3 barycentric;

        uniform mat4 MVP;
        out vec3 theColor;
        out vec4 pos;
        flat out uint face_out;
        out vec3 barycentric_vert_out;
        out vec2 UV;

        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            pos =  MVP * vec4(position,1);
            //pos =  pos4.xyz;
            theColor = colorIn;
            UV = vertexUV;
            face_out = face_id;
            barycentric_vert_out = barycentric;

        }""", GL.GL_VERTEX_SHADER)

        ERRORS_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 

            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
            in vec3 theColor;
            in vec2 UV;
            flat in uint face_out;
            in vec4 pos;
            in vec3 barycentric_vert_out;

            layout(location = 3) uniform sampler2D myTextureSampler;
            
            uniform float ww;
            uniform float wh;

            // Ouput data
            layout(location = 0) out vec3 color; 
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 barycentric1;
            layout(location = 4) out vec2 barycentric2;

            void main(){
                vec3 finalColor = theColor * texture( myTextureSampler, UV).rgb;
                color = finalColor.rgb;

                sample_pos = ((0.5*pos.xy/pos.w) + 0.5)*vec2(ww,wh);
                sample_face = face_out;
                barycentric1 = barycentric_vert_out.xy;
                barycentric2 = vec2(barycentric_vert_out.z, 0.);

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
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            layout(location = 2) uniform sampler2DMS colors;
            layout(location = 3) uniform sampler2DMS sample_positions;
            layout(location = 4) uniform usampler2DMS sample_faces;
            layout(location = 5) uniform sampler2DMS sample_barycentric_coords1;
            layout(location = 6) uniform sampler2DMS sample_barycentric_coords2;
            //layout(location = 7) uniform sampler2D imageGT;

            uniform float ww;
            uniform float wh;
            uniform int sample;

            // Ouput data
            layout(location = 0) out vec3 colorFetchOut;
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 sample_barycentric1;
            layout(location = 4) out vec2 sample_barycentric2;
            //layout(location = 5) out vec3 res;

            //out int gl_SampleMask[];
            const int all_sample_mask = 0xffff;

            void main(){
                ivec2 texcoord = ivec2(gl_FragCoord.xy);
                colorFetchOut = texelFetch(colors, texcoord, sample).xyz;
                sample_pos = texelFetch(sample_positions, texcoord, sample).xy;        
                sample_face = texelFetch(sample_faces, texcoord, sample).r;
                sample_barycentric1 = texelFetch(sample_barycentric_coords1, texcoord, sample).xy;
                sample_barycentric2 = texelFetch(sample_barycentric_coords2, texcoord, sample).xy;
                
                //vec3 imgColor = texture(imageGT, gl_FragCoord.xy/vec2(ww,wh)).rgb;
                //res = imgColor - colorFetchOut;
                
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
        # GL.glGenTextures(1, self.textureEdges)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureEdges)
        # GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)

        GL.glActiveTexture(GL.GL_TEXTURE0)

        whitePixel = np.ones([1, 1, 3])
        self.whitePixelTextureID = GL.GLuint(0)
        GL.glGenTextures(1, self.whitePixelTextureID)
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

        self.texture_errors_sample_faces = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_R32UI, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces, 0)
        #

        self.texture_errors_sample_barycentric1 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1,
                                  0)

        self.texture_errors_sample_barycentric2 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2,
                                  0)

        self.z_buf_ms_errors = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.z_buf_ms_errors)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'],
                                   False)
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

        self.render_buffer_fetch_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        #
        self.render_buffer_fetch_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)

        self.render_buffer_fetch_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)

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

        # FBO_f
        self.fbo_errors_nonms = GL.glGenFramebuffers(1)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_errors_nonms)

        render_buf_errors_render = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_render)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf_errors_render)

        render_buf_errors_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_sample_position)

        render_buf_errors_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        #

        render_buf_errors_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)

        render_buf_errors_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
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

        self.textureObjLoc = GL.glGetUniformLocation(self.errorTextureProgram, "myTextureSampler")

        # Add background cube:
        position_location = GL.glGetAttribLocation(self.errorTextureProgram, 'position')
        color_location = GL.glGetAttribLocation(self.errorTextureProgram, 'colorIn')
        uvs_location = GL.glGetAttribLocation(self.errorTextureProgram, 'vertexUV')
        face_ids_location = GL.glGetAttribLocation(self.errorTextureProgram, 'face_id')
        barycentric_location = GL.glGetAttribLocation(self.errorTextureProgram, 'barycentric')

        # self.vbo_verts_cube= vbo.VBO(np.array(self.v_bgCube).astype(np.float32))
        # self.vbo_colors_cube= vbo.VBO(np.array(self.vc_bgCube).astype(np.float32))
        # self.vbo_uvs_cube = vbo.VBO(np.array(self.ft_bgCube).astype(np.float32))
        # self.vao_bgCube = GL.GLuint(0)
        # GL.glGenVertexArrays(1, self.vao_bgCube)
        #
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube = vbo.VBO(np.array(self.f_bgCube).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        # self.vbo_f_bgCube.bind()
        # self.vbo_verts_cube.bind()
        # GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_colors_cube.bind()
        # GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_uvs_cube.bind()
        # GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        #
        # f = self.f_bgCube
        # fc = np.tile(np.arange(len(self.f), len(self.f) + len(f))[:, None], [1, 3]).ravel()
        # # fc[:, 0] = fc[:, 0] & 255
        # # fc[:, 1] = (fc[:, 1] >> 8) & 255
        # # fc[:, 2] = (fc[:, 2] >> 16) & 255
        # fc = np.asarray(fc, dtype=np.uint32)
        # vbo_face_ids_cube = vbo.VBO(fc)
        # vbo_face_ids_cube.bind()
        # GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
        # GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)
        #
        # #Barycentric cube:
        # f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
        # vbo_barycentric_cube = vbo.VBO(f_barycentric)
        # vbo_barycentric_cube.bind()
        # GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
        # GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_quad = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_quad)
        GL.glBindVertexArray(self.vao_quad)

        # Bind VAO

        self.vbo_face_ids_list = []
        self.vbo_barycentric_list = []
        self.vao_errors_mesh_list = []
        flen = 1

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_face_ids_mesh = []
            vbo_barycentric_mesh = []
            for polygons in np.arange(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]
                vbo_f.bind()
                vbo_verts = self.vbo_verts_mesh[mesh][polygons]
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_colors = self.vbo_colors_mesh[mesh][polygons]
                vbo_colors.bind()

                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_uvs = self.vbo_uvs_mesh[mesh][polygons]
                vbo_uvs.bind()
                GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                f = self.f_list[mesh][polygons]

                fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()
                # fc[:, 0] = fc[:, 0] & 255
                # fc[:, 1] = (fc[:, 1] >> 8) & 255
                # fc[:, 2] = (fc[:, 2] >> 16) & 255
                fc = np.asarray(fc, dtype=np.uint32)
                vbo_face_ids = vbo.VBO(fc)
                vbo_face_ids.bind()
                GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
                GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)

                f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
                vbo_barycentric = vbo.VBO(f_barycentric)
                vbo_barycentric.bind()
                GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                flen += len(f)

                vaos_mesh += [vao]

                vbo_face_ids_mesh += [vbo_face_ids]
                vbo_barycentric_mesh += [vbo_face_ids]

                GL.glBindVertexArray(0)

            self.vbo_face_ids_list += [vbo_face_ids_mesh]
            self.vbo_barycentric_list += [vbo_barycentric_mesh]
            self.vao_errors_mesh_list += [vaos_mesh]

    def render_image_buffers(self):

        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        self.makeCurrentContext()

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels], self.bgcolor.r[2 % self.num_channels], 1.)

        GL.glUseProgram(self.errorTextureProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        # GL.glClearBufferiv(GL.GL_COLOR​, 0​, 0)
        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        #ImageGT
        GL.glActiveTexture(GL.GL_TEXTURE1)
        # GL.glBindImageTexture(1,self.textureGT, 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RGBA8)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        self.textureGTLoc = GL.glGetUniformLocation(self.errorTextureProgram, "imageGT")
        GL.glUniform1i(self.textureGTLoc, 1)


        wwLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_errors_mesh_list[mesh][polygons]

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                # vbo_color.bind()

                f = self.f_list[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                self.vbo_colors_mesh[mesh][polygons].bind()

                if self.f.shape[1] == 2:
                    primtype = GL.GL_LINES
                else:
                    primtype = GL.GL_TRIANGLES

                assert (primtype == GL.GL_TRIANGLES)

                # GL.glUseProgram(self.errorTextureProgram)
                if self.haveUVs_list[mesh][polygons]:
                    texture = self.textureID_mesh_list[mesh][polygons]
                else:
                    texture = self.whitePixelTextureID

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glUniform1i(self.textureObjLoc, 0)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])

        # # #Background cube:
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube.bind()
        # texture = self.whitePixelTextureID
        # self.vbo_uvs_cube.bind()
        #
        # GL.glActiveTexture(GL.GL_TEXTURE0)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # GL.glUniform1i(self.textureObjLoc, 0)
        # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
        #
        # GL.glDrawElements(primtype, len(self.vbo_f_bgCube)*self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

        # self.draw_visibility_image_ms(self.v, self.f)

        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # # result_blit = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        # result_blit2 = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # result_blit_pos = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

        GL.glUseProgram(self.fetchSamplesProgram)
        # GL.glDisable(GL.GL_MULTISAMPLE)

        self.colorsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "colors")
        self.sample_positionsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_positions")
        self.sample_facesLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_faces")
        self.sample_barycentric1Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords1")
        self.sample_barycentric2Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords2")

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # GL.glActiveTexture(GL.GL_TEXTURE2)
        # GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_face)
        # GL.glUniform1i(self.sample_facesLoc, 2)

        wwLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        self.renders = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 3])
        self.renders_sample_pos = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_faces = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height']]).astype(np.uint32)
        self.renders_sample_barycentric1 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_sample_barycentric2 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 1])
        self.renders_sample_barycentric = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 3])

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_sample_fetch)
        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3,
                          GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for sample in np.arange(self.nsamples):
            sampleLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'sample')
            GL.glUniform1i(sampleLoc, sample)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
            GL.glUniform1i(self.colorsLoc, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
            GL.glUniform1i(self.sample_positionsLoc, 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
            GL.glUniform1i(self.sample_facesLoc, 2)

            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
            GL.glUniform1i(self.sample_barycentric1Loc, 3)

            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
            GL.glUniform1i(self.sample_barycentric2Loc, 4)

            GL.glBindVertexArray(self.vao_quad)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

            # GL.glBindVertexArray(self.vao_bgCube)
            # # self.vbo_f_bgCube.bind()
            # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
            #
            # GL.glDrawElements(primtype, len(self.vbo_f_bgCube) * self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_sample_fetch)

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:3].astype(np.float64))

            self.renders[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:2].astype(np.float64))
            self.renders_sample_pos[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT),
                              np.uint32).reshape(self.frustum['height'], self.frustum['height'])[:, :].astype(np.uint32))
            self.renders_faces[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:2].astype(np.float64))
            self.renders_sample_barycentric1[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:1].astype(np.float64))
            self.renders_sample_barycentric2[sample] = result

            self.renders_sample_barycentric[sample] = np.concatenate(
                [self.renders_sample_barycentric1[sample], self.renders_sample_barycentric2[sample][:, :, 0:1]], 2)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            # result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
            # self.renders_faces[sample] = result

        GL.glBindVertexArray(0)

        GL.glClearColor(0., 0., 0., 1.)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)

        ##Finally return image and derivatives

        self.render_resolved = np.mean(self.renders, 0)

        self.updateRender = True
        self.updateDerivatives_verts = True
        self.updateDerivatives_vc = True

    def draw_visibility_image_ms(self, v, f):
        """Assumes camera is set up correctly in"""
        GL.glUseProgram(self.visibilityProgram_ms)

        v = np.asarray(v)

        self.draw_visibility_image_ms(v, f)

        # Attach FBO
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        fc = np.arange(1, len(f) + 1)
        fc = np.tile(fc.reshape((-1, 1)), (1, 3))
        fc[:, 0] = fc[:, 0] & 255
        fc[:, 1] = (fc[:, 1] >> 8) & 255
        fc[:, 2] = (fc[:, 2] >> 16) & 255
        fc = np.asarray(fc, dtype=np.uint8)

        self.draw_colored_primitives_ms(self.vao_dyn_ub, v, f, fc)

    # this assumes that fc is either "by faces" or "verts by face", not "by verts"
    def draw_colored_primitives_ms(self, vao, v, f, fc=None):

        # gl.EnableClientState(GL_VERTEX_ARRAY)
        verts_by_face = np.asarray(v.reshape((-1, 3))[f.ravel()], dtype=np.float64, order='C')
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

            vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
            self.vbo_colors_ub.set_array(vc_by_face)
            self.vbo_colors_ub.bind()

        primtype = GL.GL_TRIANGLES

        self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
        self.vbo_indices_dyn.bind()

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(1, drawingBuffers)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def compute_dr_wrt(self, wrt):

        visibility = self.visibility_image

        if wrt is self.camera:
            derivatives_verts = self.get_derivatives_verts()

            return derivatives_verts

        elif wrt is self.vc:

            derivatives_vc = self.get_derivatives_vc()

            return derivatives_vc

        # Not working atm.:
        elif wrt is self.bgcolor:
            return 2. * (self.imageGT.r - self.render_image).ravel() * common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

        # Not working atm.:
        elif wrt is self.texture_stack:
            IS = np.nonzero(self.visibility_image.ravel() != 4294967295)[0]
            texcoords, texidx = self.texcoord_image_quantized
            vis_texidx = texidx.ravel()[IS]
            vis_texcoords = texcoords.ravel()[IS]
            JS = vis_texcoords * np.tile(col(vis_texidx), [1, 2]).ravel()

            clr_im = -2. * (self.imageGT.r - self.render_image) * self.renderWithoutTexture

            if False:
                cv2.imshow('clr_im', clr_im)
                # cv2.imshow('texmap', self.texture_image.r)
                cv2.waitKey(1)

            r = clr_im[:, :, 0].ravel()[IS]
            g = clr_im[:, :, 1].ravel()[IS]
            b = clr_im[:, :, 2].ravel()[IS]
            data = np.concatenate((r, g, b))

            IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
            JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))

            return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))

        return None

    def compute_r(self):
        return self.render()

    @depends_on(dterms + terms)
    def renderWithoutColor(self):
        self._call_on_changed()

        return self.render_nocolor

    @depends_on(dterms + terms)
    def renderWithoutTexture(self):
        self._call_on_changed()

        return self.render_notexture

    # @depends_on(dterms+terms)
    def render(self):
        self._call_on_changed()

        visibility = self.visibility_image

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        if self.updateRender:
            render, residuals = self.compute_image(visible, visibility, self.f)
            self.render_result = render
            self.residuals_result = residuals
            self.updateRender = False

        if self.imageGT is None:
            returnResult = self.render_result
        else:
            returnResult = self.residuals_result

        return returnResult

    def get_derivatives_verts(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_verts:
            if self.updateRender:
                self.render()
            derivatives_verts = self.compute_derivatives_verts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'],
                                                               self.v.r.size / 3, self.f)
            self.derivatives_verts = derivatives_verts
            self.updateDerivatives_verts = False
        return self.derivatives_verts

    def get_derivatives_vc(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_vc:
            if self.updateRender:
                self.render()
            derivatives_vc = self.compute_derivatives_vc(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'],
                                                         self.v.r.size / 3, self.f)
            self.derivatives_vc = derivatives_vc
            self.updateDerivatives_vc = False
        return self.derivatives_vc

    # # @depends_on(dterms+terms)
    # def image_and_derivatives(self):
    #     # self._call_on_changed()
    #     visibility = self.visibility_image
    #
    #     color = self.render_resolved
    #
    #     visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    #     num_visible = len(visible)
    #
    #     barycentric = self.barycentric_image
    #
    #     if self.updateRender:
    #         render, derivatives = self.compute_image_and_derivatives(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
    #         self.render = render
    #         self.derivatives = derivatives
    #         self.updateRender = False
    #
    #     return self.render, self.derivatives
    #

    def barycentricDerivatives(self, vertices, faces, verts):
        import chumpy as ch

        vertices = np.concatenate([vertices, np.ones([vertices.size // 3, 1])], axis=1)
        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])

        viewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])

        # # # Check with autodiff:
        # #
        # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        # # negYMat = ch.array([[1,0,self.camera.c.r[0]],[0,-1,self.camera.c.r[1]],[0,0,1]])
        # verts_hom_ch = ch.Ch(verts_hom)
        # camMtx = ch.Ch(np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])])
        # projVerts = (camMtx.dot(view_mtx)).dot(verts_hom_ch.T).T[:, :3].reshape([-1, 3])
        # viewVerts = ch.Ch(np.array(projVerts))
        # projVerts = projVerts[:, :2] / projVerts[:, 2:3]
        #
        # chViewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        # p0 = ch.Ch(viewVerticesNonBnd[:, 0, :])
        # chp0 = p0
        #
        # p1 = ch.Ch(viewVerticesNonBnd[:, 1, :])
        # chp1 = p1
        #
        # p2 = ch.Ch(viewVerticesNonBnd[:, 2, :])
        # chp2 = p2
        #
        # # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        # nt = ch.cross(p1 - p0, p2 - p0)
        # chnt = nt
        # A = 0.5 * ch.sqrt(ch.sum(nt ** 2, axis=1))
        # chnt_norm = nt / ch.sqrt(ch.sum(nt ** 2, axis=1))[:, None]
        # # nt = nt / A
        #
        # chb0part2 = ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1)
        # chb0 = 0.5 * ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1) / A
        # chb1part2 = ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1)
        # chb1 = 0.5 * ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1) / A
        # chb2part2 = ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1)
        # chb2 = 0.5 * ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1) / A
        #
        # drb0p0 = chb0.dr_wrt(p0)
        # drb0p1 = chb0.dr_wrt(p1)
        # drb0p2 = chb0.dr_wrt(p2)
        #
        # drb1p0 = chb1.dr_wrt(p0)
        # drb1p1 = chb1.dr_wrt(p1)
        # drb1p2 = chb1.dr_wrt(p2)
        #
        # drb2p0 = chb2.dr_wrt(p0)
        # drb2p1 = chb2.dr_wrt(p1)
        # drb2p2 = chb2.dr_wrt(p2)
        #
        # rows = np.tile(np.arange(drb0p0.shape[0])[None, :], [3, 1]).T.ravel()
        # cols = np.arange(drb0p0.shape[0] * 3)
        #
        # drb0p0 = np.array(drb0p0[rows, cols]).reshape([-1, 3])
        # drb0p1 = np.array(drb0p1[rows, cols]).reshape([-1, 3])
        # drb0p2 = np.array(drb0p2[rows, cols]).reshape([-1, 3])
        # drb1p0 = np.array(drb1p0[rows, cols]).reshape([-1, 3])
        # drb1p1 = np.array(drb1p1[rows, cols]).reshape([-1, 3])
        # drb1p2 = np.array(drb1p2[rows, cols]).reshape([-1, 3])
        # drb2p0 = np.array(drb2p0[rows, cols]).reshape([-1, 3])
        # drb2p1 = np.array(drb2p1[rows, cols]).reshape([-1, 3])
        # drb2p2 = np.array(drb2p2[rows, cols]).reshape([-1, 3])
        #
        # chdp0 = np.concatenate([drb0p0[:, None, :], drb1p0[:, None, :], drb2p0[:, None, :]], axis=1)
        # chdp1 = np.concatenate([drb0p1[:, None, :], drb1p1[:, None, :], drb2p1[:, None, :]], axis=1)
        # chdp2 = np.concatenate([drb0p2[:, None, :], drb1p2[:, None, :], drb2p2[:, None, :]], axis=1)
        #
        # dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)
        # dp = dp[None, :]

        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])
        viewVerts = projVerts
        projVerts = projVerts[:, :2] / projVerts[:, 2:3]

        # viewVerticesNonBnd = negYMat.dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        p0 = viewVerticesNonBnd[:, 0, :]
        p1 = viewVerticesNonBnd[:, 1, :]
        p2 = viewVerticesNonBnd[:, 2, :]

        p0_proj = p0[:, 0:2] / p0[:, 2:3]
        p1_proj = p1[:, 0:2] / p1[:, 2:3]
        p2_proj = p2[:, 0:2] / p2[:, 2:3]

        # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        nt = np.cross(p1 - p0, p2 - p0)
        nt_norm = nt / np.linalg.norm(nt, axis=1)[:, None]

        # a = -nt_norm[:, 0] / nt_norm[:, 2]
        # b = -nt_norm[:, 1] / nt_norm[:, 2]
        # c = np.sum(nt_norm * p0, 1) / nt_norm[:, 2]

        cam_f = 1

        u = p0[:, 0] / p0[:, 2]
        v = p0[:, 1] / p0[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p0[:, 2][:, None], np.zeros([len(p0), 1]), (-p0[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p0), 1]), p0[:, 2][:, None], (-p0[:, 1] / v ** 2)[:, None]]

        dxdp_0 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p1[:, 0] / p1[:, 2]
        v = p1[:, 1] / p1[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p1[:, 2][:, None], np.zeros([len(p1), 1]), (-p1[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p1), 1]), p1[:, 2][:, None], (-p1[:, 1] / v ** 2)[:, None]]

        dxdp_1 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p2[:, 0] / p2[:, 2]
        v = p2[:, 1] / p2[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p2[:, 2][:, None], np.zeros([len(p2), 1]), (-p2[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p2), 1]), p2[:, 2][:, None], (-p2[:, 1] / v ** 2)[:, None]]

        dxdp_2 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        # x = u * c / (cam_f - a * u - b * v)
        # y = v*c/(cam_f - a*u - b*v)
        # z = c*cam_f/(cam_f - a*u - b*v)

        A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        nt_mag = A * 2
        # nt = nt / A
        # db1 = 0.5*np.cross(nt_norm, p2-p1)/A[:, None]
        # db2 = 0.5*np.cross(nt_norm, p0-p2)/A[:, None]
        # db3_2 = 0.5*np.cross(nt_norm, p1-p0)/A[:, None]
        # db3 = - db1 - db2
        p = viewVerts

        pre1 = -1 / (nt_mag[:, None] ** 2) * nt_norm

        ident = np.identity(3)
        ident = np.tile(ident[None, :], [len(p2), 1, 1])

        dntdp0 = np.cross((p2 - p0)[:, None, :], -ident) + np.cross(-ident, (p1 - p0)[:, None, :])
        dntdp1 = np.cross((p2 - p0)[:, None, :], ident)
        dntdp2 = np.cross(ident, (p1 - p0)[:, None, :])

        # Pol check this!:
        dntnorm = (ident - np.einsum('ij,ik->ijk', nt_norm, nt_norm)) / nt_mag[:, None, None]
        # dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]

        dntnormdp0 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp0)
        dntnormdp1 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp1)
        dntnormdp2 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp2)

        dpart1p0 = np.einsum('ij,ijk->ik', pre1, dntdp0)
        dpart1p1 = np.einsum('ij,ijk->ik', pre1, dntdp1)
        dpart1p2 = np.einsum('ij,ijk->ik', pre1, dntdp2)

        b0 = np.sum(np.cross(nt_norm, p2 - p1) * (p - p1), axis=1)[:, None]

        db0part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident,nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1),-ident)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0.swapaxes(1,2)), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident, nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(p2-p1,nt_norm[:, :]),-ident)
        db0part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2 - p1), -ident)
        db0part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p1)

        db0dp0wrtpart1 = dpart1p0 * b0
        db0dp1wrtpart1 = dpart1p1 * b0
        db0dp2wrtpart1 = dpart1p2 * b0

        db0dp0wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p0
        db0dp1wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p1
        db0dp2wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p2

        db0dp0wrt = db0dp0wrtpart1 + db0dp0wrtpart2
        db0dp1wrt = db0dp1wrtpart1 + db0dp1wrtpart2
        db0dp2wrt = db0dp2wrtpart1 + db0dp2wrtpart2

        ######
        b1 = np.sum(np.cross(nt_norm, p0 - p2) * (p - p2), axis=1)[:, None]

        db1part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p2)
        db1part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2)
        db1part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p2) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p0 - p2), -ident)

        db1dp0wrtpart1 = dpart1p0 * b1
        db1dp1wrtpart1 = dpart1p1 * b1
        db1dp2wrtpart1 = dpart1p2 * b1

        db1dp0wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p0
        db1dp1wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p1
        db1dp2wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p2

        db1dp0wrt = db1dp0wrtpart1 + db1dp0wrtpart2
        db1dp1wrt = db1dp1wrtpart1 + db1dp1wrtpart2
        db1dp2wrt = db1dp2wrtpart1 + db1dp2wrtpart2

        ######
        b2 = np.sum(np.cross(nt_norm, p1 - p0) * (p - p0), axis=1)[:, None]

        db2part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p0) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p1 - p0), -ident)
        db2part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p0)
        db2part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0)

        db2dp0wrtpart1 = dpart1p0 * b2
        db2dp1wrtpart1 = dpart1p1 * b2
        db2dp2wrtpart1 = dpart1p2 * b2

        db2dp0wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p0
        db2dp1wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p1
        db2dp2wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p2

        db2dp0wrt = db2dp0wrtpart1 + db2dp0wrtpart2
        db2dp1wrt = db2dp1wrtpart1 + db2dp1wrtpart2
        db2dp2wrt = db2dp2wrtpart1 + db2dp2wrtpart2

        dp0 = np.concatenate([db0dp0wrt[:, None, :], db1dp0wrt[:, None, :], db2dp0wrt[:, None, :]], axis=1)
        dp1 = np.concatenate([db0dp1wrt[:, None, :], db1dp1wrt[:, None, :], db2dp1wrt[:, None, :]], axis=1)
        dp2 = np.concatenate([db0dp2wrt[:, None, :], db1dp2wrt[:, None, :], db2dp2wrt[:, None, :]], axis=1)
        #
        dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)

        # If dealing with degenerate triangles, ignore that gradient.

        # dp[nt_mag <= 1e-15] = 0

        dp = dp[None, :]

        nFaces = len(faces)
        # visTriVC = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc[vc > 1] = 1
        vc[vc < 0] = 0

        visTriVC = vc

        dxdp = np.concatenate([dxdp_0[:, None, :], dxdp_1[:, None, :], dxdp_2[:, None, :]], axis=1)

        dxdp = dxdp[None, :, None]
        # dbvc = np.sum(dp * visTriVC, 2)

        # dbvc = dp * visTriVC * t_area[None, :, None, None, None]
        dbvc = dp * visTriVC

        didp = np.sum(dbvc[:, :, :, :, :, None] * dxdp, 4).sum(2)

        # output should be shape: VC x Ninput x Tri Points x UV

        # drb0p0 # db0dp0wrt
        # drb0p1 # db0dp1wrt
        # drb0p2 # db0dp2wrt
        # drb1p0 # db1dp0wrt
        # drb1p1 # db1dp1wrt
        # drb1p2 # db1dp2wrt
        # drb2p0 # db2dp0wrt
        # drb2p1 # db2dp1wrt
        # drb2p2 # db2dp2wrt

        return didp

    def compute_image(self, visible, visibility, f):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        nsamples = self.nsamples

        if np.any(boundaryImage):
            sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
                [nsamples, -1, 2])

            # sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:,(zerosIm*boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])
            sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

            boundaryFaces = visibility[(boundaryImage) & (visibility != 4294967295)]
            nBndFaces = len(boundaryFaces)

            vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1, 1, 1])
            sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

            # if self.debug:
            #     import pdb; pdb.set_trace()

            faces = f[sampleFaces].ravel()
            vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            nv = len(vertsPerFaceProjBnd)
            p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
            p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
            p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
            t_area_bnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)
            t_area_bnd[t_area_bnd > 1] = 1

            # Trick to cap to 1 while keeping gradients.

            p1 = vertsProjBndSamples.reshape([-1,2,2])[:, 0, :]
            p2 = vertsProjBndSamples.reshape([-1,2,2])[:, 1, :]

            p = sampleV.reshape([-1,2])

            l = (p2 - p1)
            linedist = np.sqrt((np.sum(l ** 2, axis=1)))[:, None]
            self.linedist = linedist

            lnorm = l / linedist
            self.lnorm = lnorm

            v1 = p - p1
            self.v1 = v1
            d = v1[:, 0] * lnorm[:, 0] + v1[:, 1] * lnorm[:, 1]
            self.d = d
            intersectPoint = p1 + d[:, None] * lnorm

            v2 = p - p2
            self.v2 = v2
            l12 = (p1 - p2)
            linedist12 = np.sqrt((np.sum(l12 ** 2, axis=1)))[:, None]
            lnorm12 = l12 / linedist12
            d2 = v2[:, 0] * lnorm12[:, 0] + v2[:, 1] * lnorm12[:, 1]

            nonIntersect = (d2 < 0) | (d < 0)
            self.nonIntersect = nonIntersect

            argminDistNonIntersect = np.argmin(np.c_[d[nonIntersect], d2[nonIntersect]], 1)
            self.argminDistNonIntersect = argminDistNonIntersect

            intersectPoint[nonIntersect] = vertsProjBndSamples.reshape([-1,2,2])[nonIntersect][np.arange(nonIntersect.sum()), argminDistNonIntersect]

            lineToPoint = (p - intersectPoint)

            n = lineToPoint

            dist = np.sqrt((np.sum(lineToPoint ** 2, axis=1)))[:, None]

            n_norm = lineToPoint / dist

            self.n_norm = n_norm

            self.dist = dist

            d_final = dist.squeeze()

            # max_nx_ny = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))

            # d_final = d_final / max_nx_ny
            d_final = d_final

            # invViewMtx = np.linalg.inv(np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])])
            # #
            # camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
            # # invCamMtx = np.r_[np.c_[np.linalg.inv(self.camera.camera_mtx), np.array([0,0,0])], np.array([[0, 0, 0, 1]])]
            #
            # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]

            # verticesBndSamples = np.concatenate([verticesBndSamples.reshape([-1, 3]), np.ones([verticesBndSamples.size // 3, 1])], axis=1)

            # projVerticesBndOutside = (camMtx.dot(view_mtx)).dot(verticesBndSamples.T).T[:, :3].reshape([-1, 2, 3])
            # projVerticesBndDir = projVerticesBndOutside[:, 1, :] - projVerticesBndOutside[:, 0, :]
            # projVerticesBndDir = projVerticesBndDir / np.sqrt((np.sum(projVerticesBndDir ** 2, 1)))[:, None]

            # dproj = (intersectPoint[:, 0] * projVerticesBndOutside[:, 0, 2] - projVerticesBndOutside[:, 0, 0]) / (projVerticesBndDir[:, 0] - projVerticesBndDir[:, 2] * intersectPoint[:, 0])
            # # Code to check computation that dproj == dprojy
            # # dproj_y = (intersectPoint[:,1]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,1]) / (projVerticesBndDir[:,1] - projVerticesBndDir[:,2]*intersectPoint[:,1])
            #
            # projPoint = projVerticesBndOutside[:, 0, :][:, :] + dproj[:, None] * projVerticesBndDir[:, :]
            #
            # projPointVec4 = np.concatenate([projPoint, np.ones([projPoint.shape[0], 1])], axis=1)
            # viewPointIntersect = (invViewMtx.dot(np.linalg.inv(camMtx)).dot(projPointVec4.T.reshape([4, -1])).reshape([4, -1])).T[:, :3]
            #
            # barycentricVertsDistIntesect = np.linalg.norm(viewPointIntersect - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 0, :], axis=1)
            # barycentricVertsDistIntesect2 = np.linalg.norm(viewPointIntersect - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 1, :], axis=1)
            # # Code to check barycentricVertsDistIntesect + barycentricVertsDistIntesect2 = barycentricVertsDistEdge
            # barycentricVertsDistEdge = np.linalg.norm(
            #     verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 0, :] - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 1, :], axis=1)
            #
            # nonIntersect = np.abs(barycentricVertsDistIntesect + barycentricVertsDistIntesect2 - barycentricVertsDistEdge) > 1e-4
            # argminDistNonIntersect = np.argmin(np.c_[barycentricVertsDistIntesect[nonIntersect], barycentricVertsDistIntesect2[nonIntersect]], 1)
            #
            # self.viewPointIntersect = viewPointIntersect
            # self.viewPointIntersect[nonIntersect] = verticesBndSamples.reshape([-1, 2, 4])[nonIntersect, :, 0:3][np.arange(nonIntersect.sum()),
            #                                         argminDistNonIntersect, :]

            d_finalNP = d_final.copy()
            self.d_final = d_finalNP

            self.t_area_bnd = t_area_bnd

            areaWeights = np.zeros([nsamples, nBndFaces])
            areaWeights = t_area_bnd.reshape([nsamples, nBndFaces])
            areaWeightsTotal = areaWeights.sum(0)
            # areaWeightsTotal[areaWeightsTotal < 1] = 1
            self.areaWeights = areaWeights
            self.areaWeightsTotal = areaWeightsTotal

            finalColorBnd = np.ones([self.nsamples, boundaryFaces.size, 3])

            self.d_final_total = d_finalNP.reshape([self.nsamples, -1,1]).sum(0)

            # if self.imageGT is not None:
            finalColorBnd = sampleColors * d_finalNP.reshape([self.nsamples, -1,1]) / (self.d_final_total.reshape([1, -1,1]))
            # finalColorBnd = areaWeights[:,:,None] * sampleColors * d_finalNP.reshape([self.nsamples, -1,1]) / (self.d_final_total.reshape([1, -1,1]) * areaWeightsTotal[None,:,None])
            self.finalColorBnd = finalColorBnd
            # else:
            #     finalColorBnd = sampleColors

            bndColorsImage = np.zeros_like(self.color_image)
            bndColorsImage[(zerosIm * boundaryImage), :] = np.sum(finalColorBnd, axis=0)

            finalColorImageBnd = bndColorsImage

            if self.imageGT is not None:
                bndColorsResiduals = np.zeros_like(self.color_image)
                self.sampleResiduals = (sampleColors - self.imageGT.r[(zerosIm * boundaryImage),:][None,:])
                self.sampleResidualsWeighted = self.sampleResiduals**2 * d_finalNP.reshape([self.nsamples, -1,1]) / self.d_final_total.reshape([1, -1,1])

                bndColorsResiduals[(zerosIm * boundaryImage), :] = np.sum(self.sampleResidualsWeighted,0)

        if np.any(boundaryImage):
            finalColor = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * finalColorImageBnd

            if self.imageGT is not None:
                self.residuals = (self.color_image - self.imageGT.r)
                errors = self.residuals**2
                finalResidual = (1 - boundaryImage)[:, :, None] * errors + boundaryImage[:, :, None] * bndColorsResiduals
        else:
            finalColor = self.color_image

            if self.imageGT is not None:
                finalResidual = (self.color_image - self.imageGT.r)**2

        if self.imageGT is None:
            finalResidual = None

        finalColor[finalColor > 1] = 1
        finalColor[finalColor < 0] = 0

        return finalColor, finalResidual

    def compute_derivatives_verts(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        # xdiff = dEdx
        # ydiff = dEdy

        nVisF = len(visibility.ravel()[visible])
        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples
        sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
            [nsamples, -1, 2])

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage) & (visibility != 4294967295)]

        if np.any(boundaryImage):

            n_norm = self.n_norm
            dist = self.dist
            linedist = self.linedist
            d = self.d
            v1 = self.v1
            lnorm = self.lnorm

            d_final = self.d_final

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1, 1, 1])

            # Computing gradients:
            # A multisampled pixel color is given by: w R + (1-w) R' thus:
            # 1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            # 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            # 3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            # 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            # 5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            # 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            # 7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            # for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            # Finally, stack data, and IJ of nonbnd with bnd on both dwrt_v and dwrt_vc.

            ######## 1 derivatives samples outside wrt v 1: (dw * (bar*vc) - dw (bar'*vc') )/ nsamples for face sample

            # # #Chumpy autodiff code to check derivatives here:
            # chEdgeVerts = ch.Ch(vertsProjBndSamples.reshape([-1,2,2]))
            #
            # chEdgeVerts1 = chEdgeVerts[:,0,:]
            # chEdgeVerts2 = chEdgeVerts[:,1,:]
            #
            # chSampleVerts = ch.Ch(sampleV.reshape([-1,2]))
            # # c1 = (chEdgeVerts1 - chSampleVerts)
            # # c2 = (chEdgeVerts2 - chSampleVerts)
            # # n = (chEdgeVerts2 - chEdgeVerts1)
            #
            # #Code to check computation of distance below
            # # d2 = ch.abs(c1[:,:,0]*c2[:,:,1] - c1[:,:,1]*c2[:,:,0]) / ch.sqrt((ch.sum(n**2,2)))
            # # # np_mat = ch.dot(ch.array([[0,-1],[1,0]]), n)
            # # np_mat2 = -ch.concatenate([-n[:,:,1][:,:,None], n[:,:,0][:,:,None]],2)
            # # np_vec2 = np_mat2 / ch.sqrt((ch.sum(np_mat2**2,2)))[:,:,None]
            # # d2 =  d2 / ch.maximum(ch.abs(np_vec2[:,:,0]),ch.abs(np_vec2[:,:,1]))
            #
            # chl = (chEdgeVerts2 - chEdgeVerts1)
            # chlinedist = ch.sqrt((ch.sum(chl**2,axis=1)))[:,None]
            # chlnorm = chl/chlinedist
            #
            # chv1 = chSampleVerts - chEdgeVerts1
            #
            # chd = chv1[:,0]* chlnorm[:,0] + chv1[:,1]* chlnorm[:,1]
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            # # intersectPointDist1 = intersectPoint - chEdgeVerts1
            # # intersectPointDist2 = intersectPoint - chEdgeVerts2
            # # Code to check computation of distances below:
            # # lengthIntersectToPoint1 = np.linalg.norm(intersectPointDist1.r,axis=1)
            # # lengthIntersectToPoint2 = np.linalg.norm(intersectPointDist2.r,axis=1)
            #
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            #
            # chlineToPoint = (chSampleVerts - chintersectPoint)
            # chn_norm = chlineToPoint / ch.sqrt((ch.sum(chlineToPoint ** 2, axis=1)))[:, None]
            #
            # chdist = chlineToPoint[:,0]*chn_norm[:,0] + chlineToPoint[:,1]*chn_norm[:,1]
            #
            # # d_final_ch = chdist / ch.maximum(ch.abs(chn_norm[:, 0]), ch.abs(chn_norm[:, 1]))
            # d_final_ch = chdist
            #
            # d_final_ch_weights = sampleColors * (d_final_ch.reshape([self.nsamples, -1]) / ch.sum(d_final_ch.reshape([self.nsamples, -1]), 0))[:,:,None]
            #
            # d_final_outside = d_final_ch.ravel()
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts1)
            # rows = np.tile(np.arange(d_final_outside.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_outside.shape[0] * 2)
            #
            # dwdv_r_v1 = np.array(dwdv[rows, cols]).reshape([-1, 2])
            #
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts2)
            # rows = np.tile(np.arange(d_final_ch.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_ch.shape[0] * 2)
            #
            # dwdv_r_v2 = np.array(dwdv[rows, cols]).reshape([-1, 2])


            nonIntersect = self.nonIntersect
            argminDistNonIntersect = self.argminDistNonIntersect

            # max_dx_dy = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))
            d_final_np = dist
            # d_final_np = dist / max_dx_dy

            ident = np.identity(2)
            ident = np.tile(ident[None, :], [len(d_final_np), 1, 1])

            dlnorm = (ident - np.einsum('ij,ik->ijk', lnorm, lnorm)) / linedist[:, None]
            dl_normdp1 = np.einsum('ijk,ikl->ijl', dlnorm, -ident)
            dl_normdp2 = np.einsum('ijk,ikl->ijl', dlnorm, ident)

            dv1dp1 = -ident
            dv1dp2 = 0

            dddp1 = np.einsum('ijk,ij->ik', dv1dp1, lnorm) + np.einsum('ij,ijl->il', v1, dl_normdp1)
            dddp2 = 0 + np.einsum('ij,ijl->il', v1, dl_normdp2)

            dipdp1 = ident + (dddp1[:, None, :] * lnorm[:, :, None]) + d[:, None, None] * dl_normdp1
            dipdp2 = (dddp2[:, None, :] * lnorm[:, :, None]) + d[:, None, None] * dl_normdp2

            #good up to here.

            dndp1 = -dipdp1
            dndp2 = -dipdp2

            dn_norm = (ident - np.einsum('ij,ik->ijk', n_norm, n_norm)) / dist[:, None]

            # dn_normdp1 = np.einsum('ijk,ikl->ijl', dn_norm, dndp1)
            # dn_normdp2 = np.einsum('ijk,ikl->ijl', dn_norm, dndp2)

            ddistdp1 = np.einsum('ij,ijl->il', n_norm, dndp1)
            ddistdp2 = np.einsum('ij,ijl->il', n_norm, dndp2)

            # argmax_nx_ny = np.argmax(np.abs(n_norm), axis=1)
            # dmax_nx_ny_p1 = np.sign(n_norm)[np.arange(len(n_norm)), argmax_nx_ny][:, None] * dn_normdp1[np.arange(len(dn_normdp1)), argmax_nx_ny]
            # dmax_nx_ny_p2 = np.sign(n_norm)[np.arange(len(n_norm)), argmax_nx_ny][:, None] * dn_normdp2[np.arange(len(dn_normdp2)), argmax_nx_ny]

            # dd_final_dp1 = -1. / max_dx_dy[:, None] ** 2 * dmax_nx_ny_p1 * dist + 1. / max_dx_dy[:, None] * ddistdp1
            # dd_final_dp2 = -1. / max_dx_dy[:, None] ** 2 * dmax_nx_ny_p2 * dist + 1. / max_dx_dy[:, None] * ddistdp2

            dd_final_dp1 = ddistdp1
            dd_final_dp2 = ddistdp2

            # For those non intersecting points straight to the edge:

            v1 = self.v1[nonIntersect][argminDistNonIntersect == 0]
            v1_norm = v1 / np.sqrt((np.sum(v1 ** 2, axis=1)))[:, None]

            dd_final_dp1_nonintersect = -v1_norm

            v2 = self.v2[nonIntersect][argminDistNonIntersect == 1]
            v2_norm = v2 / np.sqrt((np.sum(v2 ** 2, axis=1)))[:, None]
            dd_final_dp2_nonintersect = -v2_norm

            dd_final_dp1[nonIntersect][argminDistNonIntersect == 0] = dd_final_dp1_nonintersect
            dd_final_dp1[nonIntersect][argminDistNonIntersect == 1] = 0
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 1] = dd_final_dp2_nonintersect
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 0] = 0

            dd_final_dp1_weighted_part1 = -self.d_final[:,None]* np.tile(dd_final_dp1.reshape([self.nsamples, -1, 2]).sum(0)[None,:,:],[self.nsamples,1,1]).reshape([-1, 2])/(np.tile(self.d_final_total[None,:], [self.nsamples, 1,1]).reshape([-1,1])**2)
            dd_final_dp1_weighted_part2 = dd_final_dp1 / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1])
            dd_final_dp1_weighted =  dd_final_dp1_weighted_part1 + dd_final_dp1_weighted_part2

            dd_final_dp2_weighted_part1 = -self.d_final[:,None]*np.tile(dd_final_dp2.reshape([self.nsamples, -1, 2]).sum(0)[None,:,:],[self.nsamples,1,1]).reshape([-1, 2])/(np.tile(self.d_final_total[None,:], [self.nsamples, 1,1]).reshape([-1,1])**2)
            dd_final_dp2_weighted_part2 = dd_final_dp2 / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1])
            dd_final_dp2_weighted =  dd_final_dp2_weighted_part1 + dd_final_dp2_weighted_part2

            if self.imageGT is None:
                dImage_wrt_outside_v1 = sampleColors.reshape([-1,3,1]) * dd_final_dp1_weighted[:, None, :]
                dImage_wrt_outside_v2 = sampleColors.reshape([-1,3,1]) * dd_final_dp2_weighted[:, None, :]
            else:
                dImage_wrt_outside_v1 = self.sampleResiduals.reshape([-1,3,1])**2 * dd_final_dp1_weighted[:, None, :]
                dImage_wrt_outside_v2 = self.sampleResiduals.reshape([-1,3,1])**2 * dd_final_dp2_weighted[:, None, :]

            # sampleV
            # z = dd_final_dp1.reshape([8, -1, 2])
            # eq = np.array([np.all(np.sign(z[:, i, :]) == -1) or np.all(np.sign(z[:, i, :]) == 1) for i in range(z.shape[1])])
            # dist_ns = dist.reshape([8,-1])
            # rightV = sampleV[0, :, 0] > np.max(sampleV[0, :, :], 0)[0] - 1
            # dist_ns[0, rightV]
            # dImage_wrt_outside_v1.reshape([8, -1, 3, 2])[0, rightV,:]
            # d_final_ch_weights
            # self.finalColorBnd

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 2 * 2)).ravel()

            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1, -1, 2]), [self.nsamples, 1, 1]).ravel()
            JS = col(faces)
            JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data1 = dImage_wrt_outside_v1.transpose([1, 0, 2])
            data2 = dImage_wrt_outside_v2.transpose([1, 0, 2])

            data = np.concatenate([data1[:, :, None, :], data2[:, :, None, :]], 2)

            data = data.ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))

            ######## 2 derivatives samples wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample

            verticesBnd = self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3])

            sampleBarycentricBar = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),
                                   :].reshape([-1, 3, 1])
            verts = np.sum(self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3, 3]) * sampleBarycentricBar, axis=1)

            dImage_wrt_bar_v = self.barycentricDerivatives(verticesBnd, f[sampleFaces.ravel()], verts).swapaxes(0, 1)

            if self.imageGT is None:
                # dImage_wrt_bar_v = dImage_wrt_bar_v * d_final[:, None, None, None] * self.t_area_bnd[:, None, None, None] / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                dImage_wrt_bar_v = dImage_wrt_bar_v * d_final[:, None, None, None]  / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # areaTotal = np.tile(self.areaWeightsTotal[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # d_final_total = np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # dImage_wrt_bar_v = self.areaWeights.reshape([-1,1,1,1]) * dImage_wrt_bar_v * d_final[:, None, None, None] / (areaTotal*d_final_total)
            else:
                dImage_wrt_bar_v = 2*self.sampleResiduals.reshape([-1,3])[:,:,None,None] * dImage_wrt_bar_v * d_final[:, None, None, None] * self.t_area_bnd[:, None, None, None] / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])

            ### Derivatives wrt V: 2 derivatives samples wrt v bar: (w * (dbar*vc) )/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 2 * f.shape[1])).ravel()
            faces = f[sampleFaces].ravel()
            JS = col(faces)
            JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v, [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd_bar = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        vertsPerFaceProjBnd = self.camera.r[f[nonBoundaryFaces].ravel()].reshape([-1, 3, 2])
        nv = len(vertsPerFaceProjBnd)

        p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
        p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
        p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
        t_area_nonbnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)

        t_area_nonbnd[t_area_nonbnd > 1] = 1

        bc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))

        verts = np.sum(self.v.r[f[nonBoundaryFaces.ravel()].ravel()].reshape([-1, 3, 3]) * bc[:, :, None], axis=1)

        didp = self.barycentricDerivatives(verticesNonBnd, f[nonBoundaryFaces.ravel()], verts)

        if self.imageGT is None:
            # didp = didp * t_area_nonbnd[None, :, None, None]
            didp = didp
        else:
            didp = 2 * self.residuals[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3)).T[:,:,None,None] * didp * t_area_nonbnd[None, :, None, None]

        n_channels = np.atleast_3d(observed).shape[2]

        ####### 2: Take the data and copy the corresponding dxs and dys to these new pixels.

        ### Derivatives wrt V:
        pixels = np.where(((~boundaryImage) & (visibility != 4294967295)).ravel())[0]
        IS = np.tile(col(pixels), (1, 2 * f.shape[1])).ravel()
        JS = col(f[nonBoundaryFaces].ravel())
        JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

        if n_channels > 1:
            IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        data = didp.ravel()

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result_wrt_verts_nonbnd = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))

        if np.any(boundaryImage):

            result_wrt_verts = result_wrt_verts_bnd + result_wrt_verts_bnd_bar + result_wrt_verts_nonbnd
        else:
            result_wrt_verts = result_wrt_verts_nonbnd

        return result_wrt_verts

    def compute_derivatives_vc(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        d_final = self.d_final

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        nsamples = self.nsamples

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),
                            :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage) & (visibility != 4294967295)]

        if np.any(boundaryImage):
            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)

            # Computing gradients:
            # A multisampled pixel color is given by: w R + (1-w) R' thus:
            # 1 derivatives samples wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            # 2 derivatives samples wrt v bar: (w * (dbar*vc) )/ nsamples for faces sample
            # 4 derivatives samples wrt vc : (w * (bar) )/ nsamples for faces sample

            # for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            ####### 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample

            if self.imageGT is None:
                dImage_wrt_bnd_vc = d_final[:, None] * sampleBarycentric.reshape([-1,3]) / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1,1])
            else:
                dImage_wrt_bnd_vc = d_final[:, None] * sampleBarycentric.reshape([-1,3]) / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1,1])
                dImage_wrt_bnd_vc = 2 * self.sampleResiduals.reshape([-1,3]).T[:,:,None] * dImage_wrt_bnd_vc[None,:]

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 3)).ravel()

            faces = f[sampleFaces].ravel()
            JS = col(faces)

            data = dImage_wrt_bnd_vc.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

            if self.imageGT is None:
                data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

            result_wrt_vc_bnd = result


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        ### Derivatives wrt VC:

        # Each pixel relies on three verts
        pixels = np.where(((~boundaryImage) & (visibility != 4294967295)).ravel())[0]
        IS = np.tile(col(pixels), (1, 3)).ravel()
        JS = col(f[nonBoundaryFaces].ravel())

        if self.imageGT is None:
            dImage_wrt_nonbnd_vc  = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
        else:
            dImage_wrt_nonbnd_vc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
            dImage_wrt_nonbnd_vc = 2* self.residuals[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3)).T[:,:,None] * dImage_wrt_nonbnd_vc[None,:]


        data = np.asarray(dImage_wrt_nonbnd_vc, order='C').ravel()

        IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
        JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

        if self.imageGT is None:
            data = np.concatenate([data for i in range(num_channels)])

        ij = np.vstack((IS.ravel(), JS.ravel()))
        result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

        result_wrt_vc_nonbnd = result

        if np.any(boundaryImage):
            result_wrt_vc = result_wrt_vc_bnd + result_wrt_vc_nonbnd
        else:
            result_wrt_vc = result_wrt_vc_nonbnd

        return result_wrt_vc

    def on_changed(self, which):
        super().on_changed(which)

        if 'v' or 'camera' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_verts_mesh[mesh][polygons].set_array(verts_by_face.astype(np.float32))
                    self.vbo_verts_mesh[mesh][polygons].bind()

        if 'vc' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                    self.vbo_colors_mesh[mesh][polygons].bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()
            flen = 1
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]

                    # fc = np.arange(flen, flen + len(f))
                    fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()

                    # fc[:, 0] = fc[:, 0] & 255
                    # fc[:, 1] = (fc[:, 1] >> 8) & 255
                    # fc[:, 2] = (fc[:, 2] >> 16) & 255
                    fc = np.asarray(fc, dtype=np.uint32)
                    self.vbo_face_ids_list[mesh][polygons].set_array(fc)
                    self.vbo_face_ids_list[mesh][polygons].bind()

                    flen += len(f)

                    self.vbo_indices_mesh_list[mesh][polygons].set_array(np.array(self.f_list[mesh][polygons]).astype(np.uint32))
                    self.vbo_indices_mesh_list[mesh][polygons].bind()

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

                            # Update the OpenGL textures with all the textures. (Inefficient as many might not have changed).
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)
                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size + textureCoordIdx].reshape(image.shape)

                            textureCoordIdx = textureCoordIdx + image.size
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)

                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                                               image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())

        # if 'imageGT' in which:
        #     GL.glActiveTexture(GL.GL_TEXTURE1)
        #     GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        #     image = np.array(np.flipud((self.imageGT.r)), order='C', dtype=np.float32)
        #     # GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA, image.shape[1], image.shape[0])
        #     GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

        if 'v' or 'f' or 'vc' or 'ft' or 'camera' or 'texture_stack' or 'imageGT' in which:
            self.render_image_buffers()

    def release_textures(self):
        if hasattr(self, 'textureID_mesh_list'):
            if self.textureID_mesh_list != []:
                for texture_mesh in self.textureID_mesh_list:
                    if texture_mesh != []:
                        for texture in texture_mesh:
                            if texture != None:
                                GL.glDeleteTextures(1, [texture.value])

        self.textureID_mesh_list = []

    @depends_on(dterms + terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        return no_overdraw

        # if not self.overdraw:
        #     return no_overdraw
        #
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        # overdraw = self.draw_color_image()
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #
        # # return overdraw * np.atleast_3d(self.boundarybool_image)
        #
        # boundarybool_image = self.boundarybool_image
        # if self.num_channels > 1:
        #     boundarybool_image = np.atleast_3d(boundarybool_image)
        #
        # return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_barycentric_image(self.boundarybool_image if self.overdraw else None)

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_visibility_image(self.v.r, self.f, self.boundarybool_image if self.overdraw else None)

    def image_mesh_bool(self, meshes):
        self.makeCurrentContext()
        self._call_on_changed()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0., 0., 0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)
        for mesh in meshes:
            self.draw_index(mesh)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.uint32))[:, :, 0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result != 0

    @depends_on(dterms + terms)
    def indices_image(self):
        self._call_on_changed()
        self.makeCurrentContext()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0., 0., 0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)

        for index in range(len(self.f_list)):
            self.draw_index(index)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.uint32))[:, :, 0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result

    def draw_index(self, index):

        mesh = index

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        vc = self.vc_list[mesh]

        for polygons in np.arange(len(self.f_list[mesh])):
            vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
            GL.glBindVertexArray(vao_mesh)
            f = self.f_list[mesh][polygons]
            vbo_color = self.vbo_colors_mesh[mesh][polygons]
            colors_by_face = np.asarray(vc.reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
            colors = np.array(np.ones_like(colors_by_face) * (index) / 255.0, dtype=np.float32)

            # Pol: Make a static zero vbo_color to make it more efficient?
            vbo_color.set_array(colors)

            vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

            vbo_color.bind()

            if self.f.shape[1] == 2:
                primtype = GL.GL_LINES
            else:
                primtype = GL.GL_TRIANGLES

            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

            GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])

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

        # use the third channel to identify the corresponding textures.
        color3 = np.vstack([np.ones([self.ft_list[mesh].shape[0], 1]) * mesh for mesh in range(len(self.ft_list))]).astype(np.float32) / len(
            self.ft_list)

        colors = np.asarray(np.hstack((colors, color3)), np.float64, order='C')
        self.draw_colored_primitives(self.vao_dyn, v, f, colors)

        # Why do we need this?
        if boundarybool_image is not None:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            self.draw_colored_primitives(self.vao_dyn, v, f, colors)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3)[:, :, :3].astype(np.float64)) / 255.0

        result[:, :, 1] = 1. - result[:, :, 1]
        return result

    @depends_on('ft', 'textures')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = self.ft
        # Pol: careful with this:
        data[:, 1] = 1.0 - 1.0 * data[:, 1]
        return data

    # Depends on 'f' because vpe/fpe depend on f
    # Pol: Check that depends on works on other attributes that depend_on x, if x changes.
    @depends_on('ft', 'f')
    def wireframe_tex_coords(self):
        print("wireframe_tex_coords is being computed!")
        vvt = np.zeros((self.v.r.size / 3, 2), dtype=np.float64, order='C')
        vvt[self.f.flatten()] = self.mesh_tex_coords
        edata = np.zeros((self.vpe.size, 2), dtype=np.float64, order='C')
        edata = vvt[self.ma.ravel()]
        return edata

    # TODO: can this not be inherited from base? turning off texture mapping in that instead?
    @depends_on(dterms + terms)
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
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels], self.bgcolor.r[2 % self.num_channels], 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.msaa:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_noms)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_color = self.vbo_colors_mesh[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vc = colors_by_face

                if with_vertex_colors:
                    colors = vc.astype(np.float32)
                else:
                    # Only texture.
                    colors = np.ones_like(vc).astype(np.float32)

                # Pol: Make a static zero vbo_color to make it more efficient?
                vbo_color.set_array(colors)
                vbo_color.bind()

                if self.f.shape[1] == 2:
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

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])
                # GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

        if self.msaa:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_noms)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.float64)) / 255.0

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0., 0., 0., 1.)

        if hasattr(self, 'background_image'):
            bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1, 1, 3))
            fg_px = 1 - bg_px
            result = bg_px * self.background_image + fg_px * result

        return result

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image_quantized(self):

        texcoord_image = self.texcoord_image[:, :, :2].copy()
        # Temprary:
        self.texture_image = self.textures_list[0][0].r.copy()
        texcoord_image[:, :, 0] *= self.texture_image.shape[1] - 1
        texcoord_image[:, :, 1] *= self.texture_image.shape[0] - 1
        texture_idx = (self.texcoord_image[:, :, 2] * len(self.ft_list)).astype(np.uint32)
        texcoord_image = np.round(texcoord_image)
        texcoord_image = texcoord_image[:, :, 0] + texcoord_image[:, :, 1] * self.texture_image.shape[1]

        return texcoord_image, texture_idx

    def checkBufferNum(self):
        GL.glGenBuffers(1)

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image(self):
        return self.draw_texcoord_image(self.v.r, self.f, self.ft, self.boundarybool_image if self.overdraw else None)

class ResidualRendererOpenDR(ColoredRenderer):
    terms = 'f', 'frustum', 'vt', 'ft', 'background_image', 'overdraw', 'ft_list', 'haveUVs_list', 'textures_list', 'vc_list', 'imageGT'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_stack', 'v'

    def __init__(self):
        super().__init__()

    def clear(self):
        try:
            GL.glFlush()
            GL.glFinish()
            # print ("Clearing textured renderer.")
            # for msh in self.vbo_indices_mesh_list:
            #     for vbo in msh:
            #         vbo.set_array([])
            [vbo.set_array(np.array([])) for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_indices_mesh_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_indices_mesh_list for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_colors_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_colors_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_verts_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_verts_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_uvs_mesh for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_uvs_mesh for vbo in sublist]

            [vbo.set_array(np.array([])) for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.bind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.unbind() for sublist in self.vbo_face_ids_list for vbo in sublist]
            [vbo.delete() for sublist in self.vbo_face_ids_list for vbo in sublist]

            [GL.glDeleteVertexArrays(1, [vao.value]) for sublist in self.vao_tex_mesh_list for vao in sublist]

            self.release_textures()

            if self.glMode == 'glfw':
                import glfw
                glfw.make_context_current(self.win)

            GL.glDeleteProgram(self.colorTextureProgram)

            super().clear()
        except:
            import pdb
            pdb.set_trace()
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
            color = theColor * texture( myTextureSampler, UV).rgb;
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

        self.colorTextureProgram = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

        # Define the other VAO/VBOs and shaders.
        # Text VAO and bind color, vertex indices AND uvbuffer:

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

        # GL.glEnable(GL.GL_LINE_SMOOTH)
        # GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(2.)
        self.line_width = 2.

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_indices_mesh = []
            vbo_face_ids_mesh = []
            vbo_colors_mesh = []
            vbo_vertices_mesh = []
            vbo_uvs_mesh = []
            textureIDs_mesh = []
            for polygons in range(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_verts = vbo.VBO(np.array(verts_by_face).astype(np.float32))
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_colors = vbo.VBO(np.array(colors_by_face).astype(np.float32))
                uvs_by_face = np.asarray(self.ft_list[mesh].reshape((-1, 2))[f.ravel()], dtype=np.float32, order='C')
                vbo_uvs = vbo.VBO(np.array(uvs_by_face).astype(np.float32))

                vbo_indices = vbo.VBO(np.array(self.f_list[mesh][polygons]).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
                vbo_indices.bind()
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                vbo_colors.bind()
                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                if self.haveUVs_list[mesh][polygons]:
                    vbo_uvs.bind()

                    GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                    GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                # Textures:
                texture = None
                if self.haveUVs_list[mesh][polygons]:
                    texture = GL.GLuint(0)

                    GL.glGenTextures(1, texture)
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
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
                vbo_colors_mesh = vbo_colors_mesh + [vbo_colors]
                vbo_vertices_mesh = vbo_vertices_mesh + [vbo_verts]
                vbo_uvs_mesh = vbo_uvs_mesh + [vbo_uvs]
                vaos_mesh = vaos_mesh + [vao]

            self.textureID_mesh_list = self.textureID_mesh_list + [textureIDs_mesh]
            self.vao_tex_mesh_list = self.vao_tex_mesh_list + [vaos_mesh]
            self.vbo_indices_mesh_list = self.vbo_indices_mesh_list + [vbo_indices_mesh]

            self.vbo_colors_mesh = self.vbo_colors_mesh + [vbo_colors_mesh]
            self.vbo_verts_mesh = self.vbo_verts_mesh + [vbo_vertices_mesh]
            self.vbo_uvs_mesh = self.vbo_uvs_mesh + [vbo_uvs_mesh]

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)

        self.textureID = GL.glGetUniformLocation(self.colorTextureProgram, "myTextureSampler")

    def initGL_AnalyticRenderer(self):

        self.updateRender = True
        self.updateDerivatives = True

        GL.glEnable(GL.GL_MULTISAMPLE)
        # GL.glHint(GL.GL_MULTISAMPLE_FILTER_HINT_NV, GL.GL_NICEST);
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        VERTEX_SHADER = shaders.compileShader("""#version 330 core
        // Input vertex data, different for all executions of this shader.
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 colorIn;
        layout(location = 2) in vec2 vertexUV;
        layout(location = 3) in uint face_id;
        layout(location = 4) in vec3 barycentric;

        uniform mat4 MVP;
        out vec3 theColor;
        out vec4 pos;
        flat out uint face_out;
        out vec3 barycentric_vert_out;
        out vec2 UV;

        // Values that stay constant for the whole mesh.
        void main(){
            // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP* vec4(position,1);
            pos =  MVP * vec4(position,1);
            //pos =  pos4.xyz;
            theColor = colorIn;
            UV = vertexUV;
            face_out = face_id;
            barycentric_vert_out = barycentric;

        }""", GL.GL_VERTEX_SHADER)

        ERRORS_FRAGMENT_SHADER = shaders.compileShader("""#version 330 core 

            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            //layout(early_fragment_tests) in;

            // Interpolated values from the vertex shaders
            in vec3 theColor;
            in vec2 UV;
            flat in uint face_out;
            in vec4 pos;
            in vec3 barycentric_vert_out;

            layout(location = 3) uniform sampler2D myTextureSampler;
            
            uniform float ww;
            uniform float wh;

            // Ouput data
            layout(location = 0) out vec3 color; 
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 barycentric1;
            layout(location = 4) out vec2 barycentric2;

            void main(){
                vec3 finalColor = theColor * texture( myTextureSampler, UV).rgb;
                color = finalColor.rgb;

                sample_pos = ((0.5*pos.xy/pos.w) + 0.5)*vec2(ww,wh);
                sample_face = face_out;
                barycentric1 = barycentric_vert_out.xy;
                barycentric2 = vec2(barycentric_vert_out.z, 0.);

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
            #extension GL_ARB_explicit_uniform_location : enable
            #extension GL_ARB_explicit_attrib_location : enable

            layout(location = 2) uniform sampler2DMS colors;
            layout(location = 3) uniform sampler2DMS sample_positions;
            layout(location = 4) uniform usampler2DMS sample_faces;
            layout(location = 5) uniform sampler2DMS sample_barycentric_coords1;
            layout(location = 6) uniform sampler2DMS sample_barycentric_coords2;
            //layout(location = 7) uniform sampler2D imageGT;

            uniform float ww;
            uniform float wh;
            uniform int sample;

            // Ouput data
            layout(location = 0) out vec3 colorFetchOut;
            layout(location = 1) out vec2 sample_pos;
            layout(location = 2) out uint sample_face;
            layout(location = 3) out vec2 sample_barycentric1;
            layout(location = 4) out vec2 sample_barycentric2;
            //layout(location = 5) out vec3 res;

            //out int gl_SampleMask[];
            const int all_sample_mask = 0xffff;

            void main(){
                ivec2 texcoord = ivec2(gl_FragCoord.xy);
                colorFetchOut = texelFetch(colors, texcoord, sample).xyz;
                sample_pos = texelFetch(sample_positions, texcoord, sample).xy;        
                sample_face = texelFetch(sample_faces, texcoord, sample).r;
                sample_barycentric1 = texelFetch(sample_barycentric_coords1, texcoord, sample).xy;
                sample_barycentric2 = texelFetch(sample_barycentric_coords2, texcoord, sample).xy;
                
                //vec3 imgColor = texture(imageGT, gl_FragCoord.xy/vec2(ww,wh)).rgb;
                //res = imgColor - colorFetchOut;
                
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
        # GL.glGenTextures(1, self.textureEdges)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureEdges)
        # GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)


        # texture = GL.GLuint(0)
        # GL.glGenTextures(1, texture)

        # GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # image = np.array(np.flipud((self.textures_list[mesh][polygons])), order='C', dtype=np.float32)
        # GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGB32F, image.shape[1], image.shape[0])
        # GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        GL.glActiveTexture(GL.GL_TEXTURE0)

        whitePixel = np.ones([4, 4, 3])
        self.whitePixelTextureID = GL.GLuint(0)
        GL.glGenTextures(1, self.whitePixelTextureID)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.whitePixelTextureID)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S,GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T,GL.GL_CLAMP_TO_EDGE)

        image = np.array(np.flipud((whitePixel)), order='C', dtype=np.float32)

        
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, image.shape[1], image.shape[0], 0, GL.GL_RGB, GL.GL_FLOAT, image)
        # GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA8, image.shape[1], image.shape[0])

        # GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

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

        self.texture_errors_sample_faces = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_R32UI, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces, 0)
        #

        self.texture_errors_sample_barycentric1 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1,
                                  0)

        self.texture_errors_sample_barycentric2 = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_RG32F, self.frustum['width'], self.frustum['height'], False)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D_MULTISAMPLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2,
                                  0)

        self.z_buf_ms_errors = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.z_buf_ms_errors)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.nsamples, GL.GL_DEPTH_COMPONENT, self.frustum['width'], self.frustum['height'],
                                   False)
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

        self.render_buffer_fetch_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_face)
        #
        self.render_buffer_fetch_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric1)

        self.render_buffer_fetch_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, self.render_buffer_fetch_sample_barycentric2)

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

        # FBO_f
        self.fbo_errors_nonms = GL.glGenFramebuffers(1)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_errors_nonms)

        render_buf_errors_render = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_render)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf_errors_render)

        render_buf_errors_sample_position = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_position)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_RENDERBUFFER, render_buf_errors_sample_position)

        render_buf_errors_sample_face = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_RENDERBUFFER, render_buf_errors_sample_face)
        #

        render_buf_errors_sample_barycentric1 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric1)

        render_buf_errors_sample_barycentric2 = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RG32F, self.frustum['width'], self.frustum['height'])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_RENDERBUFFER, render_buf_errors_sample_barycentric2)
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

        self.textureObjLoc = GL.glGetUniformLocation(self.errorTextureProgram, "myTextureSampler")

        # Add background cube:
        position_location = GL.glGetAttribLocation(self.errorTextureProgram, 'position')
        color_location = GL.glGetAttribLocation(self.errorTextureProgram, 'colorIn')
        uvs_location = GL.glGetAttribLocation(self.errorTextureProgram, 'vertexUV')
        face_ids_location = GL.glGetAttribLocation(self.errorTextureProgram, 'face_id')
        barycentric_location = GL.glGetAttribLocation(self.errorTextureProgram, 'barycentric')

        # self.vbo_verts_cube= vbo.VBO(np.array(self.v_bgCube).astype(np.float32))
        # self.vbo_colors_cube= vbo.VBO(np.array(self.vc_bgCube).astype(np.float32))
        # self.vbo_uvs_cube = vbo.VBO(np.array(self.ft_bgCube).astype(np.float32))
        # self.vao_bgCube = GL.GLuint(0)
        # GL.glGenVertexArrays(1, self.vao_bgCube)
        #
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube = vbo.VBO(np.array(self.f_bgCube).astype(np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER)
        # self.vbo_f_bgCube.bind()
        # self.vbo_verts_cube.bind()
        # GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_colors_cube.bind()
        # GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # self.vbo_uvs_cube.bind()
        # GL.glEnableVertexAttribArray(uvs_location) # from 'location = 0' in shader
        # GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        #
        # f = self.f_bgCube
        # fc = np.tile(np.arange(len(self.f), len(self.f) + len(f))[:, None], [1, 3]).ravel()
        # # fc[:, 0] = fc[:, 0] & 255
        # # fc[:, 1] = (fc[:, 1] >> 8) & 255
        # # fc[:, 2] = (fc[:, 2] >> 16) & 255
        # fc = np.asarray(fc, dtype=np.uint32)
        # vbo_face_ids_cube = vbo.VBO(fc)
        # vbo_face_ids_cube.bind()
        # GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
        # GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)
        #
        # #Barycentric cube:
        # f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
        # vbo_barycentric_cube = vbo.VBO(f_barycentric)
        # vbo_barycentric_cube.bind()
        # GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
        # GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vao_quad = GL.GLuint(0)
        GL.glGenVertexArrays(1, self.vao_quad)
        GL.glBindVertexArray(self.vao_quad)

        # Bind VAO

        self.vbo_face_ids_list = []
        self.vbo_barycentric_list = []
        self.vao_errors_mesh_list = []
        flen = 1

        for mesh in range(len(self.f_list)):

            vaos_mesh = []
            vbo_face_ids_mesh = []
            vbo_barycentric_mesh = []
            for polygons in np.arange(len(self.f_list[mesh])):
                vao = GL.GLuint(0)
                GL.glGenVertexArrays(1, vao)
                GL.glBindVertexArray(vao)

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]
                vbo_f.bind()
                vbo_verts = self.vbo_verts_mesh[mesh][polygons]
                vbo_verts.bind()
                GL.glEnableVertexAttribArray(position_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_colors = self.vbo_colors_mesh[mesh][polygons]
                vbo_colors.bind()

                GL.glEnableVertexAttribArray(color_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                vbo_uvs = self.vbo_uvs_mesh[mesh][polygons]
                vbo_uvs.bind()
                GL.glEnableVertexAttribArray(uvs_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(uvs_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                f = self.f_list[mesh][polygons]

                fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()
                # fc[:, 0] = fc[:, 0] & 255
                # fc[:, 1] = (fc[:, 1] >> 8) & 255
                # fc[:, 2] = (fc[:, 2] >> 16) & 255

                fc = np.asarray(fc, dtype=np.uint32)
                vbo_face_ids = vbo.VBO(fc)
                vbo_face_ids.bind()
                GL.glEnableVertexAttribArray(face_ids_location)  # from 'location = 0' in shader
                GL.glVertexAttribIPointer(face_ids_location, 1, GL.GL_UNSIGNED_INT, 0, None)

                f_barycentric = np.asarray(np.tile(np.eye(3), (f.size // 3, 1)), dtype=np.float32, order='C')
                vbo_barycentric = vbo.VBO(f_barycentric)
                vbo_barycentric.bind()
                GL.glEnableVertexAttribArray(barycentric_location)  # from 'location = 0' in shader
                GL.glVertexAttribPointer(barycentric_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

                flen += len(f)

                vaos_mesh += [vao]

                vbo_face_ids_mesh += [vbo_face_ids]
                vbo_barycentric_mesh += [vbo_face_ids]

                GL.glBindVertexArray(0)

            self.vbo_face_ids_list += [vbo_face_ids_mesh]
            self.vbo_barycentric_list += [vbo_barycentric_mesh]
            self.vao_errors_mesh_list += [vaos_mesh]

    def render_image_buffers(self):

        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_SAMPLE_SHADING)
        GL.glMinSampleShading(1.0)

        self.makeCurrentContext()

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels], self.bgcolor.r[2 % self.num_channels], 1.)

        GL.glUseProgram(self.errorTextureProgram)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        # GL.glClearBufferiv(GL.GL_COLOR​, 0​, 0)
        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        #ImageGT
        GL.glActiveTexture(GL.GL_TEXTURE1)
        # GL.glBindImageTexture(1,self.textureGT, 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RGBA8)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        self.textureGTLoc = GL.glGetUniformLocation(self.errorTextureProgram, "imageGT")
        GL.glUniform1i(self.textureGTLoc, 1)


        wwLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.errorTextureProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_errors_mesh_list[mesh][polygons]

                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                # vbo_color.bind()

                f = self.f_list[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                self.vbo_colors_mesh[mesh][polygons].bind()

                if self.f.shape[1] == 2:
                    primtype = GL.GL_LINES
                else:
                    primtype = GL.GL_TRIANGLES

                assert (primtype == GL.GL_TRIANGLES)

                # GL.glUseProgram(self.errorTextureProgram)
                if self.haveUVs_list[mesh][polygons]:
                    texture = self.textureID_mesh_list[mesh][polygons]
                else:
                    texture = self.whitePixelTextureID

                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glUniform1i(self.textureObjLoc, 0)

                GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])

        # # #Background cube:
        # GL.glBindVertexArray(self.vao_bgCube)
        # self.vbo_f_bgCube.bind()
        # texture = self.whitePixelTextureID
        # self.vbo_uvs_cube.bind()
        #
        # GL.glActiveTexture(GL.GL_TEXTURE0)
        # GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # GL.glUniform1i(self.textureObjLoc, 0)
        # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
        #
        # GL.glDrawElements(primtype, len(self.vbo_f_bgCube)*self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

        # self.draw_visibility_image_ms(self.v, self.f)

        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        # # result_blit = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        # result_blit2 = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
        #
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms_errors)
        # GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position, 0)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        # GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        # GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_errors_nonms)
        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        # result_blit_pos = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))

        GL.glUseProgram(self.fetchSamplesProgram)
        # GL.glDisable(GL.GL_MULTISAMPLE)

        self.colorsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "colors")
        self.sample_positionsLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_positions")
        self.sample_facesLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_faces")
        self.sample_barycentric1Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords1")
        self.sample_barycentric2Loc = GL.glGetUniformLocation(self.fetchSamplesProgram, "sample_barycentric_coords2")

        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # GL.glActiveTexture(GL.GL_TEXTURE2)
        # GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_face)
        # GL.glUniform1i(self.sample_facesLoc, 2)

        wwLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'ww')
        whLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'wh')
        GL.glUniform1f(wwLoc, self.frustum['width'])
        GL.glUniform1f(whLoc, self.frustum['height'])

        self.renders = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 3])
        self.renders_sample_pos = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_faces = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height']]).astype(np.uint32)
        self.renders_sample_barycentric1 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 2])
        self.renders_sample_barycentric2 = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 1])
        self.renders_sample_barycentric = np.zeros([self.nsamples, self.frustum['width'], self.frustum['height'], 3])

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_sample_fetch)
        drawingBuffers = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3,
                          GL.GL_COLOR_ATTACHMENT4]
        GL.glDrawBuffers(5, drawingBuffers)

        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for sample in np.arange(self.nsamples):
            sampleLoc = GL.glGetUniformLocation(self.fetchSamplesProgram, 'sample')
            GL.glUniform1i(sampleLoc, sample)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_render)
            GL.glUniform1i(self.colorsLoc, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_position)
            GL.glUniform1i(self.sample_positionsLoc, 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_faces)
            GL.glUniform1i(self.sample_facesLoc, 2)

            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric1)
            GL.glUniform1i(self.sample_barycentric1Loc, 3)

            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture_errors_sample_barycentric2)
            GL.glUniform1i(self.sample_barycentric2Loc, 4)

            GL.glBindVertexArray(self.vao_quad)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

            # GL.glBindVertexArray(self.vao_bgCube)
            # # self.vbo_f_bgCube.bind()
            # GL.glUniformMatrix4fv(self.MVP_texture_location, 1, GL.GL_TRUE, MVP)
            #
            # GL.glDrawElements(primtype, len(self.vbo_f_bgCube) * self.vbo_f_bgCube.data.shape[1], GL.GL_UNSIGNED_INT, None)

            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_sample_fetch)

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:3].astype(np.float64))

            self.renders[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:2].astype(np.float64))
            self.renders_sample_pos[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT),
                              np.uint32).reshape(self.frustum['height'], self.frustum['height'])[:, :].astype(np.uint32))
            self.renders_faces[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:2].astype(np.float64))
            self.renders_sample_barycentric1[sample] = result

            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
            result = np.flipud(
                np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(
                    self.frustum['height'], self.frustum['height'], 3)[:, :, 0:1].astype(np.float64))
            self.renders_sample_barycentric2[sample] = result

            self.renders_sample_barycentric[sample] = np.concatenate(
                [self.renders_sample_barycentric1[sample], self.renders_sample_barycentric2[sample][:, :, 0:1]], 2)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            # result = np.flipud(np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_FLOAT), np.float32).reshape(self.frustum['height'], self.frustum['height'], 3)[:,:,0:3].astype(np.float64))
            # self.renders_faces[sample] = result

        GL.glBindVertexArray(0)

        GL.glClearColor(0., 0., 0., 1.)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)

        ##Finally return image and derivatives

        self.render_resolved = np.mean(self.renders, 0)

        self.updateRender = True
        self.updateDerivatives_verts = True
        self.updateDerivatives_vc = True

    def draw_visibility_image_ms(self, v, f):
        """Assumes camera is set up correctly in"""
        GL.glUseProgram(self.visibilityProgram_ms)

        v = np.asarray(v)

        self.draw_visibility_image_ms(v, f)

        # Attach FBO
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        fc = np.arange(1, len(f) + 1)
        fc = np.tile(fc.reshape((-1, 1)), (1, 3))
        fc[:, 0] = fc[:, 0] & 255
        fc[:, 1] = (fc[:, 1] >> 8) & 255
        fc[:, 2] = (fc[:, 2] >> 16) & 255
        fc = np.asarray(fc, dtype=np.uint8)

        self.draw_colored_primitives_ms(self.vao_dyn_ub, v, f, fc)

    # this assumes that fc is either "by faces" or "verts by face", not "by verts"
    def draw_colored_primitives_ms(self, vao, v, f, fc=None):

        # gl.EnableClientState(GL_VERTEX_ARRAY)
        verts_by_face = np.asarray(v.reshape((-1, 3))[f.ravel()], dtype=np.float64, order='C')
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

            vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
            self.vbo_colors_ub.set_array(vc_by_face)
            self.vbo_colors_ub.bind()

        primtype = GL.GL_TRIANGLES

        self.vbo_indices_dyn.set_array(np.arange(f.size, dtype=np.uint32).ravel())
        self.vbo_indices_dyn.bind()

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms_errors)

        drawingBuffers = [GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(1, drawingBuffers)

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, np.dot(self.projectionMatrix, view_mtx))

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawElements(primtype, len(self.vbo_indices_dyn), GL.GL_UNSIGNED_INT, None)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def compute_dr_wrt(self, wrt):

        visibility = self.visibility_image

        if wrt is self.camera:
            derivatives_verts = self.get_derivatives_verts()

            return derivatives_verts

        elif wrt is self.vc:

            derivatives_vc = self.get_derivatives_vc()

            return derivatives_vc

        # Not working atm.:
        elif wrt is self.bgcolor:
            return 2. * (self.imageGT.r - self.render_image).ravel() * common.dr_wrt_bgcolor(visibility, self.frustum, num_channels=self.num_channels)

        # Not working atm.:
        elif wrt is self.texture_stack:
            IS = np.nonzero(self.visibility_image.ravel() != 4294967295)[0]
            texcoords, texidx = self.texcoord_image_quantized
            vis_texidx = texidx.ravel()[IS]
            vis_texcoords = texcoords.ravel()[IS]
            JS = vis_texcoords * np.tile(col(vis_texidx), [1, 2]).ravel()

            clr_im = -2. * (self.imageGT.r - self.render_image) * self.renderWithoutTexture

            if False:
                cv2.imshow('clr_im', clr_im)
                # cv2.imshow('texmap', self.texture_image.r)
                cv2.waitKey(1)

            r = clr_im[:, :, 0].ravel()[IS]
            g = clr_im[:, :, 1].ravel()[IS]
            b = clr_im[:, :, 2].ravel()[IS]
            data = np.concatenate((r, g, b))

            IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
            JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))

            return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))

        return None

    def compute_r(self):
        return self.render()

    @depends_on(dterms + terms)
    def renderWithoutColor(self):
        self._call_on_changed()

        return self.render_nocolor

    @depends_on(dterms + terms)
    def renderWithoutTexture(self):
        self._call_on_changed()

        return self.render_notexture

    # @depends_on(dterms+terms)
    def render(self):
        self._call_on_changed()

        visibility = self.visibility_image

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]

        if self.updateRender:
            render, residuals = self.compute_image(visible, visibility, self.f)
            self.render_result = render
            self.residuals_result = residuals
            self.updateRender = False

        if self.imageGT is None:
            returnResult = self.render_result
        else:
            returnResult = self.residuals_result

        return returnResult

    def get_derivatives_verts(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_verts:
            if self.updateRender:
                self.render()
            if self.overdraw:
                # return common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
                derivatives_verts = common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)

            else:
                derivatives_verts = common.dImage_wrt_2dVerts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
            self.derivatives_verts = derivatives_verts
            self.updateDerivatives_verts = False
        return self.derivatives_verts

    def get_derivatives_vc(self):
        self._call_on_changed()

        visibility = self.visibility_image

        color = self.render_resolved

        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        barycentric = self.barycentric_image

        if self.updateDerivatives_vc:
            if self.updateRender:
                self.render()
            derivatives_vc = self.compute_derivatives_vc(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'],
                                                         self.v.r.size / 3, self.f)
            self.derivatives_vc = derivatives_vc
            self.updateDerivatives_vc = False
        return self.derivatives_vc

    # # @depends_on(dterms+terms)
    # def image_and_derivatives(self):
    #     # self._call_on_changed()
    #     visibility = self.visibility_image
    #
    #     color = self.render_resolved
    #
    #     visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    #     num_visible = len(visible)
    #
    #     barycentric = self.barycentric_image
    #
    #     if self.updateRender:
    #         render, derivatives = self.compute_image_and_derivatives(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size / 3, self.f)
    #         self.render = render
    #         self.derivatives = derivatives
    #         self.updateRender = False
    #
    #     return self.render, self.derivatives
    #

    def barycentricDerivatives(self, vertices, faces, verts):
        import chumpy as ch

        vertices = np.concatenate([vertices, np.ones([vertices.size // 3, 1])], axis=1)
        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])

        viewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])

        # # # Check with autodiff:
        # #
        # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        # # negYMat = ch.array([[1,0,self.camera.c.r[0]],[0,-1,self.camera.c.r[1]],[0,0,1]])
        # verts_hom_ch = ch.Ch(verts_hom)
        # camMtx = ch.Ch(np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])])
        # projVerts = (camMtx.dot(view_mtx)).dot(verts_hom_ch.T).T[:, :3].reshape([-1, 3])
        # viewVerts = ch.Ch(np.array(projVerts))
        # projVerts = projVerts[:, :2] / projVerts[:, 2:3]
        #
        # chViewVerticesNonBnd = camMtx[0:3, 0:3].dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        # p0 = ch.Ch(viewVerticesNonBnd[:, 0, :])
        # chp0 = p0
        #
        # p1 = ch.Ch(viewVerticesNonBnd[:, 1, :])
        # chp1 = p1
        #
        # p2 = ch.Ch(viewVerticesNonBnd[:, 2, :])
        # chp2 = p2
        #
        # # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        # nt = ch.cross(p1 - p0, p2 - p0)
        # chnt = nt
        # A = 0.5 * ch.sqrt(ch.sum(nt ** 2, axis=1))
        # chnt_norm = nt / ch.sqrt(ch.sum(nt ** 2, axis=1))[:, None]
        # # nt = nt / A
        #
        # chb0part2 = ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1)
        # chb0 = 0.5 * ch.sum(ch.cross(chnt_norm, p2 - p1) * (viewVerts - p1), axis=1) / A
        # chb1part2 = ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1)
        # chb1 = 0.5 * ch.sum(ch.cross(chnt_norm, p0 - p2) * (viewVerts - p2), axis=1) / A
        # chb2part2 = ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1)
        # chb2 = 0.5 * ch.sum(ch.cross(chnt_norm, p1 - p0) * (viewVerts - p0), axis=1) / A
        #
        # drb0p0 = chb0.dr_wrt(p0)
        # drb0p1 = chb0.dr_wrt(p1)
        # drb0p2 = chb0.dr_wrt(p2)
        #
        # drb1p0 = chb1.dr_wrt(p0)
        # drb1p1 = chb1.dr_wrt(p1)
        # drb1p2 = chb1.dr_wrt(p2)
        #
        # drb2p0 = chb2.dr_wrt(p0)
        # drb2p1 = chb2.dr_wrt(p1)
        # drb2p2 = chb2.dr_wrt(p2)
        #
        # rows = np.tile(np.arange(drb0p0.shape[0])[None, :], [3, 1]).T.ravel()
        # cols = np.arange(drb0p0.shape[0] * 3)
        #
        # drb0p0 = np.array(drb0p0[rows, cols]).reshape([-1, 3])
        # drb0p1 = np.array(drb0p1[rows, cols]).reshape([-1, 3])
        # drb0p2 = np.array(drb0p2[rows, cols]).reshape([-1, 3])
        # drb1p0 = np.array(drb1p0[rows, cols]).reshape([-1, 3])
        # drb1p1 = np.array(drb1p1[rows, cols]).reshape([-1, 3])
        # drb1p2 = np.array(drb1p2[rows, cols]).reshape([-1, 3])
        # drb2p0 = np.array(drb2p0[rows, cols]).reshape([-1, 3])
        # drb2p1 = np.array(drb2p1[rows, cols]).reshape([-1, 3])
        # drb2p2 = np.array(drb2p2[rows, cols]).reshape([-1, 3])
        #
        # chdp0 = np.concatenate([drb0p0[:, None, :], drb1p0[:, None, :], drb2p0[:, None, :]], axis=1)
        # chdp1 = np.concatenate([drb0p1[:, None, :], drb1p1[:, None, :], drb2p1[:, None, :]], axis=1)
        # chdp2 = np.concatenate([drb0p2[:, None, :], drb1p2[:, None, :], drb2p2[:, None, :]], axis=1)
        #
        # dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)
        # dp = dp[None, :]

        view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]
        camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
        verts_hom = np.concatenate([verts.reshape([-1, 3]), np.ones([verts.size // 3, 1])], axis=1)
        # viewVerts = negYMat.dot(view_mtx.dot(verts_hom.T).T[:, :3].T).T.reshape([-1, 3])
        projVerts = (camMtx.dot(view_mtx)).dot(verts_hom.T).T[:, :3].reshape([-1, 3])
        viewVerts = projVerts
        projVerts = projVerts[:, :2] / projVerts[:, 2:3]

        # viewVerticesNonBnd = negYMat.dot(view_mtx.dot(vertices.T).T[:, :3].T).T.reshape([-1, 3, 3])
        p0 = viewVerticesNonBnd[:, 0, :]
        p1 = viewVerticesNonBnd[:, 1, :]
        p2 = viewVerticesNonBnd[:, 2, :]

        p0_proj = p0[:, 0:2] / p0[:, 2:3]
        p1_proj = p1[:, 0:2] / p1[:, 2:3]
        p2_proj = p2[:, 0:2] / p2[:, 2:3]

        # D = np.linalg.det(np.concatenate([(p3 - p1).reshape([nNonBndFaces, 1, 3]), (p1 - p2).reshape([nNonBndFaces, 1, 3])], axis=1))
        nt = np.cross(p1 - p0, p2 - p0)
        nt_norm = nt / np.linalg.norm(nt, axis=1)[:, None]

        # a = -nt_norm[:, 0] / nt_norm[:, 2]
        # b = -nt_norm[:, 1] / nt_norm[:, 2]
        # c = np.sum(nt_norm * p0, 1) / nt_norm[:, 2]

        cam_f = 1

        u = p0[:, 0] / p0[:, 2]
        v = p0[:, 1] / p0[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p0[:, 2][:, None], np.zeros([len(p0), 1]), (-p0[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p0), 1]), p0[:, 2][:, None], (-p0[:, 1] / v ** 2)[:, None]]

        dxdp_0 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p1[:, 0] / p1[:, 2]
        v = p1[:, 1] / p1[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p1[:, 2][:, None], np.zeros([len(p1), 1]), (-p1[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p1), 1]), p1[:, 2][:, None], (-p1[:, 1] / v ** 2)[:, None]]

        dxdp_1 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        u = p2[:, 0] / p2[:, 2]
        v = p2[:, 1] / p2[:, 2]

        # xudiv = (cam_f - a * u - b * v) ** 2
        # xu = np.c_[c * (cam_f - b * v) / xudiv, a * v * c / xudiv, a * cam_f * c / xudiv]
        # xv = np.c_[b * u * c / xudiv, c * (cam_f - a * u) / xudiv, b * cam_f * c / xudiv]

        xu = np.c_[p2[:, 2][:, None], np.zeros([len(p2), 1]), (-p2[:, 0] / u ** 2)[:, None]]
        xv = np.c_[np.zeros([len(p2), 1]), p2[:, 2][:, None], (-p2[:, 1] / v ** 2)[:, None]]

        dxdp_2 = np.concatenate([xu[:, :, None], xv[:, :, None]], axis=2)

        # x = u * c / (cam_f - a * u - b * v)
        # y = v*c/(cam_f - a*u - b*v)
        # z = c*cam_f/(cam_f - a*u - b*v)

        A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        nt_mag = A * 2
        # nt = nt / A
        # db1 = 0.5*np.cross(nt_norm, p2-p1)/A[:, None]
        # db2 = 0.5*np.cross(nt_norm, p0-p2)/A[:, None]
        # db3_2 = 0.5*np.cross(nt_norm, p1-p0)/A[:, None]
        # db3 = - db1 - db2
        p = viewVerts

        pre1 = -1 / (nt_mag[:, None] ** 2) * nt_norm

        ident = np.identity(3)
        ident = np.tile(ident[None, :], [len(p2), 1, 1])

        dntdp0 = np.cross((p2 - p0)[:, None, :], -ident) + np.cross(-ident, (p1 - p0)[:, None, :])
        dntdp1 = np.cross((p2 - p0)[:, None, :], ident)
        dntdp2 = np.cross(ident, (p1 - p0)[:, None, :])

        # Pol check this!:
        dntnorm = (ident - np.einsum('ij,ik->ijk', nt_norm, nt_norm)) / nt_mag[:, None, None]
        # dntnorm = (ident - np.einsum('ij,ik->ijk',nt_norm,nt_norm))/nt_mag[:,None,None]

        dntnormdp0 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp0)
        dntnormdp1 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp1)
        dntnormdp2 = np.einsum('ijk,ikl->ijl', dntnorm, dntdp2)

        dpart1p0 = np.einsum('ij,ijk->ik', pre1, dntdp0)
        dpart1p1 = np.einsum('ij,ijk->ik', pre1, dntdp1)
        dpart1p2 = np.einsum('ij,ijk->ik', pre1, dntdp2)

        b0 = np.sum(np.cross(nt_norm, p2 - p1) * (p - p1), axis=1)[:, None]

        db0part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident,nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2-p1),-ident)
        # db0part2p1 = np.einsum('ikj,ij->ik',np.cross((p2 - p1)[:, None, :], dntnormdp0.swapaxes(1,2)), p - p1) + np.einsum('ikj,ij->ik', np.cross(-ident, nt_norm[:, None, :]), p - p1) + np.einsum('ik,ikj->ik', np.cross(p2-p1,nt_norm[:, :]),-ident)
        db0part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p1) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p2 - p1), -ident)
        db0part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p2 - p1)[:, None, :]), p - p1) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p1)

        db0dp0wrtpart1 = dpart1p0 * b0
        db0dp1wrtpart1 = dpart1p1 * b0
        db0dp2wrtpart1 = dpart1p2 * b0

        db0dp0wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p0
        db0dp1wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p1
        db0dp2wrtpart2 = 1. / (nt_mag[:, None]) * db0part2p2

        db0dp0wrt = db0dp0wrtpart1 + db0dp0wrtpart2
        db0dp1wrt = db0dp1wrtpart1 + db0dp1wrtpart2
        db0dp2wrt = db0dp2wrtpart1 + db0dp2wrtpart2

        ######
        b1 = np.sum(np.cross(nt_norm, p0 - p2) * (p - p2), axis=1)[:, None]

        db1part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p2)
        db1part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2)
        db1part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p0 - p2)[:, None, :]), p - p2) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p2) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p0 - p2), -ident)

        db1dp0wrtpart1 = dpart1p0 * b1
        db1dp1wrtpart1 = dpart1p1 * b1
        db1dp2wrtpart1 = dpart1p2 * b1

        db1dp0wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p0
        db1dp1wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p1
        db1dp2wrtpart2 = 1. / (nt_mag[:, None]) * db1part2p2

        db1dp0wrt = db1dp0wrtpart1 + db1dp0wrtpart2
        db1dp1wrt = db1dp1wrtpart1 + db1dp1wrtpart2
        db1dp2wrt = db1dp2wrtpart1 + db1dp2wrtpart2

        ######
        b2 = np.sum(np.cross(nt_norm, p1 - p0) * (p - p0), axis=1)[:, None]

        db2part2p0 = np.einsum('ikj,ij->ik', np.cross(dntnormdp0.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], -ident), p - p0) + np.einsum('ik,ikj->ik', np.cross(nt_norm[:, :], p1 - p0), -ident)
        db2part2p1 = np.einsum('ikj,ij->ik', np.cross(dntnormdp1.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0) + np.einsum('ikj,ij->ik', np.cross(
            nt_norm[:, None, :], ident), p - p0)
        db2part2p2 = np.einsum('ikj,ij->ik', np.cross(dntnormdp2.swapaxes(1, 2), (p1 - p0)[:, None, :]), p - p0)

        db2dp0wrtpart1 = dpart1p0 * b2
        db2dp1wrtpart1 = dpart1p1 * b2
        db2dp2wrtpart1 = dpart1p2 * b2

        db2dp0wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p0
        db2dp1wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p1
        db2dp2wrtpart2 = 1. / (nt_mag[:, None]) * db2part2p2

        db2dp0wrt = db2dp0wrtpart1 + db2dp0wrtpart2
        db2dp1wrt = db2dp1wrtpart1 + db2dp1wrtpart2
        db2dp2wrt = db2dp2wrtpart1 + db2dp2wrtpart2

        dp0 = np.concatenate([db0dp0wrt[:, None, :], db1dp0wrt[:, None, :], db2dp0wrt[:, None, :]], axis=1)
        dp1 = np.concatenate([db0dp1wrt[:, None, :], db1dp1wrt[:, None, :], db2dp1wrt[:, None, :]], axis=1)
        dp2 = np.concatenate([db0dp2wrt[:, None, :], db1dp2wrt[:, None, :], db2dp2wrt[:, None, :]], axis=1)
        #
        dp = np.concatenate([dp0[:, :, None], dp1[:, :, None], dp2[:, :, None]], 2)

        # If dealing with degenerate triangles, ignore that gradient.

        # dp[nt_mag <= 1e-15] = 0

        dp = dp[None, :]

        nFaces = len(faces)
        # visTriVC = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc = self.vc.r[faces.ravel()].reshape([nFaces, 3, 3]).transpose([2, 0, 1])[:, :, :, None, None]
        vc[vc > 1] = 1
        vc[vc < 0] = 0

        visTriVC = vc

        dxdp = np.concatenate([dxdp_0[:, None, :], dxdp_1[:, None, :], dxdp_2[:, None, :]], axis=1)

        dxdp = dxdp[None, :, None]
        # dbvc = np.sum(dp * visTriVC, 2)

        # dbvc = dp * visTriVC * t_area[None, :, None, None, None]
        dbvc = dp * visTriVC

        didp = np.sum(dbvc[:, :, :, :, :, None] * dxdp, 4).sum(2)

        # output should be shape: VC x Ninput x Tri Points x UV

        # drb0p0 # db0dp0wrt
        # drb0p1 # db0dp1wrt
        # drb0p2 # db0dp2wrt
        # drb1p0 # db1dp0wrt
        # drb1p1 # db1dp1wrt
        # drb1p2 # db1dp2wrt
        # drb2p0 # db2dp0wrt
        # drb2p1 # db2dp1wrt
        # drb2p2 # db2dp2wrt

        return didp

    def compute_image(self, visible, visibility, f):
        """Construct a sparse jacobian that relates 2D projected vertex positions
        (in the columns) to pixel values (in the rows). This can be done
        in two steps."""

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        nsamples = self.nsamples

        if np.any(boundaryImage):
            sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
                [nsamples, -1, 2])

            # sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:,(zerosIm*boundaryImage).ravel().astype(np.bool),:].reshape([nsamples, -1, 3])
            sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

            boundaryFaces = visibility[(boundaryImage) & (visibility != 4294967295)]
            nBndFaces = len(boundaryFaces)

            vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1, 1, 1])

            sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

            # if self.debug:
            #     import pdb; pdb.set_trace()

            # faces = f[sampleFaces].ravel()
            # vertsPerFaceProjBnd = self.camera.r[faces].reshape([-1, 3, 2])
            # nv = len(vertsPerFaceProjBnd)
            # p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
            # p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
            # p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
            # t_area_bnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)
            # t_area_bnd[t_area_bnd > 1] = 1

            # Trick to cap to 1 while keeping gradients.

            p1 = vertsProjBndSamples.reshape([-1,2,2])[:, 0, :]
            p2 = vertsProjBndSamples.reshape([-1,2,2])[:, 1, :]

            p = sampleV.reshape([-1,2])

            l = (p2 - p1)
            linedist = np.sqrt((np.sum(l ** 2, axis=1)))[:, None]
            self.linedist = linedist

            lnorm = l / linedist
            self.lnorm = lnorm

            v1 = p - p1
            self.v1 = v1
            d = v1[:, 0] * lnorm[:, 0] + v1[:, 1] * lnorm[:, 1]
            self.d = d
            intersectPoint = p1 + d[:, None] * lnorm

            v2 = p - p2
            self.v2 = v2
            l12 = (p1 - p2)
            linedist12 = np.sqrt((np.sum(l12 ** 2, axis=1)))[:, None]
            lnorm12 = l12 / linedist12
            d2 = v2[:, 0] * lnorm12[:, 0] + v2[:, 1] * lnorm12[:, 1]

            nonIntersect = (d2 < 0) | (d < 0)
            self.nonIntersect = nonIntersect

            argminDistNonIntersect = np.argmin(np.c_[d[nonIntersect], d2[nonIntersect]], 1)
            self.argminDistNonIntersect = argminDistNonIntersect

            intersectPoint[nonIntersect] = vertsProjBndSamples.reshape([-1,2,2])[nonIntersect][np.arange(nonIntersect.sum()), argminDistNonIntersect]

            lineToPoint = (p - intersectPoint)

            n = lineToPoint

            dist = np.sqrt((np.sum(lineToPoint ** 2, axis=1)))[:, None]

            n_norm = lineToPoint / dist

            self.n_norm = n_norm

            self.dist = dist

            d_final = dist.squeeze()

            # max_nx_ny = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))

            # d_final = d_final / max_nx_ny
            d_final = d_final

            # invViewMtx = np.linalg.inv(np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])])
            # #
            # camMtx = np.r_[np.c_[self.camera.camera_mtx, np.array([0, 0, 0])], np.array([[0, 0, 0, 1]])]
            # # invCamMtx = np.r_[np.c_[np.linalg.inv(self.camera.camera_mtx), np.array([0,0,0])], np.array([[0, 0, 0, 1]])]
            #
            # view_mtx = np.r_[self.camera.view_mtx, np.array([[0, 0, 0, 1]])]

            # verticesBndSamples = np.concatenate([verticesBndSamples.reshape([-1, 3]), np.ones([verticesBndSamples.size // 3, 1])], axis=1)

            # projVerticesBndOutside = (camMtx.dot(view_mtx)).dot(verticesBndSamples.T).T[:, :3].reshape([-1, 2, 3])
            # projVerticesBndDir = projVerticesBndOutside[:, 1, :] - projVerticesBndOutside[:, 0, :]
            # projVerticesBndDir = projVerticesBndDir / np.sqrt((np.sum(projVerticesBndDir ** 2, 1)))[:, None]

            # dproj = (intersectPoint[:, 0] * projVerticesBndOutside[:, 0, 2] - projVerticesBndOutside[:, 0, 0]) / (projVerticesBndDir[:, 0] - projVerticesBndDir[:, 2] * intersectPoint[:, 0])
            # # Code to check computation that dproj == dprojy
            # # dproj_y = (intersectPoint[:,1]* projVerticesBndOutside[:,0,2] - projVerticesBndOutside[:,0,1]) / (projVerticesBndDir[:,1] - projVerticesBndDir[:,2]*intersectPoint[:,1])
            #
            # projPoint = projVerticesBndOutside[:, 0, :][:, :] + dproj[:, None] * projVerticesBndDir[:, :]
            #
            # projPointVec4 = np.concatenate([projPoint, np.ones([projPoint.shape[0], 1])], axis=1)
            # viewPointIntersect = (invViewMtx.dot(np.linalg.inv(camMtx)).dot(projPointVec4.T.reshape([4, -1])).reshape([4, -1])).T[:, :3]
            #
            # barycentricVertsDistIntesect = np.linalg.norm(viewPointIntersect - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 0, :], axis=1)
            # barycentricVertsDistIntesect2 = np.linalg.norm(viewPointIntersect - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 1, :], axis=1)
            # # Code to check barycentricVertsDistIntesect + barycentricVertsDistIntesect2 = barycentricVertsDistEdge
            # barycentricVertsDistEdge = np.linalg.norm(
            #     verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 0, :] - verticesBndSamples[:, 0:3].reshape([-1, 2, 3])[:, 1, :], axis=1)
            #
            # nonIntersect = np.abs(barycentricVertsDistIntesect + barycentricVertsDistIntesect2 - barycentricVertsDistEdge) > 1e-4
            # argminDistNonIntersect = np.argmin(np.c_[barycentricVertsDistIntesect[nonIntersect], barycentricVertsDistIntesect2[nonIntersect]], 1)
            #
            # self.viewPointIntersect = viewPointIntersect
            # self.viewPointIntersect[nonIntersect] = verticesBndSamples.reshape([-1, 2, 4])[nonIntersect, :, 0:3][np.arange(nonIntersect.sum()),
            #                                         argminDistNonIntersect, :]

            d_finalNP = d_final.copy()
            self.d_final = d_finalNP

            # self.t_area_bnd = t_area_bnd

            # areaWeights = np.zeros([nsamples, nBndFaces])
            # areaWeights = t_area_bnd.reshape([nsamples, nBndFaces])
            # areaWeightsTotal = areaWeights.sum(0)
            ## areaWeightsTotal[areaWeightsTotal < 1] = 1
            # self.areaWeights = areaWeights
            # self.areaWeightsTotal = areaWeightsTotal

            finalColorBnd = np.ones([self.nsamples, boundaryFaces.size, 3])

            self.d_final_total = d_finalNP.reshape([self.nsamples, -1,1]).sum(0)

            # if self.imageGT is not None:
            finalColorBnd = sampleColors * d_finalNP.reshape([self.nsamples, -1,1]) / (self.d_final_total.reshape([1, -1,1]))
            # finalColorBnd = areaWeights[:,:,None] * sampleColors * d_finalNP.reshape([self.nsamples, -1,1]) / (self.d_final_total.reshape([1, -1,1]) * areaWeightsTotal[None,:,None])
            self.finalColorBnd = finalColorBnd
            # else:
            #     finalColorBnd = sampleColors

            bndColorsImage = np.zeros_like(self.color_image)
            bndColorsImage[(zerosIm * boundaryImage), :] = np.sum(finalColorBnd, axis=0)

            finalColorImageBnd = bndColorsImage

            if self.imageGT is not None:
                bndColorsResiduals = np.zeros_like(self.color_image)
                self.sampleResiduals = (sampleColors - self.imageGT.r[(zerosIm * boundaryImage),:][None,:])
                self.sampleResidualsWeighted = self.sampleResiduals**2 * d_finalNP.reshape([self.nsamples, -1,1]) / self.d_final_total.reshape([1, -1,1])

                bndColorsResiduals[(zerosIm * boundaryImage), :] = np.sum(self.sampleResidualsWeighted,0)

        if np.any(boundaryImage):
            finalColor = (1 - boundaryImage)[:, :, None] * self.color_image + boundaryImage[:, :, None] * finalColorImageBnd

            if self.imageGT is not None:
                self.residuals = (self.color_image - self.imageGT.r)
                errors = self.residuals**2
                finalResidual = (1 - boundaryImage)[:, :, None] * errors + boundaryImage[:, :, None] * bndColorsResiduals
        else:
            finalColor = self.color_image

            if self.imageGT is not None:
                finalResidual = (self.color_image - self.imageGT.r)**2

        if self.imageGT is None:
            finalResidual = None

        finalColor[finalColor > 1] = 1
        finalColor[finalColor < 0] = 0

        return finalColor, finalResidual

    def compute_derivatives_verts(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        # xdiff = dEdx
        # ydiff = dEdy

        nVisF = len(visibility.ravel()[visible])
        # projVertices = self.camera.r[f[visibility.ravel()[visible]].ravel()].reshape([nVisF,3, 2])

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)

        rangeIm = np.arange(self.boundarybool_image.size)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        edge_visibility = self.boundaryid_image

        vertsProjBnd = self.camera.r[self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()].reshape([-1, 2, 2])

        nsamples = self.nsamples
        sampleV = self.renders_sample_pos.reshape([nsamples, -1, 2])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape(
            [nsamples, -1, 2])

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1
        if 4294967295 in sampleFaces:
            sampleFaces[sampleFaces==4294967295] = 0 #Not correct but need to check further.
        sampleColors = self.renders.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool), :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage) & (visibility != 4294967295)]

        if np.any(boundaryImage):

            n_norm = self.n_norm
            dist = self.dist
            linedist = self.linedist
            d = self.d
            v1 = self.v1
            lnorm = self.lnorm

            d_final = self.d_final

            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)

            # vertsProjBnd[None, :] - sampleV[:,None,:]
            vertsProjBndSamples = np.tile(vertsProjBnd[None, :], [self.nsamples, 1, 1, 1])

            # Computing gradients:
            # A multisampled pixel color is given by: w R + (1-w) R' thus:
            # 1 derivatives samples outside wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            # 2 derivatives samples outside wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample
            # 3 derivatives samples outside wrt v bar edge: (1-w) (dbar'*vc') )/ nsamples for faces edge (barv1', barv2', 0)
            # 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample
            # 5 derivatives samples outside wrt vc : (1-w) (bar')/ nsamples for faces edge

            # 6 derivatives samples inside wrt v : (dbar'*vc')/ nsamples for faces sample
            # 7 derivatives samples inside wrt vc : (bar)/ nsamples for faces sample

            # for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            # Finally, stack data, and IJ of nonbnd with bnd on both dwrt_v and dwrt_vc.

            ######## 1 derivatives samples outside wrt v 1: (dw * (bar*vc) - dw (bar'*vc') )/ nsamples for face sample

            # # #Chumpy autodiff code to check derivatives here:
            # chEdgeVerts = ch.Ch(vertsProjBndSamples.reshape([-1,2,2]))
            #
            # chEdgeVerts1 = chEdgeVerts[:,0,:]
            # chEdgeVerts2 = chEdgeVerts[:,1,:]
            #
            # chSampleVerts = ch.Ch(sampleV.reshape([-1,2]))
            # # c1 = (chEdgeVerts1 - chSampleVerts)
            # # c2 = (chEdgeVerts2 - chSampleVerts)
            # # n = (chEdgeVerts2 - chEdgeVerts1)
            #
            # #Code to check computation of distance below
            # # d2 = ch.abs(c1[:,:,0]*c2[:,:,1] - c1[:,:,1]*c2[:,:,0]) / ch.sqrt((ch.sum(n**2,2)))
            # # # np_mat = ch.dot(ch.array([[0,-1],[1,0]]), n)
            # # np_mat2 = -ch.concatenate([-n[:,:,1][:,:,None], n[:,:,0][:,:,None]],2)
            # # np_vec2 = np_mat2 / ch.sqrt((ch.sum(np_mat2**2,2)))[:,:,None]
            # # d2 =  d2 / ch.maximum(ch.abs(np_vec2[:,:,0]),ch.abs(np_vec2[:,:,1]))
            #
            # chl = (chEdgeVerts2 - chEdgeVerts1)
            # chlinedist = ch.sqrt((ch.sum(chl**2,axis=1)))[:,None]
            # chlnorm = chl/chlinedist
            #
            # chv1 = chSampleVerts - chEdgeVerts1
            #
            # chd = chv1[:,0]* chlnorm[:,0] + chv1[:,1]* chlnorm[:,1]
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            # # intersectPointDist1 = intersectPoint - chEdgeVerts1
            # # intersectPointDist2 = intersectPoint - chEdgeVerts2
            # # Code to check computation of distances below:
            # # lengthIntersectToPoint1 = np.linalg.norm(intersectPointDist1.r,axis=1)
            # # lengthIntersectToPoint2 = np.linalg.norm(intersectPointDist2.r,axis=1)
            #
            # chintersectPoint = chEdgeVerts1 + chd[:,None] * chlnorm
            #
            # chlineToPoint = (chSampleVerts - chintersectPoint)
            # chn_norm = chlineToPoint / ch.sqrt((ch.sum(chlineToPoint ** 2, axis=1)))[:, None]
            #
            # chdist = chlineToPoint[:,0]*chn_norm[:,0] + chlineToPoint[:,1]*chn_norm[:,1]
            #
            # # d_final_ch = chdist / ch.maximum(ch.abs(chn_norm[:, 0]), ch.abs(chn_norm[:, 1]))
            # d_final_ch = chdist
            #
            # d_final_ch_weights = sampleColors * (d_final_ch.reshape([self.nsamples, -1]) / ch.sum(d_final_ch.reshape([self.nsamples, -1]), 0))[:,:,None]
            #
            # d_final_outside = d_final_ch.ravel()
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts1)
            # rows = np.tile(np.arange(d_final_outside.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_outside.shape[0] * 2)
            #
            # dwdv_r_v1 = np.array(dwdv[rows, cols]).reshape([-1, 2])
            #
            # dwdv = d_final_outside.dr_wrt(chEdgeVerts2)
            # rows = np.tile(np.arange(d_final_ch.shape[0])[None, :], [2, 1]).T.ravel()
            # cols = np.arange(d_final_ch.shape[0] * 2)
            #
            # dwdv_r_v2 = np.array(dwdv[rows, cols]).reshape([-1, 2])


            nonIntersect = self.nonIntersect
            argminDistNonIntersect = self.argminDistNonIntersect

            # max_dx_dy = np.maximum(np.abs(n_norm[:, 0]), np.abs(n_norm[:, 1]))
            d_final_np = dist
            # d_final_np = dist / max_dx_dy

            ident = np.identity(2)
            ident = np.tile(ident[None, :], [len(d_final_np), 1, 1])

            dlnorm = (ident - np.einsum('ij,ik->ijk', lnorm, lnorm)) / linedist[:, None]
            dl_normdp1 = np.einsum('ijk,ikl->ijl', dlnorm, -ident)
            dl_normdp2 = np.einsum('ijk,ikl->ijl', dlnorm, ident)

            dv1dp1 = -ident
            dv1dp2 = 0

            dddp1 = np.einsum('ijk,ij->ik', dv1dp1, lnorm) + np.einsum('ij,ijl->il', v1, dl_normdp1)
            dddp2 = 0 + np.einsum('ij,ijl->il', v1, dl_normdp2)

            dipdp1 = ident + (dddp1[:, None, :] * lnorm[:, :, None]) + d[:, None, None] * dl_normdp1
            dipdp2 = (dddp2[:, None, :] * lnorm[:, :, None]) + d[:, None, None] * dl_normdp2

            #good up to here.

            dndp1 = -dipdp1
            dndp2 = -dipdp2

            dn_norm = (ident - np.einsum('ij,ik->ijk', n_norm, n_norm)) / dist[:, None]

            # dn_normdp1 = np.einsum('ijk,ikl->ijl', dn_norm, dndp1)
            # dn_normdp2 = np.einsum('ijk,ikl->ijl', dn_norm, dndp2)

            ddistdp1 = np.einsum('ij,ijl->il', n_norm, dndp1)
            ddistdp2 = np.einsum('ij,ijl->il', n_norm, dndp2)

            # argmax_nx_ny = np.argmax(np.abs(n_norm), axis=1)
            # dmax_nx_ny_p1 = np.sign(n_norm)[np.arange(len(n_norm)), argmax_nx_ny][:, None] * dn_normdp1[np.arange(len(dn_normdp1)), argmax_nx_ny]
            # dmax_nx_ny_p2 = np.sign(n_norm)[np.arange(len(n_norm)), argmax_nx_ny][:, None] * dn_normdp2[np.arange(len(dn_normdp2)), argmax_nx_ny]

            # dd_final_dp1 = -1. / max_dx_dy[:, None] ** 2 * dmax_nx_ny_p1 * dist + 1. / max_dx_dy[:, None] * ddistdp1
            # dd_final_dp2 = -1. / max_dx_dy[:, None] ** 2 * dmax_nx_ny_p2 * dist + 1. / max_dx_dy[:, None] * ddistdp2

            dd_final_dp1 = ddistdp1
            dd_final_dp2 = ddistdp2

            # For those non intersecting points straight to the edge:

            v1 = self.v1[nonIntersect][argminDistNonIntersect == 0]
            v1_norm = v1 / np.sqrt((np.sum(v1 ** 2, axis=1)))[:, None]

            dd_final_dp1_nonintersect = -v1_norm

            v2 = self.v2[nonIntersect][argminDistNonIntersect == 1]
            v2_norm = v2 / np.sqrt((np.sum(v2 ** 2, axis=1)))[:, None]
            dd_final_dp2_nonintersect = -v2_norm

            dd_final_dp1[nonIntersect][argminDistNonIntersect == 0] = dd_final_dp1_nonintersect
            dd_final_dp1[nonIntersect][argminDistNonIntersect == 1] = 0
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 1] = dd_final_dp2_nonintersect
            dd_final_dp2[nonIntersect][argminDistNonIntersect == 0] = 0

            dd_final_dp1_weighted_part1 = -self.d_final[:,None]* np.tile(dd_final_dp1.reshape([self.nsamples, -1, 2]).sum(0)[None,:,:],[self.nsamples,1,1]).reshape([-1, 2])/(np.tile(self.d_final_total[None,:], [self.nsamples, 1,1]).reshape([-1,1])**2)
            dd_final_dp1_weighted_part2 = dd_final_dp1 / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1])
            dd_final_dp1_weighted =  dd_final_dp1_weighted_part1 + dd_final_dp1_weighted_part2

            dd_final_dp2_weighted_part1 = -self.d_final[:,None]*np.tile(dd_final_dp2.reshape([self.nsamples, -1, 2]).sum(0)[None,:,:],[self.nsamples,1,1]).reshape([-1, 2])/(np.tile(self.d_final_total[None,:], [self.nsamples, 1,1]).reshape([-1,1])**2)
            dd_final_dp2_weighted_part2 = dd_final_dp2 / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1])
            dd_final_dp2_weighted =  dd_final_dp2_weighted_part1 + dd_final_dp2_weighted_part2

            if self.imageGT is None:
                dImage_wrt_outside_v1 = sampleColors.reshape([-1,3,1]) * dd_final_dp1_weighted[:, None, :]
                dImage_wrt_outside_v2 = sampleColors.reshape([-1,3,1]) * dd_final_dp2_weighted[:, None, :]
            else:
                dImage_wrt_outside_v1 = self.sampleResiduals.reshape([-1,3,1])**2 * dd_final_dp1_weighted[:, None, :]
                dImage_wrt_outside_v2 = self.sampleResiduals.reshape([-1,3,1])**2 * dd_final_dp2_weighted[:, None, :]

            # sampleV
            # z = dd_final_dp1.reshape([8, -1, 2])
            # eq = np.array([np.all(np.sign(z[:, i, :]) == -1) or np.all(np.sign(z[:, i, :]) == 1) for i in range(z.shape[1])])
            # dist_ns = dist.reshape([8,-1])
            # rightV = sampleV[0, :, 0] > np.max(sampleV[0, :, :], 0)[0] - 1
            # dist_ns[0, rightV]
            # dImage_wrt_outside_v1.reshape([8, -1, 3, 2])[0, rightV,:]
            # d_final_ch_weights
            # self.finalColorBnd

            ### Derivatives wrt V:
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 2 * 2)).ravel()

            faces = self.vpe[edge_visibility.ravel()[(zerosIm * boundaryImage).ravel().astype(np.bool)]].ravel()
            faces = np.tile(faces.reshape([1, -1, 2]), [self.nsamples, 1, 1]).ravel()
            JS = col(faces)
            JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data1 = dImage_wrt_outside_v1.transpose([1, 0, 2])
            data2 = dImage_wrt_outside_v2.transpose([1, 0, 2])

            data = np.concatenate([data1[:, :, None, :], data2[:, :, None, :]], 2)

            data = data.ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))

            ######## 2 derivatives samples wrt v bar outside: (w * (dbar*vc) )/ nsamples for faces sample

            verticesBnd = self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3])

            sampleBarycentricBar = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),
                                   :].reshape([-1, 3, 1])
            verts = np.sum(self.v.r[f[sampleFaces.ravel()].ravel()].reshape([-1, 3, 3]) * sampleBarycentricBar, axis=1)

            dImage_wrt_bar_v = self.barycentricDerivatives(verticesBnd, f[sampleFaces.ravel()], verts).swapaxes(0, 1)

            if self.imageGT is None:
                # dImage_wrt_bar_v = dImage_wrt_bar_v * d_final[:, None, None, None] * self.t_area_bnd[:, None, None, None] / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                dImage_wrt_bar_v = dImage_wrt_bar_v * d_final[:, None, None, None]  / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # areaTotal = np.tile(self.areaWeightsTotal[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # d_final_total = np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])
                # dImage_wrt_bar_v = self.areaWeights.reshape([-1,1,1,1]) * dImage_wrt_bar_v * d_final[:, None, None, None] / (areaTotal*d_final_total)
            else:
                dImage_wrt_bar_v = 2*self.sampleResiduals.reshape([-1,3])[:,:,None,None] * dImage_wrt_bar_v * d_final[:, None, None, None] * self.t_area_bnd[:, None, None, None] / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1, 1, 1, 1])

            ### Derivatives wrt V: 2 derivatives samples wrt v bar: (w * (dbar*vc) )/ nsamples for faces sample
            # IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 2 * f.shape[1])).ravel()
            faces = f[sampleFaces].ravel()
            JS = col(faces)
            JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

            if n_channels > 1:
                IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
                JS = np.concatenate([JS for i in range(n_channels)])

            data = np.transpose(dImage_wrt_bar_v, [1, 0, 2, 3]).ravel()

            ij = np.vstack((IS.ravel(), JS.ravel()))

            result_wrt_verts_bnd_bar = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        verticesNonBnd = self.v.r[f[nonBoundaryFaces].ravel()]

        vertsPerFaceProjBnd = self.camera.r[f[nonBoundaryFaces].ravel()].reshape([-1, 3, 2])
        nv = len(vertsPerFaceProjBnd)

        p0_proj = np.c_[vertsPerFaceProjBnd[:, 0, :], np.ones([nv, 1])]
        p1_proj = np.c_[vertsPerFaceProjBnd[:, 1, :], np.ones([nv, 1])]
        p2_proj = np.c_[vertsPerFaceProjBnd[:, 2, :], np.ones([nv, 1])]
        t_area_nonbnd = np.abs(np.linalg.det(np.concatenate([p0_proj[:, None], p1_proj[:, None], p2_proj[:, None]], axis=1)) * 0.5)

        t_area_nonbnd[t_area_nonbnd > 1] = 1

        bc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))

        verts = np.sum(self.v.r[f[nonBoundaryFaces.ravel()].ravel()].reshape([-1, 3, 3]) * bc[:, :, None], axis=1)

        didp = self.barycentricDerivatives(verticesNonBnd, f[nonBoundaryFaces.ravel()], verts)

        if self.imageGT is None:
            # didp = didp * t_area_nonbnd[None, :, None, None]
            didp = didp
        else:
            didp = 2 * self.residuals[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3)).T[:,:,None,None] * didp * t_area_nonbnd[None, :, None, None]

        n_channels = np.atleast_3d(observed).shape[2]

        ####### 2: Take the data and copy the corresponding dxs and dys to these new pixels.

        ### Derivatives wrt V:
        pixels = np.where(((~boundaryImage) & (visibility != 4294967295)).ravel())[0]
        IS = np.tile(col(pixels), (1, 2 * f.shape[1])).ravel()
        JS = col(f[nonBoundaryFaces].ravel())
        JS = np.hstack((JS * 2, JS * 2 + 1)).ravel()

        if n_channels > 1:
            IS = np.concatenate([IS * n_channels + i for i in range(n_channels)])
            JS = np.concatenate([JS for i in range(n_channels)])

        data = didp.ravel()

        ij = np.vstack((IS.ravel(), JS.ravel()))

        result_wrt_verts_nonbnd = sp.csc_matrix((data, ij), shape=(image_width * image_height * n_channels, num_verts * 2))

        if np.any(boundaryImage):

            result_wrt_verts = result_wrt_verts_bnd + result_wrt_verts_bnd_bar + result_wrt_verts_nonbnd
        else:
            result_wrt_verts = result_wrt_verts_nonbnd

        return result_wrt_verts

    def compute_derivatives_vc(self, observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
        width = self.frustum['width']
        height = self.frustum['height']
        num_channels = 3
        n_channels = num_channels
        vc_size = self.vc.size

        d_final = self.d_final

        boundaryImage = self.boundarybool_image.astype(np.bool) & (visibility != 4294967295)
        zerosIm = np.ones(self.boundarybool_image.shape).astype(np.bool)

        nsamples = self.nsamples

        sampleFaces = self.renders_faces.reshape([nsamples, -1])[:, (zerosIm * boundaryImage).ravel().astype(np.bool)].reshape([nsamples, -1]) - 1

        sampleBarycentric = self.renders_sample_barycentric.reshape([nsamples, -1, 3])[:, (zerosIm * boundaryImage).ravel().astype(np.bool),
                            :].reshape([nsamples, -1, 3])

        nonBoundaryFaces = visibility[zerosIm * (~boundaryImage) & (visibility != 4294967295)]

        if np.any(boundaryImage):
            boundaryFaces = visibility[boundaryImage]
            nBndFaces = len(boundaryFaces)

            # Computing gradients:
            # A multisampled pixel color is given by: w R + (1-w) R' thus:
            # 1 derivatives samples wrt v 1: (dw * (svc) - dw (bar'*vc') )/ nsamples for face sample
            # 2 derivatives samples wrt v bar: (w * (dbar*vc) )/ nsamples for faces sample
            # 4 derivatives samples wrt vc : (w * (bar) )/ nsamples for faces sample

            # for every boundary pixel i,j we have list of sample faces. compute gradients at each and sum them according to face identity, options:
            #   - Best: create sparse matrix for every matrix. sum them! same can be done with boundary.

            ####### 4 derivatives samples outside wrt vc : (w * (bar) )/ nsamples for faces sample

            if self.imageGT is None:
                dImage_wrt_bnd_vc = d_final[:, None] * sampleBarycentric.reshape([-1,3]) / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1,1])
            else:
                dImage_wrt_bnd_vc = d_final[:, None] * sampleBarycentric.reshape([-1,3]) / np.tile(self.d_final_total[None, :], [self.nsamples, 1, 1]).reshape([-1,1])
                dImage_wrt_bnd_vc = 2 * self.sampleResiduals.reshape([-1,3]).T[:,:,None] * dImage_wrt_bnd_vc[None,:]

            ### Derivatives wrt VC:

            # Each pixel relies on three verts
            pixels = np.tile(np.where(boundaryImage.ravel())[0][None, :], [self.nsamples, 1])
            IS = np.tile(col(pixels), (1, 3)).ravel()

            if 4294967295 in sampleFaces:
                sampleFaces[sampleFaces==4294967295] = 0 #Not correct but need to check further.

            faces = f[sampleFaces].ravel()
            JS = col(faces)

            data = dImage_wrt_bnd_vc.ravel()

            IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
            JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

            if self.imageGT is None:
                data = np.concatenate([data for i in range(num_channels)])

            ij = np.vstack((IS.ravel(), JS.ravel()))
            result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

            result_wrt_vc_bnd = result


        ########### Non boundary derivatives: ####################

        nNonBndFaces = nonBoundaryFaces.size

        ### Derivatives wrt VC:

        # Each pixel relies on three verts
        pixels = np.where(((~boundaryImage) & (visibility != 4294967295)).ravel())[0]
        IS = np.tile(col(pixels), (1, 3)).ravel()
        JS = col(f[nonBoundaryFaces].ravel())

        if self.imageGT is None:
            dImage_wrt_nonbnd_vc  = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
        else:
            dImage_wrt_nonbnd_vc = barycentric[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3))
            dImage_wrt_nonbnd_vc = 2* self.residuals[((~boundaryImage) & (visibility != 4294967295))].reshape((-1, 3)).T[:,:,None] * dImage_wrt_nonbnd_vc[None,:]


        data = np.asarray(dImage_wrt_nonbnd_vc, order='C').ravel()

        IS = np.concatenate([IS * num_channels + k for k in range(num_channels)])
        JS = np.concatenate([JS * num_channels + k for k in range(num_channels)])

        if self.imageGT is None:
            data = np.concatenate([data for i in range(num_channels)])

        ij = np.vstack((IS.ravel(), JS.ravel()))
        result = sp.csc_matrix((data, ij), shape=(width * height * num_channels, vc_size))

        result_wrt_vc_nonbnd = result

        if np.any(boundaryImage):
            result_wrt_vc = result_wrt_vc_bnd + result_wrt_vc_nonbnd
        else:
            result_wrt_vc = result_wrt_vc_nonbnd

        return result_wrt_vc

    def on_changed(self, which):
        super().on_changed(which)

        if 'v' or 'camera' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_verts_mesh[mesh][polygons].set_array(verts_by_face.astype(np.float32))
                    self.vbo_verts_mesh[mesh][polygons].bind()

        if 'vc' in which:
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]
                    colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                    self.vbo_colors_mesh[mesh][polygons].set_array(colors_by_face.astype(np.float32))
                    self.vbo_colors_mesh[mesh][polygons].bind()

        if 'f' in which:
            self.vbo_indices.set_array(self.f.astype(np.uint32))
            self.vbo_indices.bind()

            self.vbo_indices_range.set_array(np.arange(self.f.size, dtype=np.uint32).ravel())
            self.vbo_indices_range.bind()
            flen = 1
            for mesh in range(len(self.f_list)):
                for polygons in range(len(self.f_list[mesh])):
                    f = self.f_list[mesh][polygons]

                    # fc = np.arange(flen, flen + len(f))
                    fc = np.tile(np.arange(flen, flen + len(f))[:, None], [1, 3]).ravel()

                    # fc[:, 0] = fc[:, 0] & 255
                    # fc[:, 1] = (fc[:, 1] >> 8) & 255
                    # fc[:, 2] = (fc[:, 2] >> 16) & 255
                    fc = np.asarray(fc, dtype=np.uint32)
                    self.vbo_face_ids_list[mesh][polygons].set_array(fc)
                    self.vbo_face_ids_list[mesh][polygons].bind()

                    flen += len(f)

                    self.vbo_indices_mesh_list[mesh][polygons].set_array(np.array(self.f_list[mesh][polygons]).astype(np.uint32))
                    self.vbo_indices_mesh_list[mesh][polygons].bind()

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

                            # Update the OpenGL textures with all the textures. (Inefficient as many might not have changed).
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)
                            self.textures_list[mesh][polygons] = self.texture_stack[textureCoordIdx:image.size + textureCoordIdx].reshape(image.shape)

                            textureCoordIdx = textureCoordIdx + image.size
                            image = np.array(np.flipud((self.textures_list[mesh][polygons] * 255.0)), order='C', dtype=np.uint8)

                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                                               image.reshape([image.shape[1], image.shape[0], -1]).ravel().tostring())

        # if 'imageGT' in which:
        #     GL.glActiveTexture(GL.GL_TEXTURE1)
        #     GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureGT)
        #     image = np.array(np.flipud((self.imageGT.r)), order='C', dtype=np.float32)
        #     # GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA, image.shape[1], image.shape[0])
        #     GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL.GL_RGB, GL.GL_FLOAT, image)

        if 'v' or 'f' or 'vc' or 'ft' or 'camera' or 'texture_stack' or 'imageGT' in which:
            self.render_image_buffers()

    def release_textures(self):
        if hasattr(self, 'textureID_mesh_list'):
            if self.textureID_mesh_list != []:
                for texture_mesh in self.textureID_mesh_list:
                    if texture_mesh != []:
                        for texture in texture_mesh:
                            if texture != None:
                                GL.glDeleteTextures(1, [texture.value])

        self.textureID_mesh_list = []

    @depends_on(dterms + terms)
    def color_image(self):
        self._call_on_changed()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        no_overdraw = self.draw_color_image(with_vertex_colors=True, with_texture_on=True)

        return no_overdraw

        # if not self.overdraw:
        #     return no_overdraw
        #
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        # overdraw = self.draw_color_image()
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        #
        # # return overdraw * np.atleast_3d(self.boundarybool_image)
        #
        # boundarybool_image = self.boundarybool_image
        # if self.num_channels > 1:
        #     boundarybool_image = np.atleast_3d(boundarybool_image)
        #
        # return np.asarray((overdraw*boundarybool_image + no_overdraw*(1-boundarybool_image)), order='C')

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_barycentric_image(self.boundarybool_image if self.overdraw else None)

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()
        # Overload method to call without overdraw.
        return self.draw_visibility_image(self.v.r, self.f, self.boundarybool_image if self.overdraw else None)

    def image_mesh_bool(self, meshes):
        self.makeCurrentContext()
        self._call_on_changed()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0., 0., 0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)
        for mesh in meshes:
            self.draw_index(mesh)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.uint32))[:, :, 0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result != 0

    @depends_on(dterms + terms)
    def indices_image(self):
        self._call_on_changed()
        self.makeCurrentContext()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self._call_on_changed()

        GL.glClearColor(0., 0., 0., 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.colorProgram)

        for index in range(len(self.f_list)):
            self.draw_index(index)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.uint32))[:, :, 0]

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)

        return result

    def draw_index(self, index):

        mesh = index

        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        vc = self.vc_list[mesh]

        for polygons in np.arange(len(self.f_list[mesh])):
            vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
            GL.glBindVertexArray(vao_mesh)
            f = self.f_list[mesh][polygons]
            vbo_color = self.vbo_colors_mesh[mesh][polygons]
            colors_by_face = np.asarray(vc.reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
            colors = np.array(np.ones_like(colors_by_face) * (index) / 255.0, dtype=np.float32)

            # Pol: Make a static zero vbo_color to make it more efficient?
            vbo_color.set_array(colors)

            vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

            vbo_color.bind()

            if self.f.shape[1] == 2:
                primtype = GL.GL_LINES
            else:
                primtype = GL.GL_TRIANGLES

            GL.glUniformMatrix4fv(self.MVP_location, 1, GL.GL_TRUE, MVP)

            GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])

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

        # use the third channel to identify the corresponding textures.
        color3 = np.vstack([np.ones([self.ft_list[mesh].shape[0], 1]) * mesh for mesh in range(len(self.ft_list))]).astype(np.float32) / len(
            self.ft_list)

        colors = np.asarray(np.hstack((colors, color3)), np.float64, order='C')
        self.draw_colored_primitives(self.vao_dyn, v, f, colors)

        # Why do we need this?
        if boundarybool_image is not None:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            self.draw_colored_primitives(self.vao_dyn, v, f, colors)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3)[:, :, :3].astype(np.float64)) / 255.0

        result[:, :, 1] = 1. - result[:, :, 1]
        return result

    @depends_on('ft', 'textures')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = self.ft
        # Pol: careful with this:
        data[:, 1] = 1.0 - 1.0 * data[:, 1]
        return data

    # Depends on 'f' because vpe/fpe depend on f
    # Pol: Check that depends on works on other attributes that depend_on x, if x changes.
    @depends_on('ft', 'f')
    def wireframe_tex_coords(self):
        print("wireframe_tex_coords is being computed!")
        vvt = np.zeros((self.v.r.size / 3, 2), dtype=np.float64, order='C')
        vvt[self.f.flatten()] = self.mesh_tex_coords
        edata = np.zeros((self.vpe.size, 2), dtype=np.float64, order='C')
        edata = vvt[self.ma.ravel()]
        return edata

    # TODO: can this not be inherited from base? turning off texture mapping in that instead?
    @depends_on(dterms + terms)
    def boundaryid_image(self):
        self._call_on_changed()

        # self.texture_mapping_of
        self.makeCurrentContext()
        GL.glUseProgram(self.colorProgram)

        result = self.draw_boundaryid_image(self.v.r, self.f, self.vpe, self.fpe, self.camera)

        GL.glUseProgram(self.colorTextureProgram)
        # self.texture_mapping_on(with_vertex_colors=True)

        return result


    @depends_on(dterms + terms)
    def boundaryid_image_aa(self):
        self._call_on_changed()

        # self.texture_mapping_of
        self.makeCurrentContext()
        GL.glUseProgram(self.colorProgram)

        result = self.draw_boundaryid_image_aa(self.v.r, self.f, self.vpe, self.fpe, self.camera)

        GL.glUseProgram(self.colorTextureProgram)
        # self.texture_mapping_on(with_vertex_colors=True)

        return result



    def draw_color_image(self, with_vertex_colors=True, with_texture_on=True):
        self.makeCurrentContext()
        self._call_on_changed()

        GL.glEnable(GL.GL_MULTISAMPLE)

        if hasattr(self, 'bgcolor'):
            GL.glClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels], self.bgcolor.r[2 % self.num_channels], 1.)

        # use face colors if given
        # FIXME: this won't work for 2 channels
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.msaa:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo_noms)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        view_mtx = self.camera.openglMat.dot(np.asarray(np.vstack((self.camera.view_matrix, np.array([0, 0, 0, 1]))), np.float32))
        MVP = np.dot(self.projectionMatrix, view_mtx)

        for mesh in range(len(self.f_list)):

            for polygons in np.arange(len(self.f_list[mesh])):

                vao_mesh = self.vao_tex_mesh_list[mesh][polygons]
                vbo_f = self.vbo_indices_mesh_list[mesh][polygons]

                GL.glBindVertexArray(vao_mesh)
                f = self.f_list[mesh][polygons]
                verts_by_face = np.asarray(self.v_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vbo_color = self.vbo_colors_mesh[mesh][polygons]
                colors_by_face = np.asarray(self.vc_list[mesh].reshape((-1, 3))[f.ravel()], dtype=np.float32, order='C')
                vc = colors_by_face

                if with_vertex_colors:
                    colors = vc.astype(np.float32)
                else:
                    # Only texture.
                    colors = np.ones_like(vc).astype(np.float32)

                # Pol: Make a static zero vbo_color to make it more efficient?
                vbo_color.set_array(colors)
                vbo_color.bind()

                if self.f.shape[1] == 2:
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

                GL.glDrawArrays(primtype, 0, len(vbo_f) * vbo_f.data.shape[1])
                # GL.glDrawElements(primtype, len(vbo_f)*vbo_f.data.shape[1], GL.GL_UNSIGNED_INT, None)

        if self.msaa:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_ms)
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo_noms)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glBlitFramebuffer(0, 0, self.frustum['width'], self.frustum['height'], 0, 0, self.frustum['width'], self.frustum['height'],
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        result = np.flipud(
            np.frombuffer(GL.glReadPixels(0, 0, self.frustum['width'], self.frustum['height'], GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(
                self.frustum['height'], self.frustum['height'], 3).astype(np.float64)) / 255.0

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0., 0., 0., 1.)

        if hasattr(self, 'background_image'):
            bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1, 1, 3))
            fg_px = 1 - bg_px
            result = bg_px * self.background_image + fg_px * result

        return result

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image_quantized(self):

        texcoord_image = self.texcoord_image[:, :, :2].copy()
        # Temprary:
        self.texture_image = self.textures_list[0][0].r.copy()
        texcoord_image[:, :, 0] *= self.texture_image.shape[1] - 1
        texcoord_image[:, :, 1] *= self.texture_image.shape[0] - 1
        texture_idx = (self.texcoord_image[:, :, 2] * len(self.ft_list)).astype(np.uint32)
        texcoord_image = np.round(texcoord_image)
        texcoord_image = texcoord_image[:, :, 0] + texcoord_image[:, :, 1] * self.texture_image.shape[1]

        return texcoord_image, texture_idx

    def checkBufferNum(self):
        GL.glGenBuffers(1)

    @depends_on('ft', 'f', 'frustum', 'camera')
    def texcoord_image(self):
        return self.draw_texcoord_image(self.v.r, self.f, self.ft, self.boundarybool_image if self.overdraw else None)



def main():
    pass

if __name__ == '__main__':
    main()


