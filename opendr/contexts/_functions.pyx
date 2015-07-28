cdef extern from "gl_includes.h":
	ctypedef unsigned int	GLenum
	ctypedef unsigned char	GLboolean
	ctypedef unsigned int	GLbitfield
	ctypedef void		GLvoid
	ctypedef signed char	GLbyte
	ctypedef short		GLshort
	ctypedef int		GLint
	ctypedef unsigned char	GLubyte
	ctypedef unsigned short	GLushort
	ctypedef unsigned int	GLuint
	ctypedef int		GLsizei
	ctypedef float		GLfloat
	ctypedef float		GLclampf
	ctypedef double		GLdouble
	ctypedef double		GLclampd
	ctypedef void* GLeglImageOES
	ctypedef char GLchar
	ctypedef ptrdiff_t GLintptr
	ctypedef ptrdiff_t GLsizeiptr
	cdef void _glClearIndex "glClearIndex"( GLfloat c ) nogil
	cdef void _glClearColor "glClearColor"( GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha ) nogil
	cdef void _glClear "glClear"( GLbitfield mask ) nogil
	cdef void _glIndexMask "glIndexMask"( GLuint mask ) nogil
	cdef void _glColorMask "glColorMask"( GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha ) nogil
	cdef void _glAlphaFunc "glAlphaFunc"( GLenum func, GLclampf ref ) nogil
	cdef void _glBlendFunc "glBlendFunc"( GLenum sfactor, GLenum dfactor ) nogil
	cdef void _glLogicOp "glLogicOp"( GLenum opcode ) nogil
	cdef void _glCullFace "glCullFace"( GLenum mode ) nogil
	cdef void _glFrontFace "glFrontFace"( GLenum mode ) nogil
	cdef void _glPointSize "glPointSize"( GLfloat size ) nogil
	cdef void _glLineWidth "glLineWidth"( GLfloat width ) nogil
	cdef void _glLineStipple "glLineStipple"( GLint factor, GLushort pattern ) nogil
	cdef void _glPolygonMode "glPolygonMode"( GLenum face, GLenum mode ) nogil
	cdef void _glPolygonOffset "glPolygonOffset"( GLfloat factor, GLfloat units ) nogil
	cdef void _glPolygonStipple "glPolygonStipple"(  GLubyte *mask ) 
	cdef void _glGetPolygonStipple "glGetPolygonStipple"( GLubyte *mask ) 
	cdef void _glEdgeFlag "glEdgeFlag"( GLboolean flag ) nogil
	cdef void _glEdgeFlagv "glEdgeFlagv"(  GLboolean *flag ) 
	cdef void _glScissor "glScissor"( GLint x, GLint y, GLsizei width, GLsizei height) nogil
	cdef void _glClipPlane "glClipPlane"( GLenum plane,  GLdouble *equation ) 
	cdef void _glGetClipPlane "glGetClipPlane"( GLenum plane, GLdouble *equation ) 
	cdef void _glDrawBuffer "glDrawBuffer"( GLenum mode ) nogil
	cdef void _glReadBuffer "glReadBuffer"( GLenum mode ) nogil
	cdef void _glEnable "glEnable"( GLenum cap ) nogil
	cdef void _glDisable "glDisable"( GLenum cap ) nogil
	cdef GLboolean _glIsEnabled "glIsEnabled"( GLenum cap ) nogil
	cdef void _glEnableClientState "glEnableClientState"( GLenum cap ) nogil
	cdef void _glDisableClientState "glDisableClientState"( GLenum cap ) nogil
	cdef void _glGetBooleanv "glGetBooleanv"( GLenum pname, GLboolean *params ) 
	cdef void _glGetDoublev "glGetDoublev"( GLenum pname, GLdouble *params ) 
	cdef void _glGetFloatv "glGetFloatv"( GLenum pname, GLfloat *params ) 
	cdef void _glGetIntegerv "glGetIntegerv"( GLenum pname, GLint *params ) 
	cdef void _glPushAttrib "glPushAttrib"( GLbitfield mask ) nogil
	cdef void _glPopAttrib "glPopAttrib"() nogil
	cdef void _glPushClientAttrib "glPushClientAttrib"( GLbitfield mask ) nogil
	cdef void _glPopClientAttrib "glPopClientAttrib"() nogil
	cdef GLint _glRenderMode "glRenderMode"( GLenum mode ) nogil
	cdef GLenum _glGetError "glGetError"() nogil
	cdef void _glFinish "glFinish"() nogil
	cdef void _glFlush "glFlush"() nogil
	cdef void _glHint "glHint"( GLenum target, GLenum mode ) nogil
	cdef void _glClearDepth "glClearDepth"( GLclampd depth ) nogil
	cdef void _glDepthFunc "glDepthFunc"( GLenum func ) nogil
	cdef void _glDepthMask "glDepthMask"( GLboolean flag ) nogil
	cdef void _glDepthRange "glDepthRange"( GLclampd near_val, GLclampd far_val ) nogil
	cdef void _glClearAccum "glClearAccum"( GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha ) nogil
	cdef void _glAccum "glAccum"( GLenum op, GLfloat value ) nogil
	cdef void _glMatrixMode "glMatrixMode"( GLenum mode ) nogil
	cdef void _glOrtho "glOrtho"( GLdouble left, GLdouble right,
                                 GLdouble bottom, GLdouble top,
                                 GLdouble near_val, GLdouble far_val ) nogil
	cdef void _glFrustum "glFrustum"( GLdouble left, GLdouble right,
                                   GLdouble bottom, GLdouble top,
                                   GLdouble near_val, GLdouble far_val ) nogil
	cdef void _glViewport "glViewport"( GLint x, GLint y,
                                    GLsizei width, GLsizei height ) nogil
	cdef void _glPushMatrix "glPushMatrix"() nogil
	cdef void _glPopMatrix "glPopMatrix"() nogil
	cdef void _glLoadIdentity "glLoadIdentity"() nogil
	cdef void _glLoadMatrixd "glLoadMatrixd"(  GLdouble *m ) 
	cdef void _glLoadMatrixf "glLoadMatrixf"(  GLfloat *m ) 
	cdef void _glMultMatrixd "glMultMatrixd"(  GLdouble *m ) 
	cdef void _glMultMatrixf "glMultMatrixf"(  GLfloat *m ) 
	cdef void _glRotated "glRotated"( GLdouble angle,
                                   GLdouble x, GLdouble y, GLdouble z ) nogil
	cdef void _glRotatef "glRotatef"( GLfloat angle,
                                   GLfloat x, GLfloat y, GLfloat z ) nogil
	cdef void _glScaled "glScaled"( GLdouble x, GLdouble y, GLdouble z ) nogil
	cdef void _glScalef "glScalef"( GLfloat x, GLfloat y, GLfloat z ) nogil
	cdef void _glTranslated "glTranslated"( GLdouble x, GLdouble y, GLdouble z ) nogil
	cdef void _glTranslatef "glTranslatef"( GLfloat x, GLfloat y, GLfloat z ) nogil
	cdef GLboolean _glIsList "glIsList"( GLuint list ) nogil
	cdef void _glDeleteLists "glDeleteLists"( GLuint list, GLsizei range ) nogil
	cdef GLuint _glGenLists "glGenLists"( GLsizei range ) nogil
	cdef void _glNewList "glNewList"( GLuint list, GLenum mode ) nogil
	cdef void _glEndList "glEndList"() nogil
	cdef void _glCallList "glCallList"( GLuint list ) nogil
	cdef void _glCallLists "glCallLists"( GLsizei n, GLenum type,
                                      GLvoid *lists ) 
	cdef void _glListBase "glListBase"( GLuint base ) nogil
	cdef void _glBegin "glBegin"( GLenum mode ) nogil
	cdef void _glEnd "glEnd"() nogil
	cdef void _glVertex2d "glVertex2d"( GLdouble x, GLdouble y ) nogil
	cdef void _glVertex2f "glVertex2f"( GLfloat x, GLfloat y ) nogil
	cdef void _glVertex2i "glVertex2i"( GLint x, GLint y ) nogil
	cdef void _glVertex2s "glVertex2s"( GLshort x, GLshort y ) nogil
	cdef void _glVertex3d "glVertex3d"( GLdouble x, GLdouble y, GLdouble z ) nogil
	cdef void _glVertex3f "glVertex3f"( GLfloat x, GLfloat y, GLfloat z ) nogil
	cdef void _glVertex3i "glVertex3i"( GLint x, GLint y, GLint z ) nogil
	cdef void _glVertex3s "glVertex3s"( GLshort x, GLshort y, GLshort z ) nogil
	cdef void _glVertex4d "glVertex4d"( GLdouble x, GLdouble y, GLdouble z, GLdouble w ) nogil
	cdef void _glVertex4f "glVertex4f"( GLfloat x, GLfloat y, GLfloat z, GLfloat w ) nogil
	cdef void _glVertex4i "glVertex4i"( GLint x, GLint y, GLint z, GLint w ) nogil
	cdef void _glVertex4s "glVertex4s"( GLshort x, GLshort y, GLshort z, GLshort w ) nogil
	cdef void _glVertex2dv "glVertex2dv"(  GLdouble *v ) 
	cdef void _glVertex2fv "glVertex2fv"(  GLfloat *v ) 
	cdef void _glVertex2iv "glVertex2iv"(  GLint *v ) 
	cdef void _glVertex2sv "glVertex2sv"(  GLshort *v ) 
	cdef void _glVertex3dv "glVertex3dv"(  GLdouble *v ) 
	cdef void _glVertex3fv "glVertex3fv"(  GLfloat *v ) 
	cdef void _glVertex3iv "glVertex3iv"(  GLint *v ) 
	cdef void _glVertex3sv "glVertex3sv"(  GLshort *v ) 
	cdef void _glVertex4dv "glVertex4dv"(  GLdouble *v ) 
	cdef void _glVertex4fv "glVertex4fv"(  GLfloat *v ) 
	cdef void _glVertex4iv "glVertex4iv"(  GLint *v ) 
	cdef void _glVertex4sv "glVertex4sv"(  GLshort *v ) 
	cdef void _glNormal3b "glNormal3b"( GLbyte nx, GLbyte ny, GLbyte nz ) nogil
	cdef void _glNormal3d "glNormal3d"( GLdouble nx, GLdouble ny, GLdouble nz ) nogil
	cdef void _glNormal3f "glNormal3f"( GLfloat nx, GLfloat ny, GLfloat nz ) nogil
	cdef void _glNormal3i "glNormal3i"( GLint nx, GLint ny, GLint nz ) nogil
	cdef void _glNormal3s "glNormal3s"( GLshort nx, GLshort ny, GLshort nz ) nogil
	cdef void _glNormal3bv "glNormal3bv"(  GLbyte *v ) 
	cdef void _glNormal3dv "glNormal3dv"(  GLdouble *v ) 
	cdef void _glNormal3fv "glNormal3fv"(  GLfloat *v ) 
	cdef void _glNormal3iv "glNormal3iv"(  GLint *v ) 
	cdef void _glNormal3sv "glNormal3sv"(  GLshort *v ) 
	cdef void _glIndexd "glIndexd"( GLdouble c ) nogil
	cdef void _glIndexf "glIndexf"( GLfloat c ) nogil
	cdef void _glIndexi "glIndexi"( GLint c ) nogil
	cdef void _glIndexs "glIndexs"( GLshort c ) nogil
	cdef void _glIndexub "glIndexub"( GLubyte c ) nogil
	cdef void _glIndexdv "glIndexdv"(  GLdouble *c ) 
	cdef void _glIndexfv "glIndexfv"(  GLfloat *c ) 
	cdef void _glIndexiv "glIndexiv"(  GLint *c ) 
	cdef void _glIndexsv "glIndexsv"(  GLshort *c ) 
	cdef void _glIndexubv "glIndexubv"(  GLubyte *c ) 
	cdef void _glColor3b "glColor3b"( GLbyte red, GLbyte green, GLbyte blue ) nogil
	cdef void _glColor3d "glColor3d"( GLdouble red, GLdouble green, GLdouble blue ) nogil
	cdef void _glColor3f "glColor3f"( GLfloat red, GLfloat green, GLfloat blue ) nogil
	cdef void _glColor3i "glColor3i"( GLint red, GLint green, GLint blue ) nogil
	cdef void _glColor3s "glColor3s"( GLshort red, GLshort green, GLshort blue ) nogil
	cdef void _glColor3ub "glColor3ub"( GLubyte red, GLubyte green, GLubyte blue ) nogil
	cdef void _glColor3ui "glColor3ui"( GLuint red, GLuint green, GLuint blue ) nogil
	cdef void _glColor3us "glColor3us"( GLushort red, GLushort green, GLushort blue ) nogil
	cdef void _glColor4b "glColor4b"( GLbyte red, GLbyte green,
                                   GLbyte blue, GLbyte alpha ) nogil
	cdef void _glColor4d "glColor4d"( GLdouble red, GLdouble green,
                                   GLdouble blue, GLdouble alpha ) nogil
	cdef void _glColor4f "glColor4f"( GLfloat red, GLfloat green,
                                   GLfloat blue, GLfloat alpha ) nogil
	cdef void _glColor4i "glColor4i"( GLint red, GLint green,
                                   GLint blue, GLint alpha ) nogil
	cdef void _glColor4s "glColor4s"( GLshort red, GLshort green,
                                   GLshort blue, GLshort alpha ) nogil
	cdef void _glColor4ub "glColor4ub"( GLubyte red, GLubyte green,
                                    GLubyte blue, GLubyte alpha ) nogil
	cdef void _glColor4ui "glColor4ui"( GLuint red, GLuint green,
                                    GLuint blue, GLuint alpha ) nogil
	cdef void _glColor4us "glColor4us"( GLushort red, GLushort green,
                                    GLushort blue, GLushort alpha ) nogil
	cdef void _glColor3bv "glColor3bv"(  GLbyte *v ) 
	cdef void _glColor3dv "glColor3dv"(  GLdouble *v ) 
	cdef void _glColor3fv "glColor3fv"(  GLfloat *v ) 
	cdef void _glColor3iv "glColor3iv"(  GLint *v ) 
	cdef void _glColor3sv "glColor3sv"(  GLshort *v ) 
	cdef void _glColor3ubv "glColor3ubv"(  GLubyte *v ) 
	cdef void _glColor3uiv "glColor3uiv"(  GLuint *v ) 
	cdef void _glColor3usv "glColor3usv"(  GLushort *v ) 
	cdef void _glColor4bv "glColor4bv"(  GLbyte *v ) 
	cdef void _glColor4dv "glColor4dv"(  GLdouble *v ) 
	cdef void _glColor4fv "glColor4fv"(  GLfloat *v ) 
	cdef void _glColor4iv "glColor4iv"(  GLint *v ) 
	cdef void _glColor4sv "glColor4sv"(  GLshort *v ) 
	cdef void _glColor4ubv "glColor4ubv"(  GLubyte *v ) 
	cdef void _glColor4uiv "glColor4uiv"(  GLuint *v ) 
	cdef void _glColor4usv "glColor4usv"(  GLushort *v ) 
	cdef void _glTexCoord1d "glTexCoord1d"( GLdouble s ) nogil
	cdef void _glTexCoord1f "glTexCoord1f"( GLfloat s ) nogil
	cdef void _glTexCoord1i "glTexCoord1i"( GLint s ) nogil
	cdef void _glTexCoord1s "glTexCoord1s"( GLshort s ) nogil
	cdef void _glTexCoord2d "glTexCoord2d"( GLdouble s, GLdouble t ) nogil
	cdef void _glTexCoord2f "glTexCoord2f"( GLfloat s, GLfloat t ) nogil
	cdef void _glTexCoord2i "glTexCoord2i"( GLint s, GLint t ) nogil
	cdef void _glTexCoord2s "glTexCoord2s"( GLshort s, GLshort t ) nogil
	cdef void _glTexCoord3d "glTexCoord3d"( GLdouble s, GLdouble t, GLdouble r ) nogil
	cdef void _glTexCoord3f "glTexCoord3f"( GLfloat s, GLfloat t, GLfloat r ) nogil
	cdef void _glTexCoord3i "glTexCoord3i"( GLint s, GLint t, GLint r ) nogil
	cdef void _glTexCoord3s "glTexCoord3s"( GLshort s, GLshort t, GLshort r ) nogil
	cdef void _glTexCoord4d "glTexCoord4d"( GLdouble s, GLdouble t, GLdouble r, GLdouble q ) nogil
	cdef void _glTexCoord4f "glTexCoord4f"( GLfloat s, GLfloat t, GLfloat r, GLfloat q ) nogil
	cdef void _glTexCoord4i "glTexCoord4i"( GLint s, GLint t, GLint r, GLint q ) nogil
	cdef void _glTexCoord4s "glTexCoord4s"( GLshort s, GLshort t, GLshort r, GLshort q ) nogil
	cdef void _glTexCoord1dv "glTexCoord1dv"(  GLdouble *v ) 
	cdef void _glTexCoord1fv "glTexCoord1fv"(  GLfloat *v ) 
	cdef void _glTexCoord1iv "glTexCoord1iv"(  GLint *v ) 
	cdef void _glTexCoord1sv "glTexCoord1sv"(  GLshort *v ) 
	cdef void _glTexCoord2dv "glTexCoord2dv"(  GLdouble *v ) 
	cdef void _glTexCoord2fv "glTexCoord2fv"(  GLfloat *v ) 
	cdef void _glTexCoord2iv "glTexCoord2iv"(  GLint *v ) 
	cdef void _glTexCoord2sv "glTexCoord2sv"(  GLshort *v ) 
	cdef void _glTexCoord3dv "glTexCoord3dv"(  GLdouble *v ) 
	cdef void _glTexCoord3fv "glTexCoord3fv"(  GLfloat *v ) 
	cdef void _glTexCoord3iv "glTexCoord3iv"(  GLint *v ) 
	cdef void _glTexCoord3sv "glTexCoord3sv"(  GLshort *v ) 
	cdef void _glTexCoord4dv "glTexCoord4dv"(  GLdouble *v ) 
	cdef void _glTexCoord4fv "glTexCoord4fv"(  GLfloat *v ) 
	cdef void _glTexCoord4iv "glTexCoord4iv"(  GLint *v ) 
	cdef void _glTexCoord4sv "glTexCoord4sv"(  GLshort *v ) 
	cdef void _glRasterPos2d "glRasterPos2d"( GLdouble x, GLdouble y ) nogil
	cdef void _glRasterPos2f "glRasterPos2f"( GLfloat x, GLfloat y ) nogil
	cdef void _glRasterPos2i "glRasterPos2i"( GLint x, GLint y ) nogil
	cdef void _glRasterPos2s "glRasterPos2s"( GLshort x, GLshort y ) nogil
	cdef void _glRasterPos3d "glRasterPos3d"( GLdouble x, GLdouble y, GLdouble z ) nogil
	cdef void _glRasterPos3f "glRasterPos3f"( GLfloat x, GLfloat y, GLfloat z ) nogil
	cdef void _glRasterPos3i "glRasterPos3i"( GLint x, GLint y, GLint z ) nogil
	cdef void _glRasterPos3s "glRasterPos3s"( GLshort x, GLshort y, GLshort z ) nogil
	cdef void _glRasterPos4d "glRasterPos4d"( GLdouble x, GLdouble y, GLdouble z, GLdouble w ) nogil
	cdef void _glRasterPos4f "glRasterPos4f"( GLfloat x, GLfloat y, GLfloat z, GLfloat w ) nogil
	cdef void _glRasterPos4i "glRasterPos4i"( GLint x, GLint y, GLint z, GLint w ) nogil
	cdef void _glRasterPos4s "glRasterPos4s"( GLshort x, GLshort y, GLshort z, GLshort w ) nogil
	cdef void _glRasterPos2dv "glRasterPos2dv"(  GLdouble *v ) 
	cdef void _glRasterPos2fv "glRasterPos2fv"(  GLfloat *v ) 
	cdef void _glRasterPos2iv "glRasterPos2iv"(  GLint *v ) 
	cdef void _glRasterPos2sv "glRasterPos2sv"(  GLshort *v ) 
	cdef void _glRasterPos3dv "glRasterPos3dv"(  GLdouble *v ) 
	cdef void _glRasterPos3fv "glRasterPos3fv"(  GLfloat *v ) 
	cdef void _glRasterPos3iv "glRasterPos3iv"(  GLint *v ) 
	cdef void _glRasterPos3sv "glRasterPos3sv"(  GLshort *v ) 
	cdef void _glRasterPos4dv "glRasterPos4dv"(  GLdouble *v ) 
	cdef void _glRasterPos4fv "glRasterPos4fv"(  GLfloat *v ) 
	cdef void _glRasterPos4iv "glRasterPos4iv"(  GLint *v ) 
	cdef void _glRasterPos4sv "glRasterPos4sv"(  GLshort *v ) 
	cdef void _glRectd "glRectd"( GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2 ) nogil
	cdef void _glRectf "glRectf"( GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2 ) nogil
	cdef void _glRecti "glRecti"( GLint x1, GLint y1, GLint x2, GLint y2 ) nogil
	cdef void _glRects "glRects"( GLshort x1, GLshort y1, GLshort x2, GLshort y2 ) nogil
	cdef void _glRectdv "glRectdv"(  GLdouble *v1,  GLdouble *v2 ) 
	cdef void _glRectfv "glRectfv"(  GLfloat *v1,  GLfloat *v2 ) 
	cdef void _glRectiv "glRectiv"(  GLint *v1,  GLint *v2 ) 
	cdef void _glRectsv "glRectsv"(  GLshort *v1,  GLshort *v2 ) 
	cdef void _glVertexPointer "glVertexPointer"( GLint size, GLenum type,
                                       GLsizei stride,  GLvoid *ptr ) 
	cdef void _glNormalPointer "glNormalPointer"( GLenum type, GLsizei stride,
                                        GLvoid *ptr ) 
	cdef void _glColorPointer "glColorPointer"( GLint size, GLenum type,
                                      GLsizei stride,  GLvoid *ptr ) 
	cdef void _glIndexPointer "glIndexPointer"( GLenum type, GLsizei stride,
                                       GLvoid *ptr ) 
	cdef void _glTexCoordPointer "glTexCoordPointer"( GLint size, GLenum type,
                                         GLsizei stride,  GLvoid *ptr ) 
	cdef void _glEdgeFlagPointer "glEdgeFlagPointer"( GLsizei stride,  GLvoid *ptr ) 
	cdef void _glGetPointerv "glGetPointerv"( GLenum pname, GLvoid **params ) 
	cdef void _glArrayElement "glArrayElement"( GLint i ) nogil
	cdef void _glDrawArrays "glDrawArrays"( GLenum mode, GLint first, GLsizei count ) nogil
	cdef void _glDrawElements "glDrawElements"( GLenum mode, GLsizei count,
                                      GLenum type,  GLvoid *indices ) 
	cdef void _glInterleavedArrays "glInterleavedArrays"( GLenum format, GLsizei stride,
                                            GLvoid *pointer ) 
	cdef void _glShadeModel "glShadeModel"( GLenum mode ) nogil
	cdef void _glLightf "glLightf"( GLenum light, GLenum pname, GLfloat param ) nogil
	cdef void _glLighti "glLighti"( GLenum light, GLenum pname, GLint param ) nogil
	cdef void _glLightfv "glLightfv"( GLenum light, GLenum pname,
                                  GLfloat *params ) 
	cdef void _glLightiv "glLightiv"( GLenum light, GLenum pname,
                                  GLint *params ) 
	cdef void _glGetLightfv "glGetLightfv"( GLenum light, GLenum pname,
                                    GLfloat *params ) 
	cdef void _glGetLightiv "glGetLightiv"( GLenum light, GLenum pname,
                                    GLint *params ) 
	cdef void _glLightModelf "glLightModelf"( GLenum pname, GLfloat param ) nogil
	cdef void _glLightModeli "glLightModeli"( GLenum pname, GLint param ) nogil
	cdef void _glLightModelfv "glLightModelfv"( GLenum pname,  GLfloat *params ) 
	cdef void _glLightModeliv "glLightModeliv"( GLenum pname,  GLint *params ) 
	cdef void _glMaterialf "glMaterialf"( GLenum face, GLenum pname, GLfloat param ) nogil
	cdef void _glMateriali "glMateriali"( GLenum face, GLenum pname, GLint param ) nogil
	cdef void _glMaterialfv "glMaterialfv"( GLenum face, GLenum pname,  GLfloat *params ) 
	cdef void _glMaterialiv "glMaterialiv"( GLenum face, GLenum pname,  GLint *params ) 
	cdef void _glGetMaterialfv "glGetMaterialfv"( GLenum face, GLenum pname, GLfloat *params ) 
	cdef void _glGetMaterialiv "glGetMaterialiv"( GLenum face, GLenum pname, GLint *params ) 
	cdef void _glColorMaterial "glColorMaterial"( GLenum face, GLenum mode ) nogil
	cdef void _glPixelZoom "glPixelZoom"( GLfloat xfactor, GLfloat yfactor ) nogil
	cdef void _glPixelStoref "glPixelStoref"( GLenum pname, GLfloat param ) nogil
	cdef void _glPixelStorei "glPixelStorei"( GLenum pname, GLint param ) nogil
	cdef void _glPixelTransferf "glPixelTransferf"( GLenum pname, GLfloat param ) nogil
	cdef void _glPixelTransferi "glPixelTransferi"( GLenum pname, GLint param ) nogil
	cdef void _glPixelMapfv "glPixelMapfv"( GLenum map, GLsizei mapsize,
                                     GLfloat *values ) 
	cdef void _glPixelMapuiv "glPixelMapuiv"( GLenum map, GLsizei mapsize,
                                      GLuint *values ) 
	cdef void _glPixelMapusv "glPixelMapusv"( GLenum map, GLsizei mapsize,
                                      GLushort *values ) 
	cdef void _glGetPixelMapfv "glGetPixelMapfv"( GLenum map, GLfloat *values ) 
	cdef void _glGetPixelMapuiv "glGetPixelMapuiv"( GLenum map, GLuint *values ) 
	cdef void _glGetPixelMapusv "glGetPixelMapusv"( GLenum map, GLushort *values ) 
	cdef void _glBitmap "glBitmap"( GLsizei width, GLsizei height,
                                GLfloat xorig, GLfloat yorig,
                                GLfloat xmove, GLfloat ymove,
                                 GLubyte *bitmap ) 
	cdef void _glReadPixels "glReadPixels"( GLint x, GLint y,
                                    GLsizei width, GLsizei height,
                                    GLenum format, GLenum type,
                                    GLvoid *pixels ) nogil
	cdef void _glDrawPixels "glDrawPixels"( GLsizei width, GLsizei height,
                                    GLenum format, GLenum type,
                                     GLvoid *pixels ) 
	cdef void _glCopyPixels "glCopyPixels"( GLint x, GLint y,
                                    GLsizei width, GLsizei height,
                                    GLenum type ) nogil
	cdef void _glStencilFunc "glStencilFunc"( GLenum func, GLint ref, GLuint mask ) nogil
	cdef void _glStencilMask "glStencilMask"( GLuint mask ) nogil
	cdef void _glStencilOp "glStencilOp"( GLenum fail, GLenum zfail, GLenum zpass ) nogil
	cdef void _glClearStencil "glClearStencil"( GLint s ) nogil
	cdef void _glTexGend "glTexGend"( GLenum coord, GLenum pname, GLdouble param ) nogil
	cdef void _glTexGenf "glTexGenf"( GLenum coord, GLenum pname, GLfloat param ) nogil
	cdef void _glTexGeni "glTexGeni"( GLenum coord, GLenum pname, GLint param ) nogil
	cdef void _glTexGendv "glTexGendv"( GLenum coord, GLenum pname,  GLdouble *params ) 
	cdef void _glTexGenfv "glTexGenfv"( GLenum coord, GLenum pname,  GLfloat *params ) 
	cdef void _glTexGeniv "glTexGeniv"( GLenum coord, GLenum pname,  GLint *params ) 
	cdef void _glGetTexGendv "glGetTexGendv"( GLenum coord, GLenum pname, GLdouble *params ) 
	cdef void _glGetTexGenfv "glGetTexGenfv"( GLenum coord, GLenum pname, GLfloat *params ) 
	cdef void _glGetTexGeniv "glGetTexGeniv"( GLenum coord, GLenum pname, GLint *params ) 
	cdef void _glTexEnvf "glTexEnvf"( GLenum target, GLenum pname, GLfloat param ) nogil
	cdef void _glTexEnvi "glTexEnvi"( GLenum target, GLenum pname, GLint param ) nogil
	cdef void _glTexEnvfv "glTexEnvfv"( GLenum target, GLenum pname,  GLfloat *params ) 
	cdef void _glTexEnviv "glTexEnviv"( GLenum target, GLenum pname,  GLint *params ) 
	cdef void _glGetTexEnvfv "glGetTexEnvfv"( GLenum target, GLenum pname, GLfloat *params ) 
	cdef void _glGetTexEnviv "glGetTexEnviv"( GLenum target, GLenum pname, GLint *params ) 
	cdef void _glTexParameterf "glTexParameterf"( GLenum target, GLenum pname, GLfloat param ) nogil
	cdef void _glTexParameteri "glTexParameteri"( GLenum target, GLenum pname, GLint param ) nogil
	cdef void _glTexParameterfv "glTexParameterfv"( GLenum target, GLenum pname,
                                           GLfloat *params ) 
	cdef void _glTexParameteriv "glTexParameteriv"( GLenum target, GLenum pname,
                                           GLint *params ) 
	cdef void _glGetTexParameterfv "glGetTexParameterfv"( GLenum target,
                                           GLenum pname, GLfloat *params) 
	cdef void _glGetTexParameteriv "glGetTexParameteriv"( GLenum target,
                                           GLenum pname, GLint *params ) 
	cdef void _glGetTexLevelParameterfv "glGetTexLevelParameterfv"( GLenum target, GLint level,
                                                GLenum pname, GLfloat *params ) 
	cdef void _glGetTexLevelParameteriv "glGetTexLevelParameteriv"( GLenum target, GLint level,
                                                GLenum pname, GLint *params ) 
	cdef void _glTexImage1D "glTexImage1D"( GLenum target, GLint level,
                                    GLint internalFormat,
                                    GLsizei width, GLint border,
                                    GLenum format, GLenum type,
                                     GLvoid *pixels ) 
	cdef void _glTexImage2D "glTexImage2D"( GLenum target, GLint level,
                                    GLint internalFormat,
                                    GLsizei width, GLsizei height,
                                    GLint border, GLenum format, GLenum type,
                                     GLvoid *pixels ) 
	cdef void _glGetTexImage "glGetTexImage"( GLenum target, GLint level,
                                     GLenum format, GLenum type,
                                     GLvoid *pixels ) 
	cdef void _glGenTextures "glGenTextures"( GLsizei n, GLuint *textures ) 
	cdef void _glDeleteTextures "glDeleteTextures"( GLsizei n,  GLuint *textures) 
	cdef void _glBindTexture "glBindTexture"( GLenum target, GLuint texture ) nogil
	cdef void _glPrioritizeTextures "glPrioritizeTextures"( GLsizei n,
                                             GLuint *textures,
                                             GLclampf *priorities ) 
	cdef GLboolean _glAreTexturesResident "glAreTexturesResident"( GLsizei n,
                                                   GLuint *textures,
                                                  GLboolean *residences ) 
	cdef GLboolean _glIsTexture "glIsTexture"( GLuint texture ) nogil
	cdef void _glTexSubImage1D "glTexSubImage1D"( GLenum target, GLint level,
                                       GLint xoffset,
                                       GLsizei width, GLenum format,
                                       GLenum type,  GLvoid *pixels ) 
	cdef void _glTexSubImage2D "glTexSubImage2D"( GLenum target, GLint level,
                                       GLint xoffset, GLint yoffset,
                                       GLsizei width, GLsizei height,
                                       GLenum format, GLenum type,
                                        GLvoid *pixels ) 
	cdef void _glCopyTexImage1D "glCopyTexImage1D"( GLenum target, GLint level,
                                        GLenum internalformat,
                                        GLint x, GLint y,
                                        GLsizei width, GLint border ) nogil
	cdef void _glCopyTexImage2D "glCopyTexImage2D"( GLenum target, GLint level,
                                        GLenum internalformat,
                                        GLint x, GLint y,
                                        GLsizei width, GLsizei height,
                                        GLint border ) nogil
	cdef void _glCopyTexSubImage1D "glCopyTexSubImage1D"( GLenum target, GLint level,
                                           GLint xoffset, GLint x, GLint y,
                                           GLsizei width ) nogil
	cdef void _glCopyTexSubImage2D "glCopyTexSubImage2D"( GLenum target, GLint level,
                                           GLint xoffset, GLint yoffset,
                                           GLint x, GLint y,
                                           GLsizei width, GLsizei height ) nogil
	cdef void _glMap1d "glMap1d"( GLenum target, GLdouble u1, GLdouble u2,
                               GLint stride,
                               GLint order,  GLdouble *points ) 
	cdef void _glMap1f "glMap1f"( GLenum target, GLfloat u1, GLfloat u2,
                               GLint stride,
                               GLint order,  GLfloat *points ) 
	cdef void _glMap2d "glMap2d"( GLenum target,
		     GLdouble u1, GLdouble u2, GLint ustride, GLint uorder,
		     GLdouble v1, GLdouble v2, GLint vstride, GLint vorder,
		      GLdouble *points ) 
	cdef void _glMap2f "glMap2f"( GLenum target,
		     GLfloat u1, GLfloat u2, GLint ustride, GLint uorder,
		     GLfloat v1, GLfloat v2, GLint vstride, GLint vorder,
		      GLfloat *points ) 
	cdef void _glGetMapdv "glGetMapdv"( GLenum target, GLenum query, GLdouble *v ) 
	cdef void _glGetMapfv "glGetMapfv"( GLenum target, GLenum query, GLfloat *v ) 
	cdef void _glGetMapiv "glGetMapiv"( GLenum target, GLenum query, GLint *v ) 
	cdef void _glEvalCoord1d "glEvalCoord1d"( GLdouble u ) nogil
	cdef void _glEvalCoord1f "glEvalCoord1f"( GLfloat u ) nogil
	cdef void _glEvalCoord1dv "glEvalCoord1dv"(  GLdouble *u ) 
	cdef void _glEvalCoord1fv "glEvalCoord1fv"(  GLfloat *u ) 
	cdef void _glEvalCoord2d "glEvalCoord2d"( GLdouble u, GLdouble v ) nogil
	cdef void _glEvalCoord2f "glEvalCoord2f"( GLfloat u, GLfloat v ) nogil
	cdef void _glEvalCoord2dv "glEvalCoord2dv"(  GLdouble *u ) 
	cdef void _glEvalCoord2fv "glEvalCoord2fv"(  GLfloat *u ) 
	cdef void _glMapGrid1d "glMapGrid1d"( GLint un, GLdouble u1, GLdouble u2 ) nogil
	cdef void _glMapGrid1f "glMapGrid1f"( GLint un, GLfloat u1, GLfloat u2 ) nogil
	cdef void _glMapGrid2d "glMapGrid2d"( GLint un, GLdouble u1, GLdouble u2,
                                   GLint vn, GLdouble v1, GLdouble v2 ) nogil
	cdef void _glMapGrid2f "glMapGrid2f"( GLint un, GLfloat u1, GLfloat u2,
                                   GLint vn, GLfloat v1, GLfloat v2 ) nogil
	cdef void _glEvalPoint1 "glEvalPoint1"( GLint i ) nogil
	cdef void _glEvalPoint2 "glEvalPoint2"( GLint i, GLint j ) nogil
	cdef void _glEvalMesh1 "glEvalMesh1"( GLenum mode, GLint i1, GLint i2 ) nogil
	cdef void _glEvalMesh2 "glEvalMesh2"( GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2 ) nogil
	cdef void _glFogf "glFogf"( GLenum pname, GLfloat param ) nogil
	cdef void _glFogi "glFogi"( GLenum pname, GLint param ) nogil
	cdef void _glFogfv "glFogfv"( GLenum pname,  GLfloat *params ) 
	cdef void _glFogiv "glFogiv"( GLenum pname,  GLint *params ) 
	cdef void _glFeedbackBuffer "glFeedbackBuffer"( GLsizei size, GLenum type, GLfloat *buffer ) 
	cdef void _glPassThrough "glPassThrough"( GLfloat token ) nogil
	cdef void _glSelectBuffer "glSelectBuffer"( GLsizei size, GLuint *buffer ) 
	cdef void _glInitNames "glInitNames"() nogil
	cdef void _glLoadName "glLoadName"( GLuint name ) nogil
	cdef void _glPushName "glPushName"( GLuint name ) nogil
	cdef void _glPopName "glPopName"() nogil
	cdef void _glDrawRangeElements "glDrawRangeElements"( GLenum mode, GLuint start,
	GLuint end, GLsizei count, GLenum type,  GLvoid *indices ) 
	cdef void _glTexImage3D "glTexImage3D"( GLenum target, GLint level,
                                      GLint internalFormat,
                                      GLsizei width, GLsizei height,
                                      GLsizei depth, GLint border,
                                      GLenum format, GLenum type,
                                       GLvoid *pixels ) 
	cdef void _glTexSubImage3D "glTexSubImage3D"( GLenum target, GLint level,
                                         GLint xoffset, GLint yoffset,
                                         GLint zoffset, GLsizei width,
                                         GLsizei height, GLsizei depth,
                                         GLenum format,
                                         GLenum type,  GLvoid *pixels) 
	cdef void _glCopyTexSubImage3D "glCopyTexSubImage3D"( GLenum target, GLint level,
                                             GLint xoffset, GLint yoffset,
                                             GLint zoffset, GLint x,
                                             GLint y, GLsizei width,
                                             GLsizei height ) nogil
	cdef void _glColorTable "glColorTable"( GLenum target, GLenum internalformat,
                                    GLsizei width, GLenum format,
                                    GLenum type,  GLvoid *table ) 
	cdef void _glColorSubTable "glColorSubTable"( GLenum target,
                                       GLsizei start, GLsizei count,
                                       GLenum format, GLenum type,
                                        GLvoid *data ) 
	cdef void _glColorTableParameteriv "glColorTableParameteriv"(GLenum target, GLenum pname,
                                               GLint *params) 
	cdef void _glColorTableParameterfv "glColorTableParameterfv"(GLenum target, GLenum pname,
                                               GLfloat *params) 
	cdef void _glCopyColorSubTable "glCopyColorSubTable"( GLenum target, GLsizei start,
                                           GLint x, GLint y, GLsizei width ) nogil
	cdef void _glCopyColorTable "glCopyColorTable"( GLenum target, GLenum internalformat,
                                        GLint x, GLint y, GLsizei width ) nogil
	cdef void _glGetColorTable "glGetColorTable"( GLenum target, GLenum format,
                                       GLenum type, GLvoid *table ) 
	cdef void _glGetColorTableParameterfv "glGetColorTableParameterfv"( GLenum target, GLenum pname,
                                                  GLfloat *params ) 
	cdef void _glGetColorTableParameteriv "glGetColorTableParameteriv"( GLenum target, GLenum pname,
                                                  GLint *params ) 
	cdef void _glBlendEquation "glBlendEquation"( GLenum mode ) nogil
	cdef void _glBlendColor "glBlendColor"( GLclampf red, GLclampf green,
                                    GLclampf blue, GLclampf alpha ) nogil
	cdef void _glHistogram "glHistogram"( GLenum target, GLsizei width,
				   GLenum internalformat, GLboolean sink ) nogil
	cdef void _glResetHistogram "glResetHistogram"( GLenum target ) nogil
	cdef void _glGetHistogram "glGetHistogram"( GLenum target, GLboolean reset,
				      GLenum format, GLenum type,
				      GLvoid *values ) 
	cdef void _glGetHistogramParameterfv "glGetHistogramParameterfv"( GLenum target, GLenum pname,
						 GLfloat *params ) 
	cdef void _glGetHistogramParameteriv "glGetHistogramParameteriv"( GLenum target, GLenum pname,
						 GLint *params ) 
	cdef void _glMinmax "glMinmax"( GLenum target, GLenum internalformat,
				GLboolean sink ) nogil
	cdef void _glResetMinmax "glResetMinmax"( GLenum target ) nogil
	cdef void _glGetMinmax "glGetMinmax"( GLenum target, GLboolean reset,
                                   GLenum format, GLenum types,
                                   GLvoid *values ) 
	cdef void _glGetMinmaxParameterfv "glGetMinmaxParameterfv"( GLenum target, GLenum pname,
					      GLfloat *params ) 
	cdef void _glGetMinmaxParameteriv "glGetMinmaxParameteriv"( GLenum target, GLenum pname,
					      GLint *params ) 
	cdef void _glConvolutionFilter1D "glConvolutionFilter1D"( GLenum target,
	GLenum internalformat, GLsizei width, GLenum format, GLenum type,
	 GLvoid *image ) 
	cdef void _glConvolutionFilter2D "glConvolutionFilter2D"( GLenum target,
	GLenum internalformat, GLsizei width, GLsizei height, GLenum format,
	GLenum type,  GLvoid *image ) 
	cdef void _glConvolutionParameterf "glConvolutionParameterf"( GLenum target, GLenum pname,
	GLfloat params ) nogil
	cdef void _glConvolutionParameterfv "glConvolutionParameterfv"( GLenum target, GLenum pname,
	 GLfloat *params ) 
	cdef void _glConvolutionParameteri "glConvolutionParameteri"( GLenum target, GLenum pname,
	GLint params ) nogil
	cdef void _glConvolutionParameteriv "glConvolutionParameteriv"( GLenum target, GLenum pname,
	 GLint *params ) 
	cdef void _glCopyConvolutionFilter1D "glCopyConvolutionFilter1D"( GLenum target,
	GLenum internalformat, GLint x, GLint y, GLsizei width ) nogil
	cdef void _glCopyConvolutionFilter2D "glCopyConvolutionFilter2D"( GLenum target,
	GLenum internalformat, GLint x, GLint y, GLsizei width,
	GLsizei height) nogil
	cdef void _glGetConvolutionFilter "glGetConvolutionFilter"( GLenum target, GLenum format,
	GLenum type, GLvoid *image ) 
	cdef void _glGetConvolutionParameterfv "glGetConvolutionParameterfv"( GLenum target, GLenum pname,
	GLfloat *params ) 
	cdef void _glGetConvolutionParameteriv "glGetConvolutionParameteriv"( GLenum target, GLenum pname,
	GLint *params ) 
	cdef void _glSeparableFilter2D "glSeparableFilter2D"( GLenum target,
	GLenum internalformat, GLsizei width, GLsizei height, GLenum format,
	GLenum type,  GLvoid *row,  GLvoid *column ) 
	cdef void _glGetSeparableFilter "glGetSeparableFilter"( GLenum target, GLenum format,
	GLenum type, GLvoid *row, GLvoid *column, GLvoid *span ) 
	cdef void _glActiveTexture "glActiveTexture"( GLenum texture ) nogil
	cdef void _glClientActiveTexture "glClientActiveTexture"( GLenum texture ) nogil
	cdef void _glCompressedTexImage1D "glCompressedTexImage1D"( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glCompressedTexImage2D "glCompressedTexImage2D"( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glCompressedTexImage3D "glCompressedTexImage3D"( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glCompressedTexSubImage1D "glCompressedTexSubImage1D"( GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glCompressedTexSubImage2D "glCompressedTexSubImage2D"( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glCompressedTexSubImage3D "glCompressedTexSubImage3D"( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize,  GLvoid *data ) 
	cdef void _glGetCompressedTexImage "glGetCompressedTexImage"( GLenum target, GLint lod, GLvoid *img ) 
	cdef void _glMultiTexCoord1d "glMultiTexCoord1d"( GLenum target, GLdouble s ) nogil
	cdef void _glMultiTexCoord1dv "glMultiTexCoord1dv"( GLenum target,  GLdouble *v ) 
	cdef void _glMultiTexCoord1f "glMultiTexCoord1f"( GLenum target, GLfloat s ) nogil
	cdef void _glMultiTexCoord1fv "glMultiTexCoord1fv"( GLenum target,  GLfloat *v ) 
	cdef void _glMultiTexCoord1i "glMultiTexCoord1i"( GLenum target, GLint s ) nogil
	cdef void _glMultiTexCoord1iv "glMultiTexCoord1iv"( GLenum target,  GLint *v ) 
	cdef void _glMultiTexCoord1s "glMultiTexCoord1s"( GLenum target, GLshort s ) nogil
	cdef void _glMultiTexCoord1sv "glMultiTexCoord1sv"( GLenum target,  GLshort *v ) 
	cdef void _glMultiTexCoord2d "glMultiTexCoord2d"( GLenum target, GLdouble s, GLdouble t ) nogil
	cdef void _glMultiTexCoord2dv "glMultiTexCoord2dv"( GLenum target,  GLdouble *v ) 
	cdef void _glMultiTexCoord2f "glMultiTexCoord2f"( GLenum target, GLfloat s, GLfloat t ) nogil
	cdef void _glMultiTexCoord2fv "glMultiTexCoord2fv"( GLenum target,  GLfloat *v ) 
	cdef void _glMultiTexCoord2i "glMultiTexCoord2i"( GLenum target, GLint s, GLint t ) nogil
	cdef void _glMultiTexCoord2iv "glMultiTexCoord2iv"( GLenum target,  GLint *v ) 
	cdef void _glMultiTexCoord2s "glMultiTexCoord2s"( GLenum target, GLshort s, GLshort t ) nogil
	cdef void _glMultiTexCoord2sv "glMultiTexCoord2sv"( GLenum target,  GLshort *v ) 
	cdef void _glMultiTexCoord3d "glMultiTexCoord3d"( GLenum target, GLdouble s, GLdouble t, GLdouble r ) nogil
	cdef void _glMultiTexCoord3dv "glMultiTexCoord3dv"( GLenum target,  GLdouble *v ) 
	cdef void _glMultiTexCoord3f "glMultiTexCoord3f"( GLenum target, GLfloat s, GLfloat t, GLfloat r ) nogil
	cdef void _glMultiTexCoord3fv "glMultiTexCoord3fv"( GLenum target,  GLfloat *v ) 
	cdef void _glMultiTexCoord3i "glMultiTexCoord3i"( GLenum target, GLint s, GLint t, GLint r ) nogil
	cdef void _glMultiTexCoord3iv "glMultiTexCoord3iv"( GLenum target,  GLint *v ) 
	cdef void _glMultiTexCoord3s "glMultiTexCoord3s"( GLenum target, GLshort s, GLshort t, GLshort r ) nogil
	cdef void _glMultiTexCoord3sv "glMultiTexCoord3sv"( GLenum target,  GLshort *v ) 
	cdef void _glMultiTexCoord4d "glMultiTexCoord4d"( GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q ) nogil
	cdef void _glMultiTexCoord4dv "glMultiTexCoord4dv"( GLenum target,  GLdouble *v ) 
	cdef void _glMultiTexCoord4f "glMultiTexCoord4f"( GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q ) nogil
	cdef void _glMultiTexCoord4fv "glMultiTexCoord4fv"( GLenum target,  GLfloat *v ) 
	cdef void _glMultiTexCoord4i "glMultiTexCoord4i"( GLenum target, GLint s, GLint t, GLint r, GLint q ) nogil
	cdef void _glMultiTexCoord4iv "glMultiTexCoord4iv"( GLenum target,  GLint *v ) 
	cdef void _glMultiTexCoord4s "glMultiTexCoord4s"( GLenum target, GLshort s, GLshort t, GLshort r, GLshort q ) nogil
	cdef void _glMultiTexCoord4sv "glMultiTexCoord4sv"( GLenum target,  GLshort *v ) 
	cdef void _glLoadTransposeMatrixd "glLoadTransposeMatrixd"(  GLdouble m[16] ) nogil
	cdef void _glLoadTransposeMatrixf "glLoadTransposeMatrixf"(  GLfloat m[16] ) nogil
	cdef void _glMultTransposeMatrixd "glMultTransposeMatrixd"(  GLdouble m[16] ) nogil
	cdef void _glMultTransposeMatrixf "glMultTransposeMatrixf"(  GLfloat m[16] ) nogil
	cdef void _glSampleCoverage "glSampleCoverage"( GLclampf value, GLboolean invert ) nogil
	cdef void _glActiveTextureARB "glActiveTextureARB"(GLenum texture) nogil
	cdef void _glClientActiveTextureARB "glClientActiveTextureARB"(GLenum texture) nogil
	cdef void _glMultiTexCoord1dARB "glMultiTexCoord1dARB"(GLenum target, GLdouble s) nogil
	cdef void _glMultiTexCoord1dvARB "glMultiTexCoord1dvARB"(GLenum target,  GLdouble *v) 
	cdef void _glMultiTexCoord1fARB "glMultiTexCoord1fARB"(GLenum target, GLfloat s) nogil
	cdef void _glMultiTexCoord1fvARB "glMultiTexCoord1fvARB"(GLenum target,  GLfloat *v) 
	cdef void _glMultiTexCoord1iARB "glMultiTexCoord1iARB"(GLenum target, GLint s) nogil
	cdef void _glMultiTexCoord1ivARB "glMultiTexCoord1ivARB"(GLenum target,  GLint *v) 
	cdef void _glMultiTexCoord1sARB "glMultiTexCoord1sARB"(GLenum target, GLshort s) nogil
	cdef void _glMultiTexCoord1svARB "glMultiTexCoord1svARB"(GLenum target,  GLshort *v) 
	cdef void _glMultiTexCoord2dARB "glMultiTexCoord2dARB"(GLenum target, GLdouble s, GLdouble t) nogil
	cdef void _glMultiTexCoord2dvARB "glMultiTexCoord2dvARB"(GLenum target,  GLdouble *v) 
	cdef void _glMultiTexCoord2fARB "glMultiTexCoord2fARB"(GLenum target, GLfloat s, GLfloat t) nogil
	cdef void _glMultiTexCoord2fvARB "glMultiTexCoord2fvARB"(GLenum target,  GLfloat *v) 
	cdef void _glMultiTexCoord2iARB "glMultiTexCoord2iARB"(GLenum target, GLint s, GLint t) nogil
	cdef void _glMultiTexCoord2ivARB "glMultiTexCoord2ivARB"(GLenum target,  GLint *v) 
	cdef void _glMultiTexCoord2sARB "glMultiTexCoord2sARB"(GLenum target, GLshort s, GLshort t) nogil
	cdef void _glMultiTexCoord2svARB "glMultiTexCoord2svARB"(GLenum target,  GLshort *v) 
	cdef void _glMultiTexCoord3dARB "glMultiTexCoord3dARB"(GLenum target, GLdouble s, GLdouble t, GLdouble r) nogil
	cdef void _glMultiTexCoord3dvARB "glMultiTexCoord3dvARB"(GLenum target,  GLdouble *v) 
	cdef void _glMultiTexCoord3fARB "glMultiTexCoord3fARB"(GLenum target, GLfloat s, GLfloat t, GLfloat r) nogil
	cdef void _glMultiTexCoord3fvARB "glMultiTexCoord3fvARB"(GLenum target,  GLfloat *v) 
	cdef void _glMultiTexCoord3iARB "glMultiTexCoord3iARB"(GLenum target, GLint s, GLint t, GLint r) nogil
	cdef void _glMultiTexCoord3ivARB "glMultiTexCoord3ivARB"(GLenum target,  GLint *v) 
	cdef void _glMultiTexCoord3sARB "glMultiTexCoord3sARB"(GLenum target, GLshort s, GLshort t, GLshort r) nogil
	cdef void _glMultiTexCoord3svARB "glMultiTexCoord3svARB"(GLenum target,  GLshort *v) 
	cdef void _glMultiTexCoord4dARB "glMultiTexCoord4dARB"(GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q) nogil
	cdef void _glMultiTexCoord4dvARB "glMultiTexCoord4dvARB"(GLenum target,  GLdouble *v) 
	cdef void _glMultiTexCoord4fARB "glMultiTexCoord4fARB"(GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q) nogil
	cdef void _glMultiTexCoord4fvARB "glMultiTexCoord4fvARB"(GLenum target,  GLfloat *v) 
	cdef void _glMultiTexCoord4iARB "glMultiTexCoord4iARB"(GLenum target, GLint s, GLint t, GLint r, GLint q) nogil
	cdef void _glMultiTexCoord4ivARB "glMultiTexCoord4ivARB"(GLenum target,  GLint *v) 
	cdef void _glMultiTexCoord4sARB "glMultiTexCoord4sARB"(GLenum target, GLshort s, GLshort t, GLshort r, GLshort q) nogil
	cdef void _glMultiTexCoord4svARB "glMultiTexCoord4svARB"(GLenum target,  GLshort *v) 
	cdef void _glGetProgramRegisterfvMESA "glGetProgramRegisterfvMESA"(GLenum target, GLsizei len,  GLubyte *name, GLfloat *v) 
	cdef void _glBlendEquationSeparateATI "glBlendEquationSeparateATI"( GLenum modeRGB, GLenum modeA ) nogil
	cdef void _glGenerateMipmap "glGenerateMipmap"(GLenum target) nogil
	cdef void _glGenerateMipmapEXT "glGenerateMipmapEXT"(GLenum target) nogil
	cdef void _glGenerateTextureMipmapEXT "glGenerateTextureMipmapEXT"(GLuint texture, GLenum target) nogil
	cdef void _glGenerateMultiTexMipmapEXT "glGenerateMultiTexMipmapEXT"(GLenum texunit, GLenum target) nogil
	cdef void _glGenBuffers "glGenBuffers"(GLsizei n, GLuint *buffers) 
	cdef void _glGenBuffersARB "glGenBuffersARB"(GLsizei n, GLuint *buffers) 
	cdef void _glBindBuffer "glBindBuffer"(GLenum target, GLuint buffer) nogil
	cdef void _glBindBufferRange "glBindBufferRange"(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) nogil
	cdef void _glBindBufferBase "glBindBufferBase"(GLenum target, GLuint index, GLuint buffer) nogil
	cdef void _glBindBufferARB "glBindBufferARB"(GLenum target, GLuint buffer) nogil
	cdef void _glBindBufferRangeNV "glBindBufferRangeNV"(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) nogil
	cdef void _glBindBufferOffsetNV "glBindBufferOffsetNV"(GLenum target, GLuint index, GLuint buffer, GLintptr offset) nogil
	cdef void _glBindBufferBaseNV "glBindBufferBaseNV"(GLenum target, GLuint index, GLuint buffer) nogil
	cdef void _glBindBufferRangeEXT "glBindBufferRangeEXT"(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) nogil
	cdef void _glBindBufferOffsetEXT "glBindBufferOffsetEXT"(GLenum target, GLuint index, GLuint buffer, GLintptr offset) nogil
	cdef void _glBindBufferBaseEXT "glBindBufferBaseEXT"(GLenum target, GLuint index, GLuint buffer) nogil
	cdef void _glBufferData "glBufferData"(GLenum target, GLsizeiptr size,  GLvoid *data, GLenum usage) 
	cdef void _glGenVertexArrays "glGenVertexArrays"(GLsizei n, GLuint *arrays) 
	cdef void _glBindVertexArray "glBindVertexArray"(GLuint array) nogil
	cdef void _glVertexAttrib4fv "glVertexAttrib4fv"(GLuint index,  GLfloat *v) 
	cdef void _glDeleteVertexArrays "glDeleteVertexArrays"(GLsizei n,  GLuint *arrays) 
	cdef void _glClearBufferfv "glClearBufferfv"(GLenum buffer, GLint drawbuffer,  GLfloat *value) 
	cdef void _glEnableVertexAttribArray "glEnableVertexAttribArray"(GLuint index) nogil
	cdef void _glVertexAttribPointer "glVertexAttribPointer"(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride,  GLvoid *pointer) 
	cdef void _glValidateProgram "glValidateProgram"(GLuint program) nogil
	cdef void _glGetProgramInfoLog "glGetProgramInfoLog"(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) 
	cdef void _glDeleteProgram "glDeleteProgram"(GLuint program) nogil
	cdef GLuint _glCreateProgram "glCreateProgram"() nogil
	cdef GLuint _glCreateShader "glCreateShader"(GLenum type) nogil
	cdef void _glShaderSource "glShaderSource"(GLuint shader, GLsizei count,  GLchar* *string,  GLint *length) 
	cdef void _glCompileShader "glCompileShader"(GLuint shader) nogil
	cdef void _glAttachShader "glAttachShader"(GLuint program, GLuint shader) nogil
	cdef void _glLinkProgram "glLinkProgram"(GLuint program) nogil
	cdef void _glUseProgram "glUseProgram"(GLuint program) nogil
	cdef GLint _glGetUniformLocation "glGetUniformLocation"(GLuint program,  GLchar *name) 
	cdef void _glUniform1i "glUniform1i"(GLint location, GLint v0) nogil
	cdef void _glUniform4f "glUniform4f"(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) nogil
	cdef void _glUniform1f "glUniform1f"(GLint location, GLfloat v0) nogil
	cdef GLint _glGetAttribLocation "glGetAttribLocation"(GLuint program,  GLchar *name) 
	cdef void _glBindFragDataLocation "glBindFragDataLocation"(GLuint program, GLuint color,  GLchar *name) 
	cdef void _glUniformMatrix4fv "glUniformMatrix4fv"(GLint location, GLsizei count, GLboolean transpose,  GLfloat *value) 
