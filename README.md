# Modern (OpenGL 3.3+) version similar to OpenDR
This code is initially based on the work of Matt Loper's OpenDR: [https://github.com/mattloper/opendr/wiki].
API has changed to adapt and add some functionality, so it's not fully compatible with the original OpenDR, please look at the demos in this repository to see how to run it. Note: some bugs have been noticed related to people trying OpenDR and chumpy using later versions of python, I'll look into this problem asap.

For my [projects](https://github.com/polmorenoc/inversegraphics), I've had to update the implementation with:
- Updated to be used with Python 3.4+
- Requirements: 
    - GFLW 3.0+ software (http://www.glfw.org/) and python package (e.g. pip install glfw). 
    - PyOpenGL python package.
- Updated OpenDR to use OpenGL 3.3+ methods by using the modern pipepline (instead of the old fixed function pipeline). I also use shaders for rendering. This opens up the ability to use modern features (e.g. MSAA) and take more advantage of newer GPUs.
- Two main modes are used, either MESA (OSMESA) or GLFW. The first allows the code to be easily run on headless servers, but is a software based implementation of OpenGL (i.e. runs on CPU). The second one allows running on GPUs without popping up an OpenGL window. In order for it to be fully headless (to run on most servers) you'd need to have NVIDIA GPUs with a [certain X configuration](http://www.nvidia.com/content/PDF/remote-viz-tesla-gpus.pdf) (see section on "SETTING UP THE X SERVER FOR HEADLESS OPERATION"). If you want to use MESA you might need to link to the Mesa libraries and set PYOPENGL_PLATFORM=osmesa environment variable.
- The code I've implemented also allows as input lists of objects (meshes) so that one can render more complex scenes (e.g. using multiple objects and textures).
- Current work in progress:
    - Use modern OpenGL to implement analytic derivatives (see OpenDR's SQErrorRenderer class for an implementation of the squared error of an image and the OpenDR render).
    - Get derivatives wrt Texture units working.
- As for Chumpy: most of it is the same. Added some code for other optimization methods.

Run "python setup.py install" on both chumpy and opendr subdirectories to install both packages.

Two short *demos* are available that show how to use OpenDR.
- demo_fit_cube.py fits the scale of a cube to the appropriate size.
- demo_fit_teapot.py uses a deformable teapot model and fits its PCA shape parameters to the ground truth. You'll need to create a data folder and add [teapotModel.pkl](https://drive.google.com/file/d/1JO5ZsXHb_KTsjFMFx7rxY0YVAwnM3TMY/view?usp=sharing) (click to download).




For a more complicated use-case see my [Overcoming Occlusion with Inverse Graphics project](https://github.com/polmorenoc/inversegraphics).
