# New OpenDR
This code is based on the work of Matt Loper's OpenDR: (https://github.com/mattloper/opendr/wiki)[https://github.com/mattloper/opendr/wiki].

For my projects, I've had to update the implementation with:
- Updated to be used with Python 3.4+
- Updated OpenDR to use OpenGL 3.3+ methods by using the modern pipepline (instead of the old fixed function pipeline). I also use shaders for rendering. This opens up the ability to use modern features (e.g. MSAA) and take more advantage of newer GPUs.
- Two main modes are used, either MESA or GLFW. The first allows the code to be easily run on headless servers, but is a software based implementation of OpenGL (i.e. runs on CPU). The second one allows running on GPUs without popping up an OpenGL window. In order for it to be fully headless (to run on most servers) you'd need to have NVIDIA GPUs with a [certain X configuration](http://www.nvidia.com/content/PDF/remote-viz-tesla-gpus.pdf) (see section on "SETTING UP THE X SERVER FOR HEADLESS OPERATION").

#Chumpy:
Most of it is the same. Added some code for other optimization methods.

