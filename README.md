Crocodile -- NumPy Simplified Imaging for Aperture Synthesis Radio Telescopes
-----------------------------------------------------------------------------

This is a new, still experimental, project to create a reference code
in NumPy for somewhat simplified aperture synthesis imaging.

Warning: The current code is an experimental proof-of-concept. More
here soon. 


Simulate_uvw iPython Notebook
-----------------------------

This is an interactive notebook demonstrating the steps in simulate.py and t1.py. 
To run it you must have a copy of VLA_A_hor_xyz.txt from the /test directory 
available in your working directory.

> ipython notebook Simulate_uvw.ipynb

This will open the interactive notebook in your default web browser.

Degridding using the GPU
------------------------

crocodile can call the GPU Degridding module also in the SKA-SDP repository
You can find the code here: 
https://github.com/SKA-ScienceDataProcessor/GPUDegrid
To use it as part of crocodile, first modify GPUDegrid/Defines.h so that
the parameters for the size of the image and GCF and the sub-grid (Qpx)
match crocodile, then build GPUDegrid.so. Make sure the environment 
variable PYTHONPATH includes the location of GPUDegrid.so. Then, 
look for the line in synthesis.py that calls gpuconvdegrid. The line is
commented out. Uncomment it and comment out the line that calls convdegrid.
Finally, run t1.py.


