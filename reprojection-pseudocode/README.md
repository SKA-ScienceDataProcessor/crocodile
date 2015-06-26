This repo is an effort to produce a simple, stand-alone piece of software
(in C++ or pseudocode) to demonstrate reprojection and re-gridding.

I start by including a reduced and simplified 2-d version of CASA's
ImageRegrid code (ImageRegrid.h and ImageRegrid.tcc). The code has
been heavily cut in size and complexity, and is not in a compilable or
runnable state. Instead, it has been heavily commented and it intended
to describe clearly CASA's regridding and reprojection functionality.

The next stage will be to follow the CASA dependencies to understand
mathematically how reprojection has been implemented.