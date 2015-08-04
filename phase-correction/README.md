This repo is an effort to produce a simple, stand-alone piece of software
(in C++ or pseudocode) to demonstrate phase correction.

The C program 'uvfacetting.c' is a self-contained implementation of CASA's
UVWMachine class (Measures/UVWMachine.h), written from scratch
so that no CASA dependencies are required.

To compile, use:

	gcc uvfacetting.c -o uvfacetting -std=c99 -lm

The input and output coordinate systems are specified in the parameter file
'uvfacetting-params' using the parameters in_long, in_lat, in_epoch, out_long,
out_lat, out_epoch. The epoch must be one of J2000, B1950 and GALACTIC. There is
also one additional parameter uv_projection, which can be Y or N. If this parameter
is Y then the u and v coordinates will be adjusted in order to project the image
onto a new tangential plane, and returned relative to the new phase position. If
this parameter is N then the program will simply return the uvw coordinates
relative to the new phase position.

To run the program, use:

	./uvfacetting 1.1 2.2 3.3

where the 3 parameters are the u, v and w coordinates to be converted.

The equations for adjusting the uv coordinates can be found in:

Sault et al 1996 A&As 120 375-384