This repo is an effort to produce a simple, stand-alone piece of software
(in C++ or pseudocode) to demonstrate reprojection and re-gridding.

The C program 'reprojection.c' is a self-contained implementation of CASA's
regridding and reprojection class (Images/ImageRegrid.h), written from scratch
so that no CASA dependencies are required. The original CASA code uses
WCSLIB to convert from pixel to world coordinates, but here I have written
a crude spherical reprojection algorithm that ensures that everything
needed for regridding and reprojection is handled by this one program.

To compile, use:

	gcc reprojection.c -o reprojection -std=c99 -lm

This software uses standard bitmap (.BMP) files as both input and output, and
is therefore straightforward to run without the need to generate
astronomical images. Bitmap images need to be 8 bit/pixel, and can be
converted in Gimp by selecting Image -> Mode -> Greyscale. I use 'lovell.bmp'
as an input image.

The input and output coordinate systems are specified in the parameter file
'reprojection-params', and use the FITS keywords CRVALi, CRPIXi and CDi_j to
describe the orientation and scaling of the pixel axes.

To run the program, use:

	./reprojection lovell.bmp new_image.bmp

I also provide some example parameter files and output images.

I also include a reduced and simplified 2-d version of CASA's
ImageRegrid code (ImageRegrid.h and ImageRegrid.tcc). The code has
been heavily cut in size and complexity, and is not in a compilable or
runnable state. Instead, it has been heavily commented and it intended
to describe clearly CASA's regridding and reprojection functionality.