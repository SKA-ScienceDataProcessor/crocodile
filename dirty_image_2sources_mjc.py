"""
CLEAN application illustration, two point source example
"""

import numpy
import scipy
import scipy.special
import scipy.ndimage

from clean import *
from synthesis import *
from simulate import *
from simplots import *


if 1:
    """
    Read antenna positions from ASCII file (XYZ, meters)
    """
    vlas=numpy.genfromtxt("test/VLA_A_hor_xyz_5ants.txt", delimiter=",")
    plot_scatter(vlas[:,0], vlas[:,1], 'Antenna positions on the (X,Y) plane', 'X, meters', 'Y, meters')


    """
    Set the parameters of the observations, e.g. wavelength, declination of the tracking center,
    the lenth of the observations and the interval between the snapshots
    """
    uvstep = 22.9  	# an interval between the snapshots (a step in time), minutes
    uvstep = uvstep/60./12.* numpy.pi
    
    wl = 5.0 		# wavelength, meters
    
    dec = numpy.pi/4.0 	# Declination, radians
    
    obsTime = 12. 	# The length of the observations, hours
    obsTime = obsTime/12.*numpy.pi 
    
    
    """
    Generate UV plane coverage for a batch of the snapshots taken every uvstep
    during obsTime at the declination dec, wavelength is wl.
    """  
    vobs=genuv(vlas, numpy.arange(0,obsTime,uvstep) ,  dec)
    plot_scatter(vobs[:,0]/wl, vobs[:,1]/wl,'UV plane coverage for 12h observation, wavelength = 5m', 
    	'U, number of wavelengths','V, number of wavelengths')

    """
    Generate visibilities from two point sources of unit intensity, located
    at (l1,m1) and (l2,m2), units are radians w.r.t. the phase center

    """
    l1 = 0.01 # in radians
    m1 = 0.01
    l2 = -0.001
    m2 = -0.001
    yy=genvis(vobs/wl, l1, m1)
    yy=yy + genvis(vobs/wl, l2, m2)
    
    """
    No UV plane rotation to get zero W component - done later in majorcycle_imshow()
    """
#    yy=rotw(vobs/wl, yy)

    """
    Fill the Hermitian conjugated part of the UV plane, V(-u,-v) = V*(u,v)
    """
    vobs_tmp = numpy.copy(vobs)
    vobs_tmp[:,0] *= -1.0
    vobs_tmp[:,1] *= -1.0
    vobs_new = numpy.concatenate((vobs, vobs_tmp))
    yy_tmp = numpy.conjugate(yy)
    yy_new = numpy.concatenate((yy, yy_tmp))


    plot_scatter(vobs_new[:,0]/wl, vobs_new[:,1]/wl,'UV plane coverage for 12h observation with Hermitian part',
    	'U, number of wavelengths','V, number of wavelengths' )

    """
    Plot the model of the sources    
    """
    plot_scatter((l1, l2), (m1, m2), 'A model of two point sources with unit intensity',
    	'l, radians', 'm, radians')

    """
    Transfer visibilities to the square matrix nmat x nmat using
    re-gridding routine (no visibility stacking in UV cells, correct for the 
    sparse UV plane coverage)

    """
    nmat = 512
    mat_a = numpy.zeros((nmat,nmat),'D')
    maxvobs = max(vobs_new[:,0:1])[0] + 1
    mat_a = grid1(mat_a,(vobs_new/maxvobs),yy_new)
    plot_contour(numpy.abs(mat_a), 'Module of visibility V(u,v) resampled to the matrix',
    	'V, number of pixel', 'U, number of pixel', 'abs(V(u,v)')


    """
    Apply FFTSHIFT to shift the phase center to the origin
    """
    mat_b = numpy.fft.ifftshift(mat_a)
    plot_contour(numpy.abs(mat_b), 'Module of V(u,v) resampled to the matrix after FFTSHIFT',
    	'V, number of pixel', 'U, number of pixel', 'abs(V(u,v)')
    

    """
    Make inverse Fourier-transform to get the dirty image and apply FFTSHIFT again
    to return the phase center to the middle of the matrix
    Since no UV plane rotation is done so far, the phase error is high and the image is noisy.	
    """
    c = numpy.fft.ifft2(mat_b)
    c1 = numpy.fft.fftshift(c)*numpy.size(vobs_new,0)
    
    plot_contourf(numpy.abs(c1), 'Dirty image of the two point sources with phase errors',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')


    """
    Plot initial dirty image before CLEAN	
    """
if 0:
    ps,vs = sortw(vobs/wl , yy)
    T2 = 0.025 # Theta2, the half-width of the field of view to be synthetised  (radian)
    L2 = 15000 # Half-width of the uv-plane (unitless). Controls resultion of the images
    wstep = 250000
    dirty,psf = doimg(T2, L2, ps,vs, lambda *x: wslicimg(*x, wstep=wstep) )
    plot_image(numpy.abs(dirty), 'Dirty image after doimg()',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')

    plot_image(numpy.abs(psf), 'psf after doimg()',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')



"""
Application of CLEAN Cotton-Schwab algorithm, in
Synthesis Imaging in Radio Astronomy II, ed. by Taylor G.B., Carilli C.L. and Perley R.A,
ASP conf. series, vol 180, p.155 (1999)  
"""
if 1:
    """
    Call majorcycle function (Cotton-Schwab algorithm) and plot the image solution for
    every of nmajor iteration. Parameters returned are
    ps - points in UV plane
    vs - cleaned visibilities    
    """

    nmajor = 5 # number of iterations in the major cycle
    nminor = 100 # number of iterations in the minor cycle (Hogborn algorithm)
    gain = 0.1
    wstep = 250000
    T2 = 0.025 # Theta2, the half-width of the field of view to be synthetised  (radian)
    L2 = 15000 # Half-width of the uv-plane (unitless). Controls resultion of the images

    ps,vs = majorcycle_imshow(T2, L2, vobs/wl , yy, gain, nmajor, nminor, wstep)

    """
    Construct the image and the beam (PSF, point spread function) after CLEAN iterations
    and plot them.
    """
    dirty,psf = doimg(T2, L2, ps,vs, lambda *x: wslicimg(*x, wstep=wstep) )

    plot_image(numpy.abs(dirty), 'Dirty image after majorcycle',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')

    plot_image(numpy.abs(psf), 'psf after majorcycle',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')

     

if 0:
    """
    Smooth and decimate the dirty image for 3D surface plot	
    """
    ncomp = 4
    c2 = scipy.ndimage.gaussian_filter(abs(c1), ncomp) 	# smooth with a Gaussian kernel
    c3 = c2[::ncomp,::ncomp]				# decimate
    plot_3Dsurface(c3, '3D surface of the dirty image recovered (smoothed)',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m), amplitude')
	
