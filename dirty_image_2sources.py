import numpy
import scipy
import scipy.special
import scipy.ndimage

from clean import *
from synthesis import *
from simulate import *

from matplotlib import pylab
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D


"""
Scatter plot with title and labels,
x and y shoud be of the same size.
"""
def plot_scatter(x,y,title,xlabel, ylabel):
    pyplot.cla()
    pyplot.scatter(x, y)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.grid()
    pyplot.show()


"""
Contour plot with title and labels,
clabel is a label for a colorbar.
"""
def plot_contour(mat, title, xlabel, ylabel, clabel):
    pyplot.cla()
    pyplot.contour(mat)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.colorbar(label=clabel)
    pyplot.show()


"""
Filled contour plot with title and labels,
clabel is a label for a colorbar.
"""
def plot_contourf(mat, title, xlabel, ylabel, clabel):
    pyplot.cla()
    pyplot.contourf(mat)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.colorbar(label=clabel)
    pyplot.show()


"""
3D plot of the input array mat
"""
def plot_3Dsurface(mat, label, xlabel, ylabel, clabel):
    X = numpy.arange(1, numpy.size(mat,0))
    Y = numpy.arange(1, numpy.size(mat,1))
    X, Y = numpy.meshgrid(X, Y)
    Z = mat[X,Y]
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=clabel)
    pyplot.title(label)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.show()


if 1:
    """
    Read antenna positions from ASCII file (XYZ, meters)
    """
    vlas=numpy.genfromtxt("/home/vlad/software/SKA/crocodile/test/VLA_A_hor_xyz_5ants.txt", delimiter=",")
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
    Rotate UV plane to get zero W component
    """
    yy=rotw(vobs/wl, yy)

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
    (should be done by checkboarding of the Fourier image - ToDo)
    """
    c = numpy.fft.ifft2(mat_b)
    c1 = numpy.fft.fftshift(c)*numpy.size(vobs_new,0)
    
    plot_contourf(numpy.abs(c1), 'Dirty image of the two point sources recovered',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m)')
     


    """
    Smooth and decimate the dirty image for 3D surface plot	
    """
    ncomp = 4
    c2 = scipy.ndimage.gaussian_filter(abs(c1), ncomp) 	# smooth with a Gaussian kernel
    c3 = c2[::ncomp,::ncomp]				# decimate
    plot_3Dsurface(c3, '3D surface of the dirty image recovered (smoothed)',
    	'l, number of pixel', 'm, number of pixel', 'I(l,m), amplitude')
	
