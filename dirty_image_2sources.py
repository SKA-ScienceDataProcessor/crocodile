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


if 1:
    """
    Read antenna positions from ASCII file (XYZ, meters)
    """
    vlas=numpy.genfromtxt("/home/vlad/software/SKA/crocodile/test/VLA_A_hor_xyz_5ants.txt", delimiter=",")
    pyplot.scatter(vlas[:,0], vlas[:,1])
    pyplot.title('Antenna positions on the (X,Y) plane')
    pyplot.xlabel('X, meters')
    pyplot.ylabel('Y, meters')
    pyplot.grid()
    pyplot.show()
#    raw_input("press enter")

    """
    Generate UV plane coverage for a batch of the snapshots taken every 22.9 minutes (0.1 radians)
    during 12 hours (Pi radians) at 90 deg declination (Pi/2), wavelength is 5 meters.
    """
    uvstep = 0.1
    wl = 5.0
    dec = numpy.pi/4.0
    obsTime = numpy.pi
    vobs=genuv(vlas, numpy.arange(0,obsTime,uvstep) ,  dec)
    pyplot.cla()
    pyplot.scatter(vobs[:,0]/wl, vobs[:,1]/wl)
    pyplot.title('UV plane coverage for 12h observation, wavelength = 5m')
    pyplot.xlabel('U, number of wavelengths')
    pyplot.ylabel('V, number of wavelengths')
    pyplot.grid()
    pyplot.show()
#    raw_input("press enter")

    """
    Generate visibilities from two point sources of unit intensity, located
    at (l1,m1) and (l2,m2), units are radians w.r.t. the phase center

    """
    l1 = 0.01
    m1 = 0.01
    l2 = -0.001
    m2 = -0.001
    yy=genvis(vobs/wl, l1, m1)
    yy=yy + genvis(vobs/wl, l2, m2)
    """
    Rotate UV plane to exclude W component
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

    pyplot.cla()
    pyplot.scatter(vobs_new[:,0]/wl, vobs_new[:,1]/wl)
    pyplot.title('UV plane coverage for 12h observation with Hermitian part')
    pyplot.xlabel('U, number of wavelengths')
    pyplot.ylabel('V, number of wavelengths')
    pyplot.grid()
    pyplot.show()
#    raw_input("press enter")


    """
    Plot the model of the sources
    """
    pyplot.cla()
    pyplot.scatter((l1, l2), (m1, m2))
    pyplot.title('A model of two point sources with unit intensity')
    pyplot.xlabel('l, radians')
    pyplot.ylabel('m, radians')
    pyplot.grid()
    pyplot.show()


    """
    Transfer visibilities to the square matrix nmat x nmat using
    re-gridding routine (no visibility stacking in UV cells, correct for the 
    sparse UV plane coverage)

    """
    nmat = 512
    mat_a = numpy.zeros((nmat,nmat),'D')
    maxvobs = max(vobs_new[:,0:1])[0] + 1
    mat_a = grid1(mat_a,(vobs_new/maxvobs),yy_new)
    pyplot.cla()
    pyplot.contour(numpy.abs(mat_a))
    pyplot.title('Module of visibility V(u,v) resampled to the matrix')
    pyplot.xlabel('U, number of pixel')
    pyplot.ylabel('V, number of pixel')
    pyplot.colorbar(label='abs(V(u,v)')
    pyplot.show()
#    raw_input("press enter")

#    pyplot.cla()
#    pyplot.imshow(numpy.abs(mat_a))
#    pyplot.show()
#    raw_input("press enter")

    """
    Apply FFTSHIFT to shift the phase center to the origin
    """
    mat_b = numpy.fft.ifftshift(mat_a)
    pyplot.cla()
    pyplot.contour(numpy.abs(mat_b))
    pyplot.title('Module of V(u,v) resampled to the matrix after FFTSHIFT')
    pyplot.xlabel('U, number of pixel')
    pyplot.ylabel('V, number of pixel')
    pyplot.colorbar(label='abs(V(u,v)')
    pyplot.show()
#    raw_input("press enter")


    """
    Make inverse Fourier-transform to get the dirty image and apply FFTSHIFT again
    to return the phase center to the middle of the matrix
    (should be done by checkboarding of the Fourier image - ToDo)
    """
    c = numpy.fft.ifft2(mat_b)
    c1 = numpy.fft.fftshift(c)
    pyplot.cla()
    pyplot.contourf(numpy.abs(c1))
    pyplot.title('Dirty image of the two point sources recovered')
    pyplot.xlabel('l, number of pixel')
    pyplot.ylabel('m, number of pixel')
    pyplot.colorbar(label='I(l,m)')
    pyplot.show()
#    raw_input("press enter")
    


    """
    Smooth and decimate the dirty image for 3D surface plot	
    """
    ncomp = 4
    c2 = scipy.ndimage.gaussian_filter(abs(c1), ncomp)
    c3 = c2[::ncomp,::ncomp]
    X = numpy.arange(1, nmat/ncomp)
    Y = numpy.arange(1, nmat/ncomp)
    X, Y = numpy.meshgrid(X, Y)
    Z = c3[X,Y]
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#    ax.set_zlim(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=5, label='I(l,m), amplitude')
    pyplot.title('3D surface of the dirty image recovered (smoothed)')
    pyplot.xlabel('l, number of pixel')
    pyplot.ylabel('m, number of pixel')
    pyplot.show()

