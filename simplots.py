"""
Simple functions to output scatter, contour plots,
3D plots and images.
V.Stolyarov, 1.03.2015
"""

import numpy
import scipy
import scipy.special
import scipy.ndimage

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
Image plot with title and labels,
clabel is a label for a colorbar.
"""
def plot_image(mat, title, xlabel, ylabel, clabel):
    pyplot.cla()
    pyplot.imshow(mat)
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


