# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# 
# Synthetise and image interfometer data

import numpy
import scipy.special

def ucs(m):
    return numpy.mgrid[-1:1:(m.shape[0]*1j), -1:1:(m.shape[1]*1j)]

def uax(m, n, eps=0):
    "1D Array which spans nth axes of m with values between -1 and 1"
    return numpy.mgrid[(-1+eps):(1-eps):(m.shape[n]*1j)]

def aaf(a, m, c):
    """Compute the anti-aliasing function as separable product. See VLA
    Scientific Memoranda 132 by Fred Schwab.

    """
    sx,sy=map(lambda i: scipy.special.pro_ang1(m,m,c,uax(a,i,eps=1e-10))[0],
           [0,1])
    return numpy.outer(sx,sy)

def sample(a, p):
    "Take samples from array a"
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    return a[x,y]

def grid(a, aw, p, v):
    "Grid samples back onto array a"
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    for i in range(len(x)):
        a[x[i],y[i]] += v[i]
        a[-x[i],-y[i]] += numpy.conj(v[i])
        aw[x[i],y[i]] +=1
        aw[-x[i],-y[i]] +=1
    return a, aw

def convgridone(a, aw, pi, gcf, v):
    sx, sy= gcf.shape[0]/2, gcf.shape[1]/2
    a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf*v
    aw[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += 1

def convgrid(a, aw, p, v, gcf):
    "Grid after convolving with gcf" 
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    for i in range(len(x)):
        convgridone(a, aw, (x[i], y[i]), gcf, v[i])
        convgridone(a, aw, (-x[i], -y[i]), gcf, numpy.conj(v[i]))
    return a, aw    

def exmid(a, s):
    "Extract a section from middle of a map"
    cx=a.shape[0]/2
    cy=a.shape[1]/2
    return a[cx-s:cx+s+1, cy-s:cy+s+1]

def div0(a1, a2):
    "Divide a1 by a2 except pixels where a2 is zero"
    m= (a2!=0)
    res=a1.copy()
    res[m]/=a2[m]
    return res

def inv(g, w):
    return numpy.fft.ifft2(numpy.fft.ifftshift(div0(c1,c1w)))
    
def rotw(p, v):
    "Rotate visibilities to zero w plane"
    return (v * numpy.exp(2j*numpy.pi*p[:,2]))



