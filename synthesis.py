# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# 
# Synthetise and image interferometer data
"""Parameter name meanings:

T2: Theta2, the half-width of the field of view to be synthetised (radian)

L2: Half-width of the uv-plane (unitless). Controls resolution of the images


Qpx: Oversampling of pixels by the convolution kernels -- there are
(Qpx x Qpx) convolution kernels per pixels to account for fractional
pixel values. 

"""

import numpy
import scipy.special

def ucs(m):
    return numpy.mgrid[-1:1:(m.shape[0]*1j), -1:1:(m.shape[1]*1j)]

def ucsN(N):
    return numpy.mgrid[-1:1:(N*1j), -1:1:(N*1j)]

def uax(m, n, eps=0):
    "1D Array which spans the n-axis axes of m with values between -1 and 1"
    return numpy.mgrid[(-1+eps):(1-eps):(m.shape[n]*1j)]

def uax2(N, eps=0):
    """1D array which spans -1 to 1 with 0 at position N/2"""
    return numpy.mgrid[(-1+eps):(1.0-eps)*(N-2)/N:1j*N]

def aaf(a, m, c):
    """Compute the anti-aliasing function as separable product. See VLA
    Scientific Memoranda 129, 131, 132 

    """
    sx,sy=map(lambda i: scipy.special.pro_ang1(m,m,c,uax2(a.shape[i],eps=1e-10))[0],
           [0,1])
    return numpy.outer(sx,sy)

def pxoversample(ff, N, Qpx, s):
    """Takes a farfield pattern and creates oversampled convolution
    functions from it.

    :param ff: Far field pattern

    :param N:  Image size

    :param Qpx: Factor to oversample by -- there will be Qpx x Qpx convolution functions

    :param s: Size of convolution function to extract

    :returns: List of of shape (Qpx, Qpx), with the inner list on the
    u coordinate and outer indexed by j
    """
    padff=numpy.pad(ff,
                    pad_width=(N*(Qpx-1)/2,),
                    mode='constant',
                    constant_values=(0.0,))
    af=numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(padff)))
    res=[[wextract(af, i, j, Qpx, s) for i in range(Qpx)] for j in range(Qpx)]
    return res

def wkernff(N, T2, w, Qpx):
    "W beam (i.e., w effect in the far-field). T2 is half-width of map in radian"
    r2=((ucsN(N)*T2)**2).sum(axis=0)
    ph=w*(1-numpy.sqrt(1-r2))
    cp=(numpy.exp(2j*numpy.pi*ph))
    return numpy.pad(cp,
                     pad_width=(N*(Qpx-1)/2,),
                     mode='constant',
                     constant_values=(0.0,))

def wextract(a, i, j, Qpx, s):
    """Extract the (ith,jth) w-kernel from the oversampled parent and normalise

    The kernel is reversed in order to make the suitable for
    correcting the fractional coordinates
    """
    x=a[i::Qpx, j::Qpx]
    x=x[::-1,::-1]
    x*=1.0/x.sum()
    return exmid2(x,s)

def wkernaf(N, T2, w, s,
            Qpx):
    """
    The middle s pixels of W convolution kernel.

    :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
    plane than required to minimially sample T2.

    :return: (Qpx,Qpx) shaped list of convolution kernels
    """
    wff=wkernff(N, T2 , w, Qpx)
    return pxoversample(wff, N, Qpx, s)


def kinvert(a):
    """
    Pseudo-Invert a kernel: element-wise inversion (see RauThesis2010:Eq4.6)
    """ 
    return numpy.conj(a) / (numpy.abs(a)**2)

def sample(a, p):
    "Take samples from array a"
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    return a[x,y]

def grid(a, p, v):
    """Grid visibilies (v) at positions (p) into (a) without convolution

    The zeroth frequency is at N/2 where N is the length on side of
    the grid.
    """
    x=numpy.around(((1+p[:,0])*a.shape[0]/2)).astype(int)
    y=numpy.around(((1+p[:,1])*a.shape[1]/2)).astype(int)
    for i in range(len(x)):
        a[x[i],y[i]] += v[i]
    return a

def convgridone(a, pi, fi, gcf, v):
    "Convolve and grid one visibility sample"
    sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2
    # NB the order of fi below 
    a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf[fi[1]][fi[0]]*v

def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size

    :param N: Number of pixels in total 
    :param p: coordinates in range -1,1
    :param Qpx: Fractional values to round to
    """
    H=N/2
    x=(1+p)*H
    flx=numpy.floor(x+ 0.5 /Qpx)
    fracx=numpy.around(((x-flx)*Qpx))    
    return (flx).astype(int), fracx.astype(int)


def convcoords(a, p, Qpx):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    The fractional values are rounded to nearest 1/Qpx pixel value at
    fractional values greater than (Qpx-0.5)/Qpx are roundeded to next
    integer index
    """
    (x, xf), (y, yf) = [fraccoord(a.shape[i], p[:,i], Qpx) for i in [0,1]]
    return x, xf, y, yf

def convgrid(a, p, v, gcf):
    """Grid after convolving with gcf, taking into account fractional uv
    cordinate values

    :param a: Grid to add to
    :param p: UVW positions
    :param v: Visibility values
    :param gcf: List  (shape Qpx, Qpx) of convolution kernels of 
    """ 
    x, xf, y, yf=convcoords(a, p, len(gcf))
    for i in range(len(x)):
        convgridone(a,
                    (x[i], y[i]),
                    (xf[i], yf[i]),
                    gcf, v[i])
    return a

def convdegrid(a, p, gcf):
    "Convolutional-degridding"
    x, xf, y, yf=convcoords(a, p, len(gcf))
    v=[]
    sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2
    for i in range(len(x)):
        pi=(x[i], y[i])
        v.append((a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] * gcf[xf[i]][yf[i]]).sum())
    return numpy.array(v)

def exmid(a, s):
    "Extract a section from middle of a map"
    cx=a.shape[0]/2
    cy=a.shape[1]/2
    return a[cx-s:cx+s+1, cy-s:cy+s+1]

def exmid2(a, s):
    """Extract a section from middle of a map, suitable for zero frequencies at N/2"""
    cx=a.shape[0]/2
    cy=a.shape[1]/2
    return a[cx-s-1:cx+s, cy-s-1:cy+s]

def div0(a1, a2):
    "Divide a1 by a2 except pixels where a2 is zero"
    m= (a2!=0)
    res=a1.copy()
    res[m]/=a2[m]
    return res

def inv(g):
    """Invert a hermitian symetric n-dim function

    :param g: The uv grid to invert. Note that the zero frequency is
    expected at pixel N/2 where N is the size of the grid on the side.

    This function is like doing ifftshift on the x axis but not on the
    y axis
    """
    Nx,Ny=g.shape
    huv=numpy.roll(g[:,(Ny/2):], shift=-(Nx-1)/2, axis=0)
    huv=numpy.pad(huv,
                  pad_width=((0,0),(0,1)),
                  mode='constant',
                  constant_values=((0.0j, 0.0j),(0.0j, 0.0j)))
    return numpy.fft.fftshift(numpy.fft.irfft2(huv))


def rotv(p, l, m, v):
    "Rotate visibilities to direction (l,m)"
    s=numpy.array([l, m , numpy.sqrt(1 - l**2 - m**2)])
    return (v * numpy.exp(2j*numpy.pi*numpy.dot(p, s)))    
    
def rotw(p, v):
    "Rotate visibilities to zero w plane"
    return rotv(p, 0, 0, v)

def posvv(p, v):
    """Move all visibilities to the positive v half-plane"""
    s=p[:,1]<0
    p[s]*=-1.0
    v[s]=numpy.conjugate(v[s])
    return p,v

def sortw(p, v):
    """Sort on the w value. (p) are uvw coordinates and (v) are the
    visibility values
    """
    zs=numpy.argsort(p[:,2])
    if v is not None:
        return p[zs], v[zs]
    else:
        return p[zs]

def doweight(T2, L2, p, v):
    """Re-weight visibilities

    Note convolution kernels are not taken into account properly here
    """
    N= T2*L2 *4
    gw =numpy.zeros([N, N])
    p=p/L2
    x, xf, y, yf=convcoords(gw, p, 1)
    for i in range(len(x)):
        gw[x[i],y[i]] += 1
    v=v.copy()
    for i in range(len(x)):
        v[i] /= gw[x[i],y[i]]
    return v

def simpleimg(T2, L2, p, v):
    N= T2*L2 *4
    guv=numpy.zeros([N, N], dtype=complex)
    grid(guv, p/L2, v)
    return guv

def convimg(T2, L2, p, v, kv):
    """Convolve and grid with user-supplied kernels"""
    N= T2*L2 *4
    guv=numpy.zeros([N, N], dtype=complex)
    convgrid(guv, p/L2, v, kv)
    return guv

def wslicimg(T2, L2, p, v,
             wstep=2000,
             Qpx=4):
    "Basic w-projection by w-sort and slicing in w" 
    N= T2*L2 *4
    guv=numpy.zeros([N, N], dtype=complex)
    p, v = sortw(p, v)
    nv=len(v)
    ii=range( 0, nv, wstep)
    ir=zip(ii[:-1], ii[1:]) + [ (ii[-1], nv) ]
    for ilow, ihigh in ir:
        w=p[ilow:ihigh,2].mean()
        wg=wkernaf(257, T2, w, 15, Qpx)
        wg=map(lambda x: map(numpy.conj, x), wg)
        convgrid(guv,  p[ilow:ihigh]/L2, v[ilow:ihigh],  wg)
    return guv

def wslicfwd(guv,
             T2, L2, p,
             wstep=2000,
             Qpx=4):
    "Predict visibilities using w-slices"
    N= T2*L2 *4
    p= sortw(p, None)
    nv=len(p)
    ii=range( 0, nv, wstep)
    ir=zip(ii[:-1], ii[1:]) + [ (ii[-1], nv) ]
    res=[]
    for ilow, ihigh in ir:
        w=p[ilow:ihigh,2].mean()
        wg=wkernaf(257, T2, w, 15, Qpx)
        res.append (convdegrid(guv,  p[ilow:ihigh]/L2, wg))
    v=numpy.concatenate(res)
    pp=p.copy()
    pp[:,2]*=-1
    return (p, rotw(pp,v))


def doimg(T2, L2, p, v, imgfn):
    """Do imaging with imaging function (imgfn)

    Note: No longer move all visibilities to postivie v -- need to
    check the convolution kernels
    """
    p=numpy.vstack([p, p*-1])
    v=numpy.hstack([v, numpy.conj(v)])
    v=doweight(T2, L2, p, v)
    c=imgfn(T2, L2, p, rotw(p, v))
    s=inv(c)
    c=imgfn(T2, L2, p, numpy.ones(len(p)))
    p=inv(c)
    return (s,p)
    



