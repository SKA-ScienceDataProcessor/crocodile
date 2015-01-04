# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
# 
# Synthetise and image interfometer data

import numpy
import scipy.special

def ucs(m):
    return numpy.mgrid[-1:1:(m.shape[0]*1j), -1:1:(m.shape[1]*1j)]

def ucsN(N):
    return numpy.mgrid[-1:1:(N*1j), -1:1:(N*1j)]

def uax(m, n, eps=0):
    "1D Array which spans nth axes of m with values between -1 and 1"
    return numpy.mgrid[(-1+eps):(1-eps):(m.shape[n]*1j)]

def aaf(a, m, c):
    """Compute the anti-aliasing function as separable product. See VLA
    Scientific Memoranda 129, 131, 132 

    """
    sx,sy=map(lambda i: scipy.special.pro_ang1(m,m,c,uax(a,i,eps=1e-10))[0],
           [0,1])
    return numpy.outer(sx,sy)

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
    "Extract the (ith,jth) w-kernel from the oversampled parent and normalise"
    x=a[i::Qpx, j::Qpx]
    x*=1.0/x.sum()
    return exmid(x,s)

def wkernaf(N, T2, w, s,
            Qpx):
    """
    The middle s pixels of W convolution kernel.

    :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
    plane than required to minimially sample T2.

    :return: (Qpx,Qpx) shaped list of convolution kernels
    """
    wff=wkernff(N, T2 , w, Qpx)
    waf=numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(wff)))
    res=[[wextract(waf, i, j, Qpx, s) for i in range(Qpx)] for j in range(Qpx)]
    return res

def kinvert(a):
    "Pseudo-Invert a kernel" 
    return numpy.conj(a) / (numpy.abs(a)**2)

def sample(a, p):
    "Take samples from array a"
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    return a[x,y]

def grid(a, p, v):
    "Grid samples array a without convolution"
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    for i in range(len(x)):
        a[x[i],y[i]] += v[i]
    return a

def convgridone(a, pi, fi, gcf, v):
    "Convolve and grid one sample. Note the normalisation."
    sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2
    a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf[fi[0]][fi[1]]*v

def convgrid(a, p, v, gcf):
    "Grid after convolving with gcf" 
    Qpx=len(gcf)
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    xf=((((1+p[:,0])*a.shape[0]/2)-x)*Qpx).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    yf=((((1+p[:,1])*a.shape[1]/2)-y)*Qpx).astype(int)
    for i in range(len(x)):
        convgridone(a,
                    (x[i], y[i]),
                    (xf[i], yf[i]),
                    gcf, v[i])
    return a

def convdegrid(a, p, gcf):
    "Convolutional-degridding"
    Qpx=len(gcf)
    x=((1+p[:,0])*a.shape[0]/2).astype(int)
    xf=((((1+p[:,0])*a.shape[0]/2)-x)*Qpx).astype(int)
    y=((1+p[:,1])*a.shape[1]/2).astype(int)
    yf=((((1+p[:,1])*a.shape[1]/2)-y)*Qpx).astype(int)
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

def div0(a1, a2):
    "Divide a1 by a2 except pixels where a2 is zero"
    m= (a2!=0)
    res=a1.copy()
    res[m]/=a2[m]
    return res

def inv(g):
    return numpy.fft.ifft2(numpy.fft.ifftshift(g))

def rotv(p, l, m, v):
    "Rotate visibilities to direction (l,m)"
    s=numpy.array([l, m , numpy.sqrt(1 - l**2 - m**2)])
    return (v * numpy.exp(2j*numpy.pi*numpy.dot(p, s)))    
    
def rotw(p, v):
    "Rotate visibilities to zero w plane"
    return rotv(p, 0, 0, v)

def sortw(p, v):
    "Sort on the w value"
    zs=numpy.argsort(p[:,2])
    if v is not None:
        return p[zs], v[zs]
    else:
        return p[zs]

def doweight(T2, L2, p, v):
    "Re-weight visibilities"
    N= T2*L2 *4
    gw =numpy.zeros([N, N])
    p=p/L2
    x=((1+p[:,0])*N/2).astype(int)
    y=((1+p[:,1])*N/2).astype(int)
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
        wg=wkernaf(N/Qpx, T2, w, 15, Qpx)
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
        wg=wkernaf(N/Qpx, T2, w, 15, Qpx)
        res.append (convdegrid(guv,  p[ilow:ihigh]/L2, wg))
    v=numpy.concatenate(res)
    pp=p.copy()
    pp[:,2]*=-1
    return (p, rotw(pp,v))


def doimg(T2, L2, p, v, imgfn):
    v=doweight(T2, L2, p, v)
    c=imgfn(T2, L2, p, rotw(p, v))
    s=numpy.fft.fftshift(inv(c).real)
    c=imgfn(T2, L2, p, numpy.ones(len(p)))
    p=numpy.fft.fftshift(inv(c).real)
    return (s,p)
    



