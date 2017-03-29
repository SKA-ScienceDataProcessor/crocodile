# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Parameter name meanings:

- p: The uvw coordinates [*,3] (m)
- v: The Visibility values [*] (Jy)
- theta: Width of the field of view to be synthetised, as directional
  cosines (approximately radians)
- lam: Width of the uv-plane (in wavelengths). Controls resolution of the
  images.
- Qpx: Oversampling of pixels by the convolution kernels -- there are
  (Qpx x Qpx) convolution kernels per pixels to account for fractional
  pixel values.

All grids and images are considered quadratic and centered around
`N//2`, where `N` is the pixel width/height. This means that `N//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `N` the grid is not symetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

This is implemented for reference in
`coordinates`/`coordinates2`. Some noteworthy properties:
- `ceil(theta * lam)` gives the image size `N` in pixels
- `lam * coordinates2(N)` yields the `u,v` grid coordinate system
- `theta * coordinates2(N)` yields the `l,m` image coordinate system
   (radians, roughly)
"""

from __future__ import division

import numpy
import pylru
import scipy.special
import scipy.signal


def coordinateBounds(N):
    r"""
    Returns lowest and highest coordinates of an image/grid given:

    1. Step size is :math:`1/N`:

       .. math:: \frac{high-low}{N-1} = \frac{1}{N}

    2. The coordinate :math:`\lfloor N/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{N}{2}\right\rfloor * (high-low) = 0

    This is the coordinate system for shifted FFTs.
    """
    if N % 2 == 0:
        return -0.5, 0.5 * (N - 2) / N
    else:
        return -0.5 * (N - 1) / N, 0.5 * (N - 1) / N

def coordinates(N):
    """1D array which spans [-.5,.5[ with 0 at position N/2"""
    N2 = N // 2
    if N % 2 == 0:
        return numpy.mgrid[-N2:N2] / N
    else:
        return numpy.mgrid[-N2:N2+1] / N


def coordinates2(N):
    """Two dimensional grids of coordinates spanning -1 to 1 in each
    dimension, with

    1. a step size of 2/N and
    2. (0,0) at pixel (floor(n/2),floor(n/2))

    :returns: pair (cx,cy) of 2D coordinate arrays
    """

    N2 = N // 2
    if N % 2 == 0:
        return numpy.mgrid[-N2:N2, -N2:N2][::-1] / N
    else:
        return numpy.mgrid[-N2:N2+1, -N2:N2+1][::-1] / N


def fft(a):
    """ Fourier transformation from image to grid space

    :param a: image in `lm` coordinate space
    :returns: `uv` grid
    """
    return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))


def ifft(a):
    """ Fourier transformation from grid to image space

    :param a: `uv` grid to transform
    :returns: an image in `lm` coordinate space
    """
    return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))


def pad_mid(ff, N):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    :param ff: The input far field. Should be smaller than NxN.
    :param N:  The desired far field size

    """

    N0, N0w = ff.shape
    if N == N0: return ff
    assert N > N0 and N0 == N0w
    return numpy.pad(ff,
                     pad_width=2*[(N//2-N0//2, (N+1)//2-(N0+1)//2)],
                     mode='constant',
                     constant_values=0.0)

def extract_mid(a, N):
    """
    Extract a section from middle of a map

    Suitable for zero frequencies at N/2. This is the reverse
    operation to pad.

    :param a: grid from which to extract
    :param s: size of section
    """
    assert N <= a.shape[0] and N <= a.shape[1]
    cx = a.shape[0] // 2
    cy = a.shape[1] // 2
    s = N // 2
    if N % 2 != 0:
        return a[cx - s:cx + s + 1, cy - s:cy + s + 1]
    else:
        return a[cx - s:cx + s, cy - s:cy + s]

def extract_oversampled(a, xf, yf, Qpx, N):
    """
    Extract the (xf-th,yf-th) w-kernel from the oversampled parent

    Offsets are suitable for correcting of fractional coordinates,
    e.g. an offset of (xf,yf) results in the kernel for an (-xf,-yf)
    sub-grid offset.

    We do not want to make assumptions about the source grid's symetry
    here, which means that the grid's side length must be at least
    Qpx*(N+2) to contain enough information in all circumstances

    :param a: grid from which to extract
    :param ox: x offset
    :param oy: y offset
    :param Qpx: oversampling factor
    :param N: size of section
    """

    assert xf >= 0 and xf < Qpx
    assert yf >= 0 and yf < Qpx
    # Determine start offset.
    Na = a.shape[0]
    my = Na//2 - Qpx*(N//2) - yf
    mx = Na//2 - Qpx*(N//2) - xf
    assert mx >= 0 and my >= 0
    # Extract every Qpx-th pixel
    mid = a[my : my+Qpx*N : Qpx,
            mx : mx+Qpx*N : Qpx]
    # normalise
    return Qpx * Qpx * mid


def anti_aliasing_function(shape, m, c):
    """
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param m: mode parameter
    :param c: spheroidal parameter
    """

    # 2D Prolate spheroidal angular function is seperable
    sy, sx = [ scipy.special.pro_ang1(m, m, c, coordinates(N))[0]
               for N in shape ]
    return numpy.outer(sy, sx)

def kernel_transform(dl, dm):
    """Determine linear transformation matrix for a given shift that
    keeps w-kernel coordinates as constant as possible.

    To be precise: This returns a transformation matrix such that with

      (l',m') = T (l,m) + (dl,dm)

    We still have:

      sqrt(1-l'^2-m'^2) ~~ sqrt(1-l^2-m^2)

    As well as leaving l' and m' roughly aligned to l and m. This
    boils down to matching up the first-order terms of the series
    expansions (which are clearly zero for the right-hand term, as it
    is symmetric in both l and m).

    :param dl: Horizontal shift (directional cosine)
    :param dl: Vertical shift (directional cosine)
    :returns: A 2x2 transformation matrix
    """

    dn = numpy.sqrt(1-dl**2-dm**2)
    if dl != 0 or dm != 0:
        f = (dn-1) / (dl**2 + dm**2)
    else:
        f = 0
    return numpy.array([[dn-dm*dm*f,    dl*dm*f],
                        [   dl*dm*f, dn-dl*dl*f]])

def kernel_coordinates(N, theta, dl=0, dm=0, T=None):
    """
    Returns (l,m) coordinates for generation of kernels
    in a far-field of the given size.

    If coordinate transformations are passed, they must be inverse to
    the transformations applied to the visibilities using
    visibility_shift/uvw_transform.

    :param N: Desired far-field resolution
    :param theta: Field of view size (directional cosines)
    :param dl: Pattern horizontal shift (see visibility_shift)
    :param dm: Pattern vertical shift (see visibility_shift)
    :param T: Pattern transformation matrix (see uvw_transform)
    :returns: Pair of (m,l) coordinates
    """

    l,m = coordinates2(N) * theta
    if not T is None:
        l,m = T[0,0]*l+T[1,0]*m, T[0,1]*l+T[1,1]*m
    return l+dl, m+dm


def w_kernel_function(l, m, w):
    """W beam, the fresnel diffraction pattern arising from non-coplanar baselines

    For the w-kernel, shifting the kernel pattern happens to also
    shift the kernel depending on w. Use `kernel_recentre` to counter
    this effect and `visibility_recentre` to apply the reverse
    correction to visibilities.

    :param l: Horizontal image coordinates
    :param m: Vertical image coordinates
    :param N: Size of the grid in pixels
    :param w: Baseline distance to the projection plane
    :returns: N x N array with the far field

    """

    r2 = l**2 + m**2
    assert numpy.all(r2 < 1.0), "Error in image coordinate system: l %s, m %s" % (l, m)
    ph = 1 - numpy.sqrt(1.0 - r2)
    cp = numpy.exp(2j * numpy.pi * w * ph)
    return cp


def kernel_recentre(cp, theta, w, dl, dm):
    """
    Re-center the kernel in grid-space by multiplying it by a phase
    ramp in image space, allowing us to reduce kernel support. Must be
    paired with `visibility_recentre` so that we end up with the same
    uv-grid in the end.

    :param cp: Kernel pattern
    :param w: w-plane of kernel
    :param dl: Horizontal shift to add
    :param dm: Vertical shift to add
    :returns: Re-centered kernel
    """
    N = cp.shape[0]
    l,m = coordinates2(N) * theta
    return cp * numpy.exp(-2j * numpy.pi * w * (dl * l + dm * m))


def visibility_recentre(uvw, dl, dm):
    """
    Compensate for kernel re-centering - see `kernel_recentre`.

    :param uvw: Visibility coordinates
    :param dl: Horizontal shift to compensate for
    :param dm: Vertical shift to compensate for
    :returns: Visibility coordinates re-centred to the peak of their w-kernel
    """

    u, v, w = numpy.hsplit(uvw, 3)
    return numpy.hstack([u - w*dl,
                         v - w*dm,
                         w])


def kernel_oversample(ff, N, Qpx, s):
    """
    Takes a farfield pattern and creates an oversampled convolution
    function.

    If the far field size is smaller than N*Qpx, we will pad it. This
    essentially means we apply a sinc anti-aliasing kernel by default.

    :param ff: Far field pattern
    :param N:  Image size without oversampling
    :param Qpx: Factor to oversample by -- there will be Qpx x Qpx convolution arl
    :param s: Size of convolution function to extract
    :returns: Numpy array of shape [ov, ou, v, u], e.g. with sub-pixel
      offsets as the outer coordinates.
    """

    # Pad the far field to the required pixel size
    padff = pad_mid(ff, N*Qpx)

    # Obtain oversampled uv-grid
    af = ifft(padff)

    # Extract kernels
    res = [[extract_oversampled(af, x, y, Qpx, s) for x in range(Qpx)] for y in range(Qpx)]
    return numpy.array(res)


def w_kernel(theta, w, NpixFF, NpixKern, Qpx, **kwargs):
    """
    The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

    :param theta: Field of view (directional cosines)
    :param w: Baseline distance to the projection plane
    :param NpixFF: Far field size. Must be at least NpixKern+1 if Qpx > 1, otherwise NpixKern.
    :param NpixKern: Size of convolution function to extract
    :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
      plane than required to minimially sample theta.

    :returns: [Qpx,Qpx,s,s] shaped oversampled convolution kernels
    """
    assert NpixFF > NpixKern or (NpixFF == NpixKern and Qpx == 1)

    l,m = kernel_coordinates(NpixFF, theta, **kwargs)
    kern = w_kernel_function(l,m,w)
    return kernel_oversample(kern, NpixFF, Qpx, NpixKern)


def invert_kernel(a):
    """
    Pseudo-Invert a kernel: element-wise inversion (see RauThesis2010:Eq4.6)

    NOT USED
    """
    return numpy.conj(a) / (numpy.abs(a) ** 2)


def grid(a, p, v):
    """Grid visibilities (v) at positions (p) into (a) without convolution

    :param a:   The uv plane to grid to (updated in-place!)
    :param p:   The coordinates to grid to (in fraction [-.5,.5[ of grid)
    :param v:   Visibilities to grid
    """
    assert numpy.max(p) < 0.5

    N = a.shape[0]
    xy = N//2 + numpy.floor(0.5 + N * p[:,0:2]).astype(int)
    for (x, y), v in zip(xy, v):
        a[y, x] += v


def degrid(a, p):
    """DeGrid visibilities (v) at positions (p) from (a) without convolution

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at (in fraction of grid)
    :returns: Array of visibilities.
    """
    assert numpy.max(p) < 0.5

    N = a.shape[0]
    xy = N//2 + numpy.floor(0.5 + p[:,0:2] * N).astype(int)
    v = [ a[y,x] for x,y in xy ]
    return numpy.array(v)


def frac_coord(N, Qpx, p):
    """
    Compute whole and fractional parts of coordinates, rounded to
    Qpx-th fraction of pixel size

    The fractional values are rounded to nearest 1/Qpx pixel value. At
    fractional values greater than (Qpx-0.5)/Qpx coordinates are
    roundeded to next integer index.

    :param N: Number of pixels in total
    :param Qpx: Fractional values to round to
    :param p: Coordinate in range [-.5,.5[
    """
    assert (p >= -0.5).all() and (p < 0.5).all()
    x = N//2 + p * N
    flx = numpy.floor(x + 0.5 / Qpx)
    fracx = numpy.around((x - flx) * Qpx)
    return flx.astype(int), fracx.astype(int)


def frac_coords(shape, Qpx, p):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    :param shape: (height,width) grid shape
    :param Qpx: Oversampling factor
    :param p: array of (x,y) coordinates in range [-.5,.5[
    """
    h, w = shape # NB order (height,width) to match numpy!
    x, xf = frac_coord(w, Qpx, p[:,0])
    y, yf = frac_coord(h, Qpx, p[:,1])
    return x,xf, y,yf


def convgrid(gcf, a, p, v):
    """Grid after convolving with gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param a: Grid to add to
    :param p: UVW positions
    :param v: Visibility values
    :param gcf: Oversampled convolution kernel
    """

    Qpx, _, gh, gw = gcf.shape
    coords = frac_coords(a.shape, Qpx, p)
    for v, x,xf, y,yf in zip(v, *coords):
        a[y-gh//2 : y+(gh+1)//2,
          x-gw//2 : x+(gw+1)//2] += gcf[yf,xf] * v


def convdegrid(gcf, a, p):
    """Convolutional degridding

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param gcf: Oversampled convolution kernel
    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :returns: Array of visibilities.
    """
    Qpx, _, gh, gw = gcf.shape
    coords = frac_coords(a.shape, Qpx, p)
    vis = [
        numpy.sum(a[y-gh//2 : y+(gh+1)//2,
                    x-gw//2 : x+(gw+1)//2] * gcf[yf,xf])
        for x,xf, y,yf in zip(*coords)
    ]
    return numpy.array(vis)


def sort_vis_w(p, v=None):
    """Sort visibilities on the w value.
    :param p: uvw coordinates
    :param v: Visibility values (optional)
    """
    zs = numpy.argsort(p[:, 2])
    if v is not None:
        return p[zs], v[zs]
    else:
        return p[zs]


def slice_vis(step, uvw, v=None):
    """ Slice visibilities into a number of chunks.

    :param step: Maximum chunk size
    :param uvw: uvw coordinates
    :param src: visibility source
    :param v: Visibility values (optional)
    :returns: List of visibility chunk (pairs)
    """
    nv = len(uvw)
    ii = range(0, nv, step)
    if v is None:
        return [ uvw[i:i+step] for i in ii ]
    else:
        return [ (uvw[i:i+step], v[i:i+step]) for i in ii ]


def doweight(theta, lam, p, v):
    """Re-weight visibilities

    Note that as is usual, convolution kernels are not taken into account
    """
    N = int(round(theta * lam))
    assert N > 1
    gw = numpy.zeros([N, N])
    x, xf, y, yf = frac_coords(gw.shape, 1, p / lam)
    for i in range(len(x)):
        gw[x[i], y[i]] += 1
    v = v.copy()
    for i in range(len(x)):
        v[i] /= gw[x[i], y[i]]
    return v


def simple_imaging(theta, lam, uvw, src, vis):
    """Trivial function for imaging

    Does no convolution but simply puts the visibilities into a grid
    cell i.e. boxcar gridding
    """
    N = int(round(theta * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    grid(guv, uvw / lam, vis)
    return guv


def simple_predict(guv, theta, lam, uvw):
    """Trivial function for degridding

    Does no convolution but simply extracts the visibilities from a grid cell i.e. boxcar degridding

    :param theta: Field of view (directional cosines)
    :param lam: Maximum uv represented in the grid
    :param p: UVWs of visibilities
    :param v: Visibility values
    :param kv: gridding kernel
    :returns: p, v
    """
    N = int(round(theta * lam))
    assert N > 1
    v = degrid(guv, uvw / lam)
    return v


def conv_imaging(theta, lam, uvw, src, vis, kv):
    """Convolve and grid with user-supplied kernels

    :param theta: Field of view (directional cosines))
    :param lam: UV grid range
    :param uvw: UVWs of visibilities
    :param src: Visibility source information (ignored)
    :param vis: Visibility values
    :param kv: Gridding kernel
    :returns: UV grid
    """
    N = int(round(theta * lam))
    assert N > 1
    guv = numpy.zeros([N, N], dtype=complex)
    convgrid(kv, guv, uvw / lam, vis)
    return guv


def w_slice_imaging(theta, lam, uvw, src, vis,
                    wstep=2000,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection imaging using slices

    Sorts visibility by w value and splits into equally sized slices.
    W-value used for kernels is mean w per slice. Uses the same size
    for all kernels irrespective of w.

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visibilities (wavelenghts)
    :param src: Visibility source information
    :param vis: Visibility values
    :param wstep: Size of w-slices
    :param kernel_fn: Function for generating the kernels. Parameters
      `(theta, w, *ant, **kwargs)`. Default `w_kernel`.
    :returns: UV grid
    """
    N = int(round(theta * lam))
    assert N > 1
    slices = slice_vis(wstep, *sort_vis_w(p, v))
    guv = numpy.zeros([N, N], dtype=complex)
    for ps, vs in slices:
        w = numpy.mean(ps[:, 2])
        wg = numpy.conj(kernel_fn(theta, w, **kwargs))
        convgrid(wg, guv, ps / lam, vs)
    return guv


def w_slice_predict(theta, lam, uvw, src, guv,
                    wstep=2000,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection predict using w-slices

    Sorts visibility by w value and splits into equally sized slices.
    W-value used for kernels is mean w per slice. Uses the same size
    for all kernels irrespective of w.

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visiblities
    :param src: Visibility source information
    :param guv: Input uv grid to de-grid from
    :param wstep: Size of w-slices
    :param kernel_fn: Function for generating the kernels. Parameters
      `(theta, w, **kwargs)`. Default `w_kernel`.
    :returns: Visibilities, same order as p
    """
    # Calculate number of pixels in the Image
    N = int(round(theta * lam))
    assert N > 1
    # Sort the u,v,w coordinates. We cheat a little and also pass
    # visibility indices so we can easily undo the sort later.
    nv = len(p)
    slices = slice_vis(wstep, *sort_vis_w(p, numpy.arange(nv)))
    v = numpy.ndarray(nv, dtype=complex)
    for ps, ixs in slices:
        w = numpy.mean(ps[:, 2])
        wg = kernel_fn(theta, w, **kwargs)
        v[ixs] = convdegrid(wg, guv, ps / lam)
    return v


def w_conj_kernel_fn(kernel_fn):
    """Wrap a kernel function for which we know that

       kernel_fn(w) = conj(kernel_fn(-w))

    Such that we only evaluate the function for positive w. This is
    benificial when the underlying kernel function does caching, as it
    improves the cache hit rate.

    :param kernel_fn: Kernel function to wrap
    :returns: Wrapped kernel function
    """

    def fn(theta, w, *args, **kw):
        if w < 0:
            return numpy.conj(kernel_fn(theta, -w, *args, **kw))
        return kernel_fn(theta, w, *args, **kw)
    return fn

def aw_kernel_fn(a_kernel_fn, w_kernel_fn=w_kernel):
    """
    Make a kernel function to generate AW kernels.

    This convolves three kernels for every AW-kernel: Two A-kernels
    for either antenna and the w-kernel.

    We expect the two first columns of "src" to identify the two
    antennas involved in a baseline.

    :param akernel_fn: Function to generate A-kernels. Parameters
      (theta, ant, time, freq)
    :param wkernel_fn: Function to generate w-kernels. Parameters
      (theta, w)
    :returns: Kernel function to generate AW-kernels

    """

    def fn(theta, w, a1, a2, *src):

        # Convolve antenna A-kernels
        a1kern = a_kernel_fn(theta, a1, *src)
        a2kern = a_kernel_fn(theta, a2, *src)
        akern = scipy.signal.convolve2d(a1kern, a2kern, mode='same')

        # Convolve with all oversampling values to obtain Aw-kernel
        # (note that most oversampling sub-grid values will end up unused!)
        wkern = w_kernel_fn(theta, w)
        awkern = [[scipy.signal.convolve2d(akern, wk, mode='same')
                   for wk in wks]
                  for wks in wkern]
        return numpy.array(awkern)

    return fn

def w_cache_imaging(theta, lam, uvw, src, vis,
                    wstep=2000,
                    kernel_cache=None,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Basic w-projection by caching convolution arl in w

    A simple cache can be constructed externally and passed in:

      kernel_cache = pylru.FunctionCacheManager(w_kernel, cachesize)

    If applicable, consider wrapping in `w_conj_kernel_fn` to improve
    effectiveness further.

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visibilities (wavelengths)
    :param src: Visibility source information (various)
    :param vis: Visibilites to be imaged
    :param wstep: Size of w-bins (wavelengths)
    :param kernel_cache: Kernel cache. If not passed, we fall back
       to `kernel_fn`.
    :param kernel_fn: Function for generating the kernels. Parameters
       `(theta, w, **kwargs)`. Default `w_kernel`.
    :returns: UV grid

    """

    # Construct default cache, if needed. As visibilities are
    # traversed in w-order it only needs to hold the last w-kernel.
    if kernel_cache is None:
        kernel_cache = pylru.FunctionCacheManager(kernel_fn, 1)

    N = int(round(theta * lam))
    guv = numpy.zeros([N, N], dtype=complex)
    for p, s, v in zip(uvw, src, vis):
        wbin = wstep * numpy.round(p[2] / wstep)
        wg = numpy.conj(kernel_cache(theta, wbin, *s, **kwargs))
        convgrid(wg, guv, numpy.array([p / lam]), numpy.array([v]))
    return guv


def w_cache_predict(theta, lam, uvw, src, guv,
                    wstep=2000,
                    kernel_cache=None,
                    kernel_fn=w_kernel,
                    **kwargs):
    """Predict visibilities using w-kernel cache

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visibilities  (wavelengths)
    :param guv: Input uv grid to de-grid from
    :param wstep: Size of w-bins (wavelengths)
    :param kernel_cache: Kernel cache. If not passed, we fall back
       to `kernel_fn`. See `w_cache_imaging` for details.
    :param kernel_fn: Function for generating the kernels. Parameters
       `(theta, w, **kwargs)`. Default `w_kernel`.
    :returns: degridded visibilities
    """

    if kernel_cache is None:
        kernel_cache = pylru.FunctionCacheManager(kernel_fn, 1)
    def kernel_binner(theta, w, **kw):
        wbin = wstep * numpy.round(w / wstep)
        return kernel_cache(theta, wbin, **kw)
    v = numpy.ndarray(nv, dtype=complex)
    for p, s in zip(uvw, src):
        wbin = wstep * numpy.round(p[2] / wstep)
        wg = kernel_cache(theta, wbin, *s, **kwargs)
        v[ixs] = convdegrid(wg, guv, numpy.array([p / lam]))
    return v


def mirror_uvw(uvw, vis):
    """Mirror uvw with v<0 such that all visibilities have v>=0

    The result visibilities will be equivalent, as every baseline
    `a->b` has a "sister" baseline `b->a` with a complex-conjugate
    value. A dataset typically only contains one of the two, so here
    we simply choose visibilities that lie in one half of the grid.

    :param uvw: UVWs of visibilities
    :param vis: Visibilities
    :returns: new uvw, vis
    """

    # Determine indices with v<1, make copies to update
    vn = uvw[:,1] < 0
    uvw = numpy.copy(uvw)
    vis = numpy.copy(vis)

    # Flip coordinates and conjugate visibilities
    uvw[vn] = -uvw[vn]
    vis[vn] = numpy.conj(vis[vn])
    return uvw, vis


def make_grid_hermitian(guv):
    """
    Make a grid "hermitian" by adding it to its own conjugated mirror
    image.

    Just as baselines can be seen from two sides, the grid is
    "hermitian": Mirrored coordinates yield complex-conjugate
    uv-cells. However in practice, it is less computationally
    intensive to just grid one of each baseline pair, and then restore
    the hermitian property afterwards.

    :param guv: Input grid
    :returns: Hermitian grid
    """

    # Make mirror image, then add its conjugate to the original grid.
    # This is not the same concept as hermitian matrices, as:
    #  1) Not the same symetry as transposition
    #  2) We mirror on the zero point, which is off-center if the grid
    #     has even size
    if guv.shape[0] % 2 == 0:
        guv[1:,1:] += numpy.conj(guv[:0:-1,:0:-1])
        # Note that we ignore the high frequencies here
    else:
        guv += numpy.conj(guv[::-1,::-1])
    return guv


def do_imaging(theta, lam, uvw, src, vis, imgfn, **kwargs):

    """Do imaging with imaging function (imgfn)

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visibilities (wavelengths)
    :param ant: Visibility source information (various)
    :param vis: Visibilities to be imaged
    :param imgfn: imaging function e.g. `simple_imaging`, `conv_imaging`,
      `w_slice_imaging` or `w_cache_imaging`. All keyword parameters
      are passed on to the imaging function.
    :returns: dirty Image, psf
    """
    if src == None: src = numpy.ndarray((len(vis), 0))
    # Mirror baselines such that v>0
    uvw,vis = mirror_uvw(uvw, vis)
    # Determine weights
    wt = doweight(theta, lam, uvw, numpy.ones(len(uvw)))
    # Make image
    cdrt = imgfn(theta, lam, uvw, src, wt * vis, **kwargs)
    drt = numpy.real(ifft(make_grid_hermitian(cdrt)))
    # Make point spread function
    c = imgfn(theta, lam, uvw, src, wt, **kwargs)
    psf = numpy.real(ifft(make_grid_hermitian(c)))
    # Normalise
    pmax = psf.max()
    assert pmax > 0.0
    return drt / pmax, psf / pmax, pmax


def do_predict(theta, lam, uvw, modelimage, predfn, **kwargs):
    """Predict visibilities for a model Image at the phase centre using the
    specified degridding function.

    :param theta: Field of view (directional cosines)
    :param lam: UV grid range (wavelenghts)
    :param uvw: UVWs of visiblities (wavelengths)
    :param modelimage: model image as numpy.array (phase center at Nx/2,Ny/2)
    :param predfn: prediction function e.g. `simple_predict`,
      `w_slice_predict` or `w_cache_predict`.
    :returns: predicted visibilities
    """
    ximage = fft(modelimage.astype(complex))
    return predfn(theta, lam, uvw, src, ximage, **kwargs)
