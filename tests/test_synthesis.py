
from crocodile.synthesis import *
from crocodile.simulate import *

import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from astropy.coordinates import SkyCoord
from astropy import units as u

class TestSynthesis(unittest.TestCase):

    def test_coordinates(self):
        for N in [4,5,6,7,8,9,1000,1001,1002,1003]:
            low, high = coordinateBounds(N)
            c = coordinates(N)
            cx, cy = coordinates2(N)
            self.assertAlmostEqual(np.min(c), low)
            self.assertAlmostEqual(np.max(c), high)
            self.assertAlmostEqual(np.min(cx), low)
            self.assertAlmostEqual(np.max(cx), high)
            self.assertAlmostEqual(np.min(cy), low)
            self.assertAlmostEqual(np.max(cy), high)
            assert c[N//2] == 0
            assert (cx[:,N//2] == 0).all()
            assert (cy[N//2,:] == 0).all()

    def _pattern(self, N):
        return coordinates2(N)[1]+coordinates2(N)[0]*1j

    def test_pad_extract(self):
        for N, N2 in [ (1,1), (1,2), (2,3), (3,4), (2,5), (4,6) ]:
            cs = 1 + self._pattern(N)
            cs_pad = pad_mid(cs, N2)
            cs2 = 1 + self._pattern(N2) * N2 / N
            # At this point all fields in cs2 and cs_pad should either
            # be equal or zero.
            equal = numpy.abs(cs_pad - cs2) < 1e-15
            zero = numpy.abs(cs_pad) < 1e-15
            assert numpy.all(equal + zero)
            # And extracting the middle should recover the original data
            assert_allclose(extract_mid(cs_pad, N), cs)

    def test_anti_aliasing(self):
        for shape in [(4,4),(5,5),(4,6),(7,3)]:
            aaf = anti_aliasing_function(shape, 0, 10)
            self.assertEqual(aaf.shape, shape)
            self.assertAlmostEqual(aaf[shape[0]//2,shape[1]//2], 1)

    def test_w_kernel_function(self):
        l,m = kernel_coordinates(5,0.1)
        assert_allclose(w_kernel_function(l,m,0), 1)
        self.assertAlmostEqual(w_kernel_function(l,m,100)[2,2], 1)
        l,m = kernel_coordinates(10,0.1)
        self.assertAlmostEqual(w_kernel_function(l,m,100)[5,5], 1)
        self.assertAlmostEqual(w_kernel_function(l,m,1000)[5,5], 1)

    def test_w_kernel_function_recentre(self):
        for lam in [4, 1000]:
         gcf = numpy.conj(w_kernel(1/lam, 0, 1, 1, 1))
         for N in range(2,6):
          for w in [0.1, 1, 10]:
           for dl, dm in [(1,1), (2,0), (1,-3)]:
               # Generate kernels with one re-centred by (dl,dm).
               l,m = kernel_coordinates(N, N/lam)
               cp = w_kernel_function(l,m,w)
               kern = ifft(cp)
               kerns = ifft(kernel_recentre(cp, N/lam, w, dl*lam/N/w,dm*lam/N/w))
               # Check that we shifted the kernel appropriately. Note
               # that for numpy's axes 0 -> Y and 1 -> X.
               assert_allclose(numpy.roll(numpy.roll(kern, dl, axis=1), dm, axis=0),
                               kerns,
                               atol=1e-15)

    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for N in range(3,30):
            pat = self._pattern(N)
            kern = kernel_oversample(pat, 1, N-2)
            kern2 = kernel_oversample(pat, 2, N-2)
            assert_allclose(kern[0,0], kern2[0,0], atol=1e-13)
            kern3 = kernel_oversample(pat, 3, N-2)
            assert_allclose(kern[0,0], kern3[0,0], atol=1e-13)
            kern4 = kernel_oversample(pat, 4, N-2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy,ux], kern4[2*uy,2*ux], atol=1e-13)
            kern8 = kernel_oversample(pat, 8, N-2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy,ux], kern8[2*uy,2*ux], atol=1e-13)

    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        N = 10
        wff = numpy.zeros((N,N))
        wff[N//2,N//2] = 1 # Not the most interesting kernel...
        k = kernel_oversample(wff, 1, N)
        k2 = kernel_oversample(pad_mid(wff,N*2), 1, N)
        assert_allclose(k, k2)

    def _uvw(self, N, uw=0, vw=0):
        u,v = coordinates2(N)
        u=numpy.hstack(u)
        v=numpy.hstack(v)
        w = uw*u + vw*v
        return numpy.transpose([u,v,w])

    def test_grid_degrid(self):
        # The uvw we chose here correspond exactly with grid points
        # (at w=0) This is the "perfect" case in which all of this
        # should gracefully degrade to a simple FFT. There is
        # especially no information loss, so we can test
        # exhaustively.
        for lam in [6, 1e15]:
          gcf = numpy.conj(w_kernel(1/lam, 0, 1, 1, 1))
          for N in range(2,6):
            uvw = self._uvw(N)*lam
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Simulate and grid a single point source
                vis = simulate_point(uvw, x/lam, y/lam)
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvw/lam, vis)
                # Do it using convolution gridding too, which should
                # double the result
                convgrid(gcf, a, uvw/lam, None, vis)
                a /= 2
                # Image should have a single 1 at the source
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+y,N//2+x], 1)
                img[N//2+y,N//2+x] = 0
                assert_allclose(img, 0, atol=1e-14)
                # Degridding should reproduce the original visibilities
                vis_d = degrid(a, uvw/lam)
                assert_allclose(vis, vis_d)
                vis_d = convdegrid(gcf, a, uvw/lam)
                assert_allclose(vis, vis_d)

    def test_grid_shift(self):
        lam = 100
        for N in range(3,7):
          for dl, dm in [(1/lam, 1/lam), (-1/lam, 2/lam), (5/lam, 0)]:
            uvw = self._uvw(N)*lam
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Simulate and grid a single off-centre point source,
                # then shift back.
                vis = simulate_point(uvw, x/lam-dl, y/lam-dm)
                vis = visibility_shift(uvw, vis, dl, dm)
                # Should give us a point where we originally placed it
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvw/lam, vis)
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+y,N//2+x], 1)

    def test_grid_transform(self):
        lam = 100
        s = 1 / numpy.sqrt(2)
        Ts = [
            numpy.array([[1,0], [0,1]]),
            numpy.array([[-1,0], [0,1]]),
            numpy.array([[s,-s], [s,s]]),
            numpy.array([[2,0], [0,3]]),
            numpy.array([[1,2], [2,1]]),
            numpy.array([[1e5,-5e3], [6e4,-1e6]]),
            numpy.array([[0,.05], [-.05,0]])
        ]
        for T in Ts:
          # Invert transformation matrix
          Ti = numpy.linalg.inv(T)
          for N in range(3,7):
            # We will grid after the transformation. To make this
            # lossless we need to choose UVW such that they map
            # exactly to grid points *after* the transformation. We
            # can easily achieve this using the inverse transform.
            uvwt = self._uvw(N)*lam
            uvw = uvw_transform(uvwt, Ti)
            assert_allclose(uvw_transform(uvw, T), uvwt, atol=1e-13)
            xys = range(-(N//2),(N+1)//2)
            for xt, yt in itertools.product(xys, xys):
                # Same goes for grid positions: Determine position
                # before transformation such that we end up with a
                # point at x,y afterwards.
                x, y = numpy.dot([xt,yt], Ti)
                assert_allclose(numpy.dot([x,y], T), [xt,yt], atol=1e-13)
                # Now simulate at (x/y) using uvw, then grid using
                # the transformed uvwt, and the point should be at (xt/yt).
                vis = simulate_point(uvw, x/lam, y/lam)
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvwt/lam, vis)
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+yt,N//2+xt], 1)

    def test_grid_transform_shift(self):
        lam = 100
        s = 1 / numpy.sqrt(2)
        Ts = [
            numpy.array([[1,0], [0,1]]),
            numpy.array([[-1,0], [0,1]]),
            numpy.array([[s,-s], [s,s]]),
            numpy.array([[2,0], [0,3]]),
            numpy.array([[1,2], [2,1]]),
            numpy.array([[1e5,-5e3], [6e4,-1e6]]),
            numpy.array([[0,.05], [-.05,0]])
        ]
        for T in Ts:
         for dl, dm in [(1/lam, 1/lam), (-1/lam, 2/lam), (5/lam, 0)]:
          # Invert transformation matrix
          Ti = numpy.linalg.inv(T)
          for N in range(3,7):
            # We will grid after the transformation. To make this
            # lossless we need to choose UVW such that they map
            # exactly to grid points *after* the transformation. We
            # can easily achieve this using the inverse transform.
            uvwt = self._uvw(N)*lam
            uvw = uvw_transform(uvwt, Ti)
            assert_allclose(uvw_transform(uvw, T), uvwt, atol=1e-13)
            xys = range(-(N//2),(N+1)//2)
            for xt, yt in itertools.product(xys, xys):
                # Same goes for grid positions: Determine position
                # before transformation such that we end up with a
                # point at x,y afterwards.
                x, y = numpy.dot([xt,yt], Ti)
                assert_allclose(numpy.dot([x,y], T), [xt,yt], atol=1e-13)
                # Now simulate at (x/y) using uvw, then grid using
                # the transformed uvwt, and the point should be at (xt/yt).
                vis = simulate_point(uvw, x/lam-dl, y/lam-dm)
                vis = visibility_shift(uvw, vis, dl, dm)
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvwt/lam, vis)
                img = numpy.real(ifft(a))
                self.assertAlmostEqual(img[N//2+yt,N//2+xt], 1)

    def test_slice_vis(self):
        for N in range(2,10):
            for step in range(1,10):
                cs = self._uvw(N)
                slices = slice_vis(step, cs)
                assert_allclose(cs, numpy.vstack(slices))

    def test_grid_degrid_w(self):
        lam = 1000
        for uw, vw in [(.5,0),(0,.5),(-1,0),(0,-1)]:
          for N in range(1,6):
            # Generate UVW with w != 0, generate a w-kernel for every
            # unique w-value (should be exactly N by choice of uw,vw)
            uvw_all = self._uvw(N, uw, vw) * lam
            uvw_slices = slice_vis(N, sort_vis_w(uvw_all))
            # Generate kernels for every w-value, using the same far
            # field size and support as the grid size (perfect sampling).
            gcfs = [ (w_kernel(N/lam, numpy.mean(uvw[:,2]), N, N, 1)/N**2,
                      w_kernel(N/lam, numpy.mean(uvw[:,2]), N, N, 1, invert=False)/N**2)
                     for uvw in uvw_slices ]
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Generate expected image for degridding
                img_ref = numpy.zeros((N, N), dtype=float)
                img_ref[N//2+y,N//2+x] = 1
                # Gridding does not have proper border handling, so we
                # need to artificially increase our grid size here by
                # duplicating grid data.
                a_ref = numpy.fft.ifftshift(fft(img_ref))
                a_ref = numpy.vstack(2*[numpy.hstack(2*[a_ref])])
                # Make grid for gridding
                a = numpy.zeros((2*N, 2*N), dtype=complex)
                assert a.shape == a_ref.shape
                for uvw, (gcf, gcf_p) in zip(uvw_slices, gcfs):
                    # Degridding result should match direct fourier transform
                    vis = simulate_point(uvw, x/lam, y/lam)
                    vis_d = convdegrid(gcf_p, a_ref, uvw/lam/2)
                    assert_allclose(vis, vis_d)
                    # Grid
                    convgrid(gcf, a, uvw/lam/2, None, vis)
                # FFT halved generated grid (see above)
                a = numpy.fft.fftshift(a[:N,:N]+a[:N,N:]+a[N:,:N]+a[N:,N:])
                img = numpy.real(ifft(a))
                # Peak should be there, rest might be less precise as
                # we're not sampling the same w-plane any more
                assert_allclose(img[N//2+y,N//2+x], 1)
                assert_allclose(img, img_ref, atol=2e-3)

    def test_grid_shift_w(self):
        lam = 10
        for uw, vw in [(.5,0),(0,.5),(-1,0),(0,-1)]:
         for N in range(1,6):
          for dl, dm in [(1/lam, 1/lam), (-1/lam, 2/lam), (5/lam, 0)]:
            theta = N/lam
            uvw_all = self._uvw(N, uw, vw) * lam
            uvw_slices = slice_vis(N, sort_vis_w(uvw_all))
            gcfs = [ w_kernel(theta, numpy.mean(uvw[:,2]), N, N, 1, dl=-dl, dm=-dm)/N**2
                     for uvw in uvw_slices ]
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                # Make grid for gridding
                a = numpy.zeros((2*N, 2*N), dtype=complex)
                for uvw, gcf in zip(uvw_slices, gcfs):
                    vis = simulate_point(uvw, x/lam-dl, y/lam-dm)
                    # Shift. Make sure it is reversible correctly
                    # (This is enough to prove that degridding would
                    # be consistent as well)
                    viss = visibility_shift(uvw, vis, dl, dm)
                    assert_allclose(visibility_shift(uvw, viss, -dl, -dm), vis)
                    # Grid
                    convgrid(gcf, a, uvw/lam/2, None, viss)
                # FFT halved generated grid
                a2 = numpy.fft.fftshift(a[:N,:N]+a[:N,N:]+a[N:,:N]+a[N:,N:])
                img = numpy.real(ifft(a2))
                # Check peak
                assert_allclose(img[N//2+y,N//2+x], 1)

    def test_grid_transform_shift_w(self):
        lam = 10
        s = 1 / numpy.sqrt(2)
        Ts = [
            numpy.array([[1,0], [0,1]]),
            numpy.array([[2,0], [0,3]]),
            numpy.array([[0,1], [-1,0]]),
            numpy.array([[s,s], [-s,s]]),
            numpy.array([[1e5,-5e3], [6e4,-1e6]]),
            numpy.array([[0,.5], [-.5,0]])
        ]
        for T in Ts:
         Ti = numpy.linalg.inv(T)
         for uw, vw in [(.5,0),(0,.5),(0,-1)]:
          for dl, dm in [(0,0), (1/lam, 1/lam), (-1/lam, 4/lam)]:
           for N in range(1,6):
            # Generate transformed UVW with w != 0, generate a
            # w-kernel for every unique w-value (should be exactly N
            # by choice of uw,vw). Obtain "original" UVW using inverse
            # transformation.
            uvwt_all = self._uvw(N, uw, vw)*lam
            uvw_all = uvw_transform(uvwt_all, Ti)
            uvw_slices = slice_vis(N, *sort_vis_w(uvwt_all, uvw_all))
            # Generate kernels for every w-value, using the same far
            # field size and support as the grid size (perfect sampling).
            gcfs = [ w_kernel(N/lam, numpy.mean(uvwt[:,2]), N, N, 1, T=Ti, dl=-dl, dm=-dm)/N**2
                     for uvwt,_ in uvw_slices ]
            xys = range(-(N//2),(N+1)//2)
            for xt, yt in itertools.product(xys, xys):
                x, y = numpy.dot([xt,yt], Ti)
                assert_allclose(numpy.dot([x,y], T), [xt,yt], atol=1e-14)
                # Make grid for gridding
                a = numpy.zeros((2*N, 2*N), dtype=complex)
                # Grid with shift *and* using transformed UVW. This
                # should give us a point at the transformed (xt,yt)
                # position, again.
                for (uvwt, uvw), gcf in zip(uvw_slices, gcfs):
                    vis = simulate_point(uvw, x/lam-dl, y/lam-dm)
                    vis = visibility_shift(uvw, vis, dl, dm)
                    convgrid(gcf, a, uvwt/lam/2, None, vis)
                # FFT halved generated grid
                a = numpy.fft.fftshift(a[:N,:N]+a[:N,N:]+a[N:,:N]+a[N:,N:])
                img = numpy.real(ifft(a))
                # Check peak
                assert_allclose(img[N//2+yt,N//2+xt], 1)

    def test_hermitian(self):
        lam = 100
        _or = numpy.logical_or
        _and = numpy.logical_and
        for N in range(2, 20):
            # Determine UVW. Deselect zero baseline. Round to prevent
            # floating point inaccuracies 
            uvw = numpy.round(self._uvw(N)*lam)
            non_zero = _or(uvw[:,0] != 0, uvw[:,1] != 0)
            # Select non-redundant baselines
            non_red = _or(uvw[:,0] < 0, _and(uvw[:,0] == 0, uvw[:,1] < 0))
            if N % 2 == 0:
                # For even grid sizes, all baselines that correspond
                # to the outermost frequency are non-redundant, as
                # there is no valid mirrored coordinate.
                non_red = _or(non_red, _or(uvw[:,0] == -lam/2, uvw[:,1] == -lam/2))
            # Determine number of "baselines" if we remove redundant entries
            # Griding the complete set should have the same result as
            # gridding the non-redundant set followed by making the
            # grid hermitian.
            xys = range(-(N//2),(N+1)//2)
            for x, y in itertools.product(xys, xys):
                vis = simulate_point(uvw, x/lam, y/lam)
                # Grid non-zero baselines vanilla
                a = numpy.zeros((N, N), dtype=complex)
                grid(a, uvw[non_zero]/lam, vis[non_zero])
                # Grid non-redundant baselines, then make hermitian
                a_bl = numpy.zeros((N, N), dtype=complex)
                grid(a_bl, uvw[non_red]/lam, vis[non_red])
                a_blh = make_grid_hermitian(a_bl)
                # The two grids should be identical. We especially
                # expect the FFT image to be entirely real values.
                assert_allclose(a, a_blh, rtol=1e-5, atol=1e-5)
                assert_allclose(ifft(a).imag, 0, atol=1e-14)

    def test_frac_coord(self):

        for Qpx in range(1,6):
            # Check all values at *double* the oversampled accuracy
            p = numpy.arange(-2*Qpx, 2*Qpx)/4/Qpx
            flx, fracx = frac_coord(2, Qpx, p)
            # Make sure we honour the allowed values for oversampled offsets
            self.assertTrue(numpy.all(fracx >= 0).all())
            self.assertTrue(numpy.all(fracx < Qpx))
            # Check that the resulting coordinates are sound. Note
            # that because we placed coordinates exactly between
            # oversampling values above we get "even" rounding three
            # times more often than "odd" rounding. This is correct
            # behaviour of numpy.around to prevent biases in this
            # corner case.
            assert_allclose(flx - fracx / Qpx,
                            numpy.around(numpy.arange(0, 4*Qpx)/2)/Qpx)

if __name__ == '__main__':
    unittest.main()
