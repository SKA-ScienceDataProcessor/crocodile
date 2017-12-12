
# Anti-aliasing kernel generation
#
# This code is mostly by Sze Meng Tan, re-implementing the algorithms
# described in his 1986 PhD thesis "Aperture-synthesis mapping and
# parameter estimation"

import numpy

from scipy.optimize import leastsq, brent
import numpy as np
from json_tricks.np import load

from crocodile.synthesis import *

# Load kernel cache
KERNEL_CACHE = {}
with open('gridder.json', 'r') as f:
    for key, val in load(f):
        KERNEL_CACHE[tuple(key[0])] = val

def trap(vec, dx):
    # Perform trapezoidal integration
    return dx * (numpy.sum(vec) - 0.5 * (vec[0] + vec[-1]))

def func_to_min(h, x0, M, R):
    N = len(h)
    nu = (np.arange(M, dtype=float) + 0.5) / (2 * M)
    x = x0 * np.arange(N+1, dtype=float)/N
    C = calc_gridder_as_C(h, x0, nu, R)
    dnu = nu[1] - nu[0]
    dx = x[1] - x[0]
    h_ext = np.concatenate(([1.0], h))
    loss = np.zeros((len(h_ext), 2, M), dtype=float)
    for n, x_val in enumerate(x):
        one_app = 0
        for r in range(0, 2 * R):
            l = r - R + 1
            one_app += h_ext[n] * C[r, :] * np.exp(2j * np.pi * (l - nu) * x_val)
        loss[n, 0, :] = 1.0 - np.real(one_app)
        loss[n, 1, :] = np.imag(one_app)
        if n in [0, N]:
            loss[n, :, :] /= np.sqrt(2)
    loss = loss.reshape(2 * M * (N + 1))
    return loss

def optimal_grid(R, x0, N, M, h_initial=None):
    if h_initial is None:
        h_initial = np.ones(N, dtype=float)
    return leastsq(func_to_min, h_initial, args=(x0, M, R), full_output=True)

def find_cached_kernel(R, x0,  N=32, M=64, max_dx0=1/16):
    """ Returns the key for a cached gridder with th given parameters

    :param R: Half support in pixels
    :param x0: Image plane coordinates up to which coordinates are optimised
    :param N: Number of points to evaluate in image space
    :param M: Number of sub-grid points to evaluate in grid space (oversampling)
    :param max_dx0: Difference in x0 to accept when using cached kernel
    """

    # First try to look up directly
    key = (R, x0, N, M)
    if key in KERNEL_CACHE:
        return key

    # Then try to find next best one
    best = None
    dx0 = 1e15
    for kp in KERNEL_CACHE.keys():
        Rp, x0p, Np, Mp = kp

        # R must match, x0 difference must be at least as good
        if Rp != R: continue
        if abs(x0p - x0) > dx0: continue

        # Otherwise select by quality of optimisation
        if abs(x0p - x0) == dx0:
            if Np < best[2] or Mp < best[3]:
                continue
        best = kp
        dx0 = abs(x0p - x0)

    return best

def optimal_grid_cached(R, x0, N=32, M=64, max_dx0=1/16, h_initial=None):
    """ Returns an optimal gridder for the given parameters.

    :param R: Half support in pixels
    :param x0: Image plane coordinates up to which coordinates are optimised
    :param N: Number of points to evaluate in image space
    :param M: Number of sub-grid points to evaluate in grid space (oversampling)
    :param max_dx0: Difference in x0 to accept when using cached kernel
    """
    assert R >= 1
    assert x0 < 0.5

    # Best one good enough?
    best = find_cached_kernel(R, x0, N, M, max_dx0)
    if best is not None:
        Rp, x0p, Np, Mp = best
        if x0p >= x0 and x0p - x0 <= max_dx0 and Np >= N and Mp >= M:
            return KERNEL_CACHE[best]

    # Have one that is at least close enough for an initial solution?
    h_initial = None
    if best is not None:
        Rp, x0p, Np, Mp = best
        if abs(x0p - x0) <= max_dx0:
            x = x0 * (1 + np.arange(0, N, dtype=float))/N
            h_initial = calc_gridding_correction_fine(x, KERNEL_CACHE[best]['h'], x0p, M, Rp)

    # As last resort, use one with less support
    if h_initial is None:
        if R == 3:
            h_initial = optimal_grid_cached(R-1,x0,N,M, max_dx0)['h']
        elif R >= 4:
            h0 = optimal_grid_cached(R-2,x0,N,M, max_dx0)['h']
            h1 = optimal_grid_cached(R-1,x0,N,M, max_dx0)['h']
            h_initial = h1 * (h1/h0)

    # Run optimisation
    key = (R, x0, N, M)
    print("Calculating", key)
    h, cov_h, infodict, mesg, ler = optimal_grid(R, x0, N, M, h_initial)
    if ler == 0:
        return None

    # Determine error
    mean_map_err = calc_mean_map_error(x0, N, M, R, h)
    print("Mean map error: %g" % mean_map_err)

    # Make & cache result
    result = {"R": R, "x0":x0, "h":h, "err":mean_map_err}
    KERNEL_CACHE[key] = result
    return result

def calc_gridder_as_C(h, x0, nu, R):
    # Calculate gridding function C(l,nu) from gridding correction h(x) evaluated at (n+1) x_0/N where n is in range 0,...,N-1. 
    #  We assume that h(0)=1 for normalization, and use the trapezium rule for integration.
    # The gridding function is calculated for l=-R+1 to R at points nu
    M = len(nu)
    C = np.zeros((2*R, M), dtype=float)
    K = np.zeros((2*R, 2*R))
    N = len(h)
    x = x0 * np.arange(0, N+1, dtype=float)/N
    dx = x0 / N
    h_ext = np.concatenate(([1.0], h))

    for rp in range(0, 2 * R):
        lp = rp - R + 1
        for r in range(2 * R):
            l = r - R + 1
            K[rp, r] = trap((h_ext**2) * np.cos(2 * np.pi * (lp - l) * x), dx)
    #Kinv = np.linalg.pinv(K)

    rhs = np.zeros(2*R, dtype=float)
    for m, nu_val in enumerate(nu):
        for rp in range(0, 2 * R):
            lp = rp - R + 1
            rhs[rp] = trap(h_ext * np.cos(2 * np.pi * (lp - nu_val) * x), dx)
        C[:,m] = np.linalg.lstsq(K, rhs, 1e-14)[0]
        # C[:,m] = np.dot(Kinv, rhs)
    return C

def make_evaluation_grids(R, M, N):
    """Generate vectors nu and x on which the gridder and gridding correction functions need to be evaluated.
        R is the number of integer gridpoints on each side of the data point
        M determines the sampling of the nu grid, dnu = 1/(2*M)
        N determines the sampling of the x grid, dx = 1/(2*N)
    """
    nu = (numpy.arange(2 * R * M, dtype=float) + 0.5) / (2 * M)
    x = numpy.arange(N+1, dtype=float)/(2*N)
    return nu, x

def calc_gridder(h, x0, nu, M, R):
    # Calculate gridder function on a grid nu which should have been generated using make_evaluation_grids
    #  The array h is the result of an optimization process for the gridding correction function evaluated
    #  on a relatively coarse grid extending from 0 to x0
    C = calc_gridder_as_C(h, x0, nu, R)
    gridder = np.zeros(2*M*R, dtype=float)
    for m in range(M):
        for rp in range(0, 2 * R):
            lp = rp - R + 1
            indx = m - 2*lp*M
            if indx >= 0:
                gridder[indx] = C[rp, m]
            else:
                gridder[-indx-1] = C[rp, m]
    return gridder

# The functions with the "_fine" suffix allow the convolution function
# C and the gridding correction function h to be evaluated at
# arbitrary points, rather than at only the points which were used in
# performing the optimization

def calc_gridding_fine(h, x0, nu, R):
    # Calculate gridding function C(l,nu) from gridding correction h(x) evaluated at (n+1) x_0/N where n is in range 0,...,N-1.
    #  We assume that h(0)=1 for normalization, and use the trapezium rule for integration.
    # The gridding function is calculated for l=-R+1 to R at points nu
    M = len(nu)
    C = numpy.zeros((2*R, M), dtype=float)
    K = numpy.zeros((2*R, 2*R))
    N = len(h)
    x = x0 * numpy.arange(0, N+1, dtype=float)/N
    dx = x0 / N
    h_ext = numpy.concatenate(([1.0], h))

    for rp in range(0, 2 * R):
        lp = rp - R + 1
        for r in range(2 * R):
            l = r - R + 1
            K[rp, r] = trap((h_ext**2) * numpy.cos(2 * numpy.pi * (lp - l) * x), dx)
    Kinv = numpy.linalg.inv(K)

    rhs = numpy.zeros(2*R, dtype=float)
    for m, nu_val in enumerate(nu):
        for rp in range(0, 2 * R):
            lp = rp - R + 1
            rhs[rp] = trap(h_ext * numpy.cos(2 * numpy.pi * (lp - nu_val) * x), dx)
        C[:,m] = numpy.dot(Kinv, rhs)
    return C

def gridder_to_C(gridder, R):
    """Reformat gridder evaluated on the nu grid returned by make_evaluation_grids into the sampled C function
       which has an index for the closest gridpoint and an index for the fractional distance from that gridpoint
    """
    M = len(gridder) // (2 * R)
    C = np.zeros((2 * R, M), dtype=float)
    for r in range(0, 2 * R):
        l = r - R + 1
        indx = np.arange(M) - 2 * M * l
        # Use symmetry to deal with negative indices
        indx[indx<0] = -indx[indx<0] - 1
        C[r, :] = gridder[indx]
    return C

def calc_gridding_correction_fine(x, h, x0, M, R):
    nu = (numpy.arange(M, dtype=float) + 0.5) / (2 * M)
    C = calc_gridding_fine(h, x0, nu, R)
    nu = (numpy.arange(M, dtype=float) + 0.5) / (2 * M)
    dnu = 1.0 / (2 * M)
    c = numpy.zeros(x.shape, dtype=float)
    d = numpy.zeros(x.shape, dtype=float)
    for n, x_val in enumerate(x):
        for rp in range(0, 2 * R):
            lp = rp - R + 1
            for r in range(2 * R):
                l = r - R + 1
                d[n] += numpy.sum(C[rp, :] * C[r, :] * numpy.cos(2 * numpy.pi * (lp - l) * x_val)) * dnu
            c[n] += numpy.sum(C[rp, :] * numpy.cos(2 * numpy.pi * (lp - nu) * x_val)) * dnu
    hfine = c/d
    return hfine

def calc_map_error(gridder, grid_correction, nu, x, R):
    M = len(nu) // (2 * R)
    N = len(x) - 1
    dnu = nu[1] - nu[0]
    C = gridder_to_C(gridder, R)
    loss = np.zeros((len(x), 2, M), dtype=float)
    for n, x_val in enumerate(x):
        one_app = 0
        for r in range(0, 2 * R):
            l = r - R + 1
            one_app += grid_correction[n] * C[r, :] * np.exp(2j * np.pi * (l - nu[:M]) * x_val)
        loss[n, 0, :] = 1.0 - np.real(one_app)
        loss[n, 1, :] = np.imag(one_app)
    map_error = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        map_error[i] = 2 * np.sum((loss[i, :, :].flatten())**2) * dnu
    return map_error

def calc_mean_map_error(x0, N, M, R, h):
    nu, x = make_evaluation_grids(R, M, N)
    gridder = calc_gridder(h, x0, nu, M, R)
    grid_correction = calc_gridding_correction_fine(x, h, x0, M, R)
    map_err = calc_map_error(gridder, grid_correction, nu, x, R)
    which = np.digitize(x0, x)
    return numpy.sqrt( trap(map_err[:which], 1.0/(2*N)) / x[which] )

def sze_tan_grid_correction(R, x0, image_size, Nopt=64, Mopt=32, max_dx0=1/16):
    """ Generate a Sze-Tan grid correction pattern in crocodile coordinate system
    :param R: Half-support of kernel
    :param x0: Significant part of image
    :param grid_size: Size of pattern to create
    :param Nopt: Number of points to evaluate in image space
    :param Mopt: Number of sub-grid points to evaluate in grid space (oversampling)
    :param max_dx0: Difference in x0 to accept when using cached kernel
    """
    grid = optimal_grid_cached(R, x0, Nopt, Mopt, max_dx0)
    return sze_tan_grid_correction_gen(R, x0, coordinates(image_size), Nopt, Mopt, max_dx0)

def sze_tan_grid_correction_gen(R, x0, x, Nopt=64, Mopt=32, max_dx0=1/16):
    """ Generate a Sze-Tan grid correction pattern for arbitrary image coordinates
    :param R: Half-support of kernel
    :param x0: Significant part of image
    :param x: Image coordinates
    :param Nopt: Number of points to evaluate in image space
    :param Mopt: Number of sub-grid points to evaluate in grid space (oversampling)
    :param max_dx0: Difference in x0 to accept when using cached kernel
    """
    grid = optimal_grid_cached(R, x0, Nopt, Mopt, max_dx0)
    grid_correction = calc_gridding_correction_fine(numpy.array(x), grid['h'], grid['x0'], len(x), R)
    return 1 / grid_correction

def sze_tan_gridder(R, x0, over, Nopt=64, Mopt=32, max_dx0=1/16):
    grid = optimal_grid_cached(R, x0, Nopt, Mopt, max_dx0)

    # Generate gridder in crocodile coordinate system
    nu = numpy.arange(over, dtype=float) / over
    gridding = calc_gridding_fine(grid['h'], grid['x0'], nu, R)
    return numpy.transpose(gridding[::-1])

def sze_tan_mean_error(R, x0, Nopt=64, Mopt=32, max_dx0=1/16):
    grid = optimal_grid_cached(R, x0, Nopt, Mopt, max_dx0)
    return calc_mean_map_error(x0, Nopt, Mopt, R, grid['h'])
