
# AW-gridding kernel work description

This document describes possible desirable directions SDP would like
to see undertaken to study the performance of kernels on specific
architectures.

## Introduction

The kernel we have selected is gridding, because:
- The memory bandwidth and processing requirements are very demanding
- The data processed has domain specific characteristics
- It is affected by numerous parameters

We separate the work that should be undertaken into the following sections:
- Development of algorithms
- Study of optimal data layout for cache re-use and processing
- Alternative implementation of algorithms and data layout (e.g. sparse FFT)
- Organization of the parameter space
- Demonstrating the use of tools for profiling
- Reporting the behavior for implemented algorithms
- Description of the goals

## Theory

At core, radio interferometry imaging is a Fourier transform
operation, with most relevant effects mapping to simple multipliers at
various stages. Data is gathered from the interference patterns
(visibilities) of antenna pairs (baselines):

<img src="data/images/visibility_equation1.png?raw=true" width="70%">

To bring all visibility contributions into a common image plane we
further multiply by `G_w` to correct for baseline non-coplanarity
effects and by antenna weights `A_a,t,f` to correct for non-uniform
antenna reception patterns. This yields us a definition for
visibilities `V` as a contiuous function that contains complete
information about the sky.

However, in reality we can only sample this function at discrete
points given by the antenna positions at the given time. So for radio
astronomy, our input visibilities `V` is instead a sum of delta
functions, which is only non-zero where and when we have data. The
concrete points depend on the telescope layout, earth's curvature,
earth's rotation, as well as the concrete involved antennas, and the
time and frequency we took the measurements for.

Radio astronomy spends a lot of effort to compensate for this imperfect
and irregular sampling (weighting and deconvolution), but for the
purpose of this excerise let us pretend that we can actually inverse
the above equation using an inverse Fourier transform even if our
visibility function `V` is incomplete:

<img src="data/images/visibility_equation2.png?raw=true" width="70%">

To determine `I` efficiently, we would like to use fast Fourier
transforms. However, we would not want to do repeat the FFTs twice as
the formula above suggests. Fortunately, multiplication in image space
is equivalent to convolution in frequency space, so we re-express the
above equation as:

<img src="data/images/visibility_equation3.png?raw=true" width="70%">

Where `A_a,t,f` and `G_w`a re the Fourier transforms of the original
functions, which together form the grid convolution function (GCF). It
can be shown that both functions quickly tend to zero for nonzero
`(u,v)`, which means that every visibility only contributes to a very
small portion of the frequency plane. After summing up all these
contributions, we only need to perform a single FFT to collect all
visibility information into a single picture.

## Example Code

We can easily implement imaging in just a few lines of Python using
`numpy`. Note that for clarity this version is slightly
simplified. Check the [reference code](crocodile/synthesis.py) for the
full details including sub-grid coordinates.

### GCF convolution

```python
def aw_kernel_fn(theta, w, a1, a2, t, f):
        a1kern = a_kernel_fn(theta, a1, t, f)
        a2kern = a_kernel_fn(theta, a2, t, f)
        akern = scipy.signal.convolve2d(a1kern, a2kern, mode='same')
        wkern = w_kernel_fn(theta, w)
        return scipy.signal.convolve2d(akern, wkern, mode='same')
```

This computes the `A_a1,t,f A_a2,t,f G_w` convolution of two A-kernels
and one w-kernel. The parameter theta determines the grid resolution
and depends on the imaged field of view. We leave the two kernel
functions undefined here, their values should be considered input into
the gridding algorithm.

### Gridding

```python
def convgrid(gcf, guv, p, v):
    gh, gw = gcf.shape
    x,y = numpy.floor(p + 0.5)
        guv[y-gh//2 : y+(gh+1)//2,
            x-gw//2 : x+(gw+1)//2] += gcf * v
```

The visibility value v gets multiplied by the grid convolution
function and added at the appropriate position in the grid.

### Imaging

```python
def imaging(theta, lam, uvw, src, vis):
    N = int(round(theta * lam))
    guv = numpy.zeros([N, N], dtype=complex)
    for p, s, v in zip(uvw, src, vis):
    gcf = numpy.conj(aw_kernel_fn(theta, p[2], *s))
            convgrid(gcf, guv, theta * p, v)
     img = numpy.fft.ifft2(numpy.fft.ifftshift(guv))
    return numpy.real(numpy.fft.fftshift(img))
```

The parameter lambda is the size of the uv-grid, corresponding to the
output image resolution and the length of the longest
baseline. Together with theta it is used to determine the size of the
grid to allocate. The parameters uvw and src list of the values of
`u,v,w` and `a1,a2,t,f` where `V` is nonzero, with vis giving the
visibility function values at these positions.

Once we have the grid allocated, we can use the kernel function to
determine the convolution for every visibility and grid it at the
appropriate position. After the grid is completely filled, we can use
an inverse FFT to obtain an image. Note that our grid and image have
the zero frequency/position in the center, so we need to shift before
and after the FFT.

## Optimisation Opportunities

The gridding operation is mathematically simple.  However, the data
and its representation is involved and will affect the performance.

Data properties influence the computation in many ways.  The
concurrency in computing the uv-grid should be apparent in the code
shown above.  The pattern baselines in the UVW grid is sparse,
irregular, but computable.  When `V` is computed for a particular time
and frequency the result `V` may be nonzero on a sparse subset of the
UV grid.  When averaging `V` over frequencies (so called continuum
imaging) and time, the support of `V` may not be sparse.

<img src="data/images/uvw_coverage.png?raw=true" width="70%">

**Figure 1: uv-grid coverage**

The following considerations have not been studied in sufficient
detail, and leave room to modify algorithms currently in use:

1. **Data Layout:**  The visibility data at any given time are nonzero on
   a sparse subset of the u,v plane, with greatly varying density of
   samples when mapped to the `u,v,w` grid.  Figure 1 shows the
   distribution that can be expected: A lot of empty space, a densely
   filled (but computationally comparatively cheap) center region and
   multiple “arms” for longer baselines. Note that longer baselines
   have larger w-values due to the earth’s curvature.
2. **Binning:** Visibilities belonging to the same baseline snapshot cluster
   together closely, and can be seen on the left of Figure 1 as small
   lines. Creating appropriate “buckets” of visibility data exhibiting
   locality might help with caching.  The resulting function on the
   u,v grid may be sufficiently sparse to warrant a sparse encoding.
3. **Concurrency:**  Visibilities from different baselines may contribute
   to non-zero values of the gridded visibility at the same u,v grid
   points, and when they do so, the value at this gridpoint becomes
   data on which concurrency must be considered.  Handling this
   appears to be computationally costly.  However, it is likely that
   such concurrency may be avoided by clustering the computations -
   either by exploiting locality properties within the data
   (e.g. baseline, time and frequency) using the natural baseline
   structure of the data or using a checkerboard pattern of buckets.
4. **Sparseness:**  The results from the gridding operation at a
   particular time are nonzero only on a sparse subset of the u,v
   grid.  Appropriate data representation might lead to a smaller
   memory footprint.
5. **GCF locality:** Many different grid convolution functions will be
   used as the w-value, frequency channels, time and baseline
   vary. However, the usage scope of a kernel in the uv-grid is
   generally fairly local, which suggests that convolving them
   on-the-fly might be good idea. A model that relates the observed
   required memory capacity, cache misses associated with changing
   GCF, and recomputation costs to the values found in the parametric
   model is valuable.
6. **Hermitian Properties:** Every visibility has an associated
   conjugated visibility with negative u,v,w coordinates. For this
   reason visibility data only contains one of them, and the hermitian
   property is restored before or in the FFT step. This means that the
   implementation can choose whether to grid the original or the
   conjugated visibility, which can be used to increase or decrease
   locality as appropriate.

## Test Data

The visibility data reflects what a typical SKA1 Low
snapshot will look like from the point of view of a gridding
kernel. Visibility data and kernels will have the following
characteristics:

- 512 stations, therefore 130816 baselines
- Snapshot length 45 s, with 50-5 time steps (split depends on baseline)
- 7.5 MHz frequency range, with 300-20 channels (again, depends on baseline)
- just 1 polarisation

Furthermore, we assume the following gridding configuration:
- All kernels have size 15x15
- w-kernels are oversampled by a factor of 8
- A-kernel scopes are all 10s and 0.9MHz
- Theta (field of view dimension): 0.08 radians
- Lambda (uv-grid dimension): 300000 wavelengths
- Grid resolution: 0.08 * 300000 = 24000 uv-cells on each side

### Visibilies

* [data/vis/SKA1_Low_vis.h5](data/vis/SKA1_Low_vis.h5) [1.4GB, 43,930,418 visibilities]
* [data/vis/SKA1_Low_quick.h5](data/vis/SKA1_Low_quick.h5) [9MB, 130,816 visibilities]

Visibility data is packaged as an HDF5 file. The full data set
contains a group “`vis/[a1]/[a2]`” for every antenna pair (and therefore
baseline), whereas the quick dataset only contains one group of
visibilities. Each visibility group is laid out as follows:

- `frequency {nfreq}`: List of frequencies (`double`, Hz)
- `time {ntime}`: List of timeslots (`double`, UTC MJD)
- `uvw {ntime x 3}`: List of baseline coordinates (`double`, in metres)
- `vis {ntime x nfreq x npol}`: Visibility data (`complex double`)

Baseline coordinates in wavelengths can be calculated by multiplying
by frequency and dividing by the speed of light.

### W-kernels

* [data/kernels/SKA1_Low_wkern.h5](data/kernels/SKA1_Low_wkern.h5) [60MB, 268 w-planes, 1.5 lambda spacing]

W-kernels are similarly grouped in an HDF5 file. There will be groups
“wkern/[theta]/[w]” for kernels suitable for gridding a certain field
of view at a given w-value. Each group will contain the data set:

- `kern {nover x nover x nsupport x nsupport}`: Kernel data (`complex double`)

The w-kernel to apply to a visibility is the one with the closest
w-value present in the kernel set.

### A-kernels

* [data/kernels/SKA1_Low_akern.h5](data/kernels/SKA1_Low_akern.h5) [247MB, 512 antennas, 10s, 0.9 MHz]

For A-kernels we will have groups “`akern/[theta]/[a]/[t]/[f]`” for
A-kernels suitable for a certain antenna, time and frequency. Again
there will be just the kernel data set:

- `kern {nsupport x nsupport}`: Kernel data (`complex double`)

When in doubt, the kernel with the closest time and frequency should
be selected.

## Reference Code

The reference implementation is going to be the `crocodile`
repository. The goal is to implement a more efficient version of
gridding (`w_cache_imaging`) as invoked from the
[scripts/image_dataset.py](scripts/image_dataset.py) script (see below
for example). This includes all functions called from this function,
including the callbacks to cache Aw-kernel convolution and cache
functions. All of these should be considered relevant for performance
and should be benchmarked together appropriately.

Input data can be pre-processed as much as necessary to achieve an
efficient implementation, but the overhead of doing so will need to be
characterized.  Creating overlapping I/O and computation is expected
to be beneficial at least in some ranges of the parameters
values. Data can be reorganised into any layout. Furthermore,
arbitrary preparation steps are allowed on all data except the actual
visibility values.

The objective for the kernel is to produce a result that would be
equivalent to the grid output of the reference implementation after
applying `make_grid_hermitian`. This means that for every visibility,
the implementation can choose freely to either grid it unchanged or
conjugated (negate u, v and w coordinates and complex-conjugate the
visibility).

### Examples
Quickest - quick data set, simple imaging:
```bash
 scripts/image_dataset.py --theta 0.08 --lambda 300000
                          --grid out.grid --image out.img
                          quick.h5
```

Full data set, using only w-kernels:
```bash
 scripts/image_dataset.py --theta 0.08 --lambda 300000
                          --grid out.grid --image out.img
                          --wkern wkern.h5 vis.h5
```

Full data set with Aw-kernels:
```bash
 scripts/image_dataset.py --theta 0.08 --lambda 300000
                          --grid out.grid --image out.img
                          --wkern wkern.h5 --akern akern.h5
                          vis.h5
```

### Expected Results

The image result will look very similar irrespective of gridding
method, as visibilities have very low w-values, and for the purpose of
this test the A-kernels only adds noise.

<img src="data/images/wproject.img.quick.png" width="40%">
<img src="data/images/wproject.img.quick.detail.png" width="40%">

**Figure 2: Image result for w-projection of the “quick” visibility set**

<img src="data/images/wproject.img.png" width="40%">
<img src="data/images/wproject.img.detail.png" width="40%">

**Figure 3: Image result for w-projection of the full visibility set**

For both data sets a grid of points should be visible, with points 0.5
degrees (0.008726 radians, 2618 pixels) apart. Note that while the
points are more apparent in the overview of the “quick” dataset, the
signal is actually much sharper and stronger for the full data set, as
the right-side pictures show.

Reference data results can be obtained here:

* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/simple.quick.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/simple.quick.img.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/wproject.quick.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/wproject.quick.img.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/awproject.quick.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/awproject.quick.img.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/simple.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/simple.img.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/wproject.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/wproject.img.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/awproject.grid.gz
* http://www.mrao.cam.ac.uk/~pw410/crocodile/SKA1-LOW_v6_snapshot_vis_hi/awproject.img.gz

All grid and image files are 24000 x 24000 arrays of double-precision
complex numbers and double-precision real numbers respectively, in
row-major order.

## Profiling Toolchain

One of the most important tasks to be undertaken is the use and
possible development of a tool chain to understand exactly why
particular algorithms perform in a particular way.  For every kernel
developed we would like to understand:
- What % of system maxima are achieved for FLOPS/sec, bytes/sec
  (memory bandwidth), bytes/sec (IO bandwidth), cache re-use, energy
  consumption?  A direct comparison must be made with
  1. the parametric model, as it is updated to account for increased
     efficiency (e.g. using computed vs imported baselines,
     compression, precision, etc.) Note that some variations, e.g. the
     use of Fourier transforms for sparse data are not incorporated in
     the parametric model at present.
  2. The theoretical performance of the system
- What is limiting the performance?  Here we need detailed
  information, e.g. the dominating factor is data movement in stated
  instructions in the kernel from RAM to L3, it achieves xxx B/sec, %
  of max, or atomic operations on the following arguments dominate the
  computation time.
- What overhead in the observed computation is attributable to
  synchronization to handle concurrency?
- To what degree is the caching hierarchy exploited optimally?
- Document the dependency of performance when varying numerical
  parameters over a wide range and explain such
  dependencies. Numerical values that can be varied include:
  1. Grid sizes (note: kernel sizes are affected by grid sizes),
     performing the computations with the baselines with upper and
     lower bounds on length.
  2. Convolution function support sizes
  3. Threads started, cores utilized
  4. Frequency and number of channels processed
- Document and explain the dependency on non-numerical parameters, e.g.:
  1. Different sorting and grouping of visibility data
  2. Programming models employed (MPI, vs OpenMP, CUDA, others)
  3. Different data representations (e.g. float vs double, compressed,
     utilizing lookup tables)

## Deliverables
The following deliverables exist in connection with a prototype kernel:

 1. Files with performance data that was evaluated.
 2. All input sets, runtime and system configurations including code.
    Results that cannot be reproduced are not valuable.
 3. Transient observations and explanations, prepared for maximum
    re-usability, e.g.:
  - It was observed that communicating MPI processes (using the ABC
    MPI implementation version X.Y.Z) delivered better performance
    than communicating POSIX threads (using libc version U.V.W),
    because MPI communication has an optimization ….. not found in the
    thread implementation.
  - The memory bandwidth that was achieved was XX% of the maximum, due
    to ….
 4. Graphical representations of key findings underlying point 3.  If
    aspects of the parameter space are not explored, state this
    clearly and state why the results remain relevant.

We recommend that a shared SDP kernel performance study document must
be maintained summarizing important conclusions with references to
deliverables.

# FAQ

A collection of clarifications to the work description

## What is the role of gridding in the pipeline?

Gridding is currently projected to be one of the most expensive
operations. We estimate that at least ~2-4 Pflop/s of the SDP's
compute capacity will have to be spent on gridding. What makes
gridding especially interesting for the current pipeline planning is
that it will become more important the more the telescope is scaled
up, as naive scaling will lead to `O(n^4)` complexity in gridding and
kernels. There will obviously be other factors, but it looks like a
pretty safe bet that sooner or later gridding performance will become
very important.

Where does this much compute come from? To give a sense of scale, a
data set described above reflects:

 * 45 seconds (x480 for 6 hours)
 * 1 polarisation (x16 for all Mueller matrix terms)
 * 7.5 MHz (x40 for full Low band)
 * 1 facet (x25 for full FoV coverage to third null of antenna reception)
 * 1 taylor term (x5 for recovering frequency dependence)
 * 1 major loop (x10 for 10 self-calibration iterations)

So we need to process a data set like this roughly 384 million times
before a full observation is processed. The data sets will vary in
terms of visibilities and w/A-kernels, but the uv-grid distribution
will mostly stay the same.

## What about weighting and deconvolution?

You might see these pop up in the reference implementation, but it is
expected that those will not be relevant.

## Kernels switch fast. Do they have to be loaded from memory?

Visibility count per kernel depends on the baseline, and note that
generally baselines with many visibilities switch kernels more slowly.
However, it is correct that (A-)kernels get switched extremely
fast. After all, they depend not only on baseline, but also on time
and frequency, so we get at minimum ~40 kernel switches per baseline
even ignoring w-kernels.

It seems highly desirable for this reason to do convolution on-the-fly
at minimum. As argued in
[SDP Memo 028](data/docs/sdp-gridding-computational.pdf), this should
be enough to ensure that the kernel is not memory-limited on kernels
in most situations. There are also more advanced algorithms which
might have advantages in special situations, but this should be a good
starting point.

Generating w/A-kernels on-the-fly would be possible as well. However,
note that generation is generally not *that* expensive in the grand
scheme of things: It only needs to be done per individual
antenna/w-value, whereas convolution needs to be done for every
antenna-pair/w-value *combination*. For example, the A-kernel and
w-kernel sets are roughly 300 MB combined here, but all required
kernel combinations would be ~4TB!

## What is the purpose of the C gridding code in [examples/grid](examples/grid)?

This code was written as a test-run to see how hard it would be to
work with the HDF5 data set from a C-like environment. The data access
code has been tested and should be correct.

However, while the gridding portions has been used for benchmarks, it
has not been tested as thoroughly. We have verified that the results
are correct for simple and w-projection imaging, but for Aw-projection
the program is missing actual convolution. So these portions of the
program should be interpreted as illustrations, not as a reference.

## How can convolution be implemented using FFTs?

If you look at the definition of `aw_kernel_fn`, you will notice that
it is implemented as follows:

```python
    akern = scipy.signal.convolve2d(a1kern, a2kern, mode='same')
```

Where `concolve2d` implements a "naive" convolution
algorithm. However, using FFT laws we can replace this by Fourier
transformation. If we want to implement the original convolution
exactly this would boil down to:

```python
    awkern = 29*29*extract_mid(
        numpy.fft.fftshift(numpy.fft.fft2(
            numpy.fft.ifft2(numpy.fft.ifftshift(pad_mid(a1kern,29))) *
            numpy.fft.ifft2(numpy.fft.ifftshift(pad_mid(a2kern,29))) )),
        15)
```

Note that we need to pad the kernels to get the right border
behaviour. However, the kernels for Aw-gridding will normally be
chosen to /not/ overflow the borders (e.g. padding should have
happened when the kernels get generated). Therefore, the
computationally simpler "wrapping" approach is permitted as well:

```python
    awkern = 15*15*numpy.fft.fftshift(numpy.fft.fft2(
                 numpy.fft.ifft2(numpy.fft.ifftshift(a1kern)) *
                 numpy.fft.ifft2(numpy.fft.ifftshift(a2kern)) ))
```

Note that the inner FFTs of the input kernels are repeated quite
often, and should be shared as much as possible to reduce computation.
