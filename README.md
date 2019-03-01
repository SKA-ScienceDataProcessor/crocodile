
SDP Distributed Predict/Imaging I/O Prototype
=============================================

This is a prototype exploring the capability of hardware and software
to deal with the types of I/O loads that the SDP will have to support
for full-scale operation on SKA1 (and beyond).

Goals:
------

* Focused benchmarking of platform (especially buffer and network) hardware and software
* Verify parametric model assumptions concerning distributed
  performance and scaling, especially the extended analysis concerning
  I/O and memory from SDP memo 038 (pipeline-working-sets.pdf)

Main aspects:

* Distribution by facets/subgrids - involves all-to-all loosely
  synchronised communication and phase rotation work
* (De)gridding - main work that needs to be distributed. Will involve
  heavy computational work to deal with calibration and
  non-coplanarity, main motivation for requiring distribution
* Buffer I/O - needs to deal with high throughput, possibly not
  entirely sequential access patterns due to telescope layout

Technology choices:

* Written in plain C to minimise language environment as possible
  bottleneck
* Use MPI for communication - same reason
* OpenMP for parallelism - a bit limiting in terms of thread control,
  but seems good enough for a first pass
* HDF5 for data storage - seems flexible enough for a start. Might
  port to other storage back-ends at some point

Algorithmic choices:

* We consider the continuum case, but with only one polarisation,
  taylor term, snapshot (= no reprojection). These would add code
  complexity, but are likely easy to scale up.
* We start with prediction (= writing visibilities). This is clearly
  the wrong way around, the backwards step will be added in the next
  version.
* Use slightly experimental gridded phase rotation ("recombination"),
  allowing us to skip baseline-dependent averaging while still keeping
  network transfer low and separating benchmark stages more cleanly.

Design
------

The benchmark has two kinds of processes, called "producers" and
"streamers" (a third will be added once we get to the backwards
step). By default MPI ranks are evenly divided up between the two
roles. As the two types of processes are very different, it makes
sense to put them on the same node (see below for SLURM
configuration).

### Work Balance

Facet and subgrid work is assigned to producer and streamer processes
at the start of the run. Facets are large memory objects, and the same
amount of work needs to be done on each of them, therefore the number
of nodes should be chosen so we can distribute them evenly - ideally
the number of nodes should be the same as the number of facets.

On the other hand, each subgrid will be used to de-grid a different
number of baselines and therefore visibilities depending on the grid
area covered. This has to be balanced, keeping in mind that we want a
healthy mix of subgrids to visibility chunks so the streamer doesn't
run into a bottleneck on the network side of things. Right now we use
a simple round-robin scheduling, splitting the work in central
subgrids among nodes past a certain threshold.  Parallelism

Both producer and streamer scale to many cores. The facet side of
recombination / phase rotation is essentially a distributed FFT that
we do separately on two axes, which leads to ample parallelism
opportunities. However in order to keep memory residency manageable it
is best to work on subgrid columns sequentially. Subgrid work is
scheduled in rough column order to accomodate this.

The streamer employs three types of threads: One to receive data from
the network and sum up subgrids, one to write visibilities as HDF5 (we
serialise this for simplicity), and the rest to degrid baseline
visibilities from subgrids. The network thread generally has very
little work and will spawn a lot of tasks very quickly, which means
that OpenMP will often have it yield to worker threads, effectively
making it a worker thread.

### Queues

In order to get good throughput every stage has input and output
queues. We employ slightly different mechanisms depending on stage:

* The producer has only limited MPI slots per thread to send out
  subgrid contributions (current default: 8 subgrids worth)
* On the other end, the streamer has a limited number of MPI slots to
  receive facet contributions (current default: 32 subgrids worth)
* The network thread will assemble sub-grids once all contributions
  have been received, and create OpenMP tasks for degridding. The
  subgrids in question will be locked until all de-grid tasks have
  been finished, the queue is limited to the same number of entries as
  the incoming facet contribution queue (so 32 entries).
* OpenMP limits the number of degrid tasks that can be spawned, which
  means that we additionally have a degrid task queue with limited
  capacity (Seems to be around 128 for gcc). Note that a task can
  cover many baselines (current default is up to 256 - so roughly
  32768 baselines maximum).
* Finally, once visibilities have been generated, those will have to
  be written out. This is done in terms of visibility chunks (size
  configurable, the - low - default is currently 32x32). The queue has
  a length of 32768 entries (roughly half a GB worth of data with
  default parameters).

Usage
-----

To set up the repository, get data files and compile, run the
following steps. You will need Git LFS (https://git-lfs.github.com/),
HDF5 (doesn't need to have parallel support), MPI (with threading
support) and FFTW installed. It will use `mpicc` for building, and has
been tested with both GCC and ICC.

```
 $ clone https://github.com/SKA-ScienceDataProcessor/io_test.git
 $ cd io_test/src
 $ git lfs pull origin
 $ make iotest
```

### Single Node

The program can be configured using the command line. Useful
configurations:

```
  $ mpirun -n 2 ./iotest --rec-set=T05
```

Runs a small test-case with two local processes to ensure that
everything works correctly. It should show RMSE values of about 1e-8
for all recombined subgrids and 1e-7 for visibilities.

```
  $ ./iotest --rec-set=small
  $ ./iotest --rec-set=large
```

Does a dry run of the "producer" end in stand-alone mode. Primarily
useful to check whether enough memory is available. The former will
use about 10 GB memory, the latter about 350 GB.

```
  $ ./iotest --rec-set=small --vis-set=lowbd2 --facet-workers=0
```

Does a dry run of the "degrid" part in stand-alone mode. Good to check
stability and ensure that we can degrid fast enough.

```
  $ ./iotest --rec-set=small --vis-set=lowbd2 --facet-workers=0 /tmp/out.h5
```

Same as above, but actually writes data to the out the given
file. Data will be all zeroes, but this runs through the entire
back-end without involving actual distribution. Typically quite a bit
slower, as writing out data is generally the bottleneck.

```
  $ mpirun -n 2 ./iotest --rec-set=small --vis-set=lowbd2 /tmp/out.h5
```

Runs the entire thing with one producer and one streamer, this time producing actual-ish visibility data (for a random facet without grid correction).

```
  $ mpirun -n 2 ./iotest --rec-set=small --vis-set=lowbd2 --time=-230:230/512/128 --freq=225e6:300e6/8192/128 /tmp/out.h5
```

The "vis-set" and "rev-set" parameters are just default parameter sets
that can be overridden. The command line above increases time and
frequency sampling to the point where it would roughly correspond to
an SKA Low snapshot (7 minutes, 25% frequency range). The time and
frequency specification is `<start>/<end>/<steps>/<chunk>`, so in this
case 512 time steps with chunk size 128 and 8192 frequency channels
with chunks size 128. This will write roughly 9 TB of data with a
chunk granularity of 256 KB.

### Distributed

As explained the benchmark can also be run across a number of
nodes. This will distribute both the facet working set as well as the
visibility write rate pretty evenly across nodes. As noted you might
want at minimum a producer and a streamer process per node, and
configure OpenMP such that its threads take full advantage of the
machine's available cores.

For example, to run 8 producer processes (facet workers) and 32
streamer processes (subgrid workers) across 8 machines, using 8
threads each:

```
export OMP_NUM_THREADS=8
mpirun --map-by node -np 40 ./iotest --facet-workers=8 $options
```

This would allocate 4 streamer processes per node with 8 threads each,
appropriate for a node with 32 (physical) cores available. Facet
workers are typically heavily memory bound and do not interfere too
much with co-existing processes outside of reserving large amounts of
memory.

Possible options for distributed mode:

```
  options="--rec-set=large"
```

This will just do a full re-distribution of facet/subgrid data between
all nodes. This serves as a network I/O test. Note that because we are
operating the benchmark without a telescope configuration, the entire
grid is going to get transferred - not just the pieces of it that have
baselines.

```
  options="--rec-set=large --vis-set=lowbd2"
```

Will only do re-distribute data that overlaps with baselines (which is
more efficient), then do degridding.

```
  options="--rec-set=large --vis-set=lowbd2 /local/out%d.h5"
```

Also write out visibilities to the given file. Note that the benchmark
does not currently implement parallel HDF5, so different streamer
processes will have to write separate output files. The name can be
made dependent on streamer ID by putting a `%d` placeholder into it so
it won't cause conflicts on shared file systems.
