
#include "grid.h"
#include "config.h"

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>

#ifndef NO_MPI
#include <mpi.h>
#else
#define MPI_Request int
#define MPI_REQUEST_NULL 0
#endif

struct streamer
{
    struct work_config *work_cfg;
    int subgrid_worker;
    int *producer_ranks;

    struct sep_kernel_data kern;
    bool have_kern;

    // Incoming data queue (to be assembled)
    int queue_length;
    double complex *nmbf_queue;
    MPI_Request *request_queue;
    bool *skip_receive; // duplicated receive slots to skip

    // Subgrid queue (to be degridded)
    double complex *subgrid_queue;
    int *subgrid_locks;
    fftw_plan subgrid_plan;

    // Visibility chunk queue (to be written)
    int vis_queue_length;
    double complex *vis_queue;
    int *vis_a1, *vis_a2, *vis_tchunk, *vis_fchunk;
    int vis_in_ptr, vis_out_ptr;
    omp_lock_t *vis_in_lock, *vis_out_lock; // Ready to fill / write to disk

    // Visibility file
    hid_t vis_file;
    hid_t vis_group;

    // Statistics
    double wait_time;
    double critical_wait_time;
    double wait_in_time, wait_out_time;
    double read_time, write_time;
    double recombine_time, degrid_time;
    uint64_t received_data, received_subgrids, baselines_covered;
    uint64_t written_vis_data, rewritten_vis_data;
};

static double complex *nmbf_slot(struct streamer *streamer, int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    const int xM_yN_size = streamer->work_cfg->recombine.xM_yN_size;
    assert(facet >= 0 && facet < facets);
    return streamer->nmbf_queue + xM_yN_size * xM_yN_size * ((slot * facets) + facet);
}

static MPI_Request *request_slot(struct streamer *streamer, int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    assert(facet >= 0 && facet < facets);
    return streamer->request_queue + (slot * facets) + facet;
}

static double complex *subgrid_slot(struct streamer *streamer, int slot)
{
    const int xM_size = streamer->work_cfg->recombine.xM_size;
    return streamer->subgrid_queue + xM_size * xM_size * slot;
}

void streamer_ireceive(struct streamer *streamer,
                       int subgrid_work)
{

    const int xM_yN_size = streamer->work_cfg->recombine.xM_yN_size;
    struct subgrid_work *work = streamer->work_cfg->subgrid_work +
        streamer->subgrid_worker * streamer->work_cfg->subgrid_max_work;

    // Not populated? skip
    if (!work[subgrid_work].nbl)
        return;

    // Skip? This is to prevent a subgrid getting received twice
    if (streamer->skip_receive[subgrid_work])
        return;
    int iw;
    for (iw = subgrid_work+1; iw < streamer->work_cfg->subgrid_max_work; iw++)
        if (work[iw].iu == work[subgrid_work].iu && work[iw].iv == work[subgrid_work].iv)
            streamer->skip_receive[iw] = true;

    // Identify slot
    const int slot = subgrid_work % streamer->queue_length;

    // Walk through all facets we expect contributions from
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    int facet;
    for (facet = 0; facet < facets; facet++) {
        struct facet_work *fwork = streamer->work_cfg->facet_work + facet;
        if (!fwork->set) {
            *request_slot(streamer, slot, facet) = MPI_REQUEST_NULL;
            continue;
        }

#ifndef NO_MPI
        // Set up a receive slot with appropriate tag
        const int tag = make_subgrid_tag(streamer->work_cfg, streamer->subgrid_worker, subgrid_work,
                                         facet / streamer->work_cfg->facet_max_work,
                                         facet % streamer->work_cfg->facet_max_work);
        int facet_worker = facet / streamer->work_cfg->facet_max_work;
        MPI_Irecv(nmbf_slot(streamer, slot, facet),
                  xM_yN_size * xM_yN_size, MPI_DOUBLE_COMPLEX,
                  streamer->producer_ranks[facet_worker], tag, MPI_COMM_WORLD,
                  request_slot(streamer, slot, facet));
#endif
    }

}

void streamer_writer(struct streamer *streamer)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    const int vis_data_size = sizeof(double complex) * spec->time_chunk * spec->freq_chunk;
    double complex *vis_data_h5 = (double complex *) alloca(vis_data_size);

    int time_chunk_count = spec->time_count / spec->time_chunk;
    int freq_chunk_count = spec->freq_count / spec->freq_chunk;
    int bl_count = spec->cfg->ant_count * spec->cfg->ant_count
        * time_chunk_count * freq_chunk_count;
    bool *bls_written = calloc(sizeof(bool), bl_count);

    for(;;) {

        // Obtain "out" lock for writing out visibilities
        double start = get_time_ns();
        omp_set_lock(streamer->vis_out_lock + streamer->vis_out_ptr);
        streamer->wait_out_time += get_time_ns() - start;

        start = get_time_ns();

        // Obtain baseline data
        struct bl_data bl_data;
        int tchunk = streamer->vis_tchunk[streamer->vis_out_ptr];
        int fchunk = streamer->vis_fchunk[streamer->vis_out_ptr];
        if (tchunk == -1 && fchunk == -1)
            break; // Signal to end thread
        vis_spec_to_bl_data(&bl_data, spec,
                            streamer->vis_a1[streamer->vis_out_ptr],
                            streamer->vis_a2[streamer->vis_out_ptr]);
        double complex *vis_data = streamer->vis_queue +
            streamer->vis_out_ptr * spec->time_chunk * spec->freq_chunk;

        // Read visibility chunk. If it was not yet set, this will
        // just fill the buffer with zeroes.
        int chunk_index = ((bl_data.antenna2 * spec->cfg->ant_count + bl_data.antenna1)
                           * time_chunk_count + tchunk) * freq_chunk_count + fchunk;
        if (bls_written[chunk_index]) {
            read_vis_chunk(streamer->vis_group, &bl_data,
                           spec->time_chunk, spec->freq_chunk, tchunk, fchunk,
                           vis_data_h5);
        } else {
            memset(vis_data_h5, 0, vis_data_size);
	}
        streamer->read_time += get_time_ns() - start;

        // Copy over data
        start = get_time_ns();
        int i;
        for (i = 0; i < spec->time_chunk * spec->freq_chunk; i++) {
            if (vis_data[i] != 0) {
                // Make sure we never over-write data!
                assert(vis_data_h5[i] == 0);
                vis_data_h5[i] = vis_data[i];
            }
        }

        // Write chunk back
        write_vis_chunk(streamer->vis_group, &bl_data,
                        spec->time_chunk, spec->freq_chunk, tchunk, fchunk,
                        vis_data_h5);
        streamer->written_vis_data += vis_data_size;
        if (bls_written[chunk_index])
            streamer->rewritten_vis_data += vis_data_size;
        bls_written[chunk_index] = true;

        free(bl_data.time); free(bl_data.uvw_m); free(bl_data.freq);

        // Release "in" lock to mark the slot free for writing
        omp_unset_lock(streamer->vis_in_lock + streamer->vis_out_ptr);
        streamer->vis_out_ptr = (streamer->vis_out_ptr + 1) % streamer->vis_queue_length;
        streamer->write_time += get_time_ns() - start;

    }

}

static double min(double a, double b) { return a > b ? b : a; }
static double max(double a, double b) { return a < b ? b : a; }

bool streamer_degrid_chunk(struct streamer *streamer,
                           struct subgrid_work *work,
                           struct subgrid_work_bl *bl,
                           struct bl_data *bl_data,
                           int tchunk, int fchunk,
                           int slot,
                           double complex *subgrid)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;
    const double theta = streamer->work_cfg->theta;

    double sg_mid_u = work->subgrid_off_u / theta;
    double sg_mid_v = work->subgrid_off_v / theta;
    double sg_min_u = (work->subgrid_off_u - cfg->xA_size / 2) / theta;
    double sg_min_v = (work->subgrid_off_v - cfg->xA_size / 2) / theta;
    double sg_max_u = (work->subgrid_off_u + cfg->xA_size / 2) / theta;
    double sg_max_v = (work->subgrid_off_v + cfg->xA_size / 2) / theta;
    if (sg_min_v > cfg->image_size / theta / 2) {
        sg_min_v -= cfg->image_size / theta / 2;
        sg_max_v -= cfg->image_size / theta / 2;
    }

    // Determine chunk size
    int it0 = tchunk * spec->time_chunk,
        it1 = (tchunk+1) * spec->time_chunk;
    if (it1 > spec->time_count) it1 = spec->time_count;
    int if0 = fchunk * spec->freq_chunk,
        if1 = (fchunk+1) * spec->freq_chunk;
    if (if1 > spec->freq_count) if1 = spec->freq_count;

    // Check whether we actually have overlap with the subgrid (TODO:
    // again slightly naive, there's a chance we lose some
    // visibilities here)
    double f0 = uvw_m_to_l(1, bl_data->freq[if0]);
    double f1 = uvw_m_to_l(1, bl_data->freq[if1-1]);
    double *uvw0 = bl_data->uvw_m + 3*it0;
    double *uvw1 = bl_data->uvw_m + 3*(it1-1);
    double min_u = min(min(uvw0[0]*f0, uvw0[0]*f1), min(uvw1[0]*f0, uvw1[0]*f1));
    double min_v = min(min(uvw0[1]*f0, uvw0[1]*f1), min(uvw1[1]*f0, uvw1[1]*f1));
    double max_u = max(max(uvw0[0]*f0, uvw0[0]*f1), max(uvw1[0]*f0, uvw1[0]*f1));
    double max_v = max(max(uvw0[1]*f0, uvw0[1]*f1), max(uvw1[1]*f0, uvw1[1]*f1));

    // Check for overlap between baseline chunk and subgrid
    bool overlap = min_u < sg_max_u && max_u > sg_min_u &&
                   min_v < sg_max_v && max_v > sg_min_v;
    bool inv_overlap = -max_u < sg_max_u && -min_u > sg_min_u &&
                       -max_v < sg_max_v && -min_v > sg_min_v;
    if (!overlap && !inv_overlap)
        return false;

    // Allocate visibility chunk (on stack)
    const int vis_data_size = sizeof(double complex) * spec->time_chunk * spec->freq_chunk;
    double complex *vis_data = (double complex *) alloca(vis_data_size);
    memset(vis_data, 0, vis_data_size);

    // Do degridding. Twice if necessary.
    double start = get_time_ns();
    uint64_t flops = 0;

    if (overlap) {
        bl_data->vis = vis_data; // HACK
        flops += degrid_conv_bl(subgrid, cfg->xM_size, theta,
                                sg_mid_u, sg_mid_v,
                                max(0, sg_min_u), sg_max_u, sg_min_v, sg_max_v,
                                bl_data, it0, it1, if0, if1, &streamer->kern);
    }
    if (inv_overlap) {
        int i;
        // Bit of a hack. Also TODO: complex-conjugate visibilities
        for(i = 0; i < 3 * spec->time_count; i++) {
            bl_data->uvw_m[i] *= -1;
        }
        bl_data->vis = vis_data; // HACK
        flops += degrid_conv_bl(subgrid, cfg->xM_size, theta,
                                sg_mid_u, sg_mid_v,
                                max(0, sg_min_u), sg_max_u, sg_min_v, sg_max_v,
                                bl_data, it0, it1, if0, if1, &streamer->kern);
        for(i = 0; i < 3 * spec->time_count; i++) {
            bl_data->uvw_m[i] *= -1;
        }
    }

    #pragma omp atomic
    streamer->degrid_time += get_time_ns() - start;

    // No flops executed? Skip doing I/O
    if (flops == 0) {
        return true;
    }

    // Determine our slot (competing with other tasks, so need to have
    // critical section here)
    int vis_slot;
    start = get_time_ns();
    #pragma omp critical
    {
        streamer->critical_wait_time += get_time_ns() - start;
        vis_slot = streamer->vis_in_ptr;
        streamer->vis_in_ptr = (streamer->vis_in_ptr + 1) % streamer->vis_queue_length;
    }

    // Obtain lock for writing data (writer thread might not have
    // written this data to disk yet)
    start = get_time_ns();
    if(streamer->vis_group >= 0)
        omp_set_lock(streamer->vis_in_lock + vis_slot);
    #pragma omp atomic
    streamer->wait_in_time += get_time_ns() - start;

    // Set slot data
    double complex *vis_data_out = streamer->vis_queue +
        vis_slot * spec->time_chunk * spec->freq_chunk;
    memcpy(vis_data_out, vis_data, vis_data_size);
    streamer->vis_a1[vis_slot] = bl->a1;
    streamer->vis_a2[vis_slot] = bl->a2;
    streamer->vis_tchunk[vis_slot] = tchunk;
    streamer->vis_fchunk[vis_slot] = fchunk;

    // Signal slot for output
    if(streamer->vis_group >= 0)
        omp_unset_lock(streamer->vis_out_lock + vis_slot);

    return true;
}

void streamer_task(struct streamer *streamer,
                   struct subgrid_work *work,
                   struct subgrid_work_bl *bl,
                   int slot,
                   int subgrid_work,
                   double complex *subgrid)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    struct subgrid_work_bl *bl2;
    int i_bl2;
    for (bl2 = bl, i_bl2 = 0;
         bl2 && i_bl2 < streamer->work_cfg->vis_bls_per_task;
         bl2 = bl2->next, i_bl2++) {

        // Get baseline data
        struct bl_data bl_data;
        vis_spec_to_bl_data(&bl_data, spec, bl2->a1, bl2->a2);

        // Go through time/frequency chunks
        int ntchunk = (bl_data.time_count + spec->time_chunk - 1) / spec->time_chunk;
        int nfchunk = (bl_data.freq_count + spec->freq_chunk - 1) / spec->freq_chunk;
        int tchunk, fchunk;
	int nchunks = 0;
        for (tchunk = 0; tchunk < ntchunk; tchunk++)
            for (fchunk = 0; fchunk < nfchunk; fchunk++)
	      if (streamer_degrid_chunk(streamer, work,
					bl2, &bl_data, tchunk, fchunk,
					slot, subgrid))
		  nchunks++;

	// Check that plan predicted the right number of chunks. This
	// is pretty important - if this fails this means that the
	// coordinate calculations are out of synch, which might mean
	// that we have failed to account for some visibilities in the
	// plan!
	if (bl2->chunks != nchunks)
	  printf("WARNING: subgrid (%d/%d) baseline (%d-%d) %d chunks planned, %d actual!\n",
		 work->iu, work->iv, bl2->a1, bl2->a2, bl2->chunks, nchunks);

        free(bl_data.time); free(bl_data.uvw_m); free(bl_data.freq);
    }

    // Done with this chunk
#pragma omp atomic
    streamer->subgrid_locks[slot]--;

}

void streamer_work(struct streamer *streamer,
                   int subgrid_work,
                   double complex *nmbf)
{

    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;
    struct subgrid_work *const work = streamer->work_cfg->subgrid_work +
        streamer->subgrid_worker * streamer->work_cfg->subgrid_max_work + subgrid_work;
    struct facet_work *const facet_work = streamer->work_cfg->facet_work;

    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    const int nmbf_length = cfg->NMBF_NMBF_size / sizeof(double complex);

    // Identify slot to write to
    int slot = subgrid_work % streamer->queue_length;
    double start_wait = get_time_ns();
    while(streamer->subgrid_locks[slot] != 0) {
        #pragma omp taskyield
        if (get_time_ns() > start_wait + 1) {
	    //printf("Waiting on slot %d (%d)...\n", slot, streamer->subgrid_locks[slot]);
            fflush(stdout);
            start_wait = get_time_ns();
        }
    }

    double recombine_start = get_time_ns();

    // Compare with reference
    if (work->check_fct_path) {

        int i0 = work->iv, i1 = work->iu;
        int ifacet;
        for (ifacet = 0; ifacet < facets; ifacet++) {
            if (!facet_work[ifacet].set) continue;
            int j0 = facet_work[ifacet].im, j1 = facet_work[ifacet].il;
            double complex *ref = read_hdf5(cfg->NMBF_NMBF_size, work->check_hdf5,
                                            work->check_fct_path, j0, j1);
            int x; double err_sum = 0;
            for (x = 0; x < nmbf_length; x++) {
                double err = cabs(ref[x] - nmbf[nmbf_length*ifacet+x]); err_sum += err*err;
            }
            free(ref);
            double rmse = sqrt(err_sum / nmbf_length);
            if (!work->check_fct_threshold || rmse > work->check_fct_threshold) {
                printf("Subgrid %d/%d facet %d/%d checked: %g RMSE\n",
                       i0, i1, j0, j1, rmse);
            }
        }
    }

    // Accumulate contributions to this subgrid
    double complex *subgrid = subgrid_slot(streamer, slot);
    memset(subgrid, 0, cfg->SG_size);
    int ifacet;
    for (ifacet = 0; ifacet < facets; ifacet++)
        recombine2d_af0_af1(cfg, subgrid,
                            facet_work[ifacet].facet_off_m,
                            facet_work[ifacet].facet_off_l,
                            nmbf + nmbf_length*ifacet);

    // Perform Fourier transform
    fftw_execute_dft(streamer->subgrid_plan, subgrid, subgrid);

    // Check accumulated result
    if (work->check_path) {
        double complex *approx_ref = read_hdf5(cfg->SG_size, work->check_hdf5, work->check_path);
        double err_sum = 0; int y;
        for (y = 0; y < cfg->xM_size * cfg->xM_size; y++) {
            double err = cabs(subgrid[y] - approx_ref[y]); err_sum += err * err;
        }
        free(approx_ref);
        double rmse = sqrt(err_sum / cfg->xM_size / cfg->xM_size);
        printf("%sSubgrid %d/%d RMSE %g\n", rmse > work->check_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    // Check some degridded example visibilities
    if (work->check_degrid_path && streamer->have_kern) {
        int nvis = get_npoints_hdf5(work->check_hdf5, "%s/vis", work->check_degrid_path);
        double *uvw_sg = read_hdf5(3 * sizeof(double) * nvis, work->check_hdf5,
                                   "%s/uvw_subgrid", work->check_degrid_path);
        int vis_size = sizeof(double complex) * nvis;
        double complex *vis = read_hdf5(vis_size, work->check_hdf5,
                                        "%s/vis", work->check_degrid_path);

        struct bl_data bl;
        bl.antenna1 = bl.antenna2 = 0;
        bl.time_count = nvis;
        bl.freq_count = 1;
        double freq[] = { c }; // 1 m wavelength
        bl.freq = freq;
        bl.vis = (double complex *)calloc(1, vis_size);

        // Degrid and compare
        fft_shift(subgrid, cfg->xM_size);
        bl.uvw_m = uvw_sg;
        degrid_conv_bl(subgrid, cfg->xM_size, cfg->image_size, 0, 0,
                       -cfg->xM_size, cfg->xM_size, -cfg->xM_size, cfg->xM_size,
                       &bl, 0, nvis, 0, 1, &streamer->kern);
        double err_sum = 0; int y;
        for (y = 0; y < nvis; y++) {
            double err = cabs(vis[y] - bl.vis[y]); err_sum += err*err;
        }
        double rmse = sqrt(err_sum / nvis);
        printf("%sSubgrid %d/%d degrid RMSE %g\n",
               rmse > work->check_degrid_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    streamer->recombine_time += get_time_ns() - recombine_start;

    struct vis_spec *const spec = &streamer->work_cfg->spec;
    if (spec->time_count > 0 && streamer->have_kern) {

        // Loop through baselines
        struct subgrid_work_bl *bl;
        int i_bl = 0;
        for (bl = work->bls; bl; bl = bl->next, i_bl++) {
            if (i_bl % streamer->work_cfg->vis_bls_per_task != 0)
                continue;

            // We are spawning a task: Add lock to subgrid data to
            // make sure it doesn't get overwritten
            #pragma omp atomic
            streamer->subgrid_locks[slot]++;

            // Start task. Make absolutely sure it sees *everything*
            // as private, as Intel's C compiler otherwise loves to
            // generate segfaulting code. OpenMP complains here that
            // having a "private" constant is unecessary (requiring
            // the copy), but I don't trust its judgement.
            struct subgrid_work *_work = work;
            #pragma omp task firstprivate(streamer, _work, bl, slot, subgrid_work, subgrid)
                streamer_task(streamer, _work, bl, slot, subgrid_work, subgrid);

        }

        printf("Subgrid %d/%d (%d baselines)\n", work->iu, work->iv, i_bl);
        fflush(stdout);
        streamer->baselines_covered += i_bl;

    }
}

bool streamer_init(struct streamer *streamer,
                   struct work_config *wcfg, int subgrid_worker, int *producer_ranks)
{

    struct recombine2d_config *cfg = &wcfg->recombine;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;

    streamer->work_cfg = wcfg;
    streamer->subgrid_worker = subgrid_worker;
    streamer->producer_ranks = producer_ranks;

    streamer->have_kern = false;
    streamer->vis_file = streamer->vis_group = -1;
    streamer->wait_time = streamer->wait_in_time = streamer->wait_out_time =
        streamer->critical_wait_time = streamer->recombine_time = 0;
    streamer->read_time = streamer->write_time = streamer->degrid_time = 0;
    streamer->received_data = 0;
    streamer->received_subgrids = streamer->baselines_covered = 0;
    streamer->written_vis_data = streamer->rewritten_vis_data = 0;

    // Load gridding kernel
    if (wcfg->gridder_path) {
        if (load_sep_kern(wcfg->gridder_path, &streamer->kern))
            return false;
        streamer->have_kern = true;
    }

    // Create HDF5 output file if we are meant to output any amount of
    // visibilities
    if (wcfg->vis_path) {

        // Get filename to use
        char filename[512];
        sprintf(filename, wcfg->vis_path, subgrid_worker);

        // Open file and "vis" group
        printf("\nCreating %s... ", filename);
        streamer->vis_file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        streamer->vis_group = H5Gcreate(streamer->vis_file, "vis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (streamer->vis_file < 0 || streamer->vis_group < 0) {
            fprintf(stderr, "Could not open visibility file %s!\n", filename);
        } else {
            create_bl_groups(streamer->vis_group, wcfg, subgrid_worker);
        }

        H5Fflush(streamer->vis_file, H5F_SCOPE_LOCAL);
    }

    // Calculate size of queues
    streamer->queue_length = wcfg->vis_subgrid_queue_length;
    streamer->vis_queue_length = wcfg->vis_chunk_queue_length;
    const int nmbf_length = cfg->NMBF_NMBF_size / sizeof(double complex);
    const size_t queue_size = (size_t)sizeof(double complex) * nmbf_length * facets * streamer->queue_length;
    const size_t sg_queue_size = (size_t)cfg->SG_size * streamer->queue_length;
    const size_t requests_size = (size_t)sizeof(MPI_Request) * facets * streamer->queue_length;
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    const int vis_data_size = sizeof(double complex) * spec->time_chunk * spec->freq_chunk;
    printf("Allocating %.3g GB subgrid queue, %.3g GB visibility queue\n",
           (double)(queue_size+sg_queue_size+requests_size) / 1e9,
           (double)((size_t)streamer->vis_queue_length * (vis_data_size + 6 * sizeof(int))) / 1e9);

    // Allocate receive queue
    streamer->nmbf_queue = (double complex *)malloc(queue_size);
    streamer->request_queue = (MPI_Request *)malloc(requests_size);
    streamer->subgrid_queue = (double complex *)malloc(sg_queue_size);
    streamer->subgrid_locks = (int *)calloc(sizeof(int), streamer->queue_length);
    if (!streamer->nmbf_queue || !streamer->request_queue ||
        !streamer->subgrid_queue || !streamer->subgrid_locks) {

        fprintf(stderr, "ERROR: Could not allocate subgrid queue!\n");
        return false;
    }
    streamer->skip_receive = (bool *)calloc(sizeof(bool), wcfg->subgrid_max_work);

    // Plan FFTs
    streamer->subgrid_plan = fftw_plan_dft_2d(cfg->xM_size, cfg->xM_size,
                                              streamer->subgrid_queue,
                                              streamer->subgrid_queue,
                                              FFTW_BACKWARD, FFTW_MEASURE);

    // Populate receive queue
    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_max_work && iwork < streamer->queue_length; iwork++) {
        streamer_ireceive(streamer, iwork);
    }

    // Allocate visibility queue
    streamer->vis_queue = malloc((size_t)streamer->vis_queue_length * vis_data_size);
    streamer->vis_a1 = malloc((size_t)streamer->vis_queue_length * sizeof(int));
    streamer->vis_a2 = malloc((size_t)streamer->vis_queue_length * sizeof(int));
    streamer->vis_tchunk = malloc((size_t)streamer->vis_queue_length * sizeof(int));
    streamer->vis_fchunk = malloc((size_t)streamer->vis_queue_length * sizeof(int));
    streamer->vis_in_lock = malloc((size_t)streamer->vis_queue_length * sizeof(omp_lock_t));
    streamer->vis_out_lock = malloc((size_t)streamer->vis_queue_length * sizeof(omp_lock_t));
    streamer->vis_in_ptr = streamer->vis_out_ptr = 0;
    if (!streamer->vis_queue || !streamer->vis_a1 || !streamer->vis_a2 ||
        !streamer->vis_tchunk || !streamer->vis_fchunk ||
        !streamer->vis_in_lock || !streamer->vis_out_lock) {

        fprintf(stderr, "ERROR: Could not allocate visibility queue!\n");
        return false;
    }

    int i;
    for (i = 0; i < streamer->vis_queue_length; i++) {
        omp_init_lock(streamer->vis_in_lock + i);
        omp_init_lock(streamer->vis_out_lock + i);
        omp_set_lock(streamer->vis_out_lock + i);
    }

    return true;
}

void streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks)
{

    struct streamer streamer;
    if (!streamer_init(&streamer, wcfg, subgrid_worker, producer_ranks)) {
        return;
    }

    struct subgrid_work *work = wcfg->subgrid_work + subgrid_worker * wcfg->subgrid_max_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;
    const int nmbf_length = wcfg->recombine.NMBF_NMBF_size / sizeof(double complex);

#ifndef NO_MPI
    MPI_Status *status_queue = (MPI_Status *)
        malloc(sizeof(MPI_Status) * facets * streamer.queue_length);
#endif

    printf("Waiting for data...\n");
    double stream_start = get_time_ns();

    // Start doing work
    int num_workers = 0;
#pragma omp parallel sections
    {
#pragma omp section
    if(streamer.vis_group >= 0)
        streamer_writer(&streamer);

#pragma omp section
    {
    // Determine number of workers. If we are streaming to disk (see
    // above) this will tie up one thread from the pool permanently.
    num_workers = omp_get_num_threads();
    if(streamer.vis_group >= 0)
        num_workers--;

    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_max_work; iwork++) {
        int work_slot = iwork % streamer.queue_length;
        double complex *data_slot = streamer.nmbf_queue + work_slot * facets * nmbf_length;

        if (work[iwork].nbl && !streamer.skip_receive[iwork]) {

            double start = get_time_ns();
#ifndef NO_MPI
            // Wait for all facet data for this work to arrive (TODO:
            // start doing some work earlier)
            MPI_Waitall(facets, streamer.request_queue + facets * work_slot,
                        status_queue + facets * work_slot);
#endif
	    int i;
	    for (i = 0; i < facets; i++) {
	      streamer.request_queue[facets * work_slot + i] = MPI_REQUEST_NULL;
	    }
            streamer.received_data += sizeof(double complex) * facets * nmbf_length;
            streamer.received_subgrids++;
            streamer.wait_time += get_time_ns() - start;
        }

        // Do work on received data
        streamer_work(&streamer, iwork, data_slot);

        // Set up slot for new data (if appropriate)
        int iwork_r = iwork + streamer.queue_length;
        if (iwork_r < wcfg->subgrid_max_work) {
            streamer_ireceive(&streamer, iwork_r);
        }

    }

    #pragma omp taskwait

    // Signal writer task to exit
    omp_set_lock(streamer.vis_in_lock + streamer.vis_in_ptr);
    streamer.vis_tchunk[streamer.vis_in_ptr] = -1;
    streamer.vis_fchunk[streamer.vis_in_ptr] = -1;
    omp_unset_lock(streamer.vis_out_lock + streamer.vis_in_ptr);

    }
    }

    free(streamer.nmbf_queue); free(streamer.subgrid_queue);
    free(streamer.request_queue); free(streamer.subgrid_locks);
    free(streamer.skip_receive);
    fftw_free(streamer.subgrid_plan);
#ifndef NO_MPI
    free(status_queue);
#endif
    free(streamer.vis_queue);
    free(streamer.vis_a1); free(streamer.vis_a2);
    free(streamer.vis_tchunk); free(streamer.vis_fchunk);
    free(streamer.vis_in_lock); free(streamer.vis_out_lock);

    if (streamer.vis_group >= 0) {
        H5Gclose(streamer.vis_group); H5Fclose(streamer.vis_file);
    }

    double stream_time = get_time_ns() - stream_start;
    printf("Streamed for %.2fs\n", stream_time);
    printf("Received %.2f GB (%ld subgrids, %ld baselines)\n",
           (double)streamer.received_data / 1000000000, streamer.received_subgrids,
           streamer.baselines_covered);
    printf("Written %.2f GB (rewritten %.2f GB), rate %.2f GB/s (%.2f GB/s effective)\n",
           (double)streamer.written_vis_data / 1000000000,
           (double)streamer.rewritten_vis_data / 1000000000,
           (double)streamer.written_vis_data / 1000000000 / stream_time,
           (double)(streamer.written_vis_data - streamer.rewritten_vis_data)
           / 1000000000 / stream_time);
    printf("Receiver: Wait: %gs, Recombine: %gs, Idle: %gs\n",
           streamer.wait_time, streamer.recombine_time,
           stream_time - streamer.wait_time - streamer.recombine_time);
    printf("Worker: Wait: %gs, Critical: %gs, Degrid: %gs, Idle: %gs\n",
           streamer.wait_in_time, streamer.critical_wait_time,
           streamer.degrid_time,
           num_workers * stream_time
           - streamer.wait_in_time - streamer.critical_wait_time - streamer.degrid_time
           - streamer.wait_time - streamer.recombine_time);
    printf("Writer: Wait: %gs, Read: %gs, Write: %gs, Idle: %gs\n",
           streamer.wait_out_time, streamer.read_time, streamer.write_time,
           stream_time - streamer.wait_out_time - streamer.read_time - streamer.write_time);

}
