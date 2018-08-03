
#include "grid.h"
#include "config.h"

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

void streamer_ireceive(struct work_config *wcfg,
                       double complex *NMBF_NMBF_queue, MPI_Request *requests_queue,
                       int subgrid_worker, int subgrid_work,
                       int *producer_ranks)
{

    // Walk through all facets we expect contributions from
    int facet;
    for (facet = 0; facet < wcfg->facet_workers * wcfg->facet_max_work; facet++) {
        struct facet_work *fwork = wcfg->facet_work + facet;
        if (!fwork->set) {
            requests_queue[facet] = MPI_REQUEST_NULL;
            continue;
        }

        // Set up a receive slot with appropriate tag
        const int tag = make_subgrid_tag(wcfg, subgrid_worker, subgrid_work,
                                         facet / wcfg->facet_max_work, facet % wcfg->facet_max_work);
        const int elems = wcfg->recombine.xM_yN_size * wcfg->recombine.xM_yN_size;
        int facet_worker = facet / wcfg->facet_max_work;
        MPI_Irecv(NMBF_NMBF_queue + facet * elems, elems, MPI_DOUBLE_COMPLEX,
                  producer_ranks[facet_worker], tag, MPI_COMM_WORLD,
                  requests_queue + facet);
    }

}

static double min(double a, double b) { return a > b ? b : a; }
static double max(double a, double b) { return a < b ? b : a; }

void streamer_degrid_chunk(struct work_config *wcfg,
                           struct sep_kernel_data *kern,
                           hid_t vis_group,
                           struct subgrid_work *work,
                           struct subgrid_work_bl *bl,
                           struct bl_data *bl_data,
                           int tchunk, int fchunk,
                           double complex *subgrid)
{
    struct vis_spec *spec = &wcfg->spec;
    struct recombine2d_config *cfg = &wcfg->recombine;

    double sg_mid_u = work->subgrid_off_u / wcfg->theta;
    double sg_mid_v = work->subgrid_off_v / wcfg->theta;
    double sg_min_u = (work->subgrid_off_u - cfg->xA_size / 2) / wcfg->theta;
    double sg_min_v = (work->subgrid_off_v - cfg->xA_size / 2) / wcfg->theta;
    double sg_max_u = (work->subgrid_off_u + cfg->xA_size / 2) / wcfg->theta;
    double sg_max_v = (work->subgrid_off_v + cfg->xA_size / 2) / wcfg->theta;
    if (sg_min_v > cfg->image_size / wcfg->theta / 2) {
        sg_min_v -= cfg->image_size / wcfg->theta / 2;
        sg_max_v -= cfg->image_size / wcfg->theta / 2;
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
        return;

    printf("baseline %d/%d chunk %d/%d\n", bl->a1, bl->a2, tchunk, fchunk);
    printf("u: %g-%g (%g-%g)\n", min_u, max_u, sg_min_u, sg_max_u);
    printf("v: %g-%g (%g-%g)\n", min_v, max_v, sg_min_v, sg_max_v);

    // Only now we commit to doing I/O - read visibility chunk. If it
    // was not yet set, this will just fill the buffer with zeroes.
    double complex *vis_data = (double complex *)
        alloca(sizeof(double complex) * spec->time_chunk * spec->freq_chunk);

    read_vis_chunk(vis_group, bl_data,
                   spec->time_chunk, spec->freq_chunk, tchunk, fchunk,
                   vis_data);

    //printf("read visibilities:\n");
    //int i_t, i_f;
    //for(i_t = 0; i_t < spec->time_chunk; i_t++) {
    //    for(i_f = 0; i_f < spec->freq_chunk; i_f++) {
    //        printf("%g%+gi ",
    //               creal(vis_data[i_t*spec->freq_chunk+i_f]),
    //               cimag(vis_data[i_t*spec->freq_chunk+i_f]));
    //    }
    //    puts("");
    //}

    // Do degridding. Twice if necessary.
    uint64_t flops = 0;
    if (overlap) {
        bl_data->vis = vis_data; // HACK
        flops += degrid_conv_bl(subgrid, cfg->xM_size, wcfg->theta,
                                sg_mid_u, sg_mid_v,
                                max(0, sg_min_u), sg_max_u, sg_min_v, sg_max_v,
                                bl_data, it0, it1, if0, if1, kern);
    }
    if (inv_overlap) {
        int i;
        // Bit of a hack. Also TODO: complex-conjugate visibilities
        for(i = 0; i < 3 * spec->time_count; i++) {
            bl_data->uvw_m[i] *= -1;
        }
        bl_data->vis = vis_data; // HACK
        flops += degrid_conv_bl(subgrid, cfg->xM_size, wcfg->theta,
                                sg_mid_u, sg_mid_v,
                                max(0, sg_min_u), sg_max_u, sg_min_v, sg_max_v,
                                bl_data, it0, it1, if0, if1, kern);
        for(i = 0; i < 3 * spec->time_count; i++) {
            bl_data->uvw_m[i] *= -1;
        }
    }

    //printf("writing visibilities:\n");
    //for(i_t = 0; i_t < spec->time_chunk; i_t++) {
    //    for(i_f = 0; i_f < spec->freq_chunk; i_f++) {
    //        printf("%g%+gi ",
    //               creal(vis_data[i_t*spec->freq_chunk+i_f]),
    //               cimag(vis_data[i_t*spec->freq_chunk+i_f]));
    //    }
    //    puts("");
    //}
    //printf(" %d flops\n", flops);

    // Write chunk back
    write_vis_chunk(vis_group, bl_data,
                   spec->time_chunk, spec->freq_chunk, tchunk, fchunk,
                   vis_data);

    // Read visibility chunk (in case previous data exists)


    //printf("bl %d/%d du=%g dv=%g\n", bl->a1, bl->a2,


    /* int i; */
    /* for (i = 0; i < 1 /\* wcfg->spec.time_count *\/; i++) { */
    /*     printf("t=%.4f: %g %g %g / %g %g %g\n", bl_data.time[i], */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+0],bl_data.freq[0]), */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+1],bl_data.freq[0]), */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+2],bl_data.freq[0]), */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+0],bl_data.freq[bl_data.freq_count-1]), */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+1],bl_data.freq[bl_data.freq_count-1]), */
    /*            uvw_m_to_l(bl_data.uvw_m[i*3+2],bl_data.freq[bl_data.freq_count-1])); */
    /* } */

}

void streamer_work(struct work_config *wcfg,
                   struct sep_kernel_data *kern, hid_t vis_group,
                   int subgrid_worker, int subgrid_work,
                   double complex *data,
                   double complex *subgrid, fftw_plan subgrid_plan)
{

    struct recombine2d_config *cfg = &wcfg->recombine;
    struct subgrid_work *work = wcfg->subgrid_work + subgrid_worker * wcfg->subgrid_max_work + subgrid_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;
    const int data_length = wcfg->recombine.NMBF_NMBF_size / sizeof(double complex);

    // Compare with reference
    if (work->check_fct_path) {

        int i0 = work->iv, i1 = work->iu;
        int ifacet;
        for (ifacet = 0; ifacet < facets; ifacet++) {
            if (!wcfg->facet_work[ifacet].set) continue;
            int j0 = wcfg->facet_work[ifacet].im, j1 = wcfg->facet_work[ifacet].il;
            double complex *ref = read_hdf5(wcfg->recombine.NMBF_NMBF_size, work->check_hdf5,
                                            work->check_fct_path, j0, j1);
            int x; double err_sum = 0;
            for (x = 0; x < data_length; x++) {
                double err = cabs(ref[x] - data[data_length*ifacet+x]); err_sum += err*err;
            }
            free(ref);
            double rmse = sqrt(err_sum / data_length);
            if (!work->check_fct_threshold || rmse > work->check_fct_threshold) {
                printf("Subgrid %d/%d facet %d/%d checked: %g RMSE\n",
                       i0, i1, j0, j1, rmse);
            }
        }
    }

    // Accumulate contributions to this subgrid
    memset(subgrid, 0, cfg->SG_size);
    int ifacet;
    for (ifacet = 0; ifacet < facets; ifacet++)
        recombine2d_af0_af1(cfg, subgrid,
                            wcfg->facet_work[ifacet].facet_off_m,
                            wcfg->facet_work[ifacet].facet_off_l,
                            data + data_length*ifacet);

    // Perform Fourier transform
    fftw_execute(subgrid_plan);

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
    if (work->check_degrid_path && kern) {
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
                       &bl, 0, nvis, 0, 1, kern);
        double err_sum = 0; int y;
        for (y = 0; y < nvis; y++) {
            double err = cabs(vis[y] - bl.vis[y]); err_sum += err*err;
        }
        double rmse = sqrt(err_sum / nvis);
        printf("%sSubgrid %d/%d degrid RMSE %g\n",
               rmse > work->check_degrid_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    if (wcfg->spec.time_count > 0 && kern) {

        // Loop through baselines
        struct subgrid_work_bl *bl;
        for (bl = work->bls; bl; bl = bl->next) {

            // Get baseline data
            struct bl_data bl_data;
            vis_spec_to_bl_data(&bl_data, &wcfg->spec, bl->a1, bl->a2);

            // Go through time/frequency chunks
            int ntchunk = (bl_data.time_count + wcfg->spec.time_chunk - 1) / wcfg->spec.time_chunk;
            int nfchunk = (bl_data.freq_count + wcfg->spec.freq_chunk - 1) / wcfg->spec.freq_chunk;
            int tchunk, fchunk;
            for (tchunk = 0; tchunk < ntchunk; tchunk++)
                for (fchunk = 0; fchunk < nfchunk; fchunk++)
                    streamer_degrid_chunk(wcfg, kern, vis_group, work,
                                          bl, &bl_data, tchunk, fchunk,
                                          subgrid);
        }

    }
}

void streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks) {

    struct recombine2d_config *cfg = &wcfg->recombine;
    struct subgrid_work *work = wcfg->subgrid_work + subgrid_worker * wcfg->subgrid_max_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;
    hid_t vis_file = -1, vis_group = -1;
    struct sep_kernel_data kern; bool have_kern = false;

    // Create HDF5 output file if we are meant to output any amount of
    // visibilities
    if (wcfg->vis_path) {

        // Get filename to use
        char filename[512];
        sprintf(filename, wcfg->vis_path, subgrid_worker);

        // Open file and "vis" group
        printf("\nCreating %s... ", filename);
        vis_file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        vis_group = H5Gcreate(vis_file, "vis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (vis_file < 0 || vis_group < 0) {
            fprintf(stderr, "Could not open visibility file %s!\n", filename);
        } else {
            create_bl_groups(vis_group, wcfg, subgrid_worker);
        }
    }

    // Load gridding kernel
    if (wcfg->gridder_path)
        if (!load_sep_kern(wcfg->gridder_path, &kern))
            have_kern = true;

    // Allocate receive queue
    const int recv_queue_depth = 64;
    const int data_length = cfg->NMBF_NMBF_size / sizeof(double complex);
    const int queue_size = sizeof(double complex) * data_length * facets * recv_queue_depth;
    const int requests_size = sizeof(MPI_Request) * facets * recv_queue_depth;
    printf("Allocating %d MB receive queue\n", (queue_size+requests_size) / 1000000);
    double complex *NMBF_NMBF_queue = (double complex *)malloc(queue_size);
    MPI_Request *request_queue = (MPI_Request *) malloc(requests_size);
    MPI_Status *status_queue = (MPI_Status *) malloc(sizeof(MPI_Status) * facets * recv_queue_depth);

    // Work that has been done. This is needed because for split
    // subgrids, multiple work entries might correspond to just a
    // single subgrid getting transferred.
    char *done_work = calloc(sizeof(char), wcfg->subgrid_max_work);

    // Start receive queue
    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_max_work && iwork < recv_queue_depth; iwork++) {
        streamer_ireceive(wcfg,
                          NMBF_NMBF_queue + iwork * facets * data_length,
                          request_queue + iwork * facets,
                          subgrid_worker, iwork, producer_ranks);
    }

    printf("Waiting for data...\n");

    // Plan FFTs
    double complex *subgrid = (double complex *)malloc(cfg->SG_size);
    fftw_plan subgrid_plan = fftw_plan_dft_2d(cfg->xM_size, cfg->xM_size,
                                              subgrid, subgrid, FFTW_BACKWARD, FFTW_MEASURE);

    // Start doing work
    uint64_t received_data = 0; int received_subgrids = 0;
    for (iwork = 0; iwork < wcfg->subgrid_max_work; iwork++) {
        int work_slot = iwork % recv_queue_depth;
        double complex *data_slot = NMBF_NMBF_queue + work_slot * facets * data_length;
        MPI_Request *request_slot = request_queue + work_slot * facets;

        if (work[iwork].nbl && !done_work[iwork]) {

            // Wait for all facet data for this work to arrive (TODO:
            // start doing some work earlier)
            MPI_Waitall(facets, request_queue + facets * work_slot,
                        status_queue + facets * work_slot);
            memset(request_queue + facets * work_slot, MPI_REQUEST_NULL, sizeof(MPI_Request) * facets);
            received_data += sizeof(double complex) * facets * data_length;
            received_subgrids++;
        }

        // Do work on received data
        streamer_work(wcfg, have_kern ? &kern : NULL, vis_group,
                      subgrid_worker, iwork, data_slot, subgrid, subgrid_plan);

        // Mark work done
        int iw;
        done_work[iwork] = true;
        for (iw = iwork+1; iw < wcfg->subgrid_max_work; iw++)
            if (work[iw].iu == work[iwork].iu && work[iw].iv == work[iwork].iv)
                done_work[iw] = true;

        // Set up slot for new data (if appropriate)
        int iwork_r = iwork + recv_queue_depth;
        if (iwork_r < wcfg->subgrid_max_work && work[iwork_r].nbl && !done_work[iwork_r]) {
            streamer_ireceive(wcfg, data_slot, request_slot, subgrid_worker, iwork_r, producer_ranks);
        }

    }

    free(NMBF_NMBF_queue); free(request_queue); free(status_queue); free(done_work);
    fftw_free(subgrid_plan); free(subgrid);

    if (vis_group >= 0) {
        H5Gclose(vis_group); H5Fclose(vis_file);
    }

    printf("Received %.2f GB (%d subgrids)\n",
           (double)received_data / 1000000000, received_subgrids);
}
