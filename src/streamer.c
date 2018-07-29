
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

void streamer_work(struct work_config *wcfg,
                   int subgrid_worker, int subgrid_work,
                   double complex *data,
                   double complex *subgrid, fftw_plan subgrid_plan)
{

    struct recombine2d_config *cfg = &wcfg->recombine;
    struct subgrid_work *work = wcfg->subgrid_work + subgrid_worker * wcfg->subgrid_max_work + subgrid_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;
    const int data_length = wcfg->recombine.NMBF_NMBF_size / sizeof(double complex);

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

}

void streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks) {

    struct recombine2d_config *cfg = &wcfg->recombine;
    struct subgrid_work *work = wcfg->subgrid_work + subgrid_worker * wcfg->subgrid_max_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;

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
        MPI_Status *status_slot = status_queue + work_slot * facets;

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
        streamer_work(wcfg, subgrid_worker, iwork, data_slot, subgrid, subgrid_plan);

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

    printf("Received %.2f GB (%d subgrids)\n",
           (double)received_data / 1000000000, received_subgrids);
}
