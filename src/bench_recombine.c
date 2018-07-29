
#include "grid.h"
#include "config.h"

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

bool set_default_vis_spec(const char *ant_cfg_file, struct vis_spec *spec, double fov)
{
    spec->cfg = malloc(sizeof(struct ant_config));
    if (!load_ant_config(ant_cfg_file, spec->cfg))
        return false;

    spec->fov = fov;
    spec->dec = 90 * atan(1) * 4 / 180;
    spec->time_start = 10 * -45 / 3600; // h
    spec->time_count = 64;
    spec->time_chunk = 16;
    spec->time_step = 0.9 / 3600; // h
    spec->freq_start = 250e6; // Hz
    spec->freq_count = 64;
    spec->freq_chunk = 16;
    spec->freq_step = 50.e6 / spec->freq_count; // Hz

    return true;
}

bool set_default_recombine2d_config(struct work_config *cfg,
                                    int facet_workers, int subgrid_workers)
{

    struct vis_spec *spec = malloc(sizeof(struct vis_spec));
    if (!set_default_vis_spec("../data/grid/LOWBD2_north_cfg.h5", spec, 0.1))
        return false;

    return work_config_set(cfg, spec, facet_workers, subgrid_workers,
                           spec->fov * 1. / 0.75,
                           98304, 1536, "../data/grid/pswf5.00-33728.in",
                           24576, 33728, 49152, 1405, 1536, 282);
}

bool set_test_recombine2d_config(struct work_config *cfg,
                                 int facet_workers, int subgrid_workers,
                                 int rank)
{

    if (!work_config_set(cfg, 0, facet_workers, subgrid_workers,
                         0.15,
                         2000, 100, "../data/grid/T04_pswf.in",
                         400, 480, 900, 400, 500, 247))
        return false;

    load_facets_from(cfg, "../data/grid/T04_facet%d%d.in", NULL);

    // Use data from test suite. Note that not all "nmbf" reference
    // files exist in the repository, so this will show a few errors.
    // As long as no value mismatches occur this is fine
    char file[256];
    sprintf(file, "../data/grid/T04_nmbf%%d%%d%d%d.in", rank / 3, rank % 3);
    cfg->recombine.stream_check = strdup(file);
    cfg->recombine.stream_check_threshold = 1e-9;
    return true;
}

bool recombine2d_set_test5_config(struct work_config *cfg,
                                  int facet_workers, int subgrid_workers,
                                  int rank)
{
    if (!work_config_set(cfg, NULL, facet_workers, subgrid_workers,
                         0.1,
                         512, 128, "../data/grid/T05_pswf.in",
                         128, 140, 216, 128, 256, 136))
        return false;

    const char *hdf5 = "../data/grid/T05_in.h5";
    load_facets_from(cfg, "j0=%d/j1=%d/facet", hdf5);
    return true;
}

bool set_serious_test_config(struct work_config *cfg,
                             int facet_workers, int subgrid_workers,
                             int rank)
{
    struct vis_spec *spec = malloc(sizeof(struct vis_spec));
    if (!set_default_vis_spec("../data/grid/LOWBD2_north_cfg.h5", spec, 0.1))
        return false;

    if (!work_config_set(cfg, spec, facet_workers, subgrid_workers,
                         spec->fov * 1. / 0.75,
                         32768, 4, "../data/grid/T06_pswf.in",
                         8192, 8640, 16384,
                         384, 512, 276))
        return false;

    return true;
}

void streamer(struct work_config *wcfg, int rank, int streamer_count, int *producer_ranks, int producer_count) {

    struct recombine2d_config *cfg = &wcfg->recombine;
    const int recv_queue_length = 16;

    const int gather_size = cfg->NMBF_NMBF_size * producer_count;
    const int queue_size = gather_size * recv_queue_length;
    printf("Allocating %d MB receive queue\n", queue_size / 1000000);

    double complex *NMBF_NMBF = (double complex *)malloc(queue_size);
    MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * producer_count * recv_queue_length);
    MPI_Status *statuses = (MPI_Status *)malloc(sizeof(MPI_Status) * producer_count * recv_queue_length);
    memset(requests, MPI_REQUEST_NULL, sizeof(MPI_Request) * producer_count * recv_queue_length);

    int p, q;
    for (q = 0; q < recv_queue_length; q++) {
        for (p = 0; p < producer_count; p++) {
            requests[p + producer_count * q] = MPI_REQUEST_NULL;
        }
    }

    q = 0;
    uint64_t received_data = 0;

    int i0, i1;
    const int nsubgrid = cfg->image_size / cfg->xA_size;
    for (i0 = 0; i0 < nsubgrid; i0++) {
        for (i1 = 0; i1 < nsubgrid; i1++) {
            int target_rank = (i1 + i0 * nsubgrid) % streamer_count;
            if (target_rank == rank) {

                if(requests[producer_count * q] != MPI_REQUEST_NULL) {
                    MPI_Waitall(producer_count, requests + producer_count * q, statuses + producer_count * q);
                    for (p = 0; p < producer_count; p++) {
                        requests[p + producer_count * q] = MPI_REQUEST_NULL;
                    }
                    received_data += gather_size;
                }

                for (p = 0; p < producer_count; p++) {
                    MPI_Irecv(NMBF_NMBF + p * cfg->xM_yN_size * cfg->xM_yN_size,
                              cfg->xM_yN_size * cfg->xM_yN_size, MPI_DOUBLE_COMPLEX,
                              producer_ranks[p], 0, MPI_COMM_WORLD, requests + p + producer_count * q);
                }

                q = (q + 1) % recv_queue_length;

            }
        }
    }

    for (q = 0; q < recv_queue_length; q++) {
        if(requests[producer_count * q] != MPI_REQUEST_NULL) {
            MPI_Waitall(producer_count, requests + producer_count * q, statuses + producer_count * q);
            received_data += gather_size;
        }
    }
    printf("Received %.2f GB\n", (double)received_data / 1000000000);
}

int main(int argc, char *argv[]) {

    // Initialise MPI, read configuration (we need multi-threading support)
    int thread_support, world_rank, world_size;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_length = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (thread_support < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Need full thread support from MPI!\n");
        return 1;
    }
    MPI_Get_processor_name(proc_name, &proc_name_length);
    proc_name[proc_name_length] = 0;

    // Read FFTW wisdom to get through planning quicker
    fftw_import_wisdom_from_filename("recombine.wisdom");

    // Decide number of workers
    int facet_workers = (world_size + 1) / 2;
    int subgrid_workers = world_size - facet_workers;
    if (subgrid_workers == 0) subgrid_workers = 1;

    // Make imaging configuration
    struct work_config config;
    //if (!set_default_recombine2d_config(&config)) {
    //if (!set_test_recombine2d_config(&config, facet_workers, subgrid_workers, world_rank)) {
    //if (!recombine2d_set_test5_config(&config, facet_workers, subgrid_workers, world_rank)) {
    if (!set_serious_test_config(&config, facet_workers, subgrid_workers, world_rank)) {
        fprintf(stderr, "Could not set imaging configuration!\n");
        return 1;
    }

    // Local run?
    if (world_size == 1) {
        printf("%s pid %d role: Single\n", proc_name, getpid());

        producer(&config, 0, 0);

    } else {

        // Determine number of producers and streamers (pretty arbitrary for now)
        int i;

        if (world_rank < facet_workers) {
            printf("%s pid %d role: Producer %d\n", proc_name, getpid(), world_rank);

            int *streamer_ranks = (int *)malloc(sizeof(int) * subgrid_workers);
            for (i = 0; i < subgrid_workers; i++) {
                streamer_ranks[i] = facet_workers + i;
            }

            producer(&config, world_rank, streamer_ranks);

        } else if (world_rank - facet_workers < subgrid_workers) {
            printf("%s pid %d role: Streamer %d\n", proc_name, getpid(), world_rank - facet_workers);

            int *producer_ranks = (int *)malloc(sizeof(int) * facet_workers);
            for (i = 0; i < facet_workers; i++) {
                producer_ranks[i] = i;
            }

            streamer(&config, world_rank-producer_count, streamer_count, producer_ranks, producer_count);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    // Master: Write wisdom
    if (world_rank == 0) {
        fftw_export_wisdom_to_filename("recombine.wisdom");
    }
    return 0;
}
