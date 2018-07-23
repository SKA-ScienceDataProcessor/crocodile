
#include "grid.h"
#include "recombine.h"

#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

bool set_default_recombine2d_config(struct recombine2d_config *cfg)
{
    return recombine2d_set_config(cfg, 98304, 98304, 1536, "../../data/grid/pswf5.00-33728.in",
								  24576, 33728, 49152, 1405, 1536, 282);
}

bool set_test_recombine2d_config(struct recombine2d_config *cfg, int rank)
{
    if (!recombine2d_set_config(cfg, 2000, 2000, 5, "../../data/grid/T04_pswf.in", 400, 480, 900, 400, 500, 247))
		return false;
	// Use data from test suite. Note that not all "nmbf" reference
	// files exist in the repository, so this will show a few errors.
	// As long as no value mismatches occur this is fine
    char file[256];
    sprintf(file, "../../data/grid/T04_facet%d%d.in", rank / 3, rank % 3);
    cfg->facet_file = strdup(file);
    sprintf(file, "../../data/grid/T04_nmbf%%d%%d%d%d.in", rank / 3, rank % 3);
    cfg->stream_check = strdup(file);
    cfg->stream_check_threshold = 1e-9;
	return true;
}

bool recombine2d_set_test5_config(struct recombine2d_config *cfg, int rank)
{
    if (!recombine2d_set_config(cfg, 512, 512, 256, "../../data/grid/T05_pswf.in", 128, 140, 216, 128, 256, 136))
		return false;
	// Use data from test suite. Note that not all "nmbf" reference
	// files exist in the repository, so this will show a few errors.
	// As long as no value mismatches occur this is fine
    char file[256];
    sprintf(file, "../../data/grid/T05_facet%d%d.in", rank / 3, rank % 3);
    cfg->facet_file = strdup(file);
    sprintf(file, "../../data/grid/T05_nmbf%%d%%d%d%d.in", rank / 3, rank % 3);
    cfg->stream_check = strdup(file);
    cfg->stream_check_threshold = 1e-6;
    //sprintf(file, "../../data/grid/T05_nmbf%%d%%d%d%d.out", rank / 3, rank % 3);
    //cfg->stream_dump = strdup(file);
	return true;
}

struct producer_stream {

    // Stream targets
    int streamer_count;
    int *streamer_ranks;

    // Send queue
    int send_queue_length;
    MPI_Request *requests;
    int bytes_sent;

    // Private buffers
    double complex *NMBF_NMBF_queue;

    // Worker structure
    struct recombine2d_worker worker;

    // Time (in s) spent in different stages
    double mpi_wait_time, mpi_send_time;

};

void init_producer_stream(struct recombine2d_config *cfg, struct producer_stream *prod,
                          int streamer_count, int *streamer_ranks,
                          int BF_batch, fftw_plan BF_plan,
                          int send_queue_length)
{

    // Set streamers
    prod->streamer_count = streamer_count;
    prod->streamer_ranks = streamer_ranks;

    // Initialise queue
    prod->send_queue_length = send_queue_length;
    prod->requests = (MPI_Request *) malloc(sizeof(MPI_Request) * send_queue_length);
    int i;
    for (i = 0; i < send_queue_length; i++) {
        prod->requests[i] = MPI_REQUEST_NULL;
    }

    // Create buffers, initialise worker
    prod->NMBF_NMBF_queue =
        (double complex *)malloc(cfg->NMBF_NMBF_size * send_queue_length);
    recombine2d_init_worker(&prod->worker, cfg, BF_batch, BF_plan, FFTW_MEASURE);

    // Initialise statistics
    prod->bytes_sent = 0;
    prod->mpi_wait_time = prod->mpi_send_time = 0;

}

void free_producer_stream(struct producer_stream *prod)
{
    recombine2d_free_worker(&prod->worker);

    free(prod->requests);
    free(prod->NMBF_NMBF_queue);
}

void producer_add_stats(struct producer_stream *to, struct producer_stream *from)
{
    to->bytes_sent += from->bytes_sent;
    to->worker.pf1_time += from->worker.pf1_time;
    to->worker.es1_time += from->worker.es1_time;
    to->worker.ft1_time += from->worker.ft1_time;
    to->worker.pf2_time += from->worker.pf2_time;
    to->worker.es2_time += from->worker.es2_time;
    to->worker.ft2_time += from->worker.ft2_time;
    to->mpi_wait_time += from->mpi_wait_time;
    to->mpi_send_time += from->mpi_send_time;
}

void producer_dump_stats(struct producer_stream *prod, struct recombine2d_config *cfg,
                         int producer_count, double dt)
{

    double total = dt * producer_count;
    printf("\n%.2f s wall-clock, %.2f GB (%.2f GB effective), %.2f MB/s (%.2f MB/s effective)\n", dt,
           (double)prod->bytes_sent / 1000000000, (double)cfg->F_size / 1000000000,
           (double)prod->bytes_sent / dt / 1000000, (double)cfg->F_size / dt/ 1000000);
    printf("PF1: %.2f s (%.1f%%), FT1: %.2f s (%.1f%%), ES1: %.2f s (%.1f%%)\n",
           prod->worker.pf1_time, prod->worker.pf1_time / total * 100,
           prod->worker.ft1_time, prod->worker.ft1_time / total * 100,
           prod->worker.es1_time, prod->worker.es1_time / total * 100);
    printf("PF2: %.2f s (%.1f%%), FT2: %.2f s (%.1f%%), ES2: %.2f s (%.1f%%)\n",
           prod->worker.pf2_time, prod->worker.pf2_time / total * 100,
           prod->worker.ft2_time, prod->worker.ft2_time / total * 100,
           prod->worker.es2_time, prod->worker.es2_time / total * 100);
    double idle = total -
        prod->worker.pf1_time - prod->worker.ft1_time - prod->worker.es1_time -
        prod->worker.pf2_time - prod->worker.ft2_time - prod->worker.es2_time -
        prod->mpi_wait_time - prod->mpi_send_time;
    printf("mpi wait: %.2f s (%.1f%%), mpi send: %.2f s (%.1f%%), idle: %.2f s (%.1f%%)\n",
           prod->mpi_wait_time, 100 * prod->mpi_wait_time / producer_count / dt,
           prod->mpi_send_time, 100 * prod->mpi_send_time / producer_count / dt,
           idle, 100 * idle / producer_count / dt);
}

double get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

void producer_ES0_PF1_FT1_ES1(struct recombine2d_config *cfg, struct producer_stream *prod,
                              int i0, complex double *BF)
{

    // Extract subgrids along first axis, then prepare and Fourier
    // transform along second axis
    recombine2d_es1_pf0_ft0(&prod->worker, i0, BF);
    int i1;
    const int nsubgrid = cfg->image_size / cfg->xA_size;
    for (i1 = 0; i1 < nsubgrid; i1++) {

        // Select send slot if running in distributed mode
        int indx; MPI_Status status;
        if (prod->streamer_count == 0)
            indx = 0;
        else {
            for (indx = 0; indx < prod->send_queue_length; indx++) {
                if (prod->requests[indx] == MPI_REQUEST_NULL) break;
            }
            if (indx >= prod->send_queue_length) {
                double start = get_time_ns();
                MPI_Waitany(prod->send_queue_length, prod->requests, &indx, &status);
                prod->mpi_wait_time += get_time_ns() - start;
            }
            assert (indx >= 0 && indx < prod->send_queue_length);
        }

        // Extract subgrids along second axis
        double complex *NMBF_NMBF = prod->NMBF_NMBF_queue + indx * cfg->xM_yN_size * cfg->xM_yN_size;
        recombine2d_es0(&prod->worker, i0, i1, NMBF_NMBF);

        // Spread data evenly across streamers
        if (prod->streamer_count > 0) {
            int target_rank = prod->streamer_ranks[(i1 + i0 * nsubgrid) % prod->streamer_count];
            double start = get_time_ns();
            MPI_Isend(NMBF_NMBF, cfg->xM_yN_size * cfg->xM_yN_size, MPI_DOUBLE_COMPLEX,
                      target_rank, 0, MPI_COMM_WORLD, &prod->requests[indx]);
            prod->mpi_send_time += get_time_ns() - start;
            prod->bytes_sent += sizeof(double complex) * cfg->xM_yN_size * cfg->xM_yN_size;
        }

    }

}

bool producer_fill_facet(struct recombine2d_config *cfg, double complex *F, int x0_start, int x0_end) {

    if (!cfg->facet_file) {

        // Fill facet with deterministic pseudo-random numbers
        int x0, x1;
        for (x0 = x0_start; x0 < x0_end; x0++) {
            unsigned int seed = x0;
            for (x1 = 0; x1 < cfg->yB_size; x1++) {
                F[(x0-x0_start)*cfg->F_stride0+x1*cfg->F_stride1] = (double)rand_r(&seed) / RAND_MAX;
            }
        }

    } else {

        // Make sure strides are compatible
        assert (cfg->F_stride0 == cfg->yB_size && cfg->F_stride1 == 1);
        int offset = sizeof(double complex) * x0_start * cfg->yB_size;
        int size = sizeof(double complex) * (x0_end - x0_start) * cfg->yB_size;

        // Load data from file
        printf("%s\n", cfg->facet_file);
        int fd = open(cfg->facet_file, O_RDONLY, 0666);
        if (fd > 0) {
            lseek(fd, offset, SEEK_SET);
            if (read(fd, F, size) != size) {
                fprintf(stderr, "failed to read enough data from %s for range %d-%d!\n", cfg->facet_file, x0_start, x0_end);
                return false;
            }
            close(fd);
        }

    }
    return true;
}

int producer(struct recombine2d_config *cfg, int streamer_count, int *streamer_ranks) {

    printf("Using %.1f GB global, %.1f GB per thread\n",
           (double)recombine2d_global_memory(cfg) / 1000000000,
           (double)recombine2d_worker_memory(cfg) / 1000000000);

    // Create global memory buffers
    double complex *F = (double complex *)calloc(cfg->F_size, 1);
    double complex *BF = (double complex *)malloc(cfg->BF_size);
    if (!F || !BF) {
        free(F); free(BF);
        printf("Failed to allocate global buffers!\n");
        return 1;
    }

    // Fill facet with random data
    printf("Filling facet...\n"); double generate_start = get_time_ns();
    int x0; const int x0_chunk = 256;
    #pragma omp parallel for schedule(dynamic)
    for (x0 = 0; x0 < cfg->yB_size; x0+=x0_chunk) {
        int x0_end = x0 + x0+x0_chunk;
        if (x0_end > cfg->yB_size) x0_end = cfg->yB_size;
        producer_fill_facet(cfg, F+x0*cfg->F_stride0, x0, x0_end);
    }
    printf(" %.2f s\n", get_time_ns() - generate_start);

    // Debugging (TODO: remove)
    if (false) {
        int x1;
        for (x0 = 0; x0 < cfg->yB_size; x0++) {
            for (x1 = 0; x1 < cfg->yB_size; x1++) {
                printf("%8.2f%+8.2fi\t", creal(F[x0*cfg->F_stride0+x1*cfg->F_stride1]),
                       cimag(F[x0*cfg->F_stride0+x1*cfg->F_stride1]));
            }
            puts("");
        }
    }

    // Global structures
    double run_start;
    int producer_count;
    struct producer_stream *producers;

    #pragma omp parallel
    {

        const int BF_batch = 16;
        const int send_queue_length = 8;

        producer_count = omp_get_num_threads();
        #pragma omp single
        {

            // Do global planning
            printf("Planning...\n"); double planning_start = get_time_ns();
            fftw_plan BF_plan = recombine2d_bf_plan(cfg, BF_batch, BF, FFTW_MEASURE);

            // Create producers (which involves planning, and therefore is not parallelised)
            producers = (struct producer_stream *) malloc(sizeof(struct producer_stream) * producer_count);
            int i;
            for (i = 0; i < producer_count; i++) {
                init_producer_stream(cfg, producers + i, streamer_count, streamer_ranks,
                                     BF_batch, BF_plan, send_queue_length);
            }

            // End of planning phase
            printf(" %.2f s\n", get_time_ns() - planning_start);
            run_start = get_time_ns();
            printf("Streaming...\n");
        }
        struct producer_stream *prod = producers + omp_get_thread_num();

        // Do first stage preparation and Fourier Transform
        recombine2d_pf1_ft1_omp(&prod->worker, F, BF);
        // TODO: Generate facet on the fly

        int i0;
        #pragma omp for schedule(dynamic)
        for (i0 = 0; i0 < cfg->image_size / cfg->xA_size; i0++) {
            producer_ES0_PF1_FT1_ES1(cfg, prod, i0, BF);

        }

        // Wait for remaining packets to be sent
        MPI_Status statuses[send_queue_length];
        MPI_Waitall(send_queue_length, prod->requests, statuses);

        free_producer_stream(prod);
    }
    free(BF);
    free(F);

    fftw_free(producers[0].worker.BF_plan);

    // Show statistics
    int p;
    for (p = 1; p < producer_count; p++) {
        producer_add_stats(producers, producers + p);
    }
    producer_dump_stats(producers, cfg, producer_count, get_time_ns() - run_start);

    return 0;
}

void streamer(struct recombine2d_config *cfg, int rank, int streamer_count, int *producer_ranks, int producer_count) {
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
    int thread_support, rank, size;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_length = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (thread_support < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Need full thread support from MPI!\n");
        return 1;
    }
    MPI_Get_processor_name(proc_name, &proc_name_length);
    proc_name[proc_name_length] = 0;

    // Read FFTW wisdom to get through planning quicker
    fftw_import_wisdom_from_filename("recombine.wisdom");

    // Make imaging configuration
    struct recombine2d_config config;
    //if (!set_default_recombine2d_config(&config)) {
    if (!set_test_recombine2d_config(&config, rank)) {
        //if (!recombine2d_set_test5_config(&config, rank)) {
        fprintf(stderr, "Could not set imaging configuration!\n");
        return 1;
    }

    // Local run?
    if (size == 1) {
        printf("%s pid %d role: Single\n", proc_name, getpid());
        producer(&config, 0, 0);

    } else {

        // Determine number of producers and streamers (pretty arbitrary for now)
        int producer_count = size / 2;
        int streamer_count = size - producer_count;
        int i;

        if (rank < producer_count) {
            printf("%s pid %d role: Producer %d\n", proc_name, getpid(), rank);

            int *streamer_ranks = (int *)malloc(sizeof(int) * streamer_count);
            for (i = 0; i < streamer_count; i++) {
                streamer_ranks[i] = producer_count + i;
            }

            producer(&config, streamer_count, streamer_ranks);

        } else if (rank - producer_count < streamer_count) {
            printf("%s pid %d role: Streamer %d\n", proc_name, getpid(), rank - producer_count);

            int *producer_ranks = (int *)malloc(sizeof(int) * producer_count);
            for (i = 0; i < producer_count; i++) {
                producer_ranks[i] = i;
            }

            streamer(&config, rank-producer_count, streamer_count, producer_ranks, producer_count);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    // Master: Write wisdom
    if (rank == 0) {
        fftw_export_wisdom_to_filename("recombine.wisdom");
    }
    return 0;
}
