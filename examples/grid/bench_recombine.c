
#include "grid.h"

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

// Size specifications
const int image_size = 98304 * 4;
const int yB_size = 21504 * 4;
const int yN_size = 24576 * 4;
const int yP_size = 49152 * 4; // 24576;
const int xA_size = 384;
const int xM_size = 512;
//int xM_yB_size = 100;
#define nsubgrid (image_size / xA_size)

const int repeat_count = 1;

double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

int producer(int *streamer_ranks, int streamer_count) {

    const int xA_yP_size = xA_size * yP_size / image_size;
    const int xM_yP_size = xM_size * yP_size / image_size;
    const int xMxN_yP_size = 282;
    const int xM_yN_size = xM_size * yN_size / image_size;

    // Read PSWF, generate Fn, Fb and m
    double *pswf = read_dump(sizeof(double) * yN_size, "../../data/grid/T05_pswf.in");
    if (!pswf) return 1;
    double *Fb = generate_Fb(yN_size, yB_size, pswf);
    double *Fn = generate_Fn(yN_size, xM_yN_size, pswf);
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    free(pswf);

	size_t F_size = sizeof(double complex) * yB_size * yB_size;
	size_t BF_size = sizeof(double complex) * yP_size * yB_size;
    size_t MBF_size = sizeof(double complex) * xM_yP_size;
	size_t NMBF_size = sizeof(double complex) * xM_yN_size * yB_size;
	size_t NMBF_BF_size = sizeof(double complex) * xM_yN_size * yP_size;
	size_t NMBF_NMBF_size = sizeof(double complex) * xM_yN_size * xM_yN_size;

	printf("Using %.1f GB global, %.1f GB per thread\n",
		   (double)(F_size + BF_size + NMBF_size) / 1000000000,
		   (double)(MBF_size + NMBF_BF_size + NMBF_NMBF_size) / 1000000000);
    double complex *F = (double complex *)calloc(F_size, 1);
    double complex *BF = (double complex *)malloc(BF_size);
    double complex *NMBF = (double complex *)malloc(NMBF_size);
	if (!F || !BF || !NMBF) {
		free(Fb); free(Fn); free(m_trunc); free(F); free(BF); free(NMBF);
		printf("Failed to allocate global buffers!\n");
		return 1;
	}

    uint64_t output_size = 0;

    const uint64_t F_stride0 = yB_size, F_stride1 = 1;
    const uint64_t BF_stride0 = yP_size, BF_stride1 = 1;
    const uint64_t NMBF_stride0 = xM_yN_size, NMBF_stride1 = 1;
    const uint64_t NMBF_BF_stride0 = 1, NMBF_BF_stride1 = yP_size;
    const uint64_t NMBF_NMBF_stride0 = 1, NMBF_NMBF_stride1 = xM_yN_size;
    const int y_batch = 32;

    const int send_queue_length = 32;

	printf("Filling facet...\n"); double generate_start = get_time_ns();
    int y;
    #pragma omp parallel for schedule(dynamic, 1024)
    for (y = 0; y < yB_size; y++) {
        int x;
		for (x = 0; x < yB_size; x+=100) {
			F[y*F_stride0+x*F_stride1] = (double)rand() / RAND_MAX;
		}
    }
	printf(" %.2f s\n", get_time_ns() - generate_start);

    printf("Planning...\n"); double planning_start = get_time_ns();
    fftw_plan BF_plan = fftw_plan_many_dft(1, &yP_size, y_batch,
                                           BF, 0, BF_stride1, BF_stride0,
                                           BF, 0, BF_stride1, BF_stride0,
                                           FFTW_BACKWARD, FFTW_MEASURE);

    double run_start;
    double pf1_time = 0;
    double es1_time = 0;
    double ft1_time = 0;
    double pf2_time = 0;
    double es2_time = 0;
    double ft2_time = 0;
    double mpi_wait_time = 0, mpi_send_time = 0;

    #pragma omp parallel reduction(+:output_size)                     \
        reduction(+:pf1_time) reduction(+:es1_time) reduction(+:ft1_time) \
        reduction(+:pf2_time) reduction(+:es2_time) reduction(+:ft2_time) \
        reduction(+:mpi_send_time) reduction(+:mpi_wait_time)
    {

        MPI_Request requests[send_queue_length];
        int x;
        for (x = 0; x < send_queue_length; x++) {
            requests[x] = MPI_REQUEST_NULL;
        }

        double complex *MBF = (double complex *)malloc(MBF_size);
        double complex *NMBF_BF = (double complex *)malloc(NMBF_BF_size);
        double complex *NMBF_NMBF_queue =
            (double complex *)malloc(NMBF_NMBF_size * send_queue_length);

        fftw_plan NMBF_BF_plan, MBF_plan;
        #pragma omp critical
        {
            MBF_plan = fftw_plan_dft_1d(xM_yP_size, MBF, MBF,
                                        FFTW_FORWARD, FFTW_MEASURE);
            NMBF_BF_plan = fftw_plan_many_dft(1, &yP_size, xM_yN_size,
                                              NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                              NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                              FFTW_BACKWARD, FFTW_MEASURE);
        }
        #pragma omp single
        {
            printf(" %.2f s\n", get_time_ns() - planning_start);
            run_start = get_time_ns();
            printf("Streaming...\n");
        }
        int r;
        for (r = 0; r < repeat_count; r++) {

            int x;
            #pragma omp for schedule(dynamic)
            for (x = 0; x < yB_size; x+=y_batch) {
                double start = get_time_ns();
                for (y = x; y < x+y_batch; y++) {
                    prepare_facet(yB_size, yP_size, Fb,
                                  F+x*F_stride0, F_stride1,
                                  BF+x*BF_stride0, BF_stride1);
                }
                pf1_time += get_time_ns() - start;
                start = get_time_ns();
                fftw_execute_dft(BF_plan, BF+x*BF_stride0, BF+x*BF_stride0);
                ft1_time += get_time_ns() - start;
            }

            int i0;
            #pragma omp for schedule(dynamic)
            for (i0 = 0; i0 < nsubgrid; i0++) {

                int x,y;

                double start = get_time_ns();
                for (x = 0; x < yB_size; x++) {
                    extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i0*xA_yP_size, m_trunc, Fn,
                                    BF+x*BF_stride0, BF_stride1,
                                    MBF, MBF_plan,
                                    NMBF+x*NMBF_stride0, NMBF_stride1);
                }
                es1_time += get_time_ns() - start;
                start = get_time_ns();
                for (y = 0; y < xM_yN_size; y++) {
                    prepare_facet(yB_size, yP_size, Fb,
                                  NMBF+y*NMBF_stride1, NMBF_stride0,
                                  NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0);
                }
                pf2_time += get_time_ns() - start;
                start = get_time_ns();
                fftw_execute(NMBF_BF_plan);
                ft2_time += get_time_ns() - start;

                int i1;
                for (i1 = 0; i1 < nsubgrid; i1++) {

                    // Select send slot if running in distributed mode
                    int indx; MPI_Status status;
                    if (streamer_count == 0)
                        indx = 0;
                    else {
                        for (indx = 0; indx < send_queue_length; indx++) {
                            if (requests[indx] == MPI_REQUEST_NULL) break;
                        }
                        if (indx >= send_queue_length) {
                            start = get_time_ns();
                            MPI_Waitany(send_queue_length, requests, &indx, &status);
                            mpi_wait_time += get_time_ns() - start;
                        }
                        assert (indx >= 0 && indx < send_queue_length);
                    }

                    start = get_time_ns();
                    double complex *NMBF_NMBF = NMBF_NMBF_queue + indx * xM_yN_size * xM_yN_size;
                    for (y = 0; y < xM_yN_size; y++) {
                        extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i1*xA_yP_size, m_trunc, Fn,
                                        NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0,
                                        MBF, MBF_plan,
                                        NMBF_NMBF+y*NMBF_NMBF_stride1, NMBF_NMBF_stride0);
                    }
                    es2_time += get_time_ns() - start;

                    // Spread data evenly across streamers
                    if (streamer_count > 0) {
                        int target_rank = streamer_ranks[(i1 + i0 * nsubgrid) % streamer_count];
                        start = get_time_ns();
                        MPI_Isend(NMBF_NMBF, xM_yN_size * xM_yN_size, MPI_DOUBLE_COMPLEX,
                                  target_rank, 0, MPI_COMM_WORLD, &requests[indx]);
                        mpi_send_time += get_time_ns() - start;
                    }
                    output_size += sizeof(double complex) * xM_yN_size * xM_yN_size;
                }
            }

        }

        MPI_Status statuses[send_queue_length];
        MPI_Waitall(send_queue_length, requests, statuses);

        free(MBF);
        free(NMBF_NMBF_queue);
        free(NMBF_BF);
        fftw_free(MBF_plan);
    }
    free(NMBF);
    free(BF);

    double dt = get_time_ns() - run_start;
    double total = dt * omp_get_max_threads();
    printf("\n%.2f s wall-clock, %.2f GB (%.2f GB effective), %.2f MB/s (%.2f MB/s effective)\n", dt,
           (double)output_size / 1000000000, (double)F_size / 1000000000,
           (double)output_size / dt / 1000000, (double)F_size / dt/ 1000000);
    printf("PF1: %.2f s (%.1f%%), FT1: %.2f s (%.1f%%), ES1: %.2f s (%.1f%%)\n",
           pf1_time, pf1_time / total * 100, ft1_time, ft1_time / total * 100, es1_time, es1_time / total * 100);
    printf("PF2: %.2f s (%.1f%%), FT2: %.2f s (%.1f%%), ES2: %.2f s (%.1f%%)\n",
           pf2_time, pf2_time / total * 100, ft2_time, ft2_time / total * 100, es2_time, es2_time / total * 100);
	double idle = total - pf1_time - ft1_time - es1_time -
        pf2_time - ft2_time - es2_time - mpi_wait_time - mpi_send_time;
	printf("mpi wait: %.2f s (%.1f%%), mpi send: %.2f s (%.1f%%), idle: %.2f s (%.1f%%)\n",
           mpi_wait_time, 100 * mpi_wait_time / omp_get_max_threads() / dt,
           mpi_send_time, 100 * mpi_send_time / omp_get_max_threads() / dt,
           idle, 100 * idle / omp_get_max_threads() / dt);

    return 0;
}

void streamer(int rank, int streamer_count, int *producer_ranks, int producer_count) {
    const int xM_yN_size = xM_size * yN_size / image_size;
    double complex *NMBF_NMBF = (double complex *)malloc(sizeof(double complex) * xM_yN_size * xM_yN_size *
                                                         producer_count);
    MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * producer_count);
    MPI_Status *statuses = (MPI_Status *)malloc(sizeof(MPI_Status) * producer_count);

    int r, p;
    uint64_t received_data = 0;
    for (r = 0; r < repeat_count; r++) {
        int i0, i1;
        for (i0 = 0; i0 < nsubgrid; i0++) {
            for (i1 = 0; i1 < nsubgrid; i1++) {
                int target_rank = (i1 + i0 * nsubgrid) % streamer_count;
                if (target_rank == rank) {

                    for (p = 0; p < producer_count; p++) {
                        MPI_Irecv(NMBF_NMBF, xM_yN_size * xM_yN_size, MPI_DOUBLE_COMPLEX,
                                  producer_ranks[p], 0, MPI_COMM_WORLD, requests + p);
                        received_data += sizeof(double complex) * xM_yN_size * xM_yN_size;
                    }

                    MPI_Waitall(producer_count, requests, statuses);
                }
            }
        }
    }

    printf("Received %.2f GB\n", (double)received_data / 1000000000);
}

int main(int argc, char *argv[]) {
	int thread_support, rank, size;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (thread_support < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Need full thread support from MPI!\n");
        return 1;
    }

    fftw_import_wisdom_from_filename("recombine.wisdom");

    // Local run?
    if (size == 1) {
        printf("Role: Single\n");
        producer(0, 0);

    } else {

        // Determine number of producers and streamers (pretty arbitrary for now)
        int producer_count = size / 2;
        int streamer_count = size - producer_count;
        int i;

        if (rank < producer_count) {
            printf("Role: Producer %d\n", rank);

            int *streamer_ranks = (int *)malloc(sizeof(int) * streamer_count);
            for (i = 0; i < streamer_count; i++) {
                streamer_ranks[i] = producer_count + i;
            }

            producer(streamer_ranks, streamer_count);

        } else if (rank - producer_count < streamer_count) {
            printf("Role: Streamer %d\n", rank - producer_count);

            int *producer_ranks = (int *)malloc(sizeof(int) * producer_count);
            for (i = 0; i < producer_count; i++) {
                producer_ranks[i] = i;
            }

            streamer(rank-producer_count, streamer_count, producer_ranks, producer_count);
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
