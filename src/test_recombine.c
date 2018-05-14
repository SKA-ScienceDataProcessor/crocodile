
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

void *read_dump(int size, char *name, ...) {
    va_list ap;
    va_start(ap, name);
    char fname[256];
    vsnprintf(fname, 256, name, ap);
    int fd = open(fname, O_RDONLY, 0666);
    char *data = malloc(size);
    if (read(fd, data, size) != size) {
        fprintf(stderr, "failed to read enough data from %s!\n", fname);
        return 0;
    }
    close(fd);
    return data;
}

int write_dump(void *data, int size, char *name, ...) {
    va_list ap;
    va_start(ap, name);
    char fname[256];
    vsnprintf(fname, 256, name, ap);
    int fd = open(fname, O_CREAT | O_TRUNC | O_WRONLY, 0666);
    if (write(fd, data, size) != size) {
        fprintf(stderr, "failed to write data to %s!\n", fname);
        close(fd);
        return 1;
    }
    close(fd);
    return 0;
}

int T01_generate_m() {

    // Size specifications
    int image_size = 2000;
    int yN_size = 480;
    int yP_size = 900;
    int xM_size = 500;
    int xMxN_yP_size = 247;

    double *pswf = read_dump(sizeof(double) * yN_size, "../data/grid/T01_pswf.in");
    double *m_trunc_ref = read_dump(xMxN_yP_size * sizeof(double), "../data/grid/T01_m_trunc.in");
    double *m_trunc_ref_a = read_dump(yP_size * sizeof(double), "../data/grid/T01a_m_trunc.in");
    double *m_trunc_ref_b = read_dump(yP_size / 2 * sizeof(double), "../data/grid/T01b_m_trunc.in");
    if (!pswf || !m_trunc_ref) return 1;
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    double *m_trunc_a = generate_m(image_size, yP_size, yN_size, xM_size, yP_size, pswf);
    double *m_trunc_b = generate_m(image_size, yP_size, yN_size, xM_size, yP_size / 2, pswf);
    write_dump(m_trunc, xMxN_yP_size * sizeof(double), "../data/grid/T01_m_trunc.out");
    write_dump(m_trunc, yP_size * sizeof(double), "../data/grid/T01a_m_trunc.out");
    for (int i = 0; i < xMxN_yP_size; i++)
        assert(fabs(m_trunc[i] - m_trunc_ref[i]) < 1e-12);
    for (int i = 0; i < yP_size; i++)
        assert(fabs(m_trunc_a[i] - m_trunc_ref_a[i]) < 1e-12);
    for (int i = 0; i < yP_size / 2; i++)
        assert(fabs(m_trunc_b[i] - m_trunc_ref_b[i]) < 1e-12);
    free(pswf); free(m_trunc_ref); free(m_trunc); free(m_trunc_ref_a); free(m_trunc_a);

    return 0;
}

struct timespec measure_ts;
uint64_t measure_time = 0;
void measure_start() {
    clock_gettime(CLOCK_REALTIME, &measure_ts);
}
void measure_end() {
    struct timespec ts2;
    clock_gettime(CLOCK_REALTIME, &ts2);
    uint64_t t = (uint64_t)(measure_ts.tv_sec) * 1000000000 + measure_ts.tv_nsec;
    uint64_t t2 = (uint64_t)(ts2.tv_sec) * 1000000000 + ts2.tv_nsec;
    measure_time += t2 - t;
}

int T02_extract_subgrid() {

    // Size specifications (xMxN_yP_size odd/even)
    int image_size = 2000;
    int yB_size = 400;
    int yN_size = 480;
    int yP_size = 900; int yP_size_b = 920;
    int xM_size = 500;
    int xA_yP_size = 180; int xA_yP_size_b = 184;
    int xM_yP_size = 225; int xM_yP_size_b = 230;
    int xMxN_yP_size = 247; int xMxN_yP_size_b = 252;
    int xM_yN_size = 120;
    int nsubgrid = 5;

    double *pswf = read_dump(sizeof(double) * yN_size, "../data/grid/T02_pswf.in");
    double complex *facet = read_dump(sizeof(double complex) * yB_size, "../data/grid/T02_facet.in");
    double complex *bf_ref = read_dump(sizeof(double complex) * yP_size, "../data/grid/T02_bf.in");
    double complex *bf_ref_b = read_dump(sizeof(double complex) * yP_size_b, "../data/grid/T02b_bf.in");
    if (!pswf || !facet || !bf_ref || !bf_ref_b) {
        free(pswf); free(facet); free(bf_ref); free(bf_ref_b);
        return 1;
    }

    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    double *m_trunc_b = generate_m(image_size, yP_size_b, yN_size, xM_size, xMxN_yP_size_b, pswf);
    double *Fb = generate_Fb(yN_size, yB_size, pswf);
    double *Fn = generate_Fn(yN_size, xM_yN_size, pswf);
    free(pswf);

    // Test facet preparation
    double complex *bf = (double complex *)malloc(sizeof(double complex) * yP_size);
    double complex *bf_b = (double complex *)malloc(sizeof(double complex) * yP_size_b);
    prepare_facet(yB_size, yP_size, Fb, facet, 1, bf, 1);
    prepare_facet(yB_size, yP_size_b, Fb, facet, 1, bf_b, 1);
    fftw_plan bf_plan = fftw_plan_dft_1d(yP_size, bf, bf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan bf_plan_b = fftw_plan_dft_1d(yP_size_b, bf_b, bf_b, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bf_plan); fftw_execute(bf_plan_b);
    fftw_free(bf_plan); fftw_free(bf_plan_b);
    write_dump(bf, sizeof(double complex) * yP_size, "../data/grid/T02_bf.out");
    write_dump(bf_b, sizeof(double complex) * yP_size_b, "../data/grid/T02_bf_b.out");
    for (int y = 0; y < yP_size; y++)
        assert(fabs(bf[y] - bf_ref[y]) < 1e-12);
    for (int y = 0; y < yP_size_b; y++)
        assert(fabs(bf_b[y] - bf_ref_b[y]) < 1e-12);

    // Test subgrid extraction
    double complex *mbf = (double complex *)malloc(sizeof(double complex) * xM_yP_size_b);
    fftw_plan mbf_plan = fftw_plan_dft_1d(xM_yP_size, mbf, mbf, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan mbf_plan_b = fftw_plan_dft_1d(xM_yP_size_b, mbf, mbf, FFTW_FORWARD, FFTW_ESTIMATE);
    double complex *nmbf = (double complex *)malloc(sizeof(double complex) * xM_yN_size);
    double complex *nmbf_b = (double complex *)malloc(sizeof(double complex) * xM_yN_size);
    for (int i = 0; i < nsubgrid; i++) {
        double complex *nmbf_ref = read_dump(sizeof(double complex) * xM_yN_size, "../data/grid/T02_nmbf%d.in", i);
        double complex *nmbf_ref_b = read_dump(sizeof(double complex) * xM_yN_size, "../data/grid/T02b_nmbf%d.in", i);
        if (!nmbf_ref || !nmbf_ref_b) {
            free(nmbf_ref); free(nmbf_ref_b);
            return 1;
        }
        extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i*xA_yP_size,
                        m_trunc, Fn, bf, 1, mbf, mbf_plan, nmbf, 1);
        extract_subgrid(yP_size_b, xM_yP_size_b, xMxN_yP_size_b, xM_yN_size, i*xA_yP_size_b,
                        m_trunc_b, Fn, bf_b, 1, mbf, mbf_plan_b, nmbf_b, 1);
        write_dump(nmbf, sizeof(double complex) * xM_yN_size, "../data/grid/T02_nmbf%d.out", i);
        write_dump(nmbf_b, sizeof(double complex) * xM_yN_size, "../data/grid/T02b_nmbf%d.out", i);
        // Note that because of redundant data, the result actually changes when
        // we alter intermediate array sizes. However, those would still lead
        // to the same sub-grids after reassembly, to the accuracy of the approximation!
        for (int y = 0; y < xM_yN_size; y++) {
            assert(fabs(nmbf[y] - nmbf_ref[y]) < 2e-10);
            assert(fabs(nmbf_b[y] - nmbf_ref_b[y]) < 2e-10);
        }
        free(nmbf_ref); free(nmbf_ref_b);
    }

    fftw_free(mbf_plan); fftw_free(mbf_plan_b);
    free(facet); free(bf_ref); free(m_trunc); free(m_trunc_b); free(Fb); free(Fn);
    free(bf); free(mbf); free(nmbf); free(bf_b); free(nmbf_b);

    return 0;
}


int T03_add_subgrid() {

    // Size specifications
    int image_size = 2000;
    int yB_size = 400;
    int yN_size = 480;
    int yP_size = 900;
    int xM_size = 500;
    int xA_size = 400;
    int xA_yP_size = 180;
    int xM_yP_size = 225;
    int xMxN_yP_size = 247;
    int xM_yN_size = 120;
    int xM_yB_size = 100;
    const int nfacet = 5;
    const int nsubgrid = 5;

    // Read PSWF, generate Fn, Fb and m
    double *pswf = read_dump(sizeof(double) * yN_size, "../data/grid/T03_pswf.in");
    if (!pswf) return 1;
    double *Fb = generate_Fb(yN_size, yB_size, pswf);
    double *Fn = generate_Fn(yN_size, xM_yN_size, pswf);
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    free(pswf);

    // Extract all subgrids from all facets
    int i, j;
    double complex *NMBF[nsubgrid][nfacet];
    double complex *BF = (double complex *)malloc(sizeof(double complex) * yP_size);
    fftw_plan BF_plan = fftw_plan_dft_1d(yP_size, BF, BF, FFTW_BACKWARD, FFTW_ESTIMATE);
    double complex *MBF = (double complex *)malloc(sizeof(double complex) * xM_yP_size);
    fftw_plan MBF_plan = fftw_plan_dft_1d(xM_yP_size, MBF, MBF, FFTW_FORWARD, FFTW_MEASURE);
    for (j = 0; j < nfacet; j++) {
        double complex *facet = read_dump(yB_size * sizeof(double complex), "../data/grid/T03_facet%d.in", j);
        if (!facet) return 1;
        prepare_facet(yB_size, yP_size, Fb, facet, 1, BF, 1); fftw_execute(BF_plan);
        free(facet);
        for (i = 0; i < nsubgrid; i++) {
            NMBF[i][j] = (double complex *)malloc(sizeof(double complex) * xM_yN_size);
            extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i*xA_yP_size,
                            m_trunc, Fn, BF, 1, MBF, MBF_plan, NMBF[i][j], 1);
        }
    }
    free(BF); free(MBF); fftw_free(BF_plan); fftw_free(MBF_plan);

    // Recombine subgrids
    double complex *subgrid = (double complex *)malloc(xM_size * sizeof(double complex));
    fftw_plan subgrid_plan = fftw_plan_dft_1d(xM_size, subgrid, subgrid, FFTW_BACKWARD, FFTW_MEASURE);
    for (i = 0; i < nsubgrid; i++) {
        double complex *subgrid_ref = read_dump(xA_size * sizeof(double complex), "../data/grid/T03_subgrid%d.in", i);
        double complex *approx_ref = read_dump(xM_size * sizeof(double complex), "../data/grid/T03_approx%d.in", i);

        memset(subgrid, 0, xM_size * sizeof(double complex));
        for (j = 0; j < nfacet; j++) {
            add_subgrid(xM_size, xM_yN_size, j * xM_yB_size,
                        NMBF[i][j], subgrid);
            free(NMBF[i][j]);
        }
        write_dump(subgrid, xM_size * sizeof(double complex), "../data/grid/T03_approx%d.out", i);
        for (int y = 0; y < xM_size; y++) {
            assert(fabs(subgrid[y] * xM_size - approx_ref[y]) < 2e-10);
        }

        fftw_execute(subgrid_plan);
        int y;
        for (y = 0; y < xA_size / 2; y++) {
            assert(fabs(subgrid[y] - subgrid_ref[y]) < 5e-7);
        }
        for (; y < xA_size; y++) {
            assert(fabs(subgrid[y+xM_size-xA_size] - subgrid_ref[y]) < 5e-7);
        }

        free(subgrid_ref); free(approx_ref);
    }

    free(Fb); free(Fn); free(m_trunc); free(subgrid); fftw_free(subgrid_plan);

    return 0;
}

int T04_test_2d() {

    // Size specifications
    int image_size = 2000;
    int yB_size = 400;
    int yN_size = 480;
    int yP_size = 900;
    int xM_size = 500;
    //int xA_size = 400;
    int xA_yP_size = 180;
    int xM_yP_size = 225;
    int xMxN_yP_size = 247;
    int xM_yN_size = 120;
    //int xM_yB_size = 100;
    const int nfacet = 5;
    const int nsubgrid = 5;

    // Read PSWF, generate Fn, Fb and m
    double *pswf = read_dump(sizeof(double) * yN_size, "../data/grid/T04_pswf.in");
    if (!pswf) return 1;
    double *Fb = generate_Fb(yN_size, yB_size, pswf);
    double *Fn = generate_Fn(yN_size, xM_yN_size, pswf);
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    free(pswf);

    double complex *NMBF_NMBF[nsubgrid][nsubgrid][nfacet][nfacet];
    uint64_t NMBF_size = 0;
    int j0,j1;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    measure_time = 0;

    double complex *BF = (double complex *)calloc(sizeof(double complex), yP_size * yB_size);
    double complex *MBF = (double complex *)calloc(sizeof(double complex), xM_yP_size);
    double complex *NMBF = (double complex *)calloc(sizeof(double complex), xM_yN_size * yB_size);
    double complex *NMBF_BF = (double complex *)calloc(sizeof(double complex), xM_yN_size * yP_size);

    int F_stride0 = 1, F_stride1 = yB_size;
    int BF_stride0 = yP_size, BF_stride1 = 1;
    int NMBF_stride0 = xM_yN_size, NMBF_stride1 = 1;
    int NMBF_BF_stride0 = 1, NMBF_BF_stride1 = yP_size;
    int NMBF_NMBF_stride0 = 1, NMBF_NMBF_stride1 = xM_yN_size;
    const int check_outputs = 0;

    printf("Planning..."); fflush(stdout);
    fftw_plan BF_plan = fftw_plan_many_dft(1, &yP_size, yB_size,
                                           BF, 0, BF_stride1, BF_stride0,
                                           BF, 0, BF_stride1, BF_stride0,
                                           FFTW_BACKWARD, FFTW_MEASURE);
    fftw_plan MBF_plan = fftw_plan_dft_1d(xM_yP_size, MBF, MBF, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan NMBF_BF_plan = fftw_plan_many_dft(1, &yP_size, xM_yN_size,
                                                NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                                NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                                FFTW_BACKWARD, FFTW_MEASURE);
    struct timespec ts2;
    clock_gettime(CLOCK_REALTIME, &ts2);
    printf(" %.2f s\n",
           (double)ts2.tv_sec + (double)ts2.tv_nsec / 1000000000 -
           (double)ts.tv_sec - (double)ts.tv_nsec / 1000000000);

    double complex *facet[nfacet][nfacet];
    for (j0 = 0; j0 < nfacet; j0++) for (j1 = 0; j1 < nfacet; j1++) {

        facet[j0][j1] = read_dump(sizeof(double complex) * yB_size * yB_size,
                                  "../data/grid/T04_facet%d%d.in", j0, j1);
    }

    clock_gettime(CLOCK_REALTIME, &ts);
    measure_time = 0;

    for (int r = 0; r < 1; r++) {

    for (j0 = 0; j0 < nfacet; j0++) for (j1 = 0; j1 < nfacet; j1++) {

        printf("%d/%d ", j0,j1); fflush(stdout);

        for (int x = 0; x < yB_size; x++) {
            prepare_facet(yB_size, yP_size, Fb,
                          facet[j0][j1]+x*F_stride0, F_stride1,
                          BF+x*BF_stride0, BF_stride1);
        }
        fftw_execute(BF_plan);

        int i0;
        for (i0 = 0; i0 < nsubgrid; i0++) {

            for (int x = 0; x < yB_size; x++) {
                extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i0*xA_yP_size, m_trunc, Fn,
                                BF+x*BF_stride0, BF_stride1,
                                MBF, MBF_plan,
                                NMBF+x*NMBF_stride0, NMBF_stride1);
            }

            for (int y = 0; y < xM_yN_size; y++) {
                prepare_facet(yB_size, yP_size, Fb,
                              NMBF+y*NMBF_stride1, NMBF_stride0,
                              NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0);
            }
            fftw_execute(NMBF_BF_plan);

            int i1;
            for (i1 = 0; i1 < nsubgrid; i1++) {

                NMBF_NMBF[i0][i1][j0][j1] = (double complex *)malloc(sizeof(double complex) * xM_yN_size * xM_yN_size);
                NMBF_size += sizeof(double complex) * xM_yN_size * xM_yN_size;

                for (int y = 0; y < xM_yN_size; y++) {
                    extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i1*xA_yP_size, m_trunc, Fn,
                                    NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0,
                                    MBF, MBF_plan,
                                    NMBF_NMBF[i0][i1][j0][j1]+y*NMBF_NMBF_stride1, NMBF_NMBF_stride0);
                }

				//write_dump(NMBF_NMBF[i0][i1][j0][j1], sizeof(double complex) * xM_yN_size * xM_yN_size,
				//		   "../data/grid/T04_nmbf%d%d%d%d.out", i0, i1, j0, j1);
				double complex *ref = read_dump(sizeof(double complex) * xM_yN_size * xM_yN_size,
												"../data/grid/T04_nmbf%d%d%d%d.in", i0, i1, j0, j1);
				for (int y = 0; y < xM_yN_size * xM_yN_size; y++)
					assert(fabs(NMBF_NMBF[i0][i1][j0][j1][y] - ref[y]) < 3e-8);

                free(NMBF_NMBF[i0][i1][j0][j1]);

            }
        }

    }

    }

    free(NMBF);
    free(NMBF_BF);
    free(BF);

    clock_gettime(CLOCK_REALTIME, &ts2);
    uint64_t t = (uint64_t)(ts.tv_sec) * 1000000000 + ts.tv_nsec;
    uint64_t t2 = (uint64_t)(ts2.tv_sec) * 1000000000 + ts2.tv_nsec;
    double dt = (double)(t2 - t)/1000000000;
    printf("\n%.2f s, %.2f s, %.1f MB, %.2f MB/s\n", dt,
           (double)measure_time / 1000000000,
           (double)NMBF_size / 1000000,
           (double)NMBF_size / dt / 1000000);

    return 0;
}

int run_tests() {

    int count = 0,fails = 0;
#define RUN_TEST(t) count++; printf(#t"..."); fflush(stdout); if (!t()) puts(" ok"); else { puts("failed"); fails++; }
    RUN_TEST(T01_generate_m);
    RUN_TEST(T02_extract_subgrid);
    RUN_TEST(T03_add_subgrid);
    RUN_TEST(T04_test_2d);
#undef RUN_TEST
    printf(" *** %d/%d tests passed ***\n", count-fails, count);
    return fails;
}

int main(int argc, char *argv[]) {

    // Test facets
    fftw_import_wisdom_from_filename("recombine.wisdom");

    run_tests();

    fftw_export_wisdom_to_filename("recombine.wisdom");
    return 0;
}
