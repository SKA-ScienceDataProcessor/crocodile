
#include "grid.h"
#include "recombine.h"
#include "hdf5.h"

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

int T01_generate_m() {

    // Size specifications
    int image_size = 2000;
    int yN_size = 480;
    int yP_size = 900;
    int xM_size = 500;
    int xMxN_yP_size = 247;

    double *pswf = read_dump(sizeof(double) * yN_size, "../../data/grid/T01_pswf.in");
    double *m_trunc_ref = read_dump(xMxN_yP_size * sizeof(double), "../../data/grid/T01_m_trunc.in");
    double *m_trunc_ref_a = read_dump(yP_size * sizeof(double), "../../data/grid/T01a_m_trunc.in");
    double *m_trunc_ref_b = read_dump(yP_size / 2 * sizeof(double), "../../data/grid/T01b_m_trunc.in");
    if (!pswf || !m_trunc_ref) return 1;
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    double *m_trunc_a = generate_m(image_size, yP_size, yN_size, xM_size, yP_size, pswf);
    double *m_trunc_b = generate_m(image_size, yP_size, yN_size, xM_size, yP_size / 2, pswf);
    write_dump(m_trunc, xMxN_yP_size * sizeof(double), "../../data/grid/T01_m_trunc.out");
    write_dump(m_trunc, yP_size * sizeof(double), "../../data/grid/T01a_m_trunc.out");
    int i;
    for (i = 0; i < xMxN_yP_size; i++)
        assert(fabs(m_trunc[i] - m_trunc_ref[i]) < 1e-14);
    for (i = 0; i < yP_size; i++)
        assert(fabs(m_trunc_a[i] - m_trunc_ref_a[i]) < 1e-14);
    for (i = 0; i < yP_size / 2; i++)
        assert(fabs(m_trunc_b[i] - m_trunc_ref_b[i]) < 1e-14);
    free(pswf); free(m_trunc_ref); free(m_trunc); free(m_trunc_ref_a); free(m_trunc_a);

    return 0;
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

    double *pswf = read_dump(sizeof(double) * yN_size, "../../data/grid/T02_pswf.in");
    double complex *facet = read_dump(sizeof(double complex) * yB_size, "../../data/grid/T02_facet.in");
    double complex *bf_ref = read_dump(sizeof(double complex) * yP_size, "../../data/grid/T02_bf.in");
    double complex *bf_ref_b = read_dump(sizeof(double complex) * yP_size_b, "../../data/grid/T02b_bf.in");
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
    write_dump(bf, sizeof(double complex) * yP_size, "../../data/grid/T02_bf.out");
    write_dump(bf_b, sizeof(double complex) * yP_size_b, "../../data/grid/T02_bf_b.out");
    int y;
    for (y = 0; y < yP_size; y++)
        assert(fabs(bf[y] - bf_ref[y]) < 1e-12);
    for (y = 0; y < yP_size_b; y++)
        assert(fabs(bf_b[y] - bf_ref_b[y]) < 1e-12);

    // Test subgrid extraction
    double complex *mbf = (double complex *)malloc(sizeof(double complex) * xM_yP_size_b);
    fftw_plan mbf_plan = fftw_plan_dft_1d(xM_yP_size, mbf, mbf, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan mbf_plan_b = fftw_plan_dft_1d(xM_yP_size_b, mbf, mbf, FFTW_FORWARD, FFTW_ESTIMATE);
    double complex *nmbf = (double complex *)malloc(sizeof(double complex) * xM_yN_size);
    double complex *nmbf_b = (double complex *)malloc(sizeof(double complex) * xM_yN_size);
    int i;
    for (i = 0; i < nsubgrid; i++) {
        double complex *nmbf_ref = read_dump(sizeof(double complex) * xM_yN_size, "../../data/grid/T02_nmbf%d.in", i);
        double complex *nmbf_ref_b = read_dump(sizeof(double complex) * xM_yN_size, "../../data/grid/T02b_nmbf%d.in", i);
        if (!nmbf_ref || !nmbf_ref_b) {
            free(nmbf_ref); free(nmbf_ref_b);
            return 1;
        }
        extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i*xA_yP_size,
                        m_trunc, Fn, bf, 1, mbf, mbf_plan, nmbf, 1);
        extract_subgrid(yP_size_b, xM_yP_size_b, xMxN_yP_size_b, xM_yN_size, i*xA_yP_size_b,
                        m_trunc_b, Fn, bf_b, 1, mbf, mbf_plan_b, nmbf_b, 1);
        write_dump(nmbf, sizeof(double complex) * xM_yN_size, "../../data/grid/T02_nmbf%d.out", i);
        write_dump(nmbf_b, sizeof(double complex) * xM_yN_size, "../../data/grid/T02b_nmbf%d.out", i);
        // Note that because of redundant data, the result actually changes when
        // we alter intermediate array sizes. However, those would still lead
        // to the same sub-grids after reassembly, to the accuracy of the approximation!
        for (y = 0; y < xM_yN_size; y++) {
            assert(fabs(nmbf[y] - nmbf_ref[y]) < 2e-11);
            assert(fabs(nmbf_b[y] - nmbf_ref_b[y]) < 2e-11);
        }
        free(nmbf_ref); free(nmbf_ref_b);
    }

    fftw_free(mbf_plan); fftw_free(mbf_plan_b);
    free(facet); free(bf_ref); free(m_trunc); free(m_trunc_b); free(Fb); free(Fn);
    free(bf); free(mbf); free(nmbf); free(bf_b); free(nmbf_b);

    return 0;
}


int T03_add_facet() {

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
    double *pswf = read_dump(sizeof(double) * yN_size, "../../data/grid/T03_pswf.in");
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
    fftw_plan MBF_plan = fftw_plan_dft_1d(xM_yP_size, MBF, MBF, FFTW_FORWARD, FFTW_ESTIMATE);
    for (j = 0; j < nfacet; j++) {
        double complex *facet = read_dump(yB_size * sizeof(double complex), "../../data/grid/T03_facet%d.in", j);
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
    fftw_plan subgrid_plan = fftw_plan_dft_1d(xM_size, subgrid, subgrid, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (i = 0; i < nsubgrid; i++) {
        double complex *subgrid_ref = read_dump(xA_size * sizeof(double complex), "../../data/grid/T03_subgrid%d.in", i);
        double complex *approx_ref = read_dump(xM_size * sizeof(double complex), "../../data/grid/T03_approx%d.in", i);

        memset(subgrid, 0, xM_size * sizeof(double complex));
        for (j = 0; j < nfacet; j++) {
            add_facet(xM_size, xM_yN_size, j * xM_yB_size,
                        NMBF[i][j], 1, subgrid, 1);
            free(NMBF[i][j]);
        }
        write_dump(subgrid, xM_size * sizeof(double complex), "../../data/grid/T03_approx%d.out", i);
        int y;
        for (y = 0; y < xM_size; y++) {
            assert(fabs(subgrid[y] * xM_size - approx_ref[y]) < 3e-11);
        }

        fftw_execute(subgrid_plan);
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
    const int nfacet = 3;
    const int nsubgrid = 3;

    // Read PSWF, generate Fn, Fb and m
    double *pswf = read_dump(sizeof(double) * yN_size, "../../data/grid/T04_pswf.in");
    if (!pswf) return 1;
    double *Fb = generate_Fb(yN_size, yB_size, pswf);
    double *Fn = generate_Fn(yN_size, xM_yN_size, pswf);
    double *m_trunc = generate_m(image_size, yP_size, yN_size, xM_size, xMxN_yP_size, pswf);
    free(pswf);

    double complex *facet[nfacet][nfacet];
    double complex *BF = (double complex *)malloc(sizeof(double complex) * yP_size * yB_size);
    double complex *MBF = (double complex *)malloc(sizeof(double complex) * xM_yP_size);
    double complex *NMBF = (double complex *)malloc(sizeof(double complex) * xM_yN_size * yB_size);
    double complex *NMBF_BF = (double complex *)malloc(sizeof(double complex) * xM_yN_size * yP_size);
    double complex *NMBF_NMBF = (double complex *)malloc(sizeof(double complex) * xM_yN_size * xM_yN_size);

    int F_stride0 = yB_size, F_stride1 = 1;
    int BF_stride0 = yP_size, BF_stride1 = 1;
    int NMBF_stride0 = xM_yN_size, NMBF_stride1 = 1;
    int NMBF_BF_stride0 = 1, NMBF_BF_stride1 = yP_size;
    int NMBF_NMBF_stride0 = xM_yN_size, NMBF_NMBF_stride1 = 1;

    fftw_plan BF_plan = fftw_plan_many_dft(1, &yP_size, yB_size,
                                           BF, 0, BF_stride1, BF_stride0,
                                           BF, 0, BF_stride1, BF_stride0,
                                           FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan MBF_plan = fftw_plan_dft_1d(xM_yP_size, MBF, MBF, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan NMBF_BF_plan = fftw_plan_many_dft(1, &yP_size, xM_yN_size,
                                                NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                                NMBF_BF, 0, NMBF_BF_stride0, NMBF_BF_stride1,
                                                FFTW_BACKWARD, FFTW_ESTIMATE);
    int j0,j1,i0,i1,x,y;
    for (j0 = 0; j0 < nfacet; j0++) for (j1 = 0; j1 < nfacet; j1++) {

        facet[j0][j1] = read_dump(sizeof(double complex) * yB_size * yB_size,
                                  "../../data/grid/T04_facet%d%d.in", j0, j1);
        for (x = 0; x < yB_size; x++) {
            prepare_facet(yB_size, yP_size, Fb,
                          facet[j0][j1]+x*F_stride0, F_stride1,
                          BF+x*BF_stride0, BF_stride1);
        }
        fftw_execute(BF_plan);

        for (i1 = 0; i1 < nsubgrid; i1++) {

            for (x = 0; x < yB_size; x++) {
                extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i1*xA_yP_size, m_trunc, Fn,
                                BF+x*BF_stride0, BF_stride1,
                                MBF, MBF_plan,
                                NMBF+x*NMBF_stride0, NMBF_stride1);
            }

            for (y = 0; y < xM_yN_size; y++) {
                prepare_facet(yB_size, yP_size, Fb,
                              NMBF+y*NMBF_stride1, NMBF_stride0,
                              NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0);
            }
            fftw_execute(NMBF_BF_plan);

            for (i0 = 0; i0 < nsubgrid; i0++) {

                for (y = 0; y < xM_yN_size; y++) {
                    extract_subgrid(yP_size, xM_yP_size, xMxN_yP_size, xM_yN_size, i0*xA_yP_size, m_trunc, Fn,
                                    NMBF_BF+y*NMBF_BF_stride1, NMBF_BF_stride0,
                                    MBF, MBF_plan,
                                    NMBF_NMBF+y*NMBF_NMBF_stride1, NMBF_NMBF_stride0);
                }

                write_dump(NMBF_NMBF, sizeof(double complex) * xM_yN_size * xM_yN_size,
                           "../../data/grid/T04_nmbf%d%d%d%d.out", i0, i1, j0, j1);
                double complex *ref = read_dump(sizeof(double complex) * xM_yN_size * xM_yN_size,
                                                "../../data/grid/T04_nmbf%d%d%d%d.in",
                                                i0, i1, j0, j1);
                if (!ref) { free(BF); free(NMBF); free(NMBF_BF); free(NMBF_NMBF); return 1; }

                for (y = 0; y < xM_yN_size * xM_yN_size; y++)
                    assert(fabs(NMBF_NMBF[y] - ref[y]) < 2e-9);
                free(ref);
            }
        }

        free(facet[j0][j1]);

    }

    free(BF); free(NMBF); free(NMBF_BF); free(NMBF_NMBF);

    return 0;
}

// Same as last test, but using recombine2d machinery
int T04a_recombine2d() {

    const int nfacet = 3;
    const int nsubgrid = 3;
    const int BF_batch = 16;

    struct recombine2d_config cfg;
    if (!recombine2d_set_config(&cfg, 2000, 100,
                                "../../data/grid/T04_pswf.in",
                                400, 480, 900, 400, 500, 247))
        return 1;

    double complex *BF = (double complex *)malloc(cfg.BF_size);
    double complex *NMBF_NMBF = (double complex *)malloc(cfg.NMBF_NMBF_size);

    struct recombine2d_worker worker;
    fftw_plan BF_plan = recombine2d_bf_plan(&cfg, BF_batch, BF, FFTW_ESTIMATE);
    recombine2d_init_worker(&worker, &cfg, BF_batch, BF_plan, FFTW_ESTIMATE);

    int j0, j1; int ret = 0;
    for (j0 = 0; j0 < nfacet; j0++) for(j1 = 0; j1 < nfacet; j1++) {

        double complex *facet = read_dump(cfg.F_size, "../../data/grid/T04_facet%d%d.in", j0, j1);

        recombine2d_pf1_ft1_omp(&worker, facet, BF);
        int i0, i1;
        for (i1 = 0; i1 < nsubgrid; i1++) {
            recombine2d_es1_pf0_ft0(&worker, i1*cfg.xA_size, BF, worker.NMBF_BF);
            for (i0 = 0; i0 < nsubgrid; i0++) {
                recombine2d_es0(&worker, i0*cfg.xA_size, i1*cfg.xA_size, worker.NMBF_BF, NMBF_NMBF);

                double complex *ref = read_dump(cfg.NMBF_NMBF_size,
                                                "../../data/grid/T04_nmbf%d%d%d%d.in",
                                                i0, i1, j0, j1);
                if (!ref) { ret = 1; break; }
                int y;
                for (y = 0; y < cfg.xM_yN_size * cfg.xM_yN_size; y++)
                    assert(fabs(NMBF_NMBF[y] - ref[y]) < 2e-9);
                free(ref);
            }
            if(ret) break;
        }

        free(facet);
        if(ret) break;
    }

    recombine2d_free_worker(&worker);
    recombine2d_free(&cfg);
    free(BF); free(NMBF_NMBF);

    return ret;
}

int T05_frac_coord()
{
    double cs[] = {
       -2.    , -1.9375, -1.875 , -1.8125, -1.75  , -1.6875, -1.625 , -1.5625, -1.5   ,
       -1.4375, -1.375 , -1.3125, -1.25  , -1.1875, -1.125 , -1.0625, -1.    , -0.9375,
       -0.875 , -0.8125, -0.75  , -0.6875, -0.625 , -0.5625, -0.5   , -0.4375, -0.375 ,
       -0.3125, -0.25  , -0.1875, -0.125 , -0.0625,  0.    ,  0.0625,  0.125 ,  0.1875,
       0.25   , 0.3125 , 0.375  , 0.4375 , 0.5    , 0.5625 , 0.625  , 0.6875 , 0.75   ,
       0.8125 , 0.875  , 0.9375 , 1.     , 1.0625 , 1.125  , 1.1875 , 1.25   , 1.3125 ,
       1.375  , 1.4375 , 1.5    , 1.5625 , 1.625  , 1.6875 , 1.75   , 1.8125 , 1.875  ,
       1.9375
    };
    int cs_count = sizeof(cs) / sizeof(*cs);

    // Note the 3/5 pattern with ref_fx due to 0.5 getting rounded
    // towards even numbers (banker's rounding).
    double ref_x[] = {
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    double ref_fx[] = {
        0, 0, 0, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2,
        2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0,
        0, 0, 0, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0 };

    int i, x, fx;
    int grid_size = 4, oversample = 4;
    for (i = 0; i < cs_count; i++) {
        frac_coord(grid_size, oversample, cs[i], &x, &fx);
        assert(x == ref_x[i]);
        assert(fx == ref_fx[i]);
    }

    return 0;
}

int T05_degrid()
{

    const int nfacet = 4;
    const int nsubgrid = 4;
    const int BF_batch = 16;

    char *in_file = "../../data/grid/T05_in.h5";

    struct sep_kernel_data kern;
    if (load_sep_kern(in_file, &kern)) {
        return 1;
    }

    struct recombine2d_config cfg;
    if (!recombine2d_set_config(&cfg, 512, 128,
                                "../../data/grid/T05_pswf.in",
                                128, 140, 216, 128, 256, 136)) {
        return 1;
    }

    double complex *BF = (double complex *)malloc(cfg.BF_size);
    double complex *all_NMBF_NMBF = (double complex *)
        malloc(cfg.NMBF_NMBF_size * nfacet * nfacet * nsubgrid * nsubgrid);

    struct recombine2d_worker worker;
    fftw_plan BF_plan = recombine2d_bf_plan(&cfg, BF_batch, BF, FFTW_ESTIMATE);
    recombine2d_init_worker(&worker, &cfg, BF_batch, BF_plan, FFTW_ESTIMATE);

    int j0, j1; int ret = 0;
    for (j0 = 0; j0 < nfacet; j0++) for(j1 = 0; j1 < nfacet; j1++) {

        double complex *facet = read_hdf5(cfg.F_size, in_file, "j0=%d/j1=%d/facet", j0, j1);

        recombine2d_pf1_ft1_omp(&worker, facet, BF);
        int i0, i1;
        for (i1 = 0; i1 < nsubgrid; i1++) {
            recombine2d_es1_pf0_ft0(&worker, i1*cfg.xA_size, BF, worker.NMBF_BF);
            for (i0 = 0; i0 < nsubgrid; i0++) {
                int ix = ((i1 * nsubgrid + i0) * nfacet + j0) * nfacet + j1;
                double complex *nmbf_nmbf = all_NMBF_NMBF +
                    ix * (cfg.NMBF_NMBF_size / sizeof(double complex));
                recombine2d_es0(&worker, i0*cfg.xA_size, i1*cfg.xA_size, worker.NMBF_BF, nmbf_nmbf);

                double complex *ref = read_hdf5(cfg.NMBF_NMBF_size, in_file,
                                                "i0=%d/i1=%d/j0=%d/j1=%d/nmbf", i0, i1, j0, j1);

                if (!ref) { ret = 1; break; }
                int y;
                for (y = 0; y < cfg.xM_yN_size * cfg.xM_yN_size; y++) {
                    double scale = cabs(nmbf_nmbf[y]);
                    if (scale < 1e-6) scale = 1e-6;
                    assert(cabs(nmbf_nmbf[y] - ref[y]) / scale < 3.5e-7);
                }
                free(ref);
            }
            if(ret) break;
        }

        free(facet);
        if(ret) break;
    }

    recombine2d_free_worker(&worker);

    double complex *subgrid = (double complex *)malloc(cfg.SG_size);
    fftw_plan subgrid_plan = fftw_plan_dft_2d(cfg.xM_size, cfg.xM_size,
                                              subgrid, subgrid, FFTW_BACKWARD, FFTW_ESTIMATE);

    int i0, i1;
    for (i0 = 0; i0 < nsubgrid; i0++) for (i1 = 0; i1 < nsubgrid; i1++) {

        // Accumulate contributions to this subgrid
        memset(subgrid, 0, cfg.SG_size);
        int j0, j1;
        for (j0 = 0; j0 < nfacet; j0++) for (j1 = 0; j1 < nfacet; j1++) {

            int ix = ((i1 * nsubgrid + i0) * nfacet + j0) * nfacet + j1;
            double complex *nmbf_nmbf = all_NMBF_NMBF +
                ix * (cfg.NMBF_NMBF_size / sizeof(double complex));

            recombine2d_af0_af1(&cfg, subgrid, j0 * cfg.yB_size, j1 * cfg.yB_size, nmbf_nmbf);

        }

        // Perform Fourier transform
        fftw_execute(subgrid_plan);

        // Check result
        double complex *approx_ref = read_hdf5(cfg.SG_size, in_file,
                                               "i0=%d/i1=%d/approx", i0, i1);
        int y;
        for (y = 0; y < cfg.xM_size * cfg.xM_size; y++)
            assert(cabs(subgrid[y] - approx_ref[y]) / cabs(subgrid[y]) < 2.11e-7);
        free(approx_ref);

        // Read visibilities, set up baseline data
        int nvis = get_npoints_hdf5(in_file, "i0=%d/i1=%d/vis", i0, i1);
        double *uvw = read_hdf5(3 * sizeof(double) * nvis, in_file,
                                "i0=%d/i1=%d/uvw", i0, i1);
        double *uvw_sg = read_hdf5(3 * sizeof(double) * nvis, in_file,
                                   "i0=%d/i1=%d/uvw_subgrid", i0, i1);
        int vis_size = sizeof(double complex) * nvis;
        double complex *vis = read_hdf5(vis_size, in_file,
                                        "i0=%d/i1=%d/vis", i0, i1);

        struct bl_data bl;
        bl.antenna1 = bl.antenna2 = 0;
        bl.time_count = nvis;
        bl.freq_count = 1;
        double freq[] = { c }; // 1 m wavelength
        bl.freq = freq;
        bl.vis = (double complex *)calloc(1, vis_size);

        // Degrid and compare
        fft_shift(subgrid, cfg.xM_size);
        bl.uvw_m = uvw_sg;
        degrid_conv_bl(subgrid, cfg.xM_size, cfg.image_size, 0, 0,
                       -cfg.xM_size, cfg.xM_size, -cfg.xM_size, cfg.xM_size,
                       &bl, 0, nvis, 0, 1, &kern);
        for (y = 0; y < nvis; y++) {
            assert(cabs(vis[y] - bl.vis[y]) < 4e-7);
        }

        // Degrid with shift and compare - doesn't quite work for
        // subgrids overlapping the border
        double dv = (double)((i0 + nsubgrid/2) % nsubgrid - nsubgrid/2) / nsubgrid;
        double du = (double)((i1 + nsubgrid/2) % nsubgrid - nsubgrid/2) / nsubgrid;
        if (fabs(dv) < 0.5 && fabs(du) < 0.5) {
            bl.uvw_m = uvw;
            degrid_conv_bl(subgrid, cfg.xM_size, cfg.image_size, du, dv,
                           -cfg.xM_size, cfg.xM_size, -cfg.xM_size, cfg.xM_size,
                           &bl, 0, nvis, 0, 1, &kern);
            for (y = 0; y < nvis; y++) {
                assert(cabs(vis[y] - bl.vis[y]) < 4e-7);
            }
        }
        free(uvw); free(uvw_sg); free(vis);
    }

    fftw_free(subgrid_plan);

    recombine2d_free(&cfg);
    free(BF); free(all_NMBF_NMBF);

    return ret;
}

int T05_config()
{
    struct ant_config cfg;
    load_ant_config("../../data/grid/VLAA_north_cfg.h5", &cfg);

    int bl_count = 2 * (cfg.ant_count * (cfg.ant_count - 1) / 2);
    double *uvw_ref = read_dump(sizeof(double) * bl_count * 3, "../../data/grid/T05_uvw.in");

    const double ha = 10 * M_PI / 180;
    const double ha_step = M_PI / 24 / 180;
    const double dec = 80 * M_PI / 180;
    int j = 0, t, a1, a2;
    for (t = 0; t < 2; t++)
        for (a1 = 0; a1 < cfg.ant_count; a1++)
            for (a2 = a1+1; a2 < cfg.ant_count; a2++) {
                double uvw[3];
                ha_to_uvw(&cfg, a1, a2, ha + t * ha_step, dec, uvw);
                assert(fabs(uvw[0] - uvw_ref[j++] < 1e-11));
                assert(fabs(uvw[1] - uvw_ref[j++] < 1e-11));
                assert(fabs(uvw[2] - uvw_ref[j++] < 1e-11));
            }

    return 0;
}

int main(int argc, char *argv[]) {

    int count = 0,fails = 0;

#define RUN_TEST(t)                               \
      count++;                                    \
      printf(#t"...");                            \
      fflush(stdout);                             \
      if (!t())                                   \
          puts(" ok");                            \
      else                                        \
          { puts("failed"); fails++; }

    RUN_TEST(T01_generate_m);
    RUN_TEST(T02_extract_subgrid);
    RUN_TEST(T03_add_facet);
    RUN_TEST(T04_test_2d);
    RUN_TEST(T04a_recombine2d);
    RUN_TEST(T05_frac_coord);
    RUN_TEST(T05_degrid);
    RUN_TEST(T05_config);

#undef RUN_TEST

    printf(" *** %d/%d tests passed ***\n", count-fails, count);
    return fails;
}
