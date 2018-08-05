
#include "recombine.h"
#include "grid.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/stat.h>

double *generate_Fb(int yN_size, int yB_size, double *pswf) {
    double *Fb = (double *)malloc(sizeof(double) * yB_size);
    int i;
    for (i = 0; i <= yB_size/2; i++) {
        Fb[i] = 1 / pswf[i];
    }
    for (i = yB_size/2+1; i < yB_size; i++) {
        Fb[i] = Fb[yB_size-i];
    }
    return Fb;
}

double *generate_Fn(int yN_size, int xM_yN_size, double *pswf) {
    double *Fn = (double *)malloc(sizeof(double) * xM_yN_size);
    int i;
    assert(yN_size % xM_yN_size == 0);
    int xM_step = yN_size / xM_yN_size;
    for (i = 0; i < xM_yN_size; i++) {
        Fn[i] = pswf[i * xM_step];
    }
    return Fn;
}

double *generate_m(int image_size, int yP_size, int yN_size, int xM_size, int xMxN_yP_size,
                   double *pswf) {
    double *m = (double *)malloc(sizeof(double) * (yP_size/2+1));

    // Fourier transform of rectangular function with size xM, truncated by PSWF
    m[0] = pswf[0] * xM_size / image_size;
    int i;
    for (i = 1; i < yN_size/2; i++) {
        double x = (double)(i) * xM_size / image_size;
        m[i] = pswf[i] * (sin(x * M_PI) / x / M_PI) * xM_size / image_size;
    }
    for (; i <= yP_size / 2; i++) {
        m[i] = 0;
    }

    // Back to grid space
    fftw_plan plan = fftw_plan_r2r_1d(yP_size/2+1, m, m, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_free(plan);

    // Copy and mirror
    double *m_r = (double *)malloc(sizeof(double) * xMxN_yP_size);
    for (i = 0; i <= xMxN_yP_size/2; i++) {
        m_r[i] = m[i];
    }
    for (; i < xMxN_yP_size; i++) {
        m_r[i] = m[xMxN_yP_size-i];
    }
    free(m);

    return m_r;
}

void prepare_facet(int yB_size, int yP_size,
                   double *Fb,
                   double complex *facet, int facet_stride,
                   double complex *BF, int BF_stride) {
    // Multiply by Fb, pad up to yP
    int i;
    for (i = 0; i < yB_size/2; i++) {
        BF[BF_stride*i] = Fb[i] * facet[facet_stride*i] / yP_size;
    }
    for (; i < yP_size-yB_size/2; i++) {
        BF[BF_stride*i] = 0;
    }
    for (; i < yP_size; i++) {
        BF[BF_stride*i] = Fb[yB_size-yP_size+i] * facet[facet_stride*(yB_size-yP_size+i)] / yP_size;
    }
}

void extract_subgrid(int yP_size, int xM_yP_size, int xMxN_yP_size, int xM_yN_size, int subgrid_offset,
                     double *m_trunc, double *Fn,
                     complex double *BF, int BF_stride,
                     complex double *MBF, fftw_plan MBF_plan,
                     complex double *NMBF, int NMBF_stride) {
    int i;
    int xN_yP_size = xMxN_yP_size - xM_yP_size;
    assert(xN_yP_size % 2 == 0); // re-check loop borders...
    // m * b, with xN_yP_size worth of margin looping around the sides
    for (i = 0; i < xM_yP_size - xMxN_yP_size / 2; i++) {
        MBF[i] = m_trunc[i] * BF[BF_stride * ((i + subgrid_offset) % yP_size)];
    }
    for (; i < (xMxN_yP_size + 1) / 2; i++) {
        MBF[i] = m_trunc[i] * BF[BF_stride * ((i + subgrid_offset) % yP_size)];
        int bf_ix = i + subgrid_offset + yP_size - xM_yP_size;
        MBF[i] += m_trunc[xN_yP_size+i] * BF[BF_stride * (bf_ix % yP_size)];
    }
    for (; i < xM_yP_size; i++) {
        int bf_ix = i + subgrid_offset + yP_size - xM_yP_size;
        MBF[i] = m_trunc[xN_yP_size+i] * BF[BF_stride * (bf_ix % yP_size)];
    }
    fftw_execute(MBF_plan);
    for (i = 0; i < xM_yN_size / 2; i++) {
        NMBF[i * NMBF_stride] = MBF[i] * Fn[i];
    }
    for (; i < xM_yN_size; i++) {
        NMBF[i * NMBF_stride] = MBF[xM_yP_size-xM_yN_size+i] * Fn[i];
    }
}


void add_facet(int xM_size, int xM_yN_size, int facet_offset,
               complex double *NMBF, int NMBF_stride,
               complex double *out, int out_stride) {

    int i;
    for (i = 0; i < xM_yN_size; i++) {
        out[out_stride * ((i - xM_yN_size/2 + facet_offset + xM_size) % xM_size)] +=
            NMBF[NMBF_stride * ((i + xM_yN_size/2) % xM_yN_size)] / xM_size;
    }

}


bool recombine2d_set_config(struct recombine2d_config *cfg,
                            int image_size, int subgrid_spacing,
                            char *pswf_file,
                            int yB_size, int yN_size, int yP_size,
                            int xA_size, int xM_size, int xMxN_yP_size) {

    cfg->stream_check = NULL;
    cfg->stream_check_threshold = 0;
    cfg->stream_dump = NULL;

    cfg->image_size = image_size;
    cfg->subgrid_spacing = subgrid_spacing;
    assert(image_size % subgrid_spacing == 0);
    cfg->facet_spacing = image_size / subgrid_spacing;

    cfg->yB_size    = yB_size;
    cfg->yN_size    = yN_size;
    cfg->yP_size    = yP_size;
    cfg->xA_size    = xA_size;
    cfg->xM_size    = xM_size;
    cfg->xMxN_yP_size = xMxN_yP_size;

    // Check some side conditions...
    assert(image_size % xM_size == 0);
    int xM_step = image_size / xM_size;
    assert(cfg->facet_spacing % xM_step == 0);
    assert((cfg->subgrid_spacing * cfg->yP_size) % image_size == 0);

    // Only needed if we want to tile facets+subgrids without overlaps.
    // Could be generalised once we get smarter about this.
    assert(cfg->yB_size % cfg->facet_spacing == 0);
    assert(cfg->xA_size % cfg->subgrid_spacing == 0);

    cfg->yP_spacing = cfg->subgrid_spacing * cfg->yP_size / image_size;
    cfg->xM_spacing = cfg->facet_spacing * cfg->xM_size / image_size;
    assert((cfg->xM_size * cfg->yP_size) % cfg->image_size == 0);
    cfg->xM_yP_size = cfg->xM_size * cfg->yP_size / cfg->image_size;
    assert((cfg->xM_size * cfg->yN_size) % cfg->image_size == 0);
    cfg->xM_yN_size = cfg->xM_size * cfg->yN_size / cfg->image_size;

    cfg->F_size = sizeof(double complex) * cfg->yB_size * cfg->yB_size;
    cfg->BF_size = sizeof(double complex) * cfg->yP_size * cfg->yB_size;
    cfg->MBF_size = sizeof(double complex) * cfg->xM_yP_size;
    cfg->NMBF_size = sizeof(double complex) * cfg->xM_yN_size * cfg->yB_size;
    cfg->NMBF_BF_size = sizeof(double complex) * cfg->xM_yN_size * cfg->yP_size;
    cfg->NMBF_NMBF_size = sizeof(double complex) * cfg->xM_yN_size * cfg->xM_yN_size;

    cfg->SG_size = sizeof(double complex) * cfg->xM_size * cfg->xM_size;

    cfg->F_stride0 = cfg->yB_size; cfg->F_stride1 = 1;
    cfg->BF_stride0 = cfg->yP_size; cfg->BF_stride1 = 1;
    cfg->NMBF_stride0 = cfg->xM_yN_size; cfg->NMBF_stride1 = 1;
    cfg->NMBF_BF_stride0 = 1; cfg->NMBF_BF_stride1 = cfg->yP_size;
    cfg->NMBF_NMBF_stride0 = cfg->xM_yN_size; cfg->NMBF_NMBF_stride1 = 1;

    // Read PSWF (TODO: generate)
    double *pswf = read_dump(sizeof(double) * cfg->yN_size, pswf_file);
    if (!pswf) return false;

    // Generate Fn, Fb and m
    cfg->Fb = generate_Fb(cfg->yN_size, cfg->yB_size, pswf);
    cfg->Fn = generate_Fn(cfg->yN_size, cfg->xM_yN_size, pswf);
    cfg->m = generate_m(image_size, cfg->yP_size, cfg->yN_size, cfg->xM_size, cfg->xMxN_yP_size, pswf);
    free(pswf);

    return true;
}

void recombine2d_free(struct recombine2d_config *cfg)
{
    free(cfg->Fb); free(cfg->Fn); free(cfg->m);
}

// Global/per-thread memory required to run producer side
uint64_t recombine2d_global_memory(struct recombine2d_config *cfg)
{
    return cfg->F_size + cfg->BF_size;
}
uint64_t recombine2d_worker_memory(struct recombine2d_config *cfg)
{
    return cfg->MBF_size + cfg->NMBF_size + cfg->NMBF_BF_size + cfg->NMBF_NMBF_size;
}

static double get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

fftw_plan recombine2d_bf_plan(struct recombine2d_config *cfg, int BF_batch,
                              double complex *BF, unsigned planner_flags)
{
    return fftw_plan_many_dft(1, &cfg->yP_size, BF_batch,
                              BF, 0, cfg->BF_stride1, cfg->BF_stride0,
                              BF, 0, cfg->BF_stride1, cfg->BF_stride0,
                              FFTW_BACKWARD, planner_flags);
}

void recombine2d_init_worker(struct recombine2d_worker *worker, struct recombine2d_config *cfg,
                             int BF_batch, fftw_plan BF_plan, unsigned planner_flags)
{

    // Set configuration
    worker->cfg = cfg;

    // Create buffers
    worker->MBF = (double complex *)malloc(cfg->MBF_size);
    worker->NMBF = (double complex *)malloc(cfg->NMBF_size);
    worker->NMBF_BF = (double complex *)malloc(cfg->NMBF_BF_size);

    // Plan Fourier Transforms
    worker->BF_batch = BF_batch; worker->BF_plan = BF_plan;
    worker->MBF_plan = fftw_plan_dft_1d(cfg->xM_yP_size, worker->MBF, worker->MBF,
                                        FFTW_FORWARD, planner_flags);
    worker->NMBF_BF_plan = fftw_plan_many_dft(1, &cfg->yP_size, cfg->xM_yN_size,
                                            worker->NMBF_BF, 0, cfg->NMBF_BF_stride0, cfg->NMBF_BF_stride1,
                                            worker->NMBF_BF, 0, cfg->NMBF_BF_stride0, cfg->NMBF_BF_stride1,
                                            FFTW_BACKWARD, planner_flags);
    // Initialise statistics
    worker->pf1_time = worker->es1_time = worker->ft1_time =
        worker->pf2_time = worker->es2_time = worker->ft2_time = 0;

}

void recombine2d_free_worker(struct recombine2d_worker *worker)
{
    // (BF_plan is assumed to be shared)
    fftw_free(worker->MBF_plan);
    fftw_free(worker->NMBF_BF_plan);

    free(worker->MBF);
    free(worker->NMBF);
    free(worker->NMBF_BF);
}

void recombine2d_pf1_ft1_omp(struct recombine2d_worker *worker,
                             complex double *F,
                             complex double *BF)
{
    struct recombine2d_config *cfg = worker->cfg;
    int y;
#pragma omp for schedule(dynamic)
    for (y = 0; y < cfg->yB_size; y+=worker->BF_batch) {

        // Facet preparation along first axis
        double start = get_time_ns();
        int y2;
        for (y2 = y; y2 < y+worker->BF_batch && y2 < cfg->yB_size; y2++) {
            prepare_facet(cfg->yB_size, cfg->yP_size, cfg->Fb,
                          F+y2*cfg->F_stride0, cfg->F_stride1,
                          BF+y2*cfg->BF_stride0, cfg->BF_stride1);
        }
        worker->pf1_time += get_time_ns() - start;

        // Fourier transform along first axis
        start = get_time_ns();
        fftw_execute_dft(worker->BF_plan, BF+y*cfg->BF_stride0, BF+y*cfg->BF_stride0);
        worker->ft1_time += get_time_ns() - start;
    }

}

void recombine2d_pf1_ft1_es1_omp(struct recombine2d_worker *worker,
                                 int subgrid_off1,
                                 complex double *F,
                                 complex double *NMBF)
{
    struct recombine2d_config *cfg = worker->cfg;
    int y;

    int BF_chunk_size = sizeof(double complex) * cfg->yP_size * worker->BF_batch;
    double complex *BF_chunk = malloc(BF_chunk_size);
    assert(cfg->BF_stride1 == 1);
    assert(cfg->NMBF_BF_stride0 == 1);
    assert(cfg->BF_stride0 == cfg->NMBF_BF_stride1);
    assert(subgrid_off1 % cfg->subgrid_spacing == 0);

#pragma omp for schedule(dynamic)
    for (y = 0; y < cfg->yB_size; y+=worker->BF_batch) {

        // Facet preparation along first axis
        double start = get_time_ns();
        int y2;
        for (y2 = y; y2 < y+worker->BF_batch && y2 < cfg->yB_size; y2++) {
            prepare_facet(cfg->yB_size, cfg->yP_size, cfg->Fb,
                          F+y2*cfg->F_stride0, cfg->F_stride1,
                          BF_chunk+(y2-y)*cfg->BF_stride0, cfg->BF_stride1);
        }
        worker->pf1_time += get_time_ns() - start;

        // Fourier transform along first axis
        start = get_time_ns();
        fftw_execute_dft(worker->BF_plan, BF_chunk, BF_chunk);
        worker->ft1_time += get_time_ns() - start;

        // Extract subgrids along first axis
        assert(subgrid_off1 % cfg->subgrid_spacing == 0);
        int subgrid_offset = subgrid_off1 / cfg->subgrid_spacing * cfg->yP_spacing;
        start = get_time_ns();
        for (y2 = y; y2 < y+worker->BF_batch && y2 < cfg->yB_size; y2++) {
            extract_subgrid(cfg->yP_size, cfg->xM_yP_size, cfg->xMxN_yP_size, cfg->xM_yN_size,
                            subgrid_offset, cfg->m, cfg->Fn,
                            BF_chunk+(y2-y)*cfg->BF_stride0, cfg->BF_stride1,
                            worker->MBF, worker->MBF_plan,
                            NMBF+y2*cfg->NMBF_stride0, cfg->NMBF_stride1);
        }
        worker->es1_time += get_time_ns() - start;

    }

    free(BF_chunk);
}

void recombine2d_es1_pf0_ft0(struct recombine2d_worker *worker,
                             int subgrid_off1, complex double *BF, double complex *NMBF_BF)
{
    struct recombine2d_config *cfg = worker->cfg;
    int x,y;

    // Extract subgrids along first axis
    assert(subgrid_off1 % cfg->subgrid_spacing == 0);
    int subgrid_offset = subgrid_off1 / cfg->subgrid_spacing * cfg->yP_spacing;
    double start = get_time_ns();
    for (x = 0; x < cfg->yB_size; x++) {
        extract_subgrid(cfg->yP_size, cfg->xM_yP_size, cfg->xMxN_yP_size, cfg->xM_yN_size,
                        subgrid_offset, cfg->m, cfg->Fn,
                        BF+x*cfg->BF_stride0, cfg->BF_stride1,
                        worker->MBF, worker->MBF_plan,
                        worker->NMBF+x*cfg->NMBF_stride0, cfg->NMBF_stride1);
    }
    worker->es1_time += get_time_ns() - start;

    // Facet preparation along second axis
    start = get_time_ns();
    for (y = 0; y < cfg->xM_yN_size; y++) {
        prepare_facet(cfg->yB_size, cfg->yP_size, cfg->Fb,
                      worker->NMBF+y*cfg->NMBF_stride1, cfg->NMBF_stride0,
                      NMBF_BF+y*cfg->NMBF_BF_stride1, cfg->NMBF_BF_stride0);
    }
    worker->pf2_time += get_time_ns() - start;

    // Fourier transform along second axis
    start = get_time_ns();
    if (NMBF_BF == worker->NMBF_BF)
        fftw_execute(worker->NMBF_BF_plan);
    else
        fftw_execute_dft(worker->NMBF_BF_plan, NMBF_BF, NMBF_BF);
    worker->ft2_time += get_time_ns() - start;

}

void recombine2d_es1_omp(struct recombine2d_worker *worker,
                         int subgrid_off1,
                         complex double *BF,
                         double complex *NMBF)
{
    struct recombine2d_config *cfg = worker->cfg;
    int x;

    // Extract subgrids along first axis
    assert(subgrid_off1 % cfg->subgrid_spacing == 0);
    int subgrid_offset = subgrid_off1 / cfg->subgrid_spacing * cfg->yP_spacing;;
    double start = get_time_ns();
#pragma omp for schedule(dynamic, worker->BF_batch)
    for (x = 0; x < cfg->yB_size; x++) {
        extract_subgrid(cfg->yP_size, cfg->xM_yP_size, cfg->xMxN_yP_size, cfg->xM_yN_size,
                        subgrid_offset, cfg->m, cfg->Fn,
                        BF+x*cfg->BF_stride0, cfg->BF_stride1,
                        worker->MBF, worker->MBF_plan,
                        NMBF+x*cfg->NMBF_stride0, cfg->NMBF_stride1);
    }
    worker->es1_time += get_time_ns() - start;

}

void recombine2d_pf0_ft0_omp(struct recombine2d_worker *worker,
                             double complex *NMBF,
                             double complex *NMBF_BF)
{
    struct recombine2d_config *cfg = worker->cfg;

    assert(cfg->BF_stride1 == 1);
    assert(cfg->NMBF_BF_stride0 = 1);
    assert(cfg->BF_stride0 == cfg->NMBF_BF_stride1);

    int y;
#pragma omp for schedule(dynamic)
    for (y = 0; y < cfg->xM_yN_size; y+=worker->BF_batch) {

        // Facet preparation along second axis
        double start = get_time_ns();
        int y2;
        for (y2 = y; y2 < y+worker->BF_batch && y2 < cfg->xM_yN_size; y2++) {
            prepare_facet(cfg->yB_size, cfg->yP_size, cfg->Fb,
                          NMBF+y2*cfg->NMBF_stride1, cfg->NMBF_stride0,
                          NMBF_BF+y2*cfg->NMBF_BF_stride1, cfg->NMBF_BF_stride0);
        }
        worker->pf2_time += get_time_ns() - start;

        // Fourier transform along second axis.

        // Note 1: We are re-using the BF FFTW plan, which happens to
        // work because we switched strides (see assertions at start
        // of routine).

        // Note 2: We do not want to assume that xM_yN_size gets
        // evenly divided by BF_batch, the quick hack here is to just
        // make an on-the-fly plan for the last bit
        start = get_time_ns();
        fftw_plan plan = worker->BF_plan;
        if (y+worker->BF_batch >= cfg->xM_yN_size) {
            plan = recombine2d_bf_plan(worker->cfg, cfg->xM_yN_size - y,
                                       NMBF_BF+y*cfg->NMBF_BF_stride1,
                                       FFTW_ESTIMATE);
        }
        fftw_execute_dft(plan,
                         NMBF_BF+y*cfg->NMBF_BF_stride1,
                         NMBF_BF+y*cfg->NMBF_BF_stride1);
        if (plan != worker->BF_plan)
            fftw_free(plan);
        worker->ft2_time += get_time_ns() - start;

    }
}

void recombine2d_es0(struct recombine2d_worker *worker,
                     int subgrid_off0, int subgrid_off1,
                     double complex *NMBF_BF,
                     double complex *NMBF_NMBF)
{
    struct recombine2d_config *cfg = worker->cfg;
    int y;

    // Extract subgrids along second axis
    assert(subgrid_off0 % cfg->subgrid_spacing == 0);
    int subgrid_offset = subgrid_off0 / cfg->subgrid_spacing * cfg->yP_spacing;
    double start = get_time_ns();
    for (y = 0; y < cfg->xM_yN_size; y++) {
        extract_subgrid(cfg->yP_size, cfg->xM_yP_size, cfg->xMxN_yP_size, cfg->xM_yN_size,
                        subgrid_offset, cfg->m, cfg->Fn,
                        NMBF_BF+y*cfg->NMBF_BF_stride1, cfg->NMBF_BF_stride0,
                        worker->MBF, worker->MBF_plan,
                        NMBF_NMBF+y*cfg->NMBF_NMBF_stride1, cfg->NMBF_NMBF_stride0);
    }
    worker->es2_time += get_time_ns() - start;

    // Check stream contents if requested
    int i0 = subgrid_off0 / worker->cfg->xA_size;
    int i1 = subgrid_off1 / worker->cfg->xA_size;
    if (cfg->stream_check) {

        // Check whether it exists
        char filename[256]; struct stat st;
        sprintf(filename, cfg->stream_check, i0, i1);
        if (stat(filename, &st) == 0) {

            double complex *NMBF_NMBF_check = read_dump(cfg->NMBF_NMBF_size, cfg->stream_check, i0, i1);
            if (NMBF_NMBF_check) {
                int x0; int errs = 0;
                for (x0 = 0; x0 < cfg->xM_yN_size * cfg->xM_yN_size; x0++) {
                    if (cabs(NMBF_NMBF[x0] - NMBF_NMBF_check[x0]) > cabs(NMBF_NMBF[x0]) * cfg->stream_check_threshold) {
                        fprintf(stderr, "stream check failed: subgrid %d/%d at position %d/%d (%f%+f != %f%+f)\n",
                                i0, i1, x0 / cfg->xM_yN_size, x0 % cfg->xM_yN_size,
                                creal(NMBF_NMBF[x0]), cimag(NMBF_NMBF[x0]),
                                creal(NMBF_NMBF_check[x0]), cimag(NMBF_NMBF_check[x0]));
                        errs+=1;
                    }
                }
                if (!errs) {
                    printf("stream check for subgrid %d/%d passed\n", i0, i1);
                }
            }
        }
    }

    // Similarly, write dump on request
    if (cfg->stream_dump) {
        write_dump(NMBF_NMBF, cfg->NMBF_NMBF_size, cfg->stream_dump, i0, i1);
    }

}

void recombine2d_af0_af1(struct recombine2d_config *cfg,
                         double complex *subgrid,
                         int facet_off0, int facet_off1,
                         double complex *NMBF_NMBF)
{
    assert(facet_off0 % cfg->facet_spacing == 0);
    assert(facet_off1 % cfg->facet_spacing == 0);
    int facet_offset0 = facet_off0 / cfg->facet_spacing * cfg->xM_spacing;
    int facet_offset1 = facet_off1 / cfg->facet_spacing * cfg->xM_spacing;

    int j0, j1;
    for (j0 = 0; j0 < cfg->xM_yN_size; j0++) {

        int off_sg0 = ((j0 - cfg->xM_yN_size/2 + facet_offset0 + cfg->xM_size) % cfg->xM_size);
        int off_nmbf0 = ((j0 + cfg->xM_yN_size/2) % cfg->xM_yN_size);

        for (j1 = 0; j1 < cfg->xM_yN_size; j1++) {

            int off_sg1 = ((j1 - cfg->xM_yN_size/2 + facet_offset1 + cfg->xM_size) % cfg->xM_size);
            int off_nmbf1 = ((j1 + cfg->xM_yN_size/2) % cfg->xM_yN_size);

            subgrid[off_sg0*cfg->xM_size+off_sg1] +=
                NMBF_NMBF[off_nmbf0*cfg->xM_yN_size+off_nmbf1] / (cfg->xM_size * cfg->xM_size);
        }

    }

}

