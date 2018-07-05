
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>

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
    m[0] = pswf[0] * (double)(xM_size) / image_size;
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


void add_subgrid(int xM_size, int xM_yN_size, int facet_offset,
                 complex double *NMBF, complex double *out) {

    int i;
    for (i = -xM_yN_size/2; i < xM_yN_size/2; i++) {
        out[(i + facet_offset + xM_size) % xM_size] += NMBF[(i + xM_yN_size) % xM_yN_size] / xM_size;
    }

}
