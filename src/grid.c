
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <fcntl.h>
#include <stdint.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include <fenv.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "grid.h"

// Assume all uvw are zero. This eliminates all coordinate
// calculations and grids only into the middle.
//#define ASSUME_UVW_0

inline static int coord(int grid_size, double theta,
                        struct bl_data *bl_data,
                        int time, int freq) {
#ifdef ASSUME_UVW_0
    int x = 0, y = 0;
#else
    int x = (int)floor(theta * uvw_lambda(bl_data, time, freq, 0) + .5);
    int y = (int)floor(theta * uvw_lambda(bl_data, time, freq, 1) + .5);
#endif
    return (y+grid_size/2) * grid_size + (x+grid_size/2);
}

inline static void _frac_coord(int grid_size, int oversample,
                              double u, int *x, int *fx) {

    // Round to nearest oversampling value. We assume the kernel to be
    // in "natural" order, so what we are looking for is the best
    //    "x - fx / oversample"
    // approximation for "grid_size/2 + u".
    fesetround(3);
    int ox = lrint((grid_size / 2 - u) * oversample);
    *x = grid_size-(ox / oversample);
    *fx = ox % oversample;

}
void frac_coord(int grid_size, int oversample,
                double u, int *x, int *fx) {

    _frac_coord(grid_size, oversample, u, x, fx);

}

// Fractional coordinate calculation for separable 1D kernel
inline static void frac_coord_sep_uv(int grid_size, int kernel_size, int oversample,
                                     double theta,
                                     double u, double v,
                                     int *grid_offset,
                                     int *sub_offset_x, int *sub_offset_y)
{

    double x = theta * u, y = theta * v;
    // Find fractional coordinates
    int ix, iy, ixf, iyf;
    _frac_coord(grid_size, oversample, x, &ix, &ixf);
    _frac_coord(grid_size, oversample, y, &iy, &iyf);
    // Calculate grid and oversampled kernel offsets
    *grid_offset = (iy-kernel_size/2)*grid_size + (ix-kernel_size/2);
    *sub_offset_x = kernel_size * ixf;
    *sub_offset_y = kernel_size * iyf;
}

double complex degrid_conv_uv(double complex *uvgrid, int grid_size, double theta,
                              double u, double v,
                              struct sep_kernel_data *kernel,
                              uint64_t *flops)
{

    // Calculate grid and sub-grid coordinates
    int grid_offset, sub_offset_x, sub_offset_y;
    frac_coord_sep_uv(grid_size, kernel->size, kernel->oversampling,
                      theta, u, v,
                      &grid_offset, &sub_offset_x, &sub_offset_y);

#ifndef __AVX2__

    // Get visibility
    double complex vis = 0;
    int y, x;
    for (y = 0; y < kernel->size; y++) {
        double complex visy = 0;
        for (x = 0; x < kernel->size; x++)
            visy += kernel->data[sub_offset_x + x] *
                    uvgrid[grid_offset + y*grid_size + x];
        vis += kernel->data[sub_offset_y + y] * visy;
    }
    *flops += 4 * (1 + kernel->size) * kernel->size;

    return vis;

#else

    // Get visibility
    assert(kernel->size % 2 == 0);
    __m256d vis = _mm256_setzero_pd();
    int y, x;
    for (y = 0; y < kernel->size; y += 1) {
        __m256d sum = _mm256_setzero_pd();
        for (x = 0; x < kernel->size; x += 2) {
            double *pk = kernel->data + sub_offset_x + x;
            __m256d kern = _mm256_setr_pd(*pk, *pk, *(pk+1), *(pk+1));
            __m256d grid = _mm256_loadu_pd((double *)(uvgrid + grid_offset + y*grid_size + x));
            sum = _mm256_fmadd_pd(kern, grid, sum);
        }
        double kern_y = kernel->data[sub_offset_y + y];
        vis = _mm256_fmadd_pd(sum, _mm256_set1_pd(kern_y), vis);
    }
    __attribute__ ((aligned (32))) double vis_s[4];
    _mm256_store_pd(vis_s, vis);
    double complex vis_res = vis_s[0] + vis_s[2] + 1j * (vis_s[1] + vis_s[3]);

    *flops += 4 * (1 + kernel->size) * kernel->size;
    return vis_res;
#endif
}

uint64_t degrid_conv_bl(double complex *uvgrid, int grid_size, double theta,
                        double d_u, double d_v,
                        double min_u, double max_u, double min_v, double max_v,
                        struct bl_data *bl,
                        int time0, int time1, int freq0, int freq1,
                        struct sep_kernel_data *kernel)
{
    uint64_t flops = 0;
    int time, freq;
    for (time = time0; time < time1; time++) {
        for (freq = freq0; freq < freq1; freq++) {

            // Bounds check
            double u = uvw_lambda(bl, time, freq, 0);
            if (u < min_u || u >= max_u) continue;
            double v = uvw_lambda(bl, time, freq, 1);
            if (v < min_v || v >= max_v) continue;

            bl->vis[(time-time0)*(freq1 - freq0) + freq-freq0] =
                degrid_conv_uv(uvgrid, grid_size, theta,
                               u-d_u, v-d_v, kernel, &flops);
        }
    }

    return flops;
}

void fft_shift(double complex *uvgrid, int grid_size) {

    // Shift the FFT
    assert(grid_size % 2 == 0);
    int x, y;
    for (y = 0; y < grid_size; y++) {
        for (x = 0; x < grid_size/2; x++) {
            int ix0 = y * grid_size + x;
            int ix1 = (ix0 + (grid_size+1) * (grid_size/2)) % (grid_size*grid_size);
            double complex temp = uvgrid[ix0];
            uvgrid[ix0] = uvgrid[ix1];
            uvgrid[ix1] = temp;
        }
    }

}
