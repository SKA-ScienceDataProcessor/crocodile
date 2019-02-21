
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
#ifdef USE_AVX
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

// Fractional coordinate calculation for oversampled 2D kernel
inline static void frac_coord_2d(int grid_size, int kernel_size, int oversample,
                                 double theta,
                                 struct bl_data *bl_data,
                                 int time, int freq,
                                 double d_u, double d_v,
                                 int *grid_offset, int *sub_offset) {

    // Convert UVW coordinates into grid coordinates. This needs to
    // take frequency scaling, the field of view size (= grid
    // resolution) as well as the grid centre into account
#ifdef ASSUME_UVW_0
    double x = 0, y = 0;
#else
    double x = theta * (uvw_lambda(bl_data, time, freq, 0) - d_u);
    double y = theta * (uvw_lambda(bl_data, time, freq, 1) - d_v);
#endif

    // Find fractional coordinates
    int ix, iy, ixf, iyf;
    _frac_coord(grid_size, oversample, x, &ix, &ixf);
    _frac_coord(grid_size, oversample, y, &iy, &iyf);

    // Calculate grid and oversampled kernel offsets
    *grid_offset = (iy-kernel_size/2)*grid_size + (ix-kernel_size/2);
    *sub_offset = kernel_size * kernel_size * (iyf*oversample + ixf);
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
inline static void frac_coord_sep(int grid_size, int kernel_size, int oversample,
                                  double theta,
                                  struct bl_data *bl_data,
                                  int time, int freq,
                                  double d_u, double d_v,
                                  int *grid_offset,
                                  int *sub_offset_x, int *sub_offset_y)
{
#ifdef ASSUME_UVW_0
    double u = 0, v = 0;
#else
    double u = uvw_lambda(bl_data, time, freq, 0) - d_u;
    double v = uvw_lambda(bl_data, time, freq, 1) - d_v;
#endif
    frac_coord_sep_uv(grid_size, kernel_size, oversample, theta, u, v,
                      grid_offset, sub_offset_x, sub_offset_y);
}

void weight(unsigned int *wgrid, int grid_size, double theta,
            struct vis_data *vis) {

    // Simple uniform weighting
    int bl, time, freq;
    memset(wgrid, 0, grid_size * grid_size * sizeof(unsigned int));
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                wgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)]++;
            }
        }
    }
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq]
                    /= wgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)];
            }
        }
    }

}

uint64_t degrid_simple(double complex *uvgrid, int grid_size, double theta,
                       struct vis_data *vis) {

    uint64_t flops = 0;
    int bl, time, freq;
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq]
                    = uvgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)];
                flops += 2;
            }
        }
    }

    return flops;
}

uint64_t grid_simple(double complex *uvgrid, int grid_size, double theta,
                     struct vis_data *vis) {

    uint64_t flops = 0;
    int bl, time, freq;
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                uvgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)]
                    += vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq];
                flops += 2;
            }
        }
    }

    return flops;
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

#ifndef USE_AVX
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
    double test[4];
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

uint64_t degrid_conv(double complex *uvgrid, int grid_size, double theta, double d_u, double d_v,
                     struct vis_data *vis, struct sep_kernel_data *kernel)
{
    uint64_t flops = 0;
    int bl;
    for (bl = 0; bl < vis->bl_count; bl++) {
        flops += degrid_conv_bl(uvgrid, grid_size, theta,
                                d_u, d_v,
                                d_u-grid_size/theta/2, d_u+grid_size/theta/2,
                                d_v-grid_size/theta/2, d_v+grid_size/theta/2,
                                &vis->bl[bl], 0, vis->bl[bl].time_count, 0, vis->bl[bl].freq_count,
                                kernel);
    }
    return flops;
}

inline static uint64_t w_project(double complex *uvgrid, int grid_size, double theta,
                                 int time, int freq,
                                 double d_u, double d_v, double d_w,
                                 struct bl_data *bl, struct w_kernel_data *wkern) {

    // Calculate grid and sub-grid coordinates
    int grid_offset, sub_offset;
    frac_coord_2d(grid_size, wkern->size_x, wkern->oversampling,
               theta, bl, time, freq, d_u, d_v,
               &grid_offset, &sub_offset);
    // Determine w-kernel to use
    double w = uvw_lambda(bl, time, freq, 2) - d_w;
    int w_plane = (int)floor((w - wkern->w_min) / wkern->w_step + .5);
    double complex *wk = wkern->kern_by_w[w_plane].data;
    // Get visibility
    double complex v = bl->vis[time*bl->freq_count+freq];
    // Copy kernel
    int x, y;
    int wkern_size = wkern->size_x;
    assert(wkern->size_y == wkern_size);
    for (y = 0; y < wkern_size; y++) {
        for (x = 0; x < wkern_size; x++) {
            uvgrid[grid_offset + y*grid_size + x]
                += v * conj(wk[sub_offset + y*wkern_size + x]);
        }
    }

    return 8 * wkern->size_x * wkern->size_y;
}

uint64_t grid_wprojection(double complex *uvgrid, int grid_size, double theta,
                          struct vis_data *vis, struct w_kernel_data *wkern) {

    uint64_t flops = 0;
    int bl, time, freq;
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                flops += w_project(uvgrid, grid_size, theta, time, freq,
                                   0, 0, 0, &vis->bl[bl], wkern);
            }
        }
    }

    return flops;
}

static inline double lambda_min(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_max : bl_data->f_min) / c;
}
static inline double lambda_max(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_min : bl_data->f_max) / c;
}

static uint64_t w_project_bin(double complex *subgrid, int subgrid_size, double theta,
                              struct bl_data **bl_bin, int bl_count,
                              struct w_kernel_data *wkern,
                              double u_min, double u_max, double u_mid,
                              double v_min, double v_max, double v_mid,
                              double w_min, double w_max, double w_mid) {

    int bl, time, freq;
    uint64_t all_flops = 0;
    for (bl = 0; bl < bl_count; bl++) {
        struct bl_data *bl_data = bl_bin[bl];

        // Baseline cannot possible overlap uvw bin?
        if (lambda_max(bl_data, bl_data->u_max) < u_min ||
            lambda_min(bl_data, bl_data->u_min) >= u_max ||
            lambda_max(bl_data, bl_data->v_max) < v_min ||
            lambda_min(bl_data, bl_data->v_min) >= v_max ||
            lambda_max(bl_data, bl_data->w_max) < w_min ||
            lambda_min(bl_data, bl_data->w_min) >= w_max) {

            // Skip
            continue;
        }

        // Then go through individual visibilities
        uint64_t flops = 0;
        for (time = 0; time < bl_data->time_count; time++) {
            for (freq = 0; freq < bl_data->freq_count; freq++) {

                // Fine bounds check
                double u = uvw_lambda(bl_data, time, freq, 0);
                double v = uvw_lambda(bl_data, time, freq, 1);
                double w = uvw_lambda(bl_data, time, freq, 2);
                if (u < u_min || u >= u_max ||
                    v < v_min || v >= v_max ||
                    w < w_min || w >= w_max) {

                    continue;
                }

                // w-project the last mile (TODO: special case for wp=0)
                flops += w_project(subgrid, subgrid_size, theta,
                                   time, freq,
                                   u_mid, v_mid, w_mid,
                                   bl_data, wkern);
            }
        }

        #pragma omp atomic
        bl_data->flops += flops;
        all_flops += flops;
    }

    return all_flops;
}

// How is this not in the standard library somewhere?
// (stolen from Stack Overflow)
inline static double complex cipow(double complex base, int exp)
{
    double complex result = 1;
    if (exp < 0) return 1 / cipow(base, -exp);
    if (exp == 1) return base;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

inline static int cipow_flops(int exp)
{
    // Popcount, basically. Yes, there are really clever algorithms for
    // this *and* a builtin, but we are more interested in clarity here.
    if (exp < 0) return cipow_flops(exp);
    int flops = 0;
    while(exp)
    {
        if (exp & 1) flops+=6;
        exp >>= 1;
    }
    return flops;
}

double complex *make_wtransfer(double theta, double wincrement,
                               int subgrid_size, int fsample_size,
                               uint64_t *flops)
{

    if(fsample_size < subgrid_size) {
        fprintf(stderr, "Warning: Increasing Fresnel sampling to sub-grid resolution!");
        fsample_size = subgrid_size;
    }

    // Allocate grid for Fresnel sampling / w kernel
    int fsample_mem_size = sizeof(double complex) * fsample_size * fsample_size;
    int subgrid_mem_size = sizeof(double complex) * subgrid_size * subgrid_size;
    double complex *wtransfer = (double complex *)malloc(subgrid_mem_size);
    double complex *fsample = wtransfer;
    fftw_plan ff_fft_plan = 0, ff_ifft_plan = 0;
    if (fsample_size > subgrid_size) {
        // If we need to down-sample the Fresnel pattern to the
        // subgrid allocate a separate buffer
        fsample = (double complex *)malloc(fsample_mem_size);
        ff_fft_plan = fftw_plan_dft_2d(fsample_size, fsample_size, fsample, fsample, -1, FFTW_MEASURE);
        ff_ifft_plan = fftw_plan_dft_2d(subgrid_size, subgrid_size, wtransfer, wtransfer, +1, FFTW_MEASURE);
    }

    // Sample
    int x, y;
    for (y = 0; y < fsample_size; y++) {
        for (x = 0; x < fsample_size; x++) {
            double l = theta * (double)(x - fsample_size / 2) / fsample_size;
            double m = theta * (double)(y - fsample_size / 2) / fsample_size;
            double ph = 2 * M_PI * wincrement * (1 - sqrt(1 - l*l - m*m));
            fsample[y * fsample_size + x] = cos(ph) + I * sin(ph);
        }
    }
    *flops += 10 * fsample_size * fsample_size;

    // Move zero image position to (0,0). This is going to be the
    // convention for subimg, it simplifies the (frequent!) FFTs.
    fft_shift(fsample, fsample_size);

    // Downsample Fresnel pattern
    if (fsample_size > subgrid_size) {
        fftw_execute(ff_fft_plan); fftw_free(ff_fft_plan);
        *flops += (int) ceil(5 * fsample_size * fsample_size * log(fsample_size * fsample_size) / log(2) );

        fft_shift(fsample, fsample_size);

        for (y = 0; y < subgrid_size; y++) {
            for (x = 0; x < subgrid_size; x++) {
                int y2 = y - subgrid_size/2 + fsample_size/2;
                int x2 = x - subgrid_size/2 + fsample_size/2;
                wtransfer[y * subgrid_size + x] = fsample[y2 * fsample_size + x2] / fsample_size / fsample_size;
            }
        }

        free(fsample);
        fft_shift(wtransfer, subgrid_size);
        fftw_execute(ff_ifft_plan); fftw_free(ff_ifft_plan);
        *flops += (int) ceil(5 * subgrid_size * subgrid_size * log(subgrid_size * subgrid_size) / log(2) );
    }

    return wtransfer;
}

uint64_t grid_wtowers(double complex *uvgrid, int grid_size,
                      double theta,
                      struct vis_data *vis, struct w_kernel_data *wkern,
                      int subgrid_size, int fsample_size,
                      int subgrid_margin, double wincrement) {

    // Check conditions
    if(subgrid_margin < wkern->size_x) {
        fprintf(stderr, "Error: The margin is too small for this w-kernel size!\n");
        exit(1);
    }
    assert(wkern->size_x == wkern->size_y);
    if(wkern->w_min > -wincrement / 2 || wkern->w_max < wincrement / 2) {
        fprintf(stderr, "Error: The w-kernel does not cover a whole w-increment!\n");
        exit(1);
    }
    assert(subgrid_size % 2 == 0 && subgrid_margin % 2 == 0); // Not sure whether it works otherwise

    // Make w transfer pattern
    uint64_t flops = 0;
    double complex *wtransfer = make_wtransfer(theta, wincrement, subgrid_size, fsample_size, &flops);

    // Determine bounds in w
    double vis_w_min = 0, vis_w_max = 0;
    int bl;
    for (bl = 0; bl < vis->bl_count; bl++) {
        double w_min = lambda_min(&vis->bl[bl], vis->bl[bl].w_min);
        double w_max = lambda_max(&vis->bl[bl], vis->bl[bl].w_max);
        if (w_min < vis_w_min) { vis_w_min = w_min; }
        if (w_max > vis_w_max) { vis_w_max = w_max; }
    }
    int wp_min = (int) floor(vis_w_min / wincrement + 0.5);
    int wp_max = (int) floor(vis_w_max / wincrement + 0.5);

    // Bin in uv
    int chunk_size = subgrid_size - subgrid_margin;
    int chunk_count = grid_size / chunk_size + 1;
    int bins_size = sizeof(void *) * chunk_count * chunk_count;
    struct bl_data ***bins = (struct bl_data ***)malloc(bins_size);
    memset(bins, 0, bins_size);
    int bins_count_size = sizeof(int) * chunk_count * chunk_count;
    int *bins_count = (int *)malloc(bins_count_size);
    memset(bins_count, 0, bins_count_size);
    for (bl = 0; bl < vis->bl_count; bl++) {

        // Determine bounds (could be more precise, future work...)
        struct bl_data *bl_data = &vis->bl[bl];
        double u_min = lambda_min(bl_data, bl_data->u_min);
        double u_max = lambda_max(bl_data, bl_data->u_max);
        double v_min = lambda_min(bl_data, bl_data->v_min);
        double v_max = lambda_max(bl_data, bl_data->v_max);

        // Determine first/last overlapping grid chunks
        int cx0 = (floor(u_min * theta + 0.5) + grid_size/2) / chunk_size;
        int cx1 = (floor(u_max * theta + 0.5) + grid_size/2) / chunk_size;
        int cy0 = (floor(v_min * theta + 0.5) + grid_size/2) / chunk_size;
        int cy1 = (floor(v_max * theta + 0.5) + grid_size/2) / chunk_size;

        int cy, cx;
        for (cy = cy0; cy <= cy1; cy++) {
            for (cx = cx0; cx <= cx1; cx++) {

                // Lazy dynamically sized vector
                int bcount = ++bins_count[cy*chunk_count + cx];
                bins[cy*chunk_count + cx] =
                    (struct bl_data **)realloc(bins[cy*chunk_count + cx], sizeof(void *) * bcount);
                bins[cy*chunk_count + cx][bcount-1] = bl_data;

            }
        }

    }

    // Initialise FLOP counts. We will do a lot of subgrid FFTs, so cache
    // its approximate cost.
    int subgrid_cells = subgrid_size * subgrid_size;
    uint64_t subgrid_fft_flops = (int) ceil(5 * subgrid_cells * log(subgrid_cells) / log(2) );

    #pragma omp parallel reduction(+:flops)
    {

    // Make sub-grids in grid and image space, and FFT plans
    int subgrid_mem_size = sizeof(double complex) * subgrid_size * subgrid_size;
    double complex *subgrid = (double complex *)malloc(subgrid_mem_size);
    double complex *subimg = (double complex *)malloc(subgrid_mem_size);
    fftw_plan fft_plan, ifft_plan;
    #pragma omp critical
    {
        fft_plan = fftw_plan_dft_2d(subgrid_size, subgrid_size, subimg, subimg, -1, FFTW_MEASURE);
        ifft_plan = fftw_plan_dft_2d(subgrid_size, subgrid_size, subgrid, subgrid, +1, FFTW_MEASURE);
    }

    // Grid chunks
    int cc;
    #pragma omp for schedule(dynamic)
    for (cc = 0; cc < chunk_count * chunk_count; cc++) {
        int cx = cc % chunk_count, cy = cc / chunk_count;

        // Get our baselines bin
        struct bl_data **bl_bin = bins[cy*chunk_count + cx];
        int bl_count = bins_count[cy*chunk_count + cx];
        if (bl_count == 0) { continue; }
        //printf("%d %d/%d\n", omp_get_thread_num(), cx, cy);

        // Determine tower base
        int x_min = chunk_size*cx - grid_size/2;
        int y_min = chunk_size*cy - grid_size/2;
        double u_min = ((double)x_min - 0.5) / theta;
        double v_min = ((double)y_min - 0.5) / theta;
        double u_max = u_min + chunk_size / theta;
        double v_max = v_min + chunk_size / theta;

        // Midpoint in uvw. Important to note that for even-sized
        // FFTs (likely what we are using!) this is slightly
        // off-centre, so u_mid != (u_min + u_max) / 2.
        double u_mid = (double)(x_min + chunk_size / 2) / theta;
        double v_mid = (double)(y_min + chunk_size / 2) / theta;

        // Clean subgrid
        memset(subgrid, 0, subgrid_mem_size);
        memset(subimg, 0, subgrid_mem_size);

        // Go through w-planes
        int have_vis = 0;
        int last_wp = wp_min;
        int wp;
        for (wp = wp_min; wp <= wp_max; wp++) {
            double w_mid = (double)wp * wincrement;
            double w_min = ((double)wp - 0.5) * wincrement;
            double w_max = ((double)wp + 0.5) * wincrement;

            // Now grid all baselines for this uvw bin
            uint64_t bin_flops = w_project_bin(subgrid, subgrid_size, theta, bl_bin, bl_count, wkern,
                                               u_min, u_max, u_mid,
                                               v_min, v_max, v_mid,
                                               w_min, w_max, w_mid);

            // Skip the rest if we found no visibilities
            if (bin_flops == 0) { continue; }
            flops += bin_flops;
            have_vis = 1;

            // IFFT the sub-grid in-place, add to image sum
            fftw_execute(ifft_plan);
            flops += subgrid_fft_flops;

            // Bring image sum to our w-plane and add new data,
            // clean subgrid for next w-plane.
            int x, y;
            for (y = 0; y < subgrid_size; y++) {
                for (x = 0; x < subgrid_size; x++) {
                    double complex wtrans = cipow(wtransfer[y*subgrid_size + x], wp-last_wp);
                    subimg[y*subgrid_size + x] =
                        wtrans * subimg[y*subgrid_size + x] + subgrid[y*subgrid_size + x];
                    subgrid[y*subgrid_size + x] = 0;
                }
            }
            flops += subgrid_cells * (8 + cipow_flops(wp-last_wp));
            last_wp = wp;

        }

        // No visibilities? Skip
        if (!have_vis) { continue; }

        // Transfer to w=0 plane
        if (last_wp != 0) {
            int x, y;
            for (y = 0; y < subgrid_size; y++) {
                for (x = 0; x < subgrid_size; x++) {
                    subimg[y*subgrid_size + x] /= cipow(wtransfer[y*subgrid_size + x], last_wp);
                }
            }
            flops += subgrid_cells * (8 + cipow_flops(wp-last_wp));
        }

        // FFT
        fftw_execute(fft_plan);
        flops += subgrid_fft_flops;

        // Add to main grid, ignoring margins, which might be over
        // bounds (should not actually happen, but let's be safe)
        int x0 = x_min - subgrid_margin/2, x1 = x0 + subgrid_size;
        int y0 = y_min - subgrid_margin/2, y1 = y0 + subgrid_size;
        if (x0 < -grid_size/2) { x0 = -grid_size/2; }
        if (y0 < -grid_size/2) { y0 = -grid_size/2; }
        if (x1 > grid_size/2) { x1 = grid_size/2; }
        if (y1 > grid_size/2) { y1 = grid_size/2; }
        double complex *uvgrid_mid = uvgrid + (grid_size+1)*grid_size/2;
        int x, y;
        for (y = y0; y < y1; y++) {
            for (x = x0; x < x1; x++) {
                uvgrid_mid[x + y*grid_size] += subimg[(x-x_min+subgrid_margin/2) +
                                                      (y-y_min+subgrid_margin/2)*subgrid_size] / subgrid_size / subgrid_size;
            }
        }
        flops += 2 * (y1 - y0) * (x1 - x0);
    }

    // Clean up
    fftw_destroy_plan(fft_plan);
    fftw_destroy_plan(ifft_plan);
    free(subgrid);
    free(subimg);

    }

    // Check that all visibilities were actually gridded
    for (bl = 0; bl < vis->bl_count; bl++) {
        if (vis->bl[bl].flops !=
            8 * wkern->size_x * wkern->size_x * vis->bl[bl].time_count * vis->bl[bl].freq_count) {
            printf("!!! bl %d-%d: %lu flops, %d expected !!!\n",
                   vis->bl[bl].antenna1, vis->bl[bl].antenna2,
                   vis->bl[bl].flops,
                   8 * wkern->size_x * wkern->size_x * vis->bl[bl].time_count * vis->bl[bl].freq_count);
        }
    }

    int cx, cy;
    for (cy = 0; cy < grid_size / chunk_size + 1; cy++) {
        for (cx = 0; cx < grid_size / chunk_size + 1; cx++) {
            free(bins[cy*chunk_count + cx]);
        }
    }
    free(bins);
    free(bins_count);

    return flops;
}

void convolve_aw_kernels(struct bl_data *bl,
                         struct w_kernel_data *wkern,
                         struct a_kernel_data *akern) {

    assert(wkern->size_x == akern->size_x);
    assert(wkern->size_y == akern->size_y);
    int size_x = wkern->size_x, size_y = wkern->size_y;
    int ov = wkern->oversampling;

    // We assume that every time channel has their own w-kernel
    const int awkern_count = bl->time_count * akern->freq_count;
    const int awkern_size = size_x * size_y * ov * ov;
    bl->awkern = (double complex *)malloc(awkern_count * awkern_size * sizeof(double complex));

    int time, freq;
    for (time = 0; time < bl->time_count; time++) {
        for (freq = 0; freq < akern->freq_count; freq++) {
            double t = bl->time[time];
            int atime = (int)floor((t - akern->t_min) / akern->t_step + .5);
            int a1i = bl->antenna1 * akern->time_count * akern->freq_count + atime * akern->freq_count + freq;
            int a2i = bl->antenna2 * akern->time_count * akern->freq_count + atime * akern->freq_count + freq;
            struct a_kernel *a1k = &akern->kern_by_atf[a1i];
            struct a_kernel *a2k = &akern->kern_by_atf[a2i];
            double w = uvw_m_to_l(bl->uvw_m[time*3+2], a1k->freq);
            int w_plane = (int)floor((w - wkern->w_min) / wkern->w_step + .5);
            struct w_kernel *wk = &wkern->kern_by_w[w_plane];

            // Here is where we normally would convolve the kernel -
            // but it doesn't matter for this test, so we just copy
            // the w-kernel.
            memcpy(&bl->awkern[(time * akern->freq_count + freq) * awkern_size],
                   wk->data,
                   awkern_size * sizeof(double complex));

        }
    }
}

uint64_t grid_awprojection(double complex *uvgrid, int grid_size, double theta,
                           struct vis_data *vis,
                           struct w_kernel_data *wkern,
                           struct a_kernel_data *akern,
                           int bl_min, int bl_max) {

    // Note that we require awkern to be set on all baselines
    // processed here!

    uint64_t flops = 0;
    int bl, time, freq;
    for (bl = bl_min; bl < bl_max; bl++) {
        const int awkern_size = akern->size_x * akern->size_y *
                                wkern->oversampling * wkern->oversampling;
        const struct bl_data *pbl = &vis->bl[bl];
        for (time = 0; time < pbl->time_count; time++) {
            for (freq = 0; freq < pbl->freq_count; freq++) {
                // Calculate grid and sub-grid coordinates
                int grid_offset, sub_offset;
                frac_coord_2d(grid_size, wkern->size_x, wkern->oversampling,
                           theta, &vis->bl[bl], time, freq, 0, 0,
                           &grid_offset, &sub_offset);
                // Determine kernel frequency, get kernel
                int afreq = (int)floor((pbl->freq[freq] - akern->f_min)
                                       / akern->f_step + .5);
                double complex *awk = &pbl->awkern[
                     (time * akern->freq_count + afreq) * awkern_size];
                // Get visibility
                double complex v = vis->bl[bl].vis[time*vis->bl[bl].freq_count+freq];
                int x, y;
                for (y = 0; y < wkern->size_y; y++) {
                    for (x = 0; x < wkern->size_x; x++) {
                        uvgrid[grid_offset + y*grid_size + x]
                            += v * conj(awk[sub_offset + y*wkern->size_x + x]);
                    }
                }
                flops += 8 * wkern->size_x * wkern->size_y;
            }
        }
    }

    return flops;
}

void make_hermitian(double complex *uvgrid, int grid_size) {

    // Determine start index. For even-sized grids, the zero frequency
    // is at grid_size/2, so things are off by one.
    complex double *p0;
    if (grid_size % 2 == 0) {
        p0 = uvgrid + grid_size + 1;
    } else {
        p0 = uvgrid;
    }
    complex double *p1 = uvgrid + grid_size * grid_size - 1;

    // Now simply add cells from the other side of the grid
    while(p0 < p1) {
        double complex g0 = *p0;
        *p0++ += conj(*p1);
        *p1-- += conj(g0);
    }

    // Should end up exactly on the zero frequency
    assert( p0 == p1 && p0 == uvgrid + (grid_size+1) * (grid_size/2) );
    *p0 += conj(*p0);

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
