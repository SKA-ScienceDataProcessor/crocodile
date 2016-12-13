
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>

#include "grid.h"

// Assume all uvw are zero. This eliminates all coordinate
// calculations and grids only into the middle.
//#define ASSUME_UVW_0

static const double c = 299792458.0;

inline double uvw_lambda(struct bl_data *bl_data,
                         int time, int freq, int uvw) {
    return bl_data->uvw[3*time+uvw] * bl_data->freq[freq] / c;
}

inline int coord(int grid_size, double theta,
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

inline void frac_coord(int grid_size, int kernel_size, int oversample,
                       double theta,
                       struct bl_data *bl_data,
                       int time, int freq,
                       int *grid_offset, int *sub_offset) {
#ifdef ASSUME_UVW_0
    double u = 0, v = 0;
#else
    double u = theta * uvw_lambda(bl_data, time, freq, 0);
    double v = theta * uvw_lambda(bl_data, time, freq, 1);
#endif
    int flx = (int)floor(u + .5 / oversample);
    int fly = (int)floor(v + .5 / oversample);
    int xf = (int)floor((u - (double)flx) * oversample + .5);
    int yf = (int)floor((v - (double)fly) * oversample + .5);
    *grid_offset =
        (fly+grid_size/2-kernel_size/2)*grid_size +
        (flx+grid_size/2-kernel_size/2);
    *sub_offset = kernel_size * kernel_size * (yf*oversample + xf);
}

void weight(unsigned int *wgrid, int grid_size, double theta,
            struct vis_data *vis) {

    // Simple uniform weighting
    int bl, time, freq;
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

uint64_t grid_wprojection(double complex *uvgrid, int grid_size, double theta,
                          struct vis_data *vis, struct w_kernel_data *wkern) {

    uint64_t flops = 0;
    int bl, time, freq;
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                // Calculate grid and sub-grid coordinates
                int grid_offset, sub_offset;
                frac_coord(grid_size, wkern->size_x, wkern->oversampling,
                           theta, &vis->bl[bl], time, freq,
                           &grid_offset, &sub_offset);
                // Determine w-kernel to use
                double w = uvw_lambda(&vis->bl[bl], time, freq, 2);
                int w_plane = (int)floor((w - wkern->w_min) / wkern->w_step + .5);
                double complex *wk = wkern->kern[w_plane].data;
                // Copy kernel
                int x, y;
                for (y = 0; y < wkern->size_y; y++) {
                    for (x = 0; x < wkern->size_x; x++) {
                        uvgrid[grid_offset + y*grid_size + x]
                            += vis->bl[bl].vis[time*vis->bl[bl].freq_count+freq] *
                               wk[sub_offset + y*wkern->size_x + x];
                    }
                }
                flops += 8 * wkern->size_x * wkern->size_y;
            }
        }
    }

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
            double w = bl->uvw[time*3+2] * a1k->freq / c;
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
                frac_coord(grid_size, wkern->size_x, wkern->oversampling,
                           theta, &vis->bl[bl], time, freq,
                           &grid_offset, &sub_offset);
                // Determine kernel frequency, get kernel
                int afreq = (int)floor((pbl->freq[freq] - akern->f_min)
                                       / akern->f_step + .5);
                double complex *awk = &pbl->awkern[
                     (time * akern->freq_count + afreq) * awkern_size];
                int x, y;
                for (y = 0; y < wkern->size_y; y++) {
                    for (x = 0; x < wkern->size_x; x++) {
                        uvgrid[grid_offset + y*grid_size + x]
                            += pbl->vis[time*pbl->freq_count+freq] *
                               awk[sub_offset + y*wkern->size_x + x];
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
