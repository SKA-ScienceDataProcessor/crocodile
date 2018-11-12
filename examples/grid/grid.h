
#ifndef GRID_H
#define GRID_H

#include <complex.h>
#include <stdint.h>
#include <math.h>
#include <fftw3.h>
#include <stdbool.h>
#include <hdf5.h>
#include <assert.h>

// Visibility data
struct bl_data
{
    int antenna1, antenna2;
    int time_count;
    int freq_count;
    double *time;
    double *freq;
    double *uvw_m; // in m
    double complex *vis;

    // temporary from here

    double complex *awkern;
    double u_min, u_max; // in m
    double v_min, v_max; // in m
    double w_min, w_max; // in m
    double t_min, t_max; // in h
    double f_min, f_max; // in Hz

    uint64_t flops;
};
struct vis_data
{
    int antenna_count;
    int bl_count;
    struct bl_data *bl;
};

// Antenna/station configuration
struct ant_config
{
    int ant_count;
    char *name;
    double *xyz; // x geographical east, z celestial north
};

static const double c = 299792458.0;

static inline double uvw_m_to_l(double u, double f) {
    return u * f / c;
}
static inline double uvw_l_to_m(double u, double f) {
    return u / f * c;
}

// Convert hour angle / declination to UVW and back for a certain baseline
static inline void ha_to_uvw_sc(struct ant_config *cfg, int a1, int a2,
                                double ha_sin, double ha_cos,
                                double dec_sin, double dec_cos,
                                double *uvw_m)
{
    double x = cfg->xyz[3*a2+0] - cfg->xyz[3*a1+0];
    double y = cfg->xyz[3*a2+1] - cfg->xyz[3*a1+1];
    double z = cfg->xyz[3*a2+2] - cfg->xyz[3*a1+2];
    // Rotate around z axis (hour angle)
    uvw_m[0] = x * ha_cos - y * ha_sin;
    double v0 = x * ha_sin + y * ha_cos;
    // Rotate around x axis (declination)
    uvw_m[2] = z * dec_sin - v0 * dec_cos;
    uvw_m[1] = z * dec_cos + v0 * dec_sin;
}
static inline void ha_to_uvw(struct ant_config *cfg, int a1, int a2,
                             double ha, double dec,
                             double *uvw_m)
{
    ha_to_uvw_sc(cfg, a1, a2, sin(ha), cos(ha), sin(dec), cos(dec), uvw_m);
}

static inline double uv_to_ha(struct ant_config *cfg, int a1, int a2,
                              double dec, double *uv_m)
{
    double x = cfg->xyz[3*a2+0] - cfg->xyz[3*a1+0];
    double y = cfg->xyz[3*a2+1] - cfg->xyz[3*a1+1];
    double z = cfg->xyz[3*a2+2] - cfg->xyz[3*a1+2];

    assert(dec != 0 && y*y + x*x != 0);
    double v0  = (uv_m[1] - z*cos(dec)) / sin(dec);
    return asin((x*v0 - y*uv_m[0]) / (y*y + x*x));
}

// Separable kernel data
struct sep_kernel_data
{
    double *data; // Assumed to be real
    int size;
    int oversampling;
};

// W-kernel data
struct w_kernel
{
    double complex *data;
    double w;
};
struct w_kernel_data
{
    int plane_count;
    struct w_kernel *kern;
    struct w_kernel *kern_by_w;
    double w_min, w_max, w_step;
    int size_x, size_y;
    int oversampling;
};

// A-kernel data
struct a_kernel
{
    double complex *data;
    int antenna;
    double time;
    double freq;
};
struct a_kernel_data
{
    int antenna_count, time_count, freq_count;
    struct a_kernel *kern;
    struct a_kernel *kern_by_atf;
    double t_min, t_max, t_step;
    double f_min, f_max, f_step;
    int size_x, size_y;
};

// Performance counter data
struct perf_counters
{
    int x87;
    int sse_ss;
    int sse_sd;
    int sse_ps;
    int sse_pd;
    int llc_miss;
};

// Prototypes
void init_dtype_cpx();
bool load_ant_config(const char *filename, struct ant_config *ant);
bool create_vis_group(hid_t vis_g, int freq_chunk, int time_chunk, bool skip_metadata,
                      struct bl_data *bl);
bool read_vis_chunk(hid_t vis_group,
                    struct bl_data *bl,
                    int time_chunk_size, int freq_chunk_size,
                    int time_chunk_ix, int freq_chunk_ix,
                    double complex *buf);
bool write_vis_chunk(hid_t vis_group,
                    struct bl_data *bl,
                    int time_chunk_size, int freq_chunk_size,
                    int time_chunk_ix, int freq_chunk_ix,
                     double complex *buf);

int load_vis(const char *filename, struct vis_data *vis,
             double min_len, double max_len);
int load_wkern(const char *filename, double theta,
               struct w_kernel_data *wkern);
int load_akern(const char *filename, double theta,
               struct a_kernel_data *akern);
int load_sep_kern(const char *filename, struct sep_kernel_data *sepkern);

void frac_coord(int grid_size, int oversample,
                double u, int *x, int *fx);
void weight(unsigned int *wgrid, int grid_size, double theta,
            struct vis_data *vis);
uint64_t grid_simple(double complex *uvgrid, int grid_size, double theta,
                     struct vis_data *vis);
uint64_t degrid_conv_bl(double complex *uvgrid, int grid_size, double theta,
                        double d_u, double d_v,
                        double min_u, double max_u, double min_v, double max_v,
                        struct bl_data *bl, int time0, int time1, int freq0, int freq1,
                        struct sep_kernel_data *kernel);
uint64_t grid_wprojection(double complex *uvgrid, int grid_size, double theta,
                          struct vis_data *vis, struct w_kernel_data *wkern);
uint64_t grid_wtowers(double complex *uvgrid, int grid_size,
                      double theta,
                      struct vis_data *vis, struct w_kernel_data *wkern,
                      int subgrid_size, int fsample_size,
                      int subgrid_margin, double wincrement);
void convolve_aw_kernels(struct bl_data *bl,
                         struct w_kernel_data *wkern,
                         struct a_kernel_data *akern);
uint64_t grid_awprojection(double complex *uvgrid, int grid_size, double theta,
                           struct vis_data *vis,
                           struct w_kernel_data *wkern,
                           struct a_kernel_data *akern,
                           int bl_min, int bl_max);
void make_hermitian(double complex *uvgrid, int grid_size);
void fft_shift(double complex *uvgrid, int grid_size);

void open_perf_counters(struct perf_counters *counter);
void enable_perf_counters(struct perf_counters *counter);
void disable_perf_counters(struct perf_counters *counter);
void print_perf_counters(struct perf_counters *counter,
                         uint64_t expected_flops,
                         uint64_t expected_mem);

int run_tests();
void *read_dump(int size, char *name, ...);
int get_npoints_hdf5(char *file, char *name, ...);
void *read_hdf5(int size, char *file, char *name, ...);

#endif // GRID_H
