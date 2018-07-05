
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <hdf5.h>
#include <stdbool.h>

#include "grid.h"

// Complex data type
hid_t dtype_cpx;

void init_dtype_cpx() {

    // HDF5 has no native complex datatype, so we mirror h5py here and
    // declare a compound equivalent.
    dtype_cpx = H5Tcreate(H5T_COMPOUND, sizeof(double complex));
    H5Tinsert(dtype_cpx, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_cpx, "i", 8, H5T_NATIVE_DOUBLE);

}

int load_ant_config(const char *filename, struct ant_config *cfg) {

    // Open file
    printf("Reading %s...\n", filename);
    hid_t cfg_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (cfg_f < 0) {
        fprintf(stderr, "Could not open antenna configuration file %s!\n", filename);
        return 1;
    }
    hid_t cfg_g = H5Gopen(cfg_f, "cfg", H5P_DEFAULT);
    if (cfg_g < 0) {
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not open 'cfg' group in antenna configuration file %s!\n", filename);
        return 1;
    }

    // Read name
    hid_t name_a;
    if ((name_a = H5Aopen(cfg_g, "name", H5P_DEFAULT)) < 0 ||
        H5Tget_class(H5Aget_type(name_a)) != H5T_STRING ||
        H5Aread(name_a, H5Aget_type(name_a), &cfg->name) < 0) {

        H5Gclose(cfg_g);
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not read 'name' attribute from antenna configuration file %s!\n", filename);
        return 1;
    }

    // Read data, verify shape (... quite verbose ...)
    hid_t xyz_ds = H5Dopen(cfg_g, "xyz", H5P_DEFAULT);
    hsize_t xyz_dim[2];
    if (!(H5Tget_size(H5Dget_type(xyz_ds)) == sizeof(double) &&
          H5Sget_simple_extent_ndims(H5Dget_space(xyz_ds)) == 2 &&
          H5Sget_simple_extent_dims(H5Dget_space(xyz_ds), xyz_dim, NULL) >= 0 &&
          xyz_dim[1] == 3)) {

        H5Dclose(xyz_ds);
        H5Gclose(cfg_g);
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not read 'xyz' data from antenna configuration file %s!\n", filename);
        return false;
    }

    // Set/read data
    cfg->ant_count = xyz_dim[0];
    cfg->xyz = (double *)malloc(sizeof(double) * 3 * cfg->ant_count);
    H5Dread(xyz_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cfg->xyz);

    // Done
    H5Dclose(xyz_ds);
    H5Gclose(cfg_g);
    H5Fclose(cfg_f);
    return 0;
}

struct bl_stats {
    uint64_t vis_count, total_vis_count;
    double u_min, u_max;
    double v_min, v_max;
    double w_min, w_max;
    double t_min, t_max;
    double f_min, f_max;
};

static bool load_vis_group(hid_t vis_g, struct bl_data *bl,
                           int a1, int a2,
                           double min_len, double max_len,
                           struct bl_stats *stats) {

    // Read data, verify shape (... quite verbose ...)
    hid_t freq_ds = H5Dopen(vis_g, "frequency", H5P_DEFAULT);
    hid_t time_ds = H5Dopen(vis_g, "time", H5P_DEFAULT);
    hid_t uvw_ds = H5Dopen(vis_g, "uvw", H5P_DEFAULT);
    hid_t vis_ds = H5Dopen(vis_g, "vis", H5P_DEFAULT);
    hsize_t freq_dim, time_dim, uvw_dim[2], vis_dim[3];
    if (!(H5Sget_simple_extent_ndims(H5Dget_space(freq_ds)) == 1 &&
          H5Tget_size(H5Dget_type(freq_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(freq_ds), &freq_dim, NULL) >= 0 &&
          H5Sget_simple_extent_ndims(H5Dget_space(time_ds)) == 1 &&
          H5Tget_size(H5Dget_type(time_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(time_ds), &time_dim, NULL) >= 0 &&
          H5Sget_simple_extent_ndims(H5Dget_space(uvw_ds)) == 2 &&
          H5Tget_size(H5Dget_type(uvw_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(uvw_ds), uvw_dim, NULL) >= 0 &&
          uvw_dim[0] == time_dim && uvw_dim[1] == 3 &&
          H5Sget_simple_extent_ndims(H5Dget_space(vis_ds)) == 3 &&
          H5Tget_size(H5Dget_type(vis_ds)) == sizeof(double complex) &&
          H5Sget_simple_extent_dims(H5Dget_space(vis_ds), vis_dim, NULL) >= 0 &&
          vis_dim[0] == time_dim && vis_dim[1] == freq_dim && vis_dim[2] == 1)) {

        H5Dclose(freq_ds);
        H5Dclose(time_ds);
        H5Dclose(uvw_ds);
        H5Dclose(vis_ds);
        return false;
    }

    // Determine visibility count
    int vis_c = vis_dim[0] * vis_dim[1] * vis_dim[2];
    if (stats) { stats->total_vis_count += vis_c; }

    // Use first uvw to decide whether to skip baseline
    bl->uvw_m = (double *)malloc(uvw_dim[0] * uvw_dim[1] * sizeof(double));
    H5Dread(uvw_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->uvw_m);
    double len = sqrt(bl->uvw_m[0] * bl->uvw_m[0] +
                      bl->uvw_m[1] * bl->uvw_m[1]);
    if (len < min_len || len >= max_len) {
        free(bl->uvw_m);
        H5Dclose(freq_ds);
        H5Dclose(time_ds);
        H5Dclose(uvw_ds);
        H5Dclose(vis_ds);
        return false;
    }
    if (stats) { stats->vis_count += vis_c; }

    // Read the baseline
    bl->antenna1 = a1;
    bl->antenna2 = a2;
    bl->time_count = time_dim;
    bl->freq_count = freq_dim;
    bl->time = (double *)malloc(time_dim * sizeof(double));
    bl->freq = (double *)malloc(freq_dim * sizeof(double));
    bl->vis = (double complex *)malloc(vis_c * sizeof(double complex));
    H5Dread(time_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->time);
    H5Dread(freq_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->freq);
    H5Dread(vis_ds, dtype_cpx, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->vis);

    // Close groups
    H5Dclose(freq_ds);
    H5Dclose(time_ds);
    H5Dclose(uvw_ds);
    H5Dclose(vis_ds);

    // Statistics
    bl->u_min = DBL_MAX; bl->u_max = -DBL_MAX;
    bl->v_min = DBL_MAX; bl->v_max = -DBL_MAX;
    bl->w_min = DBL_MAX; bl->w_max = -DBL_MAX;
    bl->t_min = DBL_MAX; bl->t_max = -DBL_MAX;
    bl->f_min = DBL_MAX; bl->f_max = -DBL_MAX;
    int j;
    for (j = 0; j < freq_dim; j++) {
        if (bl->f_min > bl->freq[j]) { bl->f_min = bl->freq[j]; }
        if (bl->f_max < bl->freq[j]) { bl->f_max = bl->freq[j]; }
    }
    for (j = 0; j < time_dim; j++) {
        if (bl->t_min > bl->time[j])    { bl->t_min = bl->time[j]; }
        if (bl->t_max < bl->time[j])    { bl->t_max = bl->time[j]; }
        if (bl->u_min > bl->uvw_m[3*j+0]) { bl->u_min = bl->uvw_m[3*j+0]; }
        if (bl->u_max < bl->uvw_m[3*j+0]) { bl->u_max = bl->uvw_m[3*j+0]; }
        if (bl->v_min > bl->uvw_m[3*j+1]) { bl->v_min = bl->uvw_m[3*j+1]; }
        if (bl->v_max < bl->uvw_m[3*j+1]) { bl->v_max = bl->uvw_m[3*j+1]; }
        if (bl->w_min > bl->uvw_m[3*j+2]) { bl->w_min = bl->uvw_m[3*j+2]; }
        if (bl->w_max < bl->uvw_m[3*j+2]) { bl->w_max = bl->uvw_m[3*j+2]; }
    }

    if (stats) {
        if (stats->f_min > bl->f_min) { stats->f_min = bl->f_min; }
        if (stats->f_max < bl->f_max) { stats->f_max = bl->f_max; }
        if (stats->t_min > bl->t_min) { stats->t_min = bl->t_min; }
        if (stats->t_max < bl->t_max) { stats->t_max = bl->t_max; }
        if (stats->u_min > bl->u_min) { stats->u_min = bl->u_min; }
        if (stats->u_max < bl->u_max) { stats->u_max = bl->u_max; }
        if (stats->v_min > bl->v_min) { stats->v_min = bl->v_min; }
        if (stats->v_max < bl->v_max) { stats->v_max = bl->v_max; }
        if (stats->w_min > bl->w_min) { stats->w_min = bl->w_min; }
        if (stats->w_max < bl->w_max) { stats->w_max = bl->w_max; }
    }

    return true;
}

int load_vis(const char *filename, struct vis_data *vis,
             double min_len, double max_len) {

    // Open file
    printf("Reading %s...\n", filename);
    hid_t vis_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (vis_f < 0) {
        fprintf(stderr, "Could not open visibility file %s!\n", filename);
        return 1;
    }
    hid_t vis_g = H5Gopen(vis_f, "vis", H5P_DEFAULT);
    if (vis_g < 0) {
        fprintf(stderr, "Could not open 'vis' group in visibility file %s!\n", filename);
        return 1;
    }

    // Set up statistics
    struct bl_stats stats;
    stats.vis_count = stats.total_vis_count = 0;
    stats.u_min = DBL_MAX; stats.u_max = -DBL_MAX;
    stats.v_min = DBL_MAX; stats.v_max = -DBL_MAX;
    stats.w_min = DBL_MAX; stats.w_max = -DBL_MAX;
    stats.t_min = DBL_MAX; stats.t_max = -DBL_MAX;
    stats.f_min = DBL_MAX; stats.f_max = -DBL_MAX;

    // Check whether "vis" a flat visibility group (legacy - should
    // not have made a data set in this format in the first place.)
    hid_t type_a; char *type_str;
    int bl = 0;
    if (H5Aexists(vis_g, "type") &&
        (type_a = H5Aopen(vis_g, "type", H5P_DEFAULT)) >= 0 &&
        H5Tget_class(H5Aget_type(type_a)) == H5T_STRING &&
        H5Aread(type_a, H5Aget_type(type_a), &type_str) >= 0 &&
        strcmp(type_str, "Visibility") == 0) {

        // Read visibilities
        struct bl_data data;
        if (!load_vis_group(vis_g, &data, 0, 0, -DBL_MAX, DBL_MAX, &stats)) {
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;
        }

        // Read antenna datasets
        hid_t a1_ds = H5Dopen(vis_g, "antenna1", H5P_DEFAULT);
        hid_t a2_ds = H5Dopen(vis_g, "antenna2", H5P_DEFAULT);
        hsize_t a1_dim, a2_dim;
        if (!(H5Sget_simple_extent_ndims(H5Dget_space(a1_ds)) == 1 &&
              H5Tget_size(H5Dget_type(a1_ds)) == sizeof(int64_t) &&
              H5Sget_simple_extent_dims(H5Dget_space(a1_ds), &a1_dim, NULL) >= 0 &&
              a1_dim == data.time_count &&
              H5Sget_simple_extent_ndims(H5Dget_space(a2_ds)) == 1 &&
              H5Tget_size(H5Dget_type(a2_ds)) == sizeof(int64_t) &&
              H5Sget_simple_extent_dims(H5Dget_space(a2_ds), &a2_dim, NULL) >= 0 &&
              a2_dim == data.time_count)) {

            free(data.uvw_m);
            free(data.time);
            free(data.freq);
            free(data.vis);
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;

        }

        // Read antenna arrays
        int64_t *a1 = (int64_t *)malloc(a1_dim * sizeof(int64_t));
        int64_t *a2 = (int64_t *)malloc(a1_dim * sizeof(int64_t));
        H5Dread(a1_ds, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a1);
        H5Dread(a2_ds, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a2);
        H5Dclose(a1_ds);
        H5Dclose(a2_ds);

        // Split by baseline. We assume every visibility needs its own baseline.
        vis->bl_count = data.time_count;
        vis->bl = (struct bl_data *)calloc(vis->bl_count, sizeof(struct bl_data));
        stats.vis_count = 0;
        int i;
        for (i = 0; i < data.time_count; i++) {

            // Calculate baseline length (same check as in load_vis_group)
            double len = sqrt(data.uvw_m[3*i+0] * data.uvw_m[3*i+0] +
                              data.uvw_m[3*i+1] * data.uvw_m[3*i+1]);
            if (len < min_len || len >= max_len) {
                printf("asd %g\n", len);
                continue;
            }

            // Create 1-visibility baseline
            vis->bl[bl].antenna1 = a1[i];
            vis->bl[bl].antenna2 = a2[i];
            vis->bl[bl].time_count = 1;
            vis->bl[bl].freq_count = data.freq_count;
            vis->bl[bl].uvw_m = (double *)malloc(3 * sizeof(double));
            vis->bl[bl].time = (double *)malloc(sizeof(double));
            vis->bl[bl].freq = (double *)malloc(data.freq_count * sizeof(double));
            vis->bl[bl].vis = (double complex *)malloc(data.freq_count * sizeof(double complex));
            vis->bl[bl].time[0] = data.time[i];
            vis->bl[bl].uvw_m[0] = data.uvw_m[i*3+0];
            vis->bl[bl].uvw_m[1] = data.uvw_m[i*3+1];
            vis->bl[bl].uvw_m[2] = data.uvw_m[i*3+2];
            vis->bl[bl].u_min = vis->bl[bl].u_max = vis->bl[bl].uvw_m[0];
            vis->bl[bl].v_min = vis->bl[bl].v_max = vis->bl[bl].uvw_m[1];
            vis->bl[bl].w_min = vis->bl[bl].w_max = vis->bl[bl].uvw_m[2];
            vis->bl[bl].f_min = data.f_min;
            vis->bl[bl].f_max = data.f_max;
            int j;
            for (j = 0; j < data.freq_count; j++) {
                vis->bl[bl].freq[j] = data.freq[j];
                vis->bl[bl].vis[j] = data.vis[i*data.freq_count+j];
            }

            if (a1[i] > vis->antenna_count) { vis->antenna_count = a1[i]; }
            if (a2[i] > vis->antenna_count) { vis->antenna_count = a2[i]; }
            stats.vis_count++;
            bl++;
        }

        // Finish
        free(data.uvw_m);
        free(data.time);
        free(data.freq);
        free(data.vis);
        vis->bl_count = bl;

    } else {

        // Read number of baselines
        hsize_t nobjs = 0;
        H5Gget_num_objs(vis_g, &nobjs);
        vis->antenna_count = nobjs+1;
        if (vis->antenna_count == 0) {
            fprintf(stderr, "Found no antenna data in visibility file %s!\n", filename);
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;
        }
        vis->bl_count = vis->antenna_count * (vis->antenna_count - 1) / 2;

        // Read baselines
        vis->bl = (struct bl_data *)calloc(vis->bl_count, sizeof(struct bl_data));
        int a1, bl = 0;
        for (a1 = 0; a1 < vis->antenna_count-1; a1++) {
            char a1_name[64];
            sprintf(a1_name, "%d", a1);
            hid_t a1_g = H5Gopen(vis_g, a1_name, H5P_DEFAULT);
            if (a1_g < 0) {
                fprintf(stderr, "Antenna1 %s not found!", a1_name);
                continue;
            }

            int a2;
            for (a2 = a1+1; a2 < vis->antenna_count; a2++) {
                char a2_name[64];
                sprintf(a2_name, "%d", a2);
                hid_t a2_g = H5Gopen(a1_g, a2_name, H5P_DEFAULT);
                if (a2_g < 0) {
                    fprintf(stderr, "Antenna2 %s/%s not found!", a1_name, a2_name);
                    continue;
                }

                // Read group data
                if (load_vis_group(a2_g, &vis->bl[bl], a1, a2, min_len, max_len, &stats)) {

                    // Next baseline!
                    bl++;
                }

                H5Gclose(a2_g);
            }
            H5Gclose(a1_g);
        }
        vis->bl_count = bl;
    }

    H5Gclose(vis_g);
    H5Fclose(vis_f);

    printf("\n");
    if (stats.vis_count < stats.total_vis_count) {
        printf("Have %d baselines, %lu uvw positions, %lu visibilities\n", vis->bl_count, stats.vis_count, stats.total_vis_count);
    } else {
        printf("Have %d baselines and %lu visibilities\n", vis->bl_count, stats.vis_count);
    }
    printf("u range:     %.2f - %.2f lambda\n", stats.u_min*stats.f_max/c, stats.u_max*stats.f_max/c);
    printf("v range:     %.2f - %.2f lambda\n", stats.v_min*stats.f_max/c, stats.v_max*stats.f_max/c);
    printf("w range:     %.2f - %.2f lambda\n", stats.w_min*stats.f_max/c, stats.w_max*stats.f_max/c);
    printf("Antennas:    %d - %d\n"           , 0, vis->antenna_count);
    printf("t range:     %.6f - %.6f MJD UTC\n", stats.t_min, stats.t_max);
    printf("f range:     %.2f - %.2f MHz\n"    , stats.f_min/1e6, stats.f_max/1e6);

    return 0;
}

bool create_vis_group(hid_t vis_g, int freq_chunk, int time_chunk,
                      struct bl_data *bl) {

    // Create a visibility group from baseline data *without* actually
    // writing any visibility data (data in "bl" will be ignored).
    // Instead, we will write a chunked visibility dataset with zeros
    // for fill value.

    // Create properties for compact and contigous data
    hid_t compact_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(compact_ds_prop, H5D_COMPACT);
    hid_t cont_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(cont_ds_prop, H5D_CONTIGUOUS);
    hid_t chunked_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(chunked_ds_prop, H5D_CHUNKED);
    hsize_t chunks[3] = { freq_chunk, time_chunk, 1 };
    H5Pset_chunk(chunked_ds_prop, 3, chunks);
    complex double fill_value = 0;
    H5Pset_fill_value(chunked_ds_prop, dtype_cpx, &fill_value);

    // Create datasets
    hsize_t freq_size = bl->freq_count;
    hid_t freq_dsp = H5Screate_simple(1, &freq_size, NULL);
    hid_t freq_ds = H5Dcreate(vis_g, "frequency", H5T_IEEE_F64LE,
                              freq_dsp, H5P_DEFAULT, compact_ds_prop, H5P_DEFAULT);
    if (freq_ds < 0) {
        fprintf(stderr, "failed to create frequency dataset!");
        H5Sclose(freq_dsp);
        H5Pclose(compact_ds_prop); H5Pclose(cont_ds_prop); H5Pclose(chunked_ds_prop);
        return false;
    }
    hsize_t time_size = bl->time_count;
    hid_t time_dsp = H5Screate_simple(1, &time_size, NULL);
    hid_t time_ds = H5Dcreate(vis_g, "time", H5T_IEEE_F64LE,
                              H5Screate_simple(1, &time_size, NULL),
                              H5P_DEFAULT, compact_ds_prop, H5P_DEFAULT);
    if (time_ds < 0) {
        fprintf(stderr, "failed to create time dataset!");
        H5Sclose(freq_dsp); H5Sclose(time_dsp);
        H5Dclose(freq_ds);
        H5Pclose(compact_ds_prop); H5Pclose(cont_ds_prop); H5Pclose(chunked_ds_prop);
        return false;
    }
    hsize_t uvw_dims[2] = { bl->time_count, 3 };
    hid_t uvw_dsp = H5Screate_simple(2, uvw_dims, NULL);
    hid_t uvw_ds = H5Dcreate(vis_g, "uvw", H5T_IEEE_F64LE,
                             uvw_dsp, H5P_DEFAULT, cont_ds_prop, H5P_DEFAULT);
    if (uvw_ds < 0) {
        fprintf(stderr, "failed to create coordinates dataset!");
        H5Sclose(uvw_dsp); H5Sclose(freq_dsp); H5Sclose(time_dsp);
        H5Dclose(freq_ds); H5Dclose(time_ds);
        H5Pclose(compact_ds_prop); H5Pclose(cont_ds_prop); H5Pclose(chunked_ds_prop);
        return false;
        }
    hsize_t vis_dims[3] = { 0, 0, 1 };
    hsize_t max_vis_dims[3] = { bl->time_count, bl->freq_count, 1 };
    hid_t vis_dsp = H5Screate_simple(3, vis_dims, max_vis_dims);
    hid_t vis_ds = H5Dcreate(vis_g, "vis", dtype_cpx,
                             vis_dsp, H5P_DEFAULT, chunked_ds_prop, H5P_DEFAULT);
    if (vis_ds < 0) {
        fprintf(stderr, "failed to create visibility dataset!");
        H5Sclose(vis_dsp); H5Sclose(uvw_dsp); H5Sclose(freq_dsp); H5Sclose(time_dsp);
        H5Dclose(uvw_ds); H5Dclose(freq_ds); H5Dclose(time_ds);
        H5Pclose(compact_ds_prop); H5Pclose(cont_ds_prop); H5Pclose(chunked_ds_prop);
        return false;
    }

    H5Dwrite(freq_ds, H5T_NATIVE_DOUBLE, freq_dsp, freq_dsp, H5P_DEFAULT, bl->freq);
    H5Dwrite(time_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->time);
    H5Dwrite(uvw_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->uvw_m);

    H5Sclose(vis_dsp); H5Sclose(uvw_dsp); H5Sclose(freq_dsp); H5Sclose(time_dsp);
    H5Dclose(vis_ds); H5Dclose(uvw_ds); H5Dclose(freq_ds); H5Dclose(time_ds);

    H5Pclose(compact_ds_prop); H5Pclose(cont_ds_prop); H5Pclose(chunked_ds_prop);
    return true;
}


int load_wkern(const char *filename, double theta, struct w_kernel_data *wkern) {

    // Open file
    hid_t wkern_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (wkern_f < 0) {
        fprintf(stderr, "Could not open w kernel file %s!\n", filename);
        return 1;
    }

    // Access appropriate w-kernel group
    char wkern_name[64];
    sprintf(wkern_name, "wkern/%g", theta);
    hid_t wkern_g = H5Gopen(wkern_f, wkern_name, H5P_DEFAULT);
    if (wkern_g < 0) {
        fprintf(stderr, "Could not open '%s' group in w kernel file %s!\n", wkern_name, filename);
        H5Fclose(wkern_f);
        return 1;
    }

    // Read number of w-planes
    hsize_t nobjs = 0;
    H5Gget_num_objs(wkern_g, &nobjs);
    wkern->plane_count = nobjs;
    if (wkern->plane_count == 0) {
        fprintf(stderr, "Found no w-kernels in w-kernel file %s!\n", filename);
        H5Gclose(wkern_g);
        H5Fclose(wkern_f);
        return 1;
    }

    // Read kernels
    wkern->kern = (struct w_kernel *)calloc(wkern->plane_count, sizeof(struct w_kernel));
    wkern->size_x = wkern->size_y = wkern->oversampling = 0;
    int i;
    for (i = 0; i < wkern->plane_count; i++) {
        char name[64];
        H5Gget_objname_by_idx(wkern_g, i, name, sizeof(name));

        // Save w-value
        double w = atof(name);
        wkern->kern[i].w = w;
        if (w > wkern->w_max) { wkern->w_max = w; }
        if (w < wkern->w_min) { wkern->w_min = w; }

        // Open the data set
        char data_name[64];
        sprintf(data_name, "%s/kern", name);
        hid_t dset = H5Dopen(wkern_g, data_name, H5P_DEFAULT);

        // Check that it has the expected format
        if (H5Sget_simple_extent_ndims(H5Dget_space(dset)) == 4 &&
            H5Tget_size(H5Dget_type(dset)) == sizeof(double complex)) {

            // Read dimensions
            hsize_t dims[4];
            H5Sget_simple_extent_dims(H5Dget_space(dset), dims, NULL);
            if (wkern->oversampling == 0) {
                wkern->oversampling = dims[0];
                wkern->size_y = dims[2];
                wkern->size_x = dims[3];
            }
            if (wkern->oversampling == dims[0] && wkern->oversampling == dims[1] &&
                wkern->size_y == dims[2] && wkern->size_x == dims[3]) {

                // Read kernel
                hsize_t total_size = wkern->oversampling * wkern->oversampling * wkern->size_y * wkern->size_x;
                wkern->kern[i].data = (double complex *)calloc(sizeof(double complex), total_size);
                H5Dread(dset, dtype_cpx, H5S_ALL, H5S_ALL, H5P_DEFAULT, wkern->kern[i].data);

            } else {
                fprintf(stderr, "kernel %s has inconsistent dimensions - ignored!\n", data_name);
            }
        }
        H5Dclose(dset);
    }

    // Close file
    H5Gclose(wkern_g);
    H5Fclose(wkern_f);

    // Complain if anything is amiss
    if (wkern->oversampling <= 0 || wkern->size_y <= 0 || wkern->size_x <= 0) {
        fprintf(stderr, "Invalid dimensions in w-kernel file %s!\n", filename);
        return 1;
    }

    // Index kernels by w-value
    wkern->kern_by_w = (struct w_kernel *)malloc(sizeof(struct w_kernel) * wkern->plane_count);
    wkern->w_step = (wkern->w_max - wkern->w_min) / (wkern->plane_count - 1);
    for (i = 0; i < wkern->plane_count; i++) {
        double w = wkern->w_min + (i * wkern->w_step);

        // Find closest kernel. We should find an exact match if the
        // w-planes are evenly spaced, but this is more robust.
        int best = 0, j;
        for (j = 1; j < wkern->plane_count; j++) {
            if (fabs(wkern->kern[j].w - w) < fabs(wkern->kern[best].w - w)) {
                best = j;
            }
        }

        // Set
        wkern->kern_by_w[i] = wkern->kern[best];
    }
    printf("w kernels:   %.2f - %.2f lambda (step %.2f lambda)\n",
           wkern->w_min, wkern->w_max, wkern->w_step);
    printf("             %d x %d pixels (%d oversampled)",
           wkern->size_x, wkern->size_y, wkern->oversampling);

    return 0;
}

int load_akern(const char *filename, double theta, struct a_kernel_data *akern) {
    char akern_name[64];

    // Open file
    hid_t akern_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (akern_f < 0) {
        fprintf(stderr, "Could not open A kernel file %s!\n", filename);
        return 1;
    }

    // Access appropriate a-kernel group
    sprintf(akern_name, "akern/%g", theta);
    hid_t akern_g = H5Gopen(akern_f, akern_name, H5P_DEFAULT);
    if (akern_g < 0) {
        fprintf(stderr, "Could not open '%s' group in A kernel file %s!\n", akern_name, filename);
        return 1;
    }

    // Determine number of antennas, time and frequency steps
    hsize_t nobjs = 0;
    H5Gget_num_objs(akern_g, &nobjs);
    akern->antenna_count = nobjs;
    char ant_name[64]; H5Gget_objname_by_idx(akern_g, 0, ant_name, 64);
    hid_t ant_g = H5Gopen(akern_g, ant_name, H5P_DEFAULT);
    H5Gget_num_objs(ant_g, &nobjs);
    akern->time_count = nobjs;
    char time_name[64]; H5Gget_objname_by_idx(ant_g, 0, time_name, 64);
    hid_t time_g = H5Gopen(ant_g, time_name, H5P_DEFAULT);
    H5Gget_num_objs(time_g, &nobjs);
    akern->freq_count = nobjs;
    H5Gclose(time_g);
    H5Gclose(ant_g);
    if (akern->antenna_count == 0 || akern->time_count == 0 || akern->freq_count == 0) {
        fprintf(stderr, "Found no w-kernels in w-kernel file %s!\n", filename);
        H5Gclose(akern_g);
        H5Fclose(akern_f);
        return 1;
    }

    // Read kernels
    int total_kernels = akern->antenna_count * akern->time_count * akern->freq_count;
    akern->kern = (struct a_kernel *)calloc(total_kernels, sizeof(struct a_kernel));
    akern->t_min = DBL_MAX; akern->t_max = -DBL_MAX;
    akern->f_min = DBL_MAX; akern->f_max = -DBL_MAX;
    akern->size_x = akern->size_y = 0;
    int *ant_ix = (int *)calloc(akern->antenna_count, sizeof(int));
    int ant; int i = 0;
    for (ant = 0; ant < akern->antenna_count; ant++) {

        // Open antenna group, check number of children
        H5Gget_objname_by_idx(akern_g, ant, ant_name, 64);
        ant_g = H5Gopen(akern_g, ant_name, H5P_DEFAULT);
        H5Gget_num_objs(ant_g, &nobjs);
        if (nobjs != akern->time_count) {
            fprintf(stderr, "Antenna %s has inconsistent time slots - ignored!", ant_name);
            continue;
        }

        // Quick antenna index. We need to assume that antennas are zero-based.
        if (atoi(ant_name) >= 0 && atoi(ant_name) < akern->antenna_count) {
            ant_ix[atoi(ant_name)] = i;
        }

        int time = 0;
        for (time = 0; time < akern->time_count; time++) {

            // Open time slot group, check number of children
            H5Gget_objname_by_idx(ant_g, time, time_name, 64);
            time_g = H5Gopen(ant_g, time_name, H5P_DEFAULT);
            H5Gget_num_objs(time_g, &nobjs);
            if (nobjs != akern->freq_count) {
                fprintf(stderr, "Time slot %s has inconsistent frequency slots - ignored!", time_name);
                continue;
            }

            // Get time
            double t = atof(time_name);
            if (t > akern->t_max) { akern->t_max = t; }
            if (t < akern->t_min) { akern->t_min = t; }

            int freq = 0;
            for (freq = 0; freq < akern->freq_count; freq++, i++) {

                // Open kernel dataset for indexed frequency band
                char freq_name[64], data_name[64];
                H5Gget_objname_by_idx(time_g, freq, freq_name, 64);
                sprintf(data_name, "%s/kern", freq_name);
                hid_t dset = H5Dopen(time_g, data_name, H5P_DEFAULT);

                // Get frequency
                double f = atof(freq_name);
                if (f > akern->f_max) { akern->f_max = f; }
                if (f < akern->f_min) { akern->f_min = f; }

                // Check that it has the expected format
                if (H5Sget_simple_extent_ndims(H5Dget_space(dset)) == 2 &&
                    H5Tget_size(H5Dget_type(dset)) == sizeof(double complex)) {

                    // Read dimensions
                    hsize_t dims[2];
                    H5Sget_simple_extent_dims(H5Dget_space(dset), dims, NULL);
                    if (akern->size_x == 0) {
                        akern->size_y = dims[1];
                        akern->size_x = dims[0];
                    }
                    if (akern->size_y == dims[0] && akern->size_x == dims[1]) {

                        // Read kernel
                        hsize_t total_size = akern->size_y * akern->size_x;
                        akern->kern[i].antenna = atoi(ant_name);
                        akern->kern[i].time = atof(time_name);
                        akern->kern[i].freq = atof(freq_name);
                        akern->kern[i].data = (double complex *)calloc(sizeof(double complex), total_size);
                        H5Dread(dset, dtype_cpx, H5S_ALL, H5S_ALL, H5P_DEFAULT, akern->kern[i].data);

                    } else {
                        fprintf(stderr, "kernel %s has inconsistent dimensions - ignored!\n", data_name);
                    }
                }
                H5Dclose(dset);
            }
            H5Gclose(time_g);
        }
        H5Gclose(ant_g);
    }
    H5Gclose(akern_g);
    H5Fclose(akern_f);

    // Determine step length, show statistics
    akern->t_step = (akern->t_max - akern->t_min) / (akern->time_count - 1);
    akern->f_step = (akern->f_max - akern->f_min) / (akern->freq_count - 1);
    printf("A kernels:   %d antennas\n", akern->antenna_count);
    printf(" \" t range:  %.6f - %.6f MJD UTC (step %.2f s)\n", akern->t_min, akern->t_max, akern->t_step * 24 * 3600);
    printf(" \" f range:  %.2f - %.2f MHz (step %.2f MHz)\n", akern->f_min/1e6, akern->f_max/1e6, akern->f_step/1e6);

    // Index kernels by antenna, frequency and time
    printf("Indexing A kernels...\n");
    akern->kern_by_atf = (struct a_kernel *)calloc(total_kernels, sizeof(struct a_kernel));
    for (ant = 0; ant < akern->antenna_count; ant++) {
        int time;
        for (time = 0; time < akern->time_count; time++) {
            double t = akern->t_min + time * akern->t_step;
            int freq;
            for (freq = 0; freq < akern->freq_count; freq++) {
                double f = akern->f_min + freq * akern->f_step;

                // Find best kernel. As with w-kernels, we actually expect
                // things to match exactly.
                int best = ant_ix[ant], j;
                for (j = ant_ix[ant]+1; j < ant_ix[ant]+akern->time_count * akern->freq_count; j++) {
                    if(fabs(akern->kern[j].time - t) + fabs(akern->kern[j].freq - f) <
                       fabs(akern->kern[best].time - t) + fabs(akern->kern[best].freq - f)) {
                        best = j;
                    }
                }

                // Set. Do some checking to make sure we do not assign
                // an A-kernel to two slots. Not required for
                // functionality, but
                assert(akern->kern[best].data);
                int i = ant * akern->time_count * akern->freq_count + time * akern->freq_count + freq;
                akern->kern_by_atf[i] = akern->kern[best];
                akern->kern[best].data = NULL;
            }
        }
    }

    return 0;
}
