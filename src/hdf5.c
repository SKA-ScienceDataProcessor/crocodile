
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <hdf5.h>

#include "grid.h"

// Nature constant
const double c = 299792458.0;

// Complex data type
hid_t dtype_cpx;

void init_dtype_cpx() {

    // HDF5 has no native complex datatype, so we mirror h5py here and
    // declare a compound equivalent.
    dtype_cpx = H5Tcreate(H5T_COMPOUND, sizeof(double complex));
    H5Tinsert(dtype_cpx, "r", 0, H5T_IEEE_F64LE);
    H5Tinsert(dtype_cpx, "i", 8, H5T_IEEE_F64LE);

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
    int a1, bl = 0, vis_count = 0, total_vis_count = 0;
    double u_min = DBL_MAX, u_max = DBL_MIN;
    double v_min = DBL_MAX, v_max = DBL_MIN;
    double w_min = DBL_MAX, w_max = DBL_MIN;
    double t_min = DBL_MAX, t_max = DBL_MIN;
    double f_min = DBL_MAX, f_max = DBL_MIN;
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

            // Read data, verify shape (... quite verbose ...)
            hid_t freq_ds = H5Dopen(a2_g, "frequency", H5P_DEFAULT);
            hid_t time_ds = H5Dopen(a2_g, "time", H5P_DEFAULT);
            hid_t uvw_ds = H5Dopen(a2_g, "uvw", H5P_DEFAULT);
            hid_t vis_ds = H5Dopen(a2_g, "vis", H5P_DEFAULT);
            hsize_t freq_dim, time_dim, uvw_dim[2], vis_dim[3];
            if (H5Sget_simple_extent_ndims(H5Dget_space(freq_ds)) == 1 &&
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
                vis_dim[0] == time_dim && vis_dim[1] == freq_dim && vis_dim[2] == 1) {

                // Determine visibility count
                int vis_c = vis_dim[0] * vis_dim[1] * vis_dim[2];
                total_vis_count += vis_c;

                // Use first uvw to decide whether to skip baseline
                vis->bl[bl].uvw = (double *)malloc(uvw_dim[0] * uvw_dim[1] * sizeof(double));
                H5Dread(uvw_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vis->bl[bl].uvw);
                double len = sqrt(vis->bl[bl].uvw[0] * vis->bl[bl].uvw[0] +
                                  vis->bl[bl].uvw[1] * vis->bl[bl].uvw[1]);
                if (len < min_len || len >= max_len) {
                    free(vis->bl[bl].uvw);
                } else {
                    vis_count += vis_c;

                    // Read the baseline
                    vis->bl[bl].antenna1 = a1;
                    vis->bl[bl].antenna2 = a2;
                    vis->bl[bl].time_count = time_dim;
                    vis->bl[bl].freq_count = freq_dim;
                    vis->bl[bl].time = (double *)malloc(time_dim * sizeof(double));
                    vis->bl[bl].freq = (double *)malloc(freq_dim * sizeof(double));
                    vis->bl[bl].vis = (double complex *)malloc(vis_c * sizeof(double complex));
                    H5Dread(time_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vis->bl[bl].time);
                    H5Dread(freq_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vis->bl[bl].freq);
                    H5Dread(vis_ds, dtype_cpx, H5S_ALL, H5S_ALL, H5P_DEFAULT, vis->bl[bl].vis);

                    // Statistics
                    int j;
                    for (j = 0; j < freq_dim; j++) {
                        if (f_min > vis->bl[bl].freq[j]) { f_min = vis->bl[bl].freq[j]; }
                        if (f_max < vis->bl[bl].freq[j]) { f_max = vis->bl[bl].freq[j]; }
                    }
                    for (j = 0; j < time_dim; j++) {
                        if (t_min > vis->bl[bl].time[j]) { t_min = vis->bl[bl].time[j]; }
                        if (t_max < vis->bl[bl].time[j]) { t_max = vis->bl[bl].time[j]; }
                        if (u_min > vis->bl[bl].uvw[3*j+0]) { u_min = vis->bl[bl].uvw[3*j+0]; }
                        if (u_max < vis->bl[bl].uvw[3*j+0]) { u_max = vis->bl[bl].uvw[3*j+0]; }
                        if (v_min > vis->bl[bl].uvw[3*j+1]) { v_min = vis->bl[bl].uvw[3*j+1]; }
                        if (v_max < vis->bl[bl].uvw[3*j+1]) { v_max = vis->bl[bl].uvw[3*j+1]; }
                        if (w_min > vis->bl[bl].uvw[3*j+2]) { w_min = vis->bl[bl].uvw[3*j+2]; }
                        if (w_max < vis->bl[bl].uvw[3*j+2]) { w_max = vis->bl[bl].uvw[3*j+2]; }
                    }

                    bl++;
                }

            } else {
                fprintf(stderr, "Baseline %s/%s has not the expected format!", a1_name, a2_name);
            }

            H5Dclose(freq_ds);
            H5Dclose(time_ds);
            H5Dclose(uvw_ds);
            H5Dclose(vis_ds);
            H5Gclose(a2_g);
        }
        H5Gclose(a1_g);
    }
    vis->bl_count = bl;

    H5Gclose(vis_g);
    H5Fclose(vis_f);

    printf("\n");
    if (vis_count < total_vis_count) {
        printf("Have %d visibilities (%d total)\n", vis_count, total_vis_count);
    } else {
        printf("Have %d visibilities\n", vis_count);
    }
    printf("u range:     %.2f - %.2f lambda\n", u_min*f_max/c, u_max*f_max/c);
    printf("v range:     %.2f - %.2f lambda\n", v_min*f_max/c, v_max*f_max/c);
    printf("w range:     %.2f - %.2f lambda\n", w_min*f_max/c, w_max*f_max/c);
    printf("Antennas:    %d - %d\n"           , 0, vis->antenna_count);
    printf("t range:     %.6f - %.6f MJD UTC\n", t_min, t_max);
    printf("f range:     %.2f - %.2f MHz\n"    , f_min/1e6, f_max/1e6);

    return 0;
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
    akern->t_min = DBL_MAX; akern->t_max = DBL_MIN;
    akern->f_min = DBL_MAX; akern->f_max = DBL_MIN;
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
