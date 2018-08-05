
#include "grid.h"
#include "config.h"

#include <hdf5.h>
#include <stdlib.h>
#include <time.h>

void simple_benchmark(const char *filename,
                      struct work_config *work_cfg,
                      bool write, int worker)
{

    struct subgrid_work *work = work_cfg->subgrid_work +
        worker*work_cfg->subgrid_max_work;

    // Some test data to write
    int j;
    int time_chunk = work_cfg->spec.time_chunk;
    int freq_chunk = work_cfg->spec.freq_chunk;
    double complex *data = (double complex *)
        malloc(sizeof(double complex) * time_chunk * freq_chunk);
    for (j = 0; j < time_chunk * freq_chunk; j++) {
            data[j] = 1111111111;
    }

    hid_t vis_f = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t vis_g = H5Gopen(vis_f, "vis", H5P_DEFAULT);

    double write_start = get_time_ns();
    printf(write ? "Writing... " : "Reading... ");
    fflush(stdout);

    // Run through all work we are supposed to do, writing data
    uint64_t bytes = 0;
    int iwork;
    for (iwork = 0; iwork < work_cfg->subgrid_max_work; iwork++) {
        if (!work[iwork].nbl) continue; 
        struct subgrid_work_bl *wbl;
        printf("%d ", iwork); fflush(stdout);
        // Loop through baselines
        for (wbl = work[iwork].bls; wbl; wbl = wbl->next) {
            struct bl_data bl;
            vis_spec_to_bl_data(&bl, &work_cfg->spec, wbl->a1, wbl->a2);
            // Loop through time/frequency chunks, assuming we'd
            // write them sequentially like this. Note that
            // baselines might overlap, leading to chunks getting
            // written multiple times. This will also happen in
            // reality, just not as often.
            int itime, ifreq;
            for (itime = 0; itime < work_cfg->spec.time_count / time_chunk; itime++)
                for (ifreq = 0; ifreq < work_cfg->spec.freq_count / freq_chunk; ifreq++) {
                    if (write)
                        write_vis_chunk(vis_g, &bl, time_chunk, freq_chunk, itime, ifreq, data);
                    else
                        read_vis_chunk(vis_g, &bl, time_chunk, freq_chunk, itime, ifreq, data);
                    bytes += sizeof(double complex) * time_chunk * freq_chunk;
                }
            free(bl.time); free(bl.freq); free(bl.uvw_m);
        }
    }

    H5Gclose(vis_g); H5Fclose(vis_f);
    double write_time = get_time_ns() - write_start;
    printf(write ? "\nWrote %g GB in %gs (%g GB/s)\n" : "\nRead %g GB in %gs (%g GB/s)\n",
           (double)bytes / 1000000000, write_time, (double)bytes / write_time / 1000000000);

    free(data);
}

int main(int argc, char *argv[])
{

    init_dtype_cpx();

    struct ant_config cfg;
    load_ant_config("../data/grid/LOWBD2_north_cfg.h5", &cfg);

    struct vis_spec spec;
    spec.cfg = &cfg;
    spec.fov = 0.1;
    spec.dec = 90 * atan(1) * 4 / 180;
    spec.time_start = 10 * -45 / 3600; // h
    spec.time_count = 64;
    spec.time_chunk = 64;
    spec.time_step = 0.9 / 3600; // h
    spec.freq_start = 250e6; // Hz
    spec.freq_count = 64;
    spec.freq_chunk = 64;
    spec.freq_step = 50.e6 / spec.freq_count; // Hz

    struct work_config work_cfg;
    config_init(&work_cfg, 9, 10);
    if (!config_set(&work_cfg,
                    32768, 4,
                    "../data/grid/T06_pswf.in",
                    8192, 8640, 16384,
                    384, 512, 276)) {
        return 1;
    }
    config_set_visibilities(&work_cfg, &spec, spec.fov * 1. / 0.75, NULL);
    config_assign_work(&work_cfg);

    int i;
    for (i = 0; i < work_cfg.subgrid_workers; i++) {

        // Get filename to use
        char filename[512];
        sprintf(filename, argc > 1 ? argv[1] : "out%d.h5", i);

        // Open file
        printf("\nCreating %s... ", filename);
        hid_t vis_f = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (vis_f < 0) {
            fprintf(stderr, "Could not visibility file %s!\n", filename);
            return 1;
        }
        // Create "vis" group
        hid_t vis_g = H5Gcreate(vis_f, "vis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (vis_g < 0) {
            fprintf(stderr, "Could not open 'vis' group in visibility file %s!\n", filename);
            return 1;
        }

        // Create all baseline groups
        create_bl_groups(vis_g, &work_cfg, i);
        H5Gclose(vis_g); H5Fclose(vis_f);

        // Run simple write+read benchmark
        simple_benchmark(filename, &work_cfg, true, i);
        simple_benchmark(filename, &work_cfg, false, i);

    }
    return 0;
}
