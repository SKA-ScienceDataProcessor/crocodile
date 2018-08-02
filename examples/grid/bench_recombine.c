
#include "grid.h"
#include "config.h"

#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

bool set_default_vis_spec(const char *ant_cfg_file, struct vis_spec *spec, double fov)
{
    spec->cfg = malloc(sizeof(struct ant_config));
    if (!load_ant_config(ant_cfg_file, spec->cfg))
        return false;

    spec->fov = fov;
    spec->dec = 90 * atan(1) * 4 / 180;
    spec->time_start = 10 * -45 / 3600; // h
    spec->time_count = 64;
    spec->time_chunk = 16;
    spec->time_step = 0.9 / 3600; // h
    spec->freq_start = 250e6; // Hz
    spec->freq_count = 64;
    spec->freq_chunk = 16;
    spec->freq_step = 50.e6 / spec->freq_count; // Hz

    return true;
}

bool set_default_recombine2d_config(struct work_config *cfg,
                                    int facet_workers, int subgrid_workers)
{

    struct vis_spec *spec = malloc(sizeof(struct vis_spec));
    if (!set_default_vis_spec("../../data/grid/LOWBD2_north_cfg.h5", spec, 0.1))
        return false;

    return work_config_set(cfg, spec, facet_workers, subgrid_workers,
                           spec->fov * 1. / 0.75,
                           98304, 1536, "../../data/grid/pswf5.00-33728.in",
                           24576, 33728, 49152, 1405, 1536, 282);
}

bool set_test_recombine2d_config(struct work_config *cfg,
                                 int facet_workers, int subgrid_workers,
                                 int rank)
{

    if (!work_config_set(cfg, 0, facet_workers, subgrid_workers,
                         0.15,
                         2000, 100, "../../data/grid/T04_pswf.in",
                         400, 480, 900, 400, 500, 247))
        return false;

    load_facets_from(cfg, "../../data/grid/T04_facet%d%d.in", NULL);

    // Use data from test suite. Note that not all "nmbf" reference
    // files exist in the repository, so this will show a few errors.
    // As long as no value mismatches occur this is fine
    char file[256];
    sprintf(file, "../../data/grid/T04_nmbf%%d%%d%d%d.in", rank / 3, rank % 3);
    cfg->recombine.stream_check = strdup(file);
    cfg->recombine.stream_check_threshold = 1e-9;
    return true;
}

bool recombine2d_set_test5_config(struct work_config *cfg,
                                  int facet_workers, int subgrid_workers,
                                  int rank)
{
    if (!work_config_set(cfg, NULL, facet_workers, subgrid_workers,
                         0.1,
                         512, 128, "../../data/grid/T05_pswf.in",
                         128, 140, 216, 128, 256, 136))
        return false;

    const char *hdf5 = "../../data/grid/T05_in.h5";
    load_facets_from(cfg, "j0=%d/j1=%d/facet", hdf5);
    check_subgrids_against(cfg, 1.2e-8, 5.5e-6,
                           "i0=%d/i1=%d/approx",
                           "i0=%d/i1=%d/j0=%%d/j1=%%d/nmbf", hdf5);
    return true;
}

bool set_small_test_config(struct work_config *cfg,
                           int facet_workers, int subgrid_workers,
                           int rank)
{
    struct vis_spec *spec = malloc(sizeof(struct vis_spec));
    if (!set_default_vis_spec("../../data/grid/LOWBD2_north_cfg.h5", spec, 0.05))
        return false;

    if (!work_config_set(cfg, spec, facet_workers, subgrid_workers,
                         spec->fov * 1. / 0.75,
                         16384, 16, "../../data/grid/T06_pswf_small.in",
                         5120, 5280, 8192,
                         432, 512, 274))
        return false;

    return true;
}

bool set_serious_test_config(struct work_config *cfg,
                             int facet_workers, int subgrid_workers,
                             int rank)
{
    struct vis_spec *spec = malloc(sizeof(struct vis_spec));
    if (!set_default_vis_spec("../../data/grid/LOWBD2_north_cfg.h5", spec, 0.1))
        return false;

    if (!work_config_set(cfg, spec, facet_workers, subgrid_workers,
                         spec->fov * 1. / 0.75,
                         32768, 4, "../../data/grid/T06_pswf.in",
                         8192, 8640, 16384,
                         384, 512, 276))
        return false;

    return true;
}

enum Opts
    {
        Opt_flag,

        Opt_telescope, Opt_fov, Opt_dec, Opt_time, Opt_freq, Opt_grid, Opt_vis_set,
        Opt_recombine, Opt_rec_aa, Opt_rec_set, Opt_rec_load_facet, Opt_rec_load_facet_hdf5,
        Opt_facet_workers, Opt_parallel_cols, Opt_dont_retain_bf,
    };


bool set_cmdarg_config(int argc, char **argv,
                       struct work_config *cfg, int world_rank, int world_size)
{
    // Read parameters
    int produce_parallel_cols = false;
    int produce_retain_bf = true;
    struct option options[] =
      {
        {"telescope",  required_argument, 0, Opt_telescope },
        {"dec",        required_argument, 0, Opt_dec },
        {"fov",        required_argument, 0, Opt_fov },
        {"time",       required_argument, 0, Opt_time },
        {"freq",       required_argument, 0, Opt_freq },
        {"grid",       required_argument, 0, Opt_grid },
        {"vis-set",    required_argument, 0, Opt_vis_set},

        {"recombine",  required_argument, 0, Opt_recombine },
        {"rec-aa",     required_argument, 0, Opt_rec_aa },
        {"rec-set",    required_argument, 0, Opt_rec_set },
        {"load-facet", required_argument, 0, Opt_rec_load_facet },
        {"load-facet-hdf5", required_argument, 0, Opt_rec_load_facet_hdf5 },

        {"facet-workers", required_argument, 0, Opt_facet_workers },
        {"parallel-columns", no_argument, &produce_parallel_cols, true },
        {"dont-retain-bf", no_argument,   &produce_retain_bf, false },

        {0, 0, 0, 0}
      };

    bool have_vis_spec = false;
    struct vis_spec spec;
    int recombine_pars[8]; char aa_path[256];
    char facet_path[256]; char facet_path_hdf5[256];
    int facet_workers = (world_size + 1) / 2;
    double grid_x0 = 0.4; char gridder_path[256];
    memset(&spec, 0, sizeof(spec));
    spec.dec = 90 * atan(1) * 4 / 180;
    memset(&recombine_pars, 0, sizeof(recombine_pars));
    aa_path[0] = gridder_path[0] = facet_path[0] = facet_path_hdf5[0] = 0;

    int option_index = 0, invalid = 0, c, nscan;
    while ((c = getopt_long(argc, argv, ":", options, &option_index)) != -1) {
        switch(c) {
        case Opt_flag: break;
        case Opt_telescope:
            free(spec.cfg);
            spec.cfg = malloc(sizeof(struct ant_config));
            if (!load_ant_config(optarg, spec.cfg))
                return false;
            have_vis_spec = true;
            break;
        case Opt_fov: spec.fov = atof(optarg); break;
        case Opt_dec: spec.dec = atof(optarg); break;
        case Opt_time:
            nscan = sscanf(optarg, "%lg:%lg/%d/%d",
                           &spec.time_start, &spec.time_step,
                           &spec.time_count, &spec.time_chunk);
            if (nscan < 3) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'time' option!\n");
            } else {
                if (nscan == 3) spec.time_chunk = spec.time_count;
                spec.time_start /= 3600; spec.time_step /= 3600;
                spec.time_step = (spec.time_step - spec.time_start) / spec.time_count;
            }
            break;
        case Opt_freq:
            nscan = sscanf(optarg, "%lg:%lg/%d/%d",
                           &spec.freq_start, &spec.freq_step,
                           &spec.freq_count, &spec.freq_chunk);
            if (nscan < 3) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'freq' option!\n");
            } else {
                if (nscan == 3) spec.freq_chunk = spec.freq_count;
                spec.freq_step = (spec.freq_step - spec.freq_start) / spec.freq_count;
            }
            break;
        case Opt_grid:
            nscan = sscanf(optarg, "%lg:%255s", &grid_x0, gridder_path);
            if (nscan < 2) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'grid' option!\n");
            }
            break;
        case Opt_vis_set:
            if (!strcasecmp(optarg, "vlaa")) {
                free(spec.cfg);
                spec.cfg = malloc(sizeof(struct ant_config));
                if (!load_ant_config("../../data/grid/LOWBD2_north_cfg.h5", spec.cfg))
                    return false;
                spec.fov = 0.08;
                spec.dec = 90 * atan(1) * 4 / 180;
                spec.time_start = 10 * -45 / 3600; // h
                spec.time_count = 64;
                spec.time_chunk = 16;
                spec.time_step = 10 * 0.9 / 3600; // h
                spec.freq_start = 250e6; // Hz
                spec.freq_count = 64;
                spec.freq_chunk = 16;
                spec.freq_step = 50.e6 / spec.freq_count; // Hz
                have_vis_spec = true;
            } else {
                invalid=true; fprintf(stderr, "ERROR: Unknown visibility parameter set: %s!\n", optarg);
            }
            break;
        case Opt_recombine:
            nscan = sscanf(optarg, "%d,%d,%d,%d,%d,%d,%d,%d",
                           recombine_pars,recombine_pars+1,recombine_pars+2,recombine_pars+3,
                           recombine_pars+4,recombine_pars+5,recombine_pars+6,recombine_pars+7);
            if (nscan < 8) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'recombine' option!\n");
            }
            break;
        case Opt_rec_aa:
            strncpy(aa_path, optarg, 254); aa_path[255] = 0;
            break;
        case Opt_rec_load_facet:
            strncpy(facet_path, optarg, 254); facet_path[255] = 0;
            break;
        case Opt_rec_load_facet_hdf5:
            strncpy(facet_path_hdf5, optarg, 254); facet_path[255] = 0;
            break;
        case Opt_rec_set:
            if (!strcasecmp(optarg, "T04")) {
                recombine_pars[0] = 2000;
                recombine_pars[1] = 100;
                recombine_pars[2] = 400;
                recombine_pars[3] = 480;
                recombine_pars[4] = 900;
                recombine_pars[5] = 400;
                recombine_pars[6] = 500;
                recombine_pars[7] = 247;
                strcpy(aa_path, "../../data/grid/T04_pswf.in");
                strcpy(facet_path, "../../data/grid/T04_facet%d%d.in");
            } else if (!strcasecmp(optarg, "T05")) {
                recombine_pars[0] = 512;
                recombine_pars[1] = 128;
                recombine_pars[2] = 128;
                recombine_pars[3] = 140;
                recombine_pars[4] = 216;
                recombine_pars[5] = 128;
                recombine_pars[6] = 256;
                recombine_pars[7] = 136;
                strcpy(aa_path, "../../data/grid/T05_pswf.in");
                strcpy(facet_path, "j0=%d/j1=%d/facet");
                strcpy(facet_path_hdf5, "../../data/grid/T05_in.h5");
            } else if (!strcasecmp(optarg, "small")) {
                recombine_pars[0] = 16384;
                recombine_pars[1] = 64;
                recombine_pars[2] = 5120;
                recombine_pars[3] = 6144;
                recombine_pars[4] = 8192;
                recombine_pars[5] = 448;
                recombine_pars[6] = 512;
                recombine_pars[7] = 272;
                strcpy(aa_path, "../../data/grid/T06_pswf_small.in");
            } else if (!strcasecmp(optarg, "large")) {
                recombine_pars[0] = 98304;
                recombine_pars[1] = 64;
                recombine_pars[2] = 7680;
                recombine_pars[3] = 9216;
                recombine_pars[4] = 12288;
                recombine_pars[5] = 704;
                recombine_pars[6] = 1024;
                recombine_pars[7] = 146;
                strcpy(aa_path, "../../data/grid/T06_pswf_large.in");
            } else {
                invalid=true; fprintf(stderr, "ERROR: Unknown recombination parameter set: %s!\n", optarg);
            }
            break;
        default: invalid = 1; break;
        }
    }

    if (have_vis_spec) {
        if (!spec.fov) { invalid=1; fprintf(stderr, "ERROR: Please specify non-zero field of view!\n"); }
        if (!spec.dec) { invalid=1; fprintf(stderr, "ERROR: Please specify non-zero declination!\n"); }
        if (!spec.time_count) { invalid=1; fprintf(stderr, "ERROR: Please specify dump times!\n"); }
        if (!spec.freq_count) { invalid=1; fprintf(stderr, "ERROR: Please specify frequency channels!\n"); }
    }

    if (!recombine_pars[0]) {
        invalid=1; fprintf(stderr, "ERROR: Please supply recombination parameters!\n");
    }

    if (invalid) {
        printf("Usage: %s...\n", argv[0]);
        printf("\n");
        printf("Visibility Parameters:\n");
        printf("  --telescope=<path>     Read stations from file\n");
        printf("  --fov=<val>            Set field of view (radians)\n");
        printf("  --dec=<val>            Set source declination (default 90 degrees)\n");
        printf("  --time=<start>:<end>/<steps>[/<chunks>]  Set dump times (hour angle, in s)\n");
        printf("  --freq=<start>:<end>/<steps>[/<chunks>]  Set frequency channels (in Hz).\n");
        printf("  --grid=<x0>,<path>     Gridding function to use\n");
        printf("  --vis=[vlaa/ska_low]   Use standard configuration sets\n");
        printf("\n");
        printf("Recombination Parameters:\n");
        printf("  --recombine=<N>,<Ny>,<yBs>,<yNs>,<yPs>,<xAs>,<xMs>,<xMxMyPs>\n");
        printf("                         Facet/subgrid recombination parameters\n");
        printf("  --rec-aa=<path>        Anti-aliasing function to use for recombination\n");
        printf("  --rec-set=[test]");
        printf("\n");
        printf("Distribution Parameters:\n");
        printf("  --facet-workers=<val>  Number of workers holding facets (default: half)\n");
        printf("  --dont-keep-bf         Discard BF term. Saves memory at expense of compute.\n");
        printf("  --parallel-columns     Work on grid columns in parallel. Worse for distribution.\n");
        printf("\n");
        printf("Positional Parameters:\n");
        printf("  <path>                 Visibility file. '%%d' will be replaced by rank.");
        return false;
    }

    // Set configuration
    int subgrid_workers = world_size - facet_workers;
    if (subgrid_workers < 1) subgrid_workers = 1;

    printf("aa_path=%s sg=%d fa=%d\n", aa_path, subgrid_workers, facet_workers);
    if (!work_config_set(cfg, have_vis_spec ? &spec : NULL,
                         facet_workers, subgrid_workers,
                         spec.fov / 2 / grid_x0,
                         recombine_pars[0], recombine_pars[1],
                         aa_path,
                         recombine_pars[2], recombine_pars[3],
                         recombine_pars[4], recombine_pars[5],
                         recombine_pars[6], recombine_pars[7]))
       return false;

    // Extra testing options, where appropriate
    if (facet_path[0]) {
        load_facets_from(cfg, facet_path, facet_path_hdf5[0] ? facet_path_hdf5 : NULL);
    }

    return true;
}

int main(int argc, char *argv[]) {

    // Initialise MPI, read configuration (we need multi-threading support)
    int thread_support, world_rank, world_size;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_length = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (thread_support < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Need full thread support from MPI!\n");
        return 1;
    }
    MPI_Get_processor_name(proc_name, &proc_name_length);
    proc_name[proc_name_length] = 0;

    // Read FFTW wisdom to get through planning quicker
    fftw_import_wisdom_from_filename("recombine.wisdom");

    // Decide number of workers
    int facet_workers = (world_size + 1) / 2;
    int subgrid_workers = world_size - facet_workers;
    if (subgrid_workers == 0) subgrid_workers = 1;

    // Make imaging configuration
    struct work_config config;
    if (!set_cmdarg_config(argc, argv, &config, world_rank, world_size)) {

    //if (!set_default_recombine2d_config(&config)) {
    //if (!set_test_recombine2d_config(&config, facet_workers, subgrid_workers, world_rank)) {
    //if (!recombine2d_set_test5_config(&config, facet_workers, subgrid_workers, world_rank)) {
    //if (!set_small_test_config(&config, facet_workers, subgrid_workers, world_rank)) {
        fprintf(stderr, "Could not set imaging configuration!\n");
        return 1;
    }

    // Local run?
    if (world_size == 1) {
        printf("%s pid %d role: Single\n", proc_name, getpid());

        producer(&config, 0, 0);

    } else {

        // Determine number of producers and streamers (pretty arbitrary for now)
        int i;

        if (world_rank < facet_workers) {
            printf("%s pid %d role: Producer %d\n", proc_name, getpid(), world_rank);

            int *streamer_ranks = (int *)malloc(sizeof(int) * subgrid_workers);
            for (i = 0; i < subgrid_workers; i++) {
                streamer_ranks[i] = facet_workers + i;
            }

            producer(&config, world_rank, streamer_ranks);

        } else if (world_rank - facet_workers < subgrid_workers) {
            printf("%s pid %d role: Streamer %d\n", proc_name, getpid(), world_rank - facet_workers);

            int *producer_ranks = (int *)malloc(sizeof(int) * facet_workers);
            for (i = 0; i < facet_workers; i++) {
                producer_ranks[i] = i;
            }

            streamer(&config, world_rank-facet_workers, producer_ranks);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    // Master: Write wisdom
    if (world_rank == 0) {
        fftw_export_wisdom_to_filename("recombine.wisdom");
    }
    return 0;
}
