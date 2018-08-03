
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

bool load_vis_parset(const char *set_name, int image_size, struct vis_spec *spec)
{

    if (!strcasecmp(set_name, "lowbd2")) {
        free(spec->cfg);
        spec->cfg = malloc(sizeof(struct ant_config));
        if (!load_ant_config("../data/grid/LOWBD2_north_cfg.h5", spec->cfg))
            return false;
        spec->fov = 1300. / image_size;
        spec->dec = 90 * atan(1) * 4 / 180;
        spec->time_start = 10 * -45 / 3600; // h
        spec->time_count = 64;
        spec->time_chunk = 16;
        spec->time_step = 10 * 0.9 / 3600; // h
        spec->freq_start = 250e6; // Hz
        spec->freq_count = 64;
        spec->freq_chunk = 16;
        spec->freq_step = 50.e6 / spec->freq_count; // Hz
        return true;
    }

    if (!strcasecmp(set_name, "vlaa")) {
        free(spec->cfg);
        spec->cfg = malloc(sizeof(struct ant_config));
        if (!load_ant_config("../data/grid/VLAA_north_cfg.h5", spec->cfg))
            return false;
        spec->fov = 3000. / image_size;
        spec->dec = 90 * atan(1) * 4 / 180;
        spec->time_start = 10 * -45 / 3600; // h
        spec->time_count = 64;
        spec->time_chunk = 16;
        spec->time_step = 10 * 0.9 / 3600; // h
        spec->freq_start = 250e6; // Hz
        spec->freq_count = 64;
        spec->freq_chunk = 16;
        spec->freq_step = 50.e6 / spec->freq_count; // Hz
        return true;
    }

    return false;
}

bool load_recombine_parset(const char *parset,
                           int *recombine_pars, char *aa_path,
                           char *facet_path, char *facet_path_hdf5,
                           char *subgrid_path, char *subgrid_fct_path,
                           char *subgrid_degrid_path, char *subgrid_path_hdf5,
                           double *subgrid_threshold, double *subgrid_fct_threshold,
                           double *subgrid_degrid_threshold,
                           double *gridder_x0, char *gridder_path)
{
    if (!strcasecmp(parset, "T04")) {
        recombine_pars[0] = 2000;
        recombine_pars[1] = 100;
        recombine_pars[2] = 400;
        recombine_pars[3] = 480;
        recombine_pars[4] = 900;
        recombine_pars[5] = 400;
        recombine_pars[6] = 500;
        recombine_pars[7] = 247;
        strcpy(aa_path, "../data/grid/T04_pswf.in");
        strcpy(facet_path, "../data/grid/T04_facet%d%d.in");
        return true;
    }
    if (!strcasecmp(parset, "T05")) {
        recombine_pars[0] = 512;
        recombine_pars[1] = 128;
        recombine_pars[2] = 128;
        recombine_pars[3] = 140;
        recombine_pars[4] = 216;
        recombine_pars[5] = 128;
        recombine_pars[6] = 256;
        recombine_pars[7] = 136;
        strcpy(aa_path, "../data/grid/T05_pswf.in");
        strcpy(facet_path, "j0=%d/j1=%d/facet");
        strcpy(facet_path_hdf5, "../data/grid/T05_in.h5");
        strcpy(subgrid_fct_path, "i0=%d/i1=%d/j0=%%d/j1=%%d/nmbf");
        *subgrid_fct_threshold = 5.5e-6;
        strcpy(subgrid_path, "i0=%d/i1=%d/approx");
        *subgrid_threshold = 1.2e-8;
        strcpy(subgrid_degrid_path, "i0=%d/i1=%d");
        strcpy(subgrid_path_hdf5, "../data/grid/T05_in.h5");
        *subgrid_degrid_threshold = 2e-7;
        strcpy(gridder_path, "../data/grid/T05_in.h5");
        *gridder_x0 = 0.4;
        return true;
    }
    if (!strcasecmp(parset, "small")) {
        recombine_pars[0] = 16384;
        recombine_pars[1] = 64;
        recombine_pars[2] = 5120;
        recombine_pars[3] = 6144;
        recombine_pars[4] = 8192;
        recombine_pars[5] = 448;
        recombine_pars[6] = 512;
        recombine_pars[7] = 272;
        strcpy(aa_path, "../data/grid/T06_pswf_small.in");
        return true;
    }
    if (!strcasecmp(parset, "large")) {
        recombine_pars[0] = 98304;
        recombine_pars[1] = 64;
        recombine_pars[2] = 7680;
        recombine_pars[3] = 9216;
        recombine_pars[4] = 12288;
        recombine_pars[5] = 704;
        recombine_pars[6] = 1024;
        recombine_pars[7] = 146;
        strcpy(aa_path, "../data/grid/T06_pswf_large.in");
        return true;
    }
    return false;
}

enum Opts
    {
        Opt_flag,

        Opt_telescope, Opt_fov, Opt_dec, Opt_time, Opt_freq, Opt_grid, Opt_vis_set,
        Opt_recombine, Opt_rec_aa, Opt_rec_set,
        Opt_rec_load_facet, Opt_rec_load_facet_hdf5,
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
    char subgrid_path[256]; char subgrid_fct_path[256];
    char subgrid_degrid_path[256]; char subgrid_path_hdf5[256];
    double subgrid_threshold = 1e-8, subgrid_fct_threshold = 1e-8,
           subgrid_degrid_threshold = 1e-8;
    int facet_workers = (world_size + 1) / 2;
    double grid_x0 = 0.4; char gridder_path[256]; char vis_path[256];
    memset(&spec, 0, sizeof(spec));
    spec.dec = 90 * atan(1) * 4 / 180;
    memset(&recombine_pars, 0, sizeof(recombine_pars));
    aa_path[0] = gridder_path[0] = facet_path[0] = facet_path_hdf5[0] =
        subgrid_path[0] = subgrid_fct_path[0] = subgrid_degrid_path[0] =
        subgrid_path_hdf5[0] = vis_path[0] = 0;

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
            nscan = sscanf(optarg, "%lg,%255s", &grid_x0, gridder_path);
            printf("Got grid\n");
            if (nscan < 2) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'grid' option!\n");
            }
            break;
        case Opt_vis_set:
            if (load_vis_parset(optarg, recombine_pars[0], &spec))
                have_vis_spec = true;
            else {
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
            if (!load_recombine_parset(optarg, recombine_pars, aa_path,
                                       facet_path, facet_path_hdf5,
                                       subgrid_path, subgrid_fct_path,
                                       subgrid_degrid_path, subgrid_path_hdf5,
                                       &subgrid_threshold, &subgrid_fct_threshold,
                                       &subgrid_degrid_threshold,
                                       &grid_x0, gridder_path)) {
                invalid=true; fprintf(stderr, "ERROR: Unknown recombination parameter set: %s!\n", optarg);
            }
            break;
        default: invalid = 1; break;
        }
    }

    // Get positional arguments (visibility file path)
    if (!invalid && optind+1 == argc) {
        strncpy(vis_path, argv[argc-1], 255);
        vis_path[255] = 0;
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
        printf("  --dont-retain-bf       Discard BF term. Saves memory at expense of compute.\n");
        printf("  --parallel-columns     Work on grid columns in parallel. Worse for distribution.\n");
        printf("\n");
        printf("Positional Parameters:\n");
        printf("  <path>                 Visibility file. '%%d' will be replaced by rank.\n\n");
        return false;
    }

    // Set configuration
    int subgrid_workers = world_size - facet_workers;
    if (subgrid_workers < 1) subgrid_workers = 1;

    config_init(cfg, facet_workers, subgrid_workers);
    cfg->produce_parallel_cols = produce_parallel_cols;
    cfg->produce_retain_bf = produce_retain_bf;

    if (!config_set(cfg,
                    recombine_pars[0], recombine_pars[1],
                    aa_path,
                    recombine_pars[2], recombine_pars[3],
                    recombine_pars[4], recombine_pars[5],
                    recombine_pars[6], recombine_pars[7])) {
        return false;
    }
    if (have_vis_spec) {
        config_set_visibilities(cfg, &spec, spec.fov / 2 / grid_x0,
                                vis_path[0] ? vis_path : NULL);
    }
    if (gridder_path[0]) {
        config_set_degrid(cfg, gridder_path[0] ? gridder_path : NULL);
    }

    // Make work assignment
    if (!config_assign_work(cfg))
        return false;

    // Extra testing options, where appropriate
    if (facet_path[0]) {
        config_load_facets(cfg, facet_path, facet_path_hdf5[0] ? facet_path_hdf5 : NULL);
    }
    if (subgrid_path[0] || subgrid_fct_path[0]) {
        config_check_subgrids(cfg,
                              subgrid_threshold, subgrid_fct_threshold, subgrid_degrid_threshold,
                              subgrid_path[0] ? subgrid_path : NULL,
                              subgrid_fct_path[0] ? subgrid_fct_path : NULL,
                              subgrid_degrid_path[0] ? subgrid_degrid_path : NULL,
                              subgrid_path_hdf5[0] ? subgrid_path_hdf5 : NULL);
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

    config_free(&config);

    // Master: Write wisdom
    if (world_rank == 0) {
        fftw_export_wisdom_to_filename("recombine.wisdom");
    }
    return 0;
}