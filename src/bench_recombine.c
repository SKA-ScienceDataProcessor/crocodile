
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
#ifndef NO_MPI
#include <mpi.h>
#endif

bool load_vis_parset(const char *set_name, int image_size, struct vis_spec *spec)
{

    if (!strcasecmp(set_name, "lowbd2")) {
        free(spec->cfg);
        spec->cfg = malloc(sizeof(struct ant_config));
        if (!load_ant_config("../data/grid/LOWBD2_north_cfg.h5", spec->cfg))
            return false;
        spec->fov = (double)image_size / 210000;
        spec->dec = 90 * atan(1) * 4 / 180;
        spec->time_start = -230 / 3600; // h
        spec->time_count = 128;
        spec->time_chunk = 64;
        spec->time_step = 460 / 3600; // h
        spec->freq_start = 225e6; // Hz
        spec->freq_count = 2048;
        spec->freq_chunk = 64;
        spec->freq_step = 75.e6 / spec->freq_count; // Hz
        return true;
    }

    if (!strcasecmp(set_name, "vlaa")) {
        free(spec->cfg);
        spec->cfg = malloc(sizeof(struct ant_config));
        if (!load_ant_config("../data/grid/VLAA_north_cfg.h5", spec->cfg))
            return false;
        spec->fov = (double)image_size / 100000;
        spec->dec = 90 * atan(1) * 4 / 180;
        spec->time_start = 10 * -45 / 3600; // h
        spec->time_count = 64;
        spec->time_chunk = 32;
        spec->time_step = 10 * 0.9 / 3600; // h
        spec->freq_start = 250e6; // Hz
        spec->freq_count = 64;
        spec->freq_chunk = 32;
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
                           char *gridder_path)
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
        return true;
    }
    if (!strcasecmp(parset, "T05_") || !strcasecmp(parset, "512-216-256")) {
        recombine_pars[0] = 512;
        recombine_pars[1] = 128;
        recombine_pars[2] = 128;
        recombine_pars[3] = 140;
        recombine_pars[4] = 216;
        recombine_pars[5] = 128;
        recombine_pars[6] = 256;
        recombine_pars[7] = 136;
        strcpy(aa_path, "../data/grid/T05_pswf.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "tiny") || !strcasecmp(parset, "8k-2k-512")) {
        recombine_pars[0] = 8192;
        recombine_pars[1] = 128;
        recombine_pars[2] = 1280;
        recombine_pars[3] = 1536;
        recombine_pars[4] = 2048;
        recombine_pars[5] = 384;
        recombine_pars[6] = 512;
        recombine_pars[7] = 144;
        strcpy(aa_path, "../data/grid/T06_pswf_tiny.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "small") || !strcasecmp(parset, "16k-8k-512")) {
        recombine_pars[0] = 16384;
        recombine_pars[1] = 32;
        recombine_pars[2] = 5120;
        recombine_pars[3] = 5632;
        recombine_pars[4] = 8192;
        recombine_pars[5] = 416;
        recombine_pars[6] = 512;
        recombine_pars[7] = 280;
        strcpy(aa_path, "../data/grid/T06_pswf_small.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "smallish") || !strcasecmp(parset, "32k-8k-1k")) {
        recombine_pars[0] = 32768;
        recombine_pars[1] = 64;
        recombine_pars[2] = 5120;
        recombine_pars[3] = 5632;
        recombine_pars[4] = 8192;
        recombine_pars[5] = 832;
        recombine_pars[6] = 1024;
        recombine_pars[7] = 280;
        strcpy(aa_path, "../data/grid/T06_pswf_smallish.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "medium") || !strcasecmp(parset, "64k-16k-1k")) {
        recombine_pars[0] = 65536;
        recombine_pars[1] = 64;
        recombine_pars[2] = 10240;
        recombine_pars[3] = 11264;
        recombine_pars[4] = 16384;
        recombine_pars[5] = 832;
        recombine_pars[6] = 1024;
        recombine_pars[7] = 280;
        strcpy(aa_path, "../data/grid/T06_pswf_medium.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "large") || !strcasecmp(parset, "96k-12k-1k")) {
        recombine_pars[0] = 98304;
        recombine_pars[1] = 256;
        recombine_pars[2] = 6912;
        recombine_pars[3] = 8832;
        recombine_pars[4] = 12288;
        recombine_pars[5] = 768;
        recombine_pars[6] = 1024;
        recombine_pars[7] = 144;
        strcpy(aa_path, "../data/grid/T06_pswf_96k_12k_1k_1e-07.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "96k-24k-1k")) {
        recombine_pars[0] = 98304;
        recombine_pars[1] = 64;
        recombine_pars[2] = 15360;
        recombine_pars[3] = 16896;
        recombine_pars[4] = 24576;
        recombine_pars[5] = 832;
        recombine_pars[6] = 1024;
        recombine_pars[7] = 280;
        strcpy(aa_path, "../data/grid/T06_pswf_large.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "tremendous") || !strcasecmp(parset, "128k-32k-2k")) {
        recombine_pars[0] = 131072;
        recombine_pars[1] = 64;
        recombine_pars[2] = 20480;
        recombine_pars[3] = 22528;
        recombine_pars[4] = 32768;
        recombine_pars[5] = 1856;
        recombine_pars[6] = 2048;
        recombine_pars[7] = 536;
        strcpy(aa_path, "../data/grid/T06_pswf_128k_32k_2k.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    if (!strcasecmp(parset, "huge") || !strcasecmp(parset, "256k-32k-2k")) {
        recombine_pars[0] = 262144;
        recombine_pars[1] = 128;
        recombine_pars[2] = 20480;
        recombine_pars[3] = 22528;
        recombine_pars[4] = 32768;
        recombine_pars[5] = 1664;
        recombine_pars[6] = 2048;
        recombine_pars[7] = 280;
        strcpy(aa_path, "../data/grid/T06_pswf_huge.in");
        strcpy(gridder_path, "../data/grid/T05b_kern.h5");
        return true;
    }
    return false;
}

enum Opts
    {
        Opt_flag = 0,

        Opt_telescope, Opt_fov, Opt_dec, Opt_time, Opt_freq, Opt_grid, Opt_vis_set,
        Opt_recombine, Opt_rec_aa, Opt_rec_set,
        Opt_rec_load_facet, Opt_rec_load_facet_hdf5, Opt_batch_rows,
        Opt_facet_workers, Opt_parallel_cols, Opt_dont_retain_bf,
        Opt_source_count, Opt_send_queue,
        Opt_bls_per_task, Opt_subgrid_queue, Opt_task_queue, Opt_visibility_queue,
        Opt_statsd, Opt_statsd_port,
    };

bool set_cmdarg_config(int argc, char **argv,
                       struct work_config *cfg, int world_rank, int world_size)
{
    // Read parameters
    config_init(cfg);

    struct option options[] =
      {
        {"telescope",  required_argument, 0, Opt_telescope },
        {"dec",        required_argument, 0, Opt_dec },
        {"fov",        required_argument, 0, Opt_fov },
        {"time",       required_argument, 0, Opt_time },
        {"freq",       required_argument, 0, Opt_freq },
        {"grid",       required_argument, 0, Opt_grid },
        {"vis-set",    required_argument, 0, Opt_vis_set},
        {"add-meta",   no_argument,       &cfg->vis_skip_metadata, false },

        {"recombine",  required_argument, 0, Opt_recombine },
        {"rec-aa",     required_argument, 0, Opt_rec_aa },
        {"rec-set",    required_argument, 0, Opt_rec_set },
        {"load-facet", required_argument, 0, Opt_rec_load_facet },
        {"load-facet-hdf5", required_argument, 0, Opt_rec_load_facet_hdf5 },
        {"batch-rows", required_argument, 0, Opt_batch_rows },

        {"facet-workers", required_argument, 0, Opt_facet_workers },
        {"parallel-columns", no_argument, &cfg->produce_parallel_cols, true },
        {"dont-retain-bf", no_argument,   &cfg->produce_retain_bf, false },
        {"source-count", required_argument, 0, Opt_source_count },
        {"bls-per-task", required_argument, 0, Opt_bls_per_task },
        {"send-queue", required_argument, 0, Opt_send_queue },
        {"subgrid-queue", required_argument, 0, Opt_subgrid_queue },
        {"task-queue", required_argument, 0, Opt_task_queue },
        {"visibility-queue", required_argument, 0, Opt_visibility_queue },

        {"statsd",     optional_argument, 0, Opt_statsd },
        {"statsd-port",required_argument, 0, Opt_statsd_port },

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
    char gridder_path[256]; char vis_path[256];
    char statsd_addr[256]; char statsd_port[256] = "8125";
    memset(&spec, 0, sizeof(spec));
    spec.dec = 90 * atan(1) * 4 / 180;
    memset(&recombine_pars, 0, sizeof(recombine_pars));
    aa_path[0] = gridder_path[0] = facet_path[0] = facet_path_hdf5[0] =
        subgrid_path[0] = subgrid_fct_path[0] = subgrid_degrid_path[0] =
        subgrid_path_hdf5[0] = vis_path[0] = statsd_addr[0] = 0;

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
            nscan = sscanf(optarg, "%255s", gridder_path);
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
                                       gridder_path)) {
                invalid=true; fprintf(stderr, "ERROR: Unknown recombination parameter set: %s!\n", optarg);
            }
            break;
        case Opt_batch_rows:
            nscan = sscanf(optarg, "%d", &cfg->produce_batch_rows);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'batch-rows' option!\n");
            }
            break;
        case Opt_facet_workers:
            nscan = sscanf(optarg, "%d", &facet_workers);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'facet-workers' option!\n");
            }
            break;
        case Opt_source_count:
            nscan = sscanf(optarg, "%d", &cfg->produce_source_count);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'source-count' option!\n");
            }
            break;
        case Opt_bls_per_task:
            nscan = sscanf(optarg, "%d", &cfg->vis_bls_per_task);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'bls-per-task' option!\n");
            }
            break;
        case Opt_send_queue:
            nscan = sscanf(optarg, "%d", &cfg->produce_queue_length);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'send-queue' option!\n");
            }
            break;
        case Opt_subgrid_queue:
            nscan = sscanf(optarg, "%d", &cfg->vis_subgrid_queue_length);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'subgrid-queue' option!\n");
            }
            break;
        case Opt_task_queue:
            nscan = sscanf(optarg, "%d", &cfg->vis_task_queue_length);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'task-queue' option!\n");
            }
            break;
        case Opt_visibility_queue:
            nscan = sscanf(optarg, "%d", &cfg->vis_chunk_queue_length);
            if (nscan != 1) {
                invalid=true; fprintf(stderr, "ERROR: Could not parse 'visibility-queue' option!\n");
            }
            break;
        case Opt_statsd:
            strncpy(statsd_addr, optarg, 254); statsd_addr[255] = 0;
            break;
        case Opt_statsd_port:
            strncpy(statsd_port, optarg, 254); statsd_port[255] = 0;
            break;
        case '?':
        default:
            invalid=true; fprintf(stderr, "ERROR: Unknown option '%s'!\n", argv[opterr]);
            break;
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
    } else {
        if (cfg->produce_source_count) { invalid=1; fprintf(stderr, "ERROR: Please specify visibilities (FoV) when generating sources!\n"); }
    }

    if (!recombine_pars[0]) {
        invalid=1; fprintf(stderr, "ERROR: Please supply recombination parameters!\n");
    }

    if (invalid) {
        printf("Usage: %s [options] <path>\n", argv[0]);
        printf("\n");
        printf("Image Parameters:\n");
        printf("  --source-count=<n>     Number of sources (default 0 = entire image random)\n");
        printf("\n");
        printf("Visibility Parameters:\n");
        printf("  --telescope=<path>     Read stations from file\n");
        printf("  --fov=<val>            Set field of view (radians)\n");
        printf("  --dec=<val>            Set source declination (default 90 degrees)\n");
        printf("  --time=<start>:<end>/<steps>[/<chunk>]  Set dump times (hour angle, in s)\n");
        printf("  --freq=<start>:<end>/<steps>[/<chunk>]  Set frequency channels (in Hz).\n");
        printf("  --grid=<x0>,<path>     Gridding function to use\n");
        printf("  --vis=[vlaa/ska_low]   Use standard configuration sets\n");
        printf("\n");
        printf("Recombination Parameters:\n");
        printf("  --recombine=<N>,<Ny>,<yBs>,<yNs>,<yPs>,<xAs>,<xMs>,<xMxMyPs>\n");
        printf("                         Facet/subgrid recombination parameters\n");
        printf("  --rec-aa=<path>        Anti-aliasing function to use for recombination\n");
        printf("  --rec-set=[test]       Selection recombination parameter set");
        printf("  --batch-rows=<N>       Image rows to batch per thread");
        printf("\n");
        printf("Distribution Parameters:\n");
        printf("  --facet-workers=<val>  Number of workers holding facets (default: half)\n");
        printf("  --dont-retain-bf       Discard BF term. Saves memory at expense of compute.\n");
        printf("  --parallel-columns     Work on grid columns in parallel. Worse for distribution.\n");
        printf("  --send-queue=<N>       Outgoing subgrid queue length (default 8)");
        printf("  --bls-per-task=<N>     Number of baselines per OpenMP task (default 256)\n");
        printf("  --subgrid-queue=<N>    Incoming subgrid queue length (default 8)\n");
        printf("  --visibility-queue=<N> Outgoing visibility queue length (default 32768)\n");
        printf("\n");
        printf("Positional Parameters:\n");
        printf("  <path>                 Visibility file. '%%d' will be replaced by rank.\n\n");
        return false;
    }

    // Set configuration
    int subgrid_workers = world_size - facet_workers;
    if (subgrid_workers < 1) subgrid_workers = 1;

    if (statsd_addr[0]) {
        if (!config_set_statsd(cfg, statsd_addr, statsd_port)) {
            return false;
        }
    }

    if (!config_set(cfg,
                    recombine_pars[0], recombine_pars[1],
                    aa_path,
                    recombine_pars[2], recombine_pars[3],
                    recombine_pars[4], recombine_pars[5],
                    recombine_pars[6], recombine_pars[7])) {
        return false;
    }
    if (gridder_path[0]) {
        if (!config_set_degrid(cfg, gridder_path[0] ? gridder_path : NULL)) {
            invalid = 1;
            fprintf(stderr, "ERROR: Could not access gridder at %s!", gridder_path);
        }
    }
    if (have_vis_spec) {
        config_set_visibilities(cfg, &spec, spec.fov / 2 / cfg->gridder_x0,
                                vis_path[0] ? vis_path : NULL);
    }

    // Make work assignment
    if (!config_assign_work(cfg, facet_workers, subgrid_workers))
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
    int world_rank, world_size;
    char proc_name[256];
#ifndef NO_MPI
    int thread_support, proc_name_length = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (thread_support < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Need full thread support from MPI!\n");
        return 1;
    }
    MPI_Get_processor_name(proc_name, &proc_name_length);
    proc_name[proc_name_length] = 0;
#else
    world_rank = 0; world_size = 1;
    gethostname(proc_name, 256);
#endif

    // Read FFTW wisdom to get through planning quicker
    fftw_import_wisdom_from_filename("recombine.wisdom");

    // HDF5 initialisation
    init_dtype_cpx();

    // Make imaging configuration
    struct work_config config;
    if (!set_cmdarg_config(argc, argv, &config, world_rank, world_size)) {
        return 1;
    }

    // Local run?
    if (world_size == 1) {

        if (config.facet_workers > 0) {
            printf("%s pid %d role: Standalone producer\n", proc_name, getpid());
            producer(&config, 0, 0);
        } else {
            printf("%s pid %d role: Standalone streamer\n", proc_name, getpid());
            streamer(&config, 0, 0);
        }

    } else {

        // Determine number of producers and streamers (pretty arbitrary for now)
        int i;

        if (world_rank < config.facet_workers) {
            printf("%s pid %d role: Producer %d\n", proc_name, getpid(), world_rank);

            int *streamer_ranks = NULL;
            if (config.facet_workers < world_size) {
                streamer_ranks = (int *)malloc(sizeof(int) * config.subgrid_workers);
                for (i = 0; i < config.subgrid_workers; i++) {
                    streamer_ranks[i] = config.facet_workers + i;
                }
            }
            printf("%d subgrid workers, streamer_ranks = %p\n", config.subgrid_workers, streamer_ranks);

            producer(&config, world_rank, streamer_ranks);

            free(streamer_ranks);

        } else if (world_rank - config.facet_workers < config.subgrid_workers) {
            printf("%s pid %d role: Streamer %d\n", proc_name, getpid(), world_rank - config.facet_workers);

            int *producer_ranks = NULL;
            if (config.subgrid_workers < world_size) {
                producer_ranks = (int *)malloc(sizeof(int) * config.facet_workers);
                for (i = 0; i < config.facet_workers; i++) {
                    producer_ranks[i] = i;
                }
            }
            printf("%d facet workers, producer_ranks = %p\n", config.facet_workers, producer_ranks);

            streamer(&config, world_rank - config.facet_workers, producer_ranks);
        }

    }

#ifndef NO_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
#endif

    config_free(&config);

    // Master: Write wisdom
    if (world_rank == 0) {
        fftw_export_wisdom_to_filename("recombine.wisdom");
    }
    return 0;
}
