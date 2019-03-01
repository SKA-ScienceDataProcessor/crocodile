
#ifndef CONFIG_H

#include "recombine.h"

// Specification of a visibility set
struct vis_spec
{
    double fov; // (true) field of view
    struct ant_config *cfg; // antennas
    double dec; // declination (radian)
    double time_start; int time_count; int time_chunk; double time_step; // hour angle (radian)
    double freq_start; int freq_count; int freq_chunk; double freq_step; // (Hz)
    // Cached hour angle / declination cosinus & sinus
    double *ha_sin, *ha_cos;
    double dec_sin, dec_cos;
};

void bl_bounding_box(struct vis_spec *spec,
                     int a1, int a2,
                     int tstep0, int tstep1,
                     int fstep0, int fstep1,
                     double *uvw_l_min, double *uvw_l_max);

// Work to do on a facet
struct facet_work
{
    int il, im;
    int facet_off_l, facet_off_m;
    char *path, *hdf5; // random if not set
    bool set; // empty otherwise
};

// Work to do for a subgrid on a baseline
struct subgrid_work_bl
{
    int a1, a2;
    int chunks;
    struct subgrid_work_bl *next;
};

// Work to do for a subgrid
struct subgrid_work
{
    int iu, iv; // Column/row number. Used for grouping, so must be consistent across work items!
    int subgrid_off_u, subgrid_off_v; // Midpoint offset in grid coordinates
    int nbl; // Baselines in this work bin
    char *check_path, *check_fct_path,
         *check_degrid_path, *check_hdf5; // check data if set
    double check_threshold, check_fct_threshold,
           check_degrid_threshold; // at what discrepancy to fail
    struct subgrid_work_bl *bls; // Baselines
};

struct work_config {

    // Fundamental dimensions
    // double lam; // size of entire grid in wavelenghts
    double theta; // size of image in radians
    struct vis_spec spec;
    char *vis_path; // Visibility file (pattern)
    char *gridder_path; // Gridding kernel file
    double gridder_x0; // Accuracy limit of gridder
    double *grid_correction; // Grid correction function

    // Worker configuration
    int facet_workers; // number of facet workers
    int facet_max_work; // work list length per worker
    int facet_count; // Number of facets
    struct facet_work *facet_work; // facet work list (2d array - worker x work)
    int subgrid_workers; // number of subgrid workers
    int subgrid_max_work; // work list length per worker
    struct subgrid_work *subgrid_work; // subgrid work list (2d array - worker x work)
    int iu_min, iu_max, iv_min, iv_max; // subgrid columns/rows

    // Recombination configuration
    struct recombine2d_config recombine;

    // Parameters
    int config_dump_baseline_bins;
    int config_dump_subgrid_work;
    int produce_parallel_cols;
    int produce_retain_bf;
    int produce_source_count;
    int produce_source_checks;
    int produce_batch_rows;
    int produce_queue_length;
    int vis_skip_metadata;
    int vis_bls_per_task;
    int vis_subgrid_queue_length;
    int vis_task_queue_length;
    int vis_chunk_queue_length;

    // Statsd connection
    int statsd_socket;
    double statsd_rate;
};

double get_time_ns();

void config_init(struct work_config *cfg);
bool config_set(struct work_config *cfg,
                int image_size, int subgrid_spacing,
                char *pswf_file,
                int yB_size, int yN_size, int yP_size,
                int xA_size, int xM_size, int xMxN_yP_size);
bool config_assign_work(struct work_config *cfg,
                        int facet_workers, int subgrid_workers);

void config_free(struct work_config *cfg);

void config_set_visibilities(struct work_config *cfg,
                             struct vis_spec *spec, double theta,
                             const char *vis_path);
bool config_set_degrid(struct work_config *cfg,
                       const char *gridder_path);

bool config_set_statsd(struct work_config *cfg,
                       const char *node, const char *service);
void config_send_statsd(struct work_config *cfg, const char *stat);

void config_load_facets(struct work_config *cfg, const char *path_fmt, const char *hdf5);
void config_check_subgrids(struct work_config *cfg,
                           double threshold, double fct_threshold, double degrid_threshold,
                           const char *check_fmt, const char *check_fct_fmt,
                           const char *check_degrid_fmt, const char *hdf5);

void vis_spec_to_bl_data(struct bl_data *bl, struct vis_spec *spec,
                         int a1, int a2);
bool create_bl_groups(hid_t vis_group, struct work_config *work_cfg, int worker);

int make_subgrid_tag(struct work_config *wcfg,
                     int subgrid_worker_ix, int subgrid_work_ix,
                     int facet_worker_ix, int facet_work_ix);

int producer(struct work_config *wcfg, int facet_worker, int *streamer_ranks);
void streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks);

#endif // CONFIG_H
