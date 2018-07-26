
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
};

// Work to do on a facet
struct facet_work
{
    int il, im;
    int facet_off0, facet_off1;
    int tag;
    double d_l, d_m;
};

// Work to do for a subgrid on a baseline
struct subgrid_work_bl
{
	int a1, a2;
	struct subgrid_work_bl *next;
};

// Work to do for a subgrid
struct subgrid_work
{
    int iu, iv;
    double d_u, d_v;
	int subgrid_off_u, subgrid_off_v;
	int start_bl;
    int nbl;
	struct subgrid_work_bl *bls;
};

struct work_config {

    // Fundamental dimensions
    double lam; // size of entire grid in wavelenghts
    double theta; // size of image in radians
	struct vis_spec spec;

	// Worker configuration
    int facet_workers; // number of facet workers
	int facet_max_work; // work list length per worker
    struct facet_work *facet_work; // facet work list
    int subgrid_workers; // number of subgrid workers
    int subgrid_max_work; // work list length per worker
    struct subgrid_work *subgrid_work; // subgrid work list

	// Recombination configuration
	struct recombine2d_config recombine;
};

void bl_bounding_box(struct vis_spec *spec, int a1, int a2,
                     double *uvw_l_min, double *uvw_l_max);
void bl_bounding_subgrids(struct vis_spec *spec, double lam, double xA, int a1, int a2,
                          int *sg_min, int *sg_max);

bool work_config_set(struct work_config *cfg,
                     struct vis_spec *spec,
                     int facet_workers, int subgrid_workers,
                     double theta, int image_size, int subgrid_spacing,
                     char *pswf_file,
                     int yB_size, int yN_size, int yP_size,
                     int xA_size, int xM_size, int xMxN_yP_size);

void vis_spec_to_bl_data(struct bl_data *bl, struct vis_spec *spec,
                         int a1, int a2);
bool create_bl_groups(hid_t vis_group, struct work_config *work_cfg, int worker);

#endif // CONFIG_H
