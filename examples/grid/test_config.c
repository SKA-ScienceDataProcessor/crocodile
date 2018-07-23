
#define _GNU_SOURCE

#include "grid.h"
#include "recombine.h"

#include <hdf5.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

double min(double a, double b) { return a > b ? b : a; }
double max(double a, double b) { return a < b ? b : a; }

void bl_bounding_box(struct ant_config *cfg, struct bl_data *bl, double dec,
					 double *uvw_l_min, double *uvw_l_max)
{

	// Check time start and end (TODO - that's simplifying quite a bit)
	double uvw0[3], uvw1[3];
	ha_to_uvw(cfg, bl->antenna1, bl->antenna2, bl->time[0], dec, uvw0);
	ha_to_uvw(cfg, bl->antenna1, bl->antenna2, bl->time[bl->time_count-1], dec, uvw1);

	// Conversion factor to uvw in lambda
	double f0 = uvw_m_to_l(1, bl->freq[0]);
	double f1 = uvw_m_to_l(1, bl->freq[bl->freq_count-1]);

	// Determine bounding box
	int i = 0;
	for (i = 0; i < 3; i++) {
		uvw_l_min[i] = min(min(uvw0[i]*f0, uvw0[i]*f1), min(uvw1[i]*f0, uvw1[i]*f1));
		uvw_l_max[i] = max(max(uvw0[i]*f0, uvw0[i]*f1), max(uvw1[i]*f0, uvw1[i]*f1));
	}
}

void bl_bounding_box_int(struct ant_config *cfg, struct bl_data *bl,
						 double dec, double lam, double xA,
                         int *sg_min, int *sg_max)
{
    double uvw_l_min[3], uvw_l_max[3];
    bl_bounding_box(cfg, bl, dec, uvw_l_min, uvw_l_max);

    sg_min[0] = (int)round((uvw_l_min[0]/lam+0.5) / xA);
    sg_min[1] = (int)round((uvw_l_min[1]/lam+0.5) / xA);
    sg_max[0] = (int)round((uvw_l_max[0]/lam+0.5) / xA);
    sg_max[1] = (int)round((uvw_l_max[1]/lam+0.5) / xA);
}

double get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

int compare_work_nvis(const void *_sg1, const void *_sg2, void *_nvis) {
	const struct work *w1 = (const struct work *)_sg1;
    const struct work *w2 = (const struct work *)_sg2;
	return w1->nvis < w2->nvis;
}

bool generate_work_assignment(struct recombine2d_config *recombine_cfg,
							  struct vis_spec *spec,
                              struct work_assignment *work)
{
    struct ant_config *cfg = spec->cfg;

    // Create baseline structure
    struct bl_data bl;
    bl.time_count = spec->time_count;
    bl.time = (double *)malloc(sizeof(double) * spec->time_count);
    bl.uvw_m = (double *)malloc(sizeof(double) * spec->time_count * 3);
    int i;
    for (i = 0; i < spec->time_count; i++) {
        bl.time[i] = spec->time_start + spec->time_step * i;
    }
    bl.freq_count = spec->freq_count;
    bl.freq = (double *)malloc(sizeof(double) * spec->freq_count);
    for (i = 0; i < spec->freq_count; i++) {
        bl.freq[i] = spec->freq_start + spec->freq_step * i;
    }

	// Count visibilities per sub-grid
	int nsubgrid = (int)ceil(1 / work->xA);
	int *nvis = (int *)calloc(sizeof(int), nsubgrid * nsubgrid);

    int a1, a2;
    for (a1 = 0; a1 < cfg->ant_count; a1++) {
        for (a2 = a1+1; a2 < cfg->ant_count; a2++) {

            // Determine baseline bounding box
            int sg_min[2], sg_max[2];
            bl.antenna1 = a1;
            bl.antenna2 = a2;
            bl_bounding_box_int(cfg, &bl, spec->dec, work->lam, work->xA, sg_min, sg_max);

			// Fill bins, for both the baseline and its conjugated mirror
			int iu, iv;
			for (iv = sg_min[1]; iv <= sg_max[1]; iv++) {
				assert (! (iv < 0 || iv >= nsubgrid));
				for (iu = sg_min[0]; iu <= sg_max[0]; iu++) {
					assert (! (iu < 0 || iu >= nsubgrid));
					nvis[iv * nsubgrid + iu] += 1;
				}
			}
			for (iv = nsubgrid-sg_max[1]; iv <= nsubgrid-sg_min[1]; iv++) {
				assert (! (iv < 0 || iv >= nsubgrid));
				for (iu = nsubgrid-sg_max[0]; iu <= nsubgrid-sg_min[0]; iu++) {
					assert (! (iu < 0 || iu >= nsubgrid));
					// Don't double-count if conjugated area overlaps
					// with un-conjugated area: Clearly we don't want
					// to grid those visibilities twice.
					if (iv < sg_min[1] || iv > sg_max[1] ||
						iu < sg_min[0] || iu > sg_max[0])
						nvis[iv * nsubgrid + iu] += 1;
				}
			}

        }
    }


	// Count how many sub-grids actually have visibilities
	int npop = 0;
	int iu, iv;
	for (iu = nsubgrid/2; iu < nsubgrid; iu++)
		for (iv = 0; iv < nsubgrid; iv++)
			if (nvis[iv * nsubgrid + iu])
				npop++;

    // Allocate work description
	work->max_work = (npop + work->nworkers - 1) / work->nworkers;
	work->work = (struct work *)calloc(sizeof(struct work), work->nworkers * work->max_work);

	// Go through columns and assign work
	struct work *column = (struct work *)calloc(sizeof(struct work), nsubgrid);
    int iworker = 0, iwork = 0;
	for (iu = nsubgrid/2; iu < nsubgrid; iu++) {

        // Generate column of work
		int ncol = 0;
		for (iv = 0; iv < nsubgrid; iv++)
			if (nvis[iv * nsubgrid + iu]) {
				column[ncol].iu = iu;
                column[ncol].iv = iv;
				column[ncol].d_u = work->xA * iu;
                column[ncol].d_v = work->xA * iv;
                column[ncol].nvis = nvis[iv * nsubgrid + iu];
                ncol++;
            }

        // Sort
		qsort(column, ncol, sizeof(struct work), (void *)compare_work_nvis);

        // Assign in round-robin fashion. Not exactly optimal.
		int i;
		for (i = 0; i < ncol; i++) {
			work->work[iworker * work->max_work + iwork] = column[i];
            iworker++;
            if (iworker >= work->nworkers) {
                iworker = 0;
                iwork++;
            }
        }
	}

	// Statistics
	int min_vis = INT_MAX, max_vis = 0;
	for (i = 0; i < work->nworkers; i++) {
		int j; int vis = 0;
		for (j = 0; j < work->max_work; j++) {
			vis += work->work[i* work->max_work+j].nvis;
		}
		min_vis = min(vis, min_vis);
		max_vis = max(vis, max_vis);
	}
	printf("%d populated sub-grids, %d per worker\n", npop, work->max_work);
	printf("%d baseline bins minimum, %d baseline bins maximum\n", min_vis, max_vis);

    return true;
}

bool create_vis_file(char *filename, struct vis_spec *spec,
                     struct work_assignment *work, int worker)
{
    struct ant_config *cfg = spec->cfg;

    double pi = atan(1) * 4;

    // Create baseline structure
    struct bl_data bl;
    bl.time_count = spec->time_count;
    bl.time = (double *)malloc(sizeof(double) * spec->time_count);
    bl.uvw_m = (double *)malloc(sizeof(double) * spec->time_count * 3);
    int i;
    for (i = 0; i < spec->time_count; i++) {
        bl.time[i] = spec->time_start + spec->time_step * i;
    }
    bl.freq_count = spec->freq_count;
    bl.freq = (double *)malloc(sizeof(double) * spec->freq_count);
    for (i = 0; i < spec->freq_count; i++) {
        bl.freq[i] = spec->freq_start + spec->freq_step * i;
    }

    // Open file
    printf("Writing %s... ", filename);
    hid_t vis_f = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (vis_f < 0) {
        fprintf(stderr, "Could not visibility file %s!\n", filename);
        return false;
    }

    // Create "vis" group
    hid_t vis_g = H5Gcreate(vis_f, "vis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (vis_g < 0) {
        fprintf(stderr, "Could not open 'vis' group in visibility file %s!\n", filename);
        H5Fclose(vis_f);
        return false;
    }

    double start_write = get_time_ns();
    int a1, a2;
    for (a1 = 0; a1 < spec->cfg->ant_count; a1++) {
        if (a1 % 32 == 0) { printf("%d ", a1); fflush(stdout); }

        // Create outer antenna group
        char a1name[12];
        sprintf(a1name, "%d", a1);
        hid_t a1_g = H5Gcreate(vis_g, a1name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (a1_g < 0) {
            fprintf(stderr, "Could not open '%s' antenna group in visibility file %s!\n", a1name, filename);
            H5Gclose(vis_g); H5Fclose(vis_f);
            return false;
        }

        for (a2 = a1+1; a2 < spec->cfg->ant_count; a2++) {

            // Determine baseline bounding box
            int sg_min[2], sg_max[2];
            bl.antenna1 = a1;
            bl.antenna2 = a2;
            bl_bounding_box_int(cfg, &bl, spec->dec, work->lam, work->xA, sg_min, sg_max);

            // Check whether baseline overlaps our work box
            int i;
            for (i = 0; i < work->max_work; i++) {
                int iu = work->work[worker*work->max_work + i].iu;
                int iv = work->work[worker*work->max_work + i].iv;
                if (iu >= sg_min[0] && iu <= sg_max[0] &&
                    iv >= sg_min[1] && iv <= sg_max[1])
                    break;
            }
            if (i >= work->max_work)
                continue; // skip

            // Create inner antenna group
            char a2name[12];
            sprintf(a2name, "%d", a2);
            hid_t a2_g = H5Gcreate(a1_g, a2name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (a2_g < 0) {
                fprintf(stderr, "Could not open '%s' antenna group in visibility file %s!\n", a2name, filename);
                H5Gclose(a1_g); H5Gclose(vis_g); H5Fclose(vis_f); 
                return false;
            }

			double dec = 45 * pi / 180;
			double uvw_l_min[3], uvw_l_max[3];
            bl.antenna1 = a1;
            bl.antenna2 = a2;
			bl_bounding_box(spec->cfg, &bl, dec, uvw_l_min, uvw_l_max);
			//printf("min: %g %g %g\n", uvw_l_min[0], uvw_l_min[1], uvw_l_min[2]);
			//printf("max: %g %g %g\n", uvw_l_max[0], uvw_l_max[1], uvw_l_max[2]);

            if (!create_vis_group(a2_g, spec->freq_chunk, spec->time_chunk, &bl)) {
                H5Gclose(a2_g); H5Gclose(a1_g); H5Gclose(vis_g); H5Fclose(vis_f);
                return 1;
            }

            H5Gclose(a2_g);
        }
        H5Gclose(a1_g);
    }

    printf("\ndone\n");
    H5Gclose(vis_g);
    H5Fclose(vis_f);

    printf("took %.2f s\n", get_time_ns() - start_write);
    return true;
}


int main(int argc, char *argv[])
{

    init_dtype_cpx();

    struct ant_config cfg;
    load_ant_config("../../data/grid/LOWBD2_north_cfg.h5", &cfg);
	
    struct vis_spec spec;
    spec.cfg = &cfg;
    spec.dec = 45 * atan(1) * 4 / 180;
    spec.time_start = 10 * -45 / 3600; // h
    spec.time_count = 100;
    spec.time_chunk = 16;
    spec.time_step = 0.9 / 3600; // h
    spec.freq_start = 250e6; // Hz
    spec.freq_count = 500;
    spec.freq_chunk = 16;
    spec.freq_step = .1e6; // Hz

    struct work_assignment work;
    work.lam = 327680;
	work.theta = 0.1;
    work.xA = 400 / work.lam;
    work.nworkers = 10;
    generate_work_assignment(&spec, &work);

    create_vis_file("out.h5", &spec, &work, 0);
    return 0;
}
