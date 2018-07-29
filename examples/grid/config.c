
#include "grid.h"
#include "config.h"

#include <hdf5.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

const int WORK_SPLIT_THRESHOLD = 10;

double min(double a, double b) { return a > b ? b : a; }
double max(double a, double b) { return a < b ? b : a; }

void bl_bounding_box(struct vis_spec *spec, int a1, int a2,
                     double *uvw_l_min, double *uvw_l_max)
{
    struct ant_config *cfg = spec->cfg;

    // Calculate time/frequency end points
    double time_end = spec->time_start + (spec->time_count-1) * spec->time_step;
    double freq_end = spec->freq_start + (spec->freq_count-1) * spec->freq_step;

    // Check time start and end (TODO - that's simplifying quite a bit)
    double uvw0[3], uvw1[3];
    ha_to_uvw(cfg, a1, a2, spec->time_start, spec->dec, uvw0);
    ha_to_uvw(cfg, a1, a2, time_end, spec->dec, uvw1);

    // Conversion factor to uvw in lambda
    double f0 = uvw_m_to_l(1, spec->freq_start);
    double f1 = uvw_m_to_l(1, freq_end);

    // Determine bounding box
    int i = 0;
    for (i = 0; i < 3; i++) {
        uvw_l_min[i] = min(min(uvw0[i]*f0, uvw0[i]*f1), min(uvw1[i]*f0, uvw1[i]*f1));
        uvw_l_max[i] = max(max(uvw0[i]*f0, uvw0[i]*f1), max(uvw1[i]*f0, uvw1[i]*f1));
    }
}

void bl_bounding_subgrids(struct vis_spec *spec, double lam, double xA, int a1, int a2,
                          int *sg_min, int *sg_max)
{
    double uvw_l_min[3], uvw_l_max[3];
    bl_bounding_box(spec, a1, a2, uvw_l_min, uvw_l_max);

    // Convert into subgrid indices
    sg_min[0] = (int)round(uvw_l_min[0]/lam/xA);
    sg_min[1] = (int)round(uvw_l_min[1]/lam/xA);
    sg_max[0] = (int)round(uvw_l_max[0]/lam/xA);
    sg_max[1] = (int)round(uvw_l_max[1]/lam/xA);
}

static int compare_work_nbl(const void *_w1, const void *_w2) {
    const struct subgrid_work *w1 = (const struct subgrid_work *)_w1;
    const struct subgrid_work *w2 = (const struct subgrid_work *)_w2;
    return w1->nbl < w2->nbl;
}

struct worker_prio
{
    int worker;
    int nbl;
};

static int compare_prio_nbl(const void *_w1, const void *_w2)
{
    const struct worker_prio *w1 = (const struct worker_prio *)_w1;
    const struct worker_prio *w2 = (const struct worker_prio *)_w2;
    return w1->nbl > w2->nbl;
}

static void bin_baseline(int *nbl, struct subgrid_work_bl **bls, int nsubgrid,
                         int a1, int a2, int iu, int iv)
{
    assert (iu >= 0 && iu < nsubgrid);
    assert (iv >= 0 && iv < nsubgrid);

    // Count
    nbl[iv*nsubgrid + iu]++;

    // Add work structure
    struct subgrid_work_bl *wbl = (struct subgrid_work_bl *)
        malloc(sizeof(struct subgrid_work_bl));
    wbl->a1 = a1; wbl->a2 = a2;
    wbl->next = bls[iv * nsubgrid + iu];
    bls[iv * nsubgrid + iu] = wbl;
}

// Bin baselines per overlapping subgrid
static int collect_baselines(struct vis_spec *spec,
                             double lam, double xA,
                             int **pnbl, struct subgrid_work_bl ***pbls)
{

    // Determine number of subgrid bins we need
    int nsubgrid = 2 * (int)ceil(1. / 2 / xA) + 1;
    int *nbl = (int *)calloc(sizeof(int), nsubgrid * nsubgrid);
    struct subgrid_work_bl **bls = (struct subgrid_work_bl **)
        calloc(sizeof(struct subgrid_work_bl *), nsubgrid * nsubgrid);

    int a1, a2;
    for (a1 = 0; a1 < spec->cfg->ant_count; a1++) {
        for (a2 = a1+1; a2 < spec->cfg->ant_count; a2++) {

            // Determine baseline bounding box
            int sg_min[2], sg_max[2];
            bl_bounding_subgrids(spec, lam, xA, a1, a2, sg_min, sg_max);

            // Fill bins, for both the baseline and its conjugated mirror
            int iu, iv;
            for (iv = nsubgrid/2+sg_min[1]; iv <= nsubgrid/2+sg_max[1]; iv++) {
                for (iu = nsubgrid/2+sg_min[0]; iu <= nsubgrid/2+sg_max[0]; iu++) {
                    bin_baseline(nbl, bls, nsubgrid, a1, a2, iu, iv);
                }
            }
            for (iv = nsubgrid/2-sg_max[1]; iv <= nsubgrid/2-sg_min[1]; iv++) {
                assert (iv >= 0 && iv < nsubgrid);
                for (iu = nsubgrid/2-sg_max[0]; iu <= nsubgrid/2-sg_min[0]; iu++) {
                    assert (iu >= 0 && iu < nsubgrid);
                    // Don't double-count if conjugated area overlaps
                    // with un-conjugated area: Clearly we don't want
                    // to grid those visibilitise twice.
                    if (iv < sg_min[1] || iv > sg_max[1] ||
                        iu < sg_min[0] || iu > sg_max[0]) {

                        bin_baseline(nbl, bls, nsubgrid, a1, a2, iu, iv);
                    }
                }
            }

        }
    }

    *pnbl = nbl;
    *pbls = bls;
    return nsubgrid;
}

// Pop given number of baselines from the start of the linked list
static struct subgrid_work_bl *pop_bls(struct subgrid_work_bl **bls, int n)
{
    struct subgrid_work_bl *first = *bls;
    struct subgrid_work_bl *bl = *bls;
    assert(n >= 1);
    if (!bl) return bl;
    while (n > 1 && bl->next) { n--; bl = bl->next; }
    *bls = bl->next;
    bl->next = NULL;
    return first;
}


static bool generate_subgrid_work_assignment(struct work_config *cfg)
{
    struct vis_spec *spec = &cfg->spec;

    // Count visibilities per sub-grid
    double xA = (double)cfg->recombine.xA_size / cfg->recombine.image_size;
    int *nbl; struct subgrid_work_bl **bls;
    int nsubgrid = collect_baselines(spec, cfg->lam, xA, &nbl, &bls);

    // Count how many sub-grids actually have visibilities
    int npop = 0, nbl_total = 0;
    int iu, iv;
    for (iu = nsubgrid/2; iu < nsubgrid; iu++)
        for (iv = 0; iv < nsubgrid; iv++)
            if (nbl[iv * nsubgrid + iu]) {
                npop++;
                nbl_total+=nbl[iv * nsubgrid + iu];
            }

    // We don't want bins that are too full compared to the average -
    // determine at what point we're going to split them.
    int work_max_nbl = WORK_SPLIT_THRESHOLD * nbl_total / npop;
    printf("%d subgrid baseline bins, %.4g average per subgrid, splitting above %d\n",
           npop, (double)nbl_total / npop, work_max_nbl);

    // Now count again how much work we have total, and per
    // column. Note that we ignore grid data at u < 0, as transferring
    // half the grid is enough to reconstruct a real-valued image.
    int nwork = 0, max_work_column = 0;
    for (iu = nsubgrid/2; iu < nsubgrid; iu++) {
        int nwork_start = nwork;
        for (iv = 0; iv < nsubgrid; iv++) {
            int nv = nbl[iv * nsubgrid + iu];
            nwork += (nv + work_max_nbl - 1) / work_max_nbl;
        }
        // How much work in this column?
        if (nwork - nwork_start > max_work_column)
            max_work_column = nwork-nwork_start;
    }

    // Allocate work description
    cfg->subgrid_max_work = (nwork + cfg->subgrid_workers - 1) / cfg->subgrid_workers;
    cfg->subgrid_work = (struct subgrid_work *)
        calloc(sizeof(struct subgrid_work), cfg->subgrid_workers * cfg->subgrid_max_work);
    printf("%d split subgrid baseline bins, %d per worker\n", nwork, cfg->subgrid_max_work);

    // Worker priority order for acquiring new work
    struct worker_prio *worker_prio = malloc(sizeof(worker_prio) * cfg->subgrid_workers);
    int i;
    for (i = 0; i < cfg->subgrid_workers; i++) {
        worker_prio[i].worker = i;
        worker_prio[i].nbl = 0;
    }

    // Go through columns and assign work
    struct subgrid_work *column = (struct subgrid_work *)
        calloc(sizeof(struct subgrid_work), max_work_column);
    int iworker = 0, iwork = 0;
    for (iu = nsubgrid/2; iu < nsubgrid; iu++) {

        // Generate column of work
        int ncol = 0; int start_bl;
        for (iv = 0; iv < nsubgrid; iv++) {
            int nv = nbl[iv * nsubgrid + iu];
            for (start_bl = 0; start_bl < nv; start_bl += work_max_nbl) {
                column[ncol].iu = iu - nsubgrid/2;
                column[ncol].iv = iv - nsubgrid/2;
                column[ncol].subgrid_off_u = cfg->recombine.xA_size * ((column[ncol].iu + nsubgrid) % nsubgrid);
                column[ncol].subgrid_off_v = cfg->recombine.xA_size * ((column[ncol].iv + nsubgrid) % nsubgrid);
                column[ncol].nbl = min(nv-start_bl, work_max_nbl);
                column[ncol].bls = pop_bls(&bls[iv * nsubgrid + iu], work_max_nbl);
                ncol++;
            }
        }

        // Sort
        qsort(column, ncol, sizeof(struct subgrid_work), compare_work_nbl);

        // Assign in roughly prioritised round-robin fashion. Note
        // that there's two kinds of work here - I/O and actualy
        // degridding work, and we want to distribute both roughly
        // equally so neither becomes a bottleneck. So we are pretty
        // conservative, this just makes the worst cases less
        // likely. There are likely better ways.
        int i;
        for (i = 0; i < ncol; i++) {
            // Assign work to next worker in priority list
            cfg->subgrid_work[worker_prio[iworker].worker * cfg->subgrid_max_work
                              + iwork] = column[i];
            worker_prio[iworker].nbl += column[i].nbl;
            iworker++;
            // Gone through list? Re-sort, start from the beginning
            if (iworker >= cfg->subgrid_workers) {
                qsort(worker_prio, cfg->subgrid_workers, sizeof(struct worker_prio), compare_prio_nbl);
                iworker = 0;
                iwork++;
            }
        }
    }

    // Statistics
    int min_vis = INT_MAX, max_vis = 0;
    cfg->iu_min = INT_MAX; cfg->iu_max = INT_MIN;
    cfg->iv_min = INT_MAX; cfg->iv_max = INT_MIN;
    for (i = 0; i < cfg->subgrid_workers; i++) {
        int j; int vis = 0;
        for (j = 0; j < cfg->subgrid_max_work; j++) {
            struct subgrid_work *work = cfg->subgrid_work + i* cfg->subgrid_max_work+j;
            if (work->iu < cfg->iu_min) cfg->iu_min = work->iu;
            if (work->iu > cfg->iu_max) cfg->iu_max = work->iu;
            if (work->iv < cfg->iv_min) cfg->iv_min = work->iv;
            if (work->iv > cfg->iv_max) cfg->iv_max = work->iv;
            vis += work->nbl;
            //printf("%d ", work->nbl);
        }
        //printf(" -> %d\n", vis);
        min_vis = min(vis, min_vis);
        max_vis = max(vis, max_vis);
    }
    printf("%d baseline bins minimum, %d baseline bins maximum\n", min_vis, max_vis);

    return true;
}

static bool generate_facet_work_assignment(struct work_config *cfg)
{

    // This is straightforward: We just assume that all facets within
    // the field of view are set. Note that theta is generally larger
    // than the FoV, so this won't cover the entire image.
    double yB = (double)cfg->recombine.yB_size / cfg->recombine.image_size;
    int nfacet = 2 * ceil(cfg->spec.fov / cfg->theta / yB / 2 - 0.5) + 1;
    printf("%dx%d facets covering %g FoV (facet %g, grid theta %g)\n",
           nfacet, nfacet, cfg->spec.fov, cfg->theta * yB, cfg->theta);

    // Allocate work array
    cfg->facet_max_work = (nfacet * nfacet + cfg->facet_workers - 1) / cfg->facet_workers;
    cfg->facet_work = (struct facet_work *)
        calloc(sizeof(struct facet_work), cfg->facet_workers * cfg->facet_max_work);

    int i;
    for (i = 0; i < nfacet * nfacet; i++) {
        int iworker = i % cfg->facet_workers, iwork = i / cfg->facet_workers;
        struct facet_work *work = cfg->facet_work + cfg->facet_max_work * iworker + iwork;
        work->il = (i / nfacet) - nfacet/2;
        work->im = (i % nfacet) - nfacet/2;
        work->facet_off_l = work->il * cfg->recombine.yB_size;
        work->facet_off_m = work->im * cfg->recombine.yB_size;
        work->set = true;
    }

    return true;
}

static bool generate_full_redistribute_assignment(struct work_config *cfg)
{

    // No visibilities involved, so generate work assignment where we
    // simply redistribute all data from a number of facets matching
    // the number of facet workers.
    assert(!cfg->spec.time_count);

    int nsubgrid = cfg->recombine.image_size / cfg->recombine.xA_size;
    int subgrid_work = nsubgrid * nsubgrid;
    cfg->subgrid_max_work = (subgrid_work + cfg->subgrid_workers - 1) / cfg->subgrid_workers;
    cfg->subgrid_work = (struct subgrid_work *)
        calloc(sizeof(struct subgrid_work), cfg->subgrid_max_work * cfg->subgrid_workers);
    int i;
    for (i = 0; i < subgrid_work; i++) {
        struct subgrid_work *work = cfg->subgrid_work + i;
        work->iu = i / nsubgrid;
        work->iv = i % nsubgrid;
        work->subgrid_off_u = work->iu * cfg->recombine.xA_size;
        work->subgrid_off_v = work->iv * cfg->recombine.xA_size;
        work->nbl = 1;
        // Dummy 0-0 baseline
        work->bls = (struct subgrid_work_bl *)calloc(sizeof(struct  subgrid_work_bl), 1);
    }
    cfg->iu_min = cfg->iv_min = 0;
    cfg->iu_max = cfg->iv_max = nsubgrid-1;

    int nfacet = cfg->recombine.image_size / cfg->recombine.yB_size;
    cfg->facet_max_work = (nfacet * nfacet + cfg->facet_workers - 1) / cfg->facet_workers;
    cfg->facet_work = (struct facet_work *)
        calloc(sizeof(struct facet_work), cfg->facet_max_work * cfg->facet_workers);
    for (i = 0; i < nfacet * nfacet; i++) {
        int iworker = i % cfg->facet_workers, iwork = i / cfg->facet_workers;
        struct facet_work *work = cfg->facet_work + cfg->facet_max_work * iworker + iwork;
        work->il = i / nfacet;
        work->im = i % nfacet;
        work->facet_off_l = work->il * cfg->recombine.yB_size;
        work->facet_off_m = work->im * cfg->recombine.yB_size;
        work->set = true;
    }

    return true;
}

bool work_config_set(struct work_config *cfg,
                     struct vis_spec *spec,
                     int facet_workers, int subgrid_workers,
                     double theta, int image_size, int subgrid_spacing,
                     char *pswf_file,
                     int yB_size, int yN_size, int yP_size,
                     int xA_size, int xM_size, int xMxN_yP_size) {

    // Initialise structure
    cfg->lam = image_size / theta;
    cfg->theta = theta;
    if (spec) {
        cfg->spec = *spec;
    } else {
        memset(&cfg->spec, 0, sizeof(cfg->spec));
    }
    cfg->facet_workers = facet_workers;
    cfg->facet_max_work = 0;
    cfg->facet_work = NULL;
    cfg->subgrid_workers = subgrid_workers;
    cfg->subgrid_max_work = 0;
    cfg->subgrid_work = NULL;

    // Set recombination configuration
    printf("\nInitialising recombination...\n");
    if (!recombine2d_set_config(&cfg->recombine, image_size, subgrid_spacing, pswf_file,
                                yB_size, yN_size, yP_size,
                                xA_size, xM_size, xMxN_yP_size))
        return false;

    // Generate work assignments
    if (spec) {
        printf("\nGenerating work assignments...\n");
        if (!generate_facet_work_assignment(cfg))
            return false;
        if (!generate_subgrid_work_assignment(cfg))
            return false;
    } else {
        if (!generate_full_redistribute_assignment(cfg))
            return false;
    }

    // Warn if we have multiple facets per worker
    if (cfg->facet_max_work > 1) {
        printf("WARNING: Multiple facets per worker is inefficient. Consider more MPI ranks.\n");
    }

    return true;
}

void load_facets_from(struct work_config *cfg,
                      const char *path_fmt,
                      const char *hdf5)
{

    int i;
    for (i = 0; i < cfg->facet_workers * cfg->facet_max_work; i++) {
        struct facet_work *work = cfg->facet_work + i;
        if (!work->set) continue;
        char path[256];
        snprintf(path, 256, path_fmt, work->im, work->il);
        work->path = strdup(path);
        work->hdf5 = hdf5 ? strdup(hdf5) : NULL;
    }

}

void check_subgrids_against(struct work_config *cfg,
                            double threshold, double fct_threshold,
                            const char *check_fmt,
                            const char *check_fct_fmt,
                            const char *hdf5)
{

    int i;
    for (i = 0; i < cfg->subgrid_workers * cfg->subgrid_max_work; i++) {
        struct subgrid_work *work = cfg->subgrid_work + i;
        if (!work->nbl) continue;
        char path[256];
        if (check_fmt) {
            snprintf(path, 256, check_fmt, work->iv, work->iu);
            work->check_path = strdup(path);
        }
        if (check_fct_fmt) {
            snprintf(path, 256, check_fct_fmt, work->iv, work->iu);
            work->check_fct_path = strdup(path);
        }
        work->check_hdf5 = hdf5 ? strdup(hdf5) : NULL;
        work->check_threshold = threshold;
        work->check_fct_threshold = fct_threshold;
    }

}
// Make baseline specification. Right now this is the same for every
// baseline, but this will change for baseline dependent averaging.
void vis_spec_to_bl_data(struct bl_data *bl, struct vis_spec *spec,
                         int a1, int a2)
{
    int i;

    // Create baseline structure
    bl->time_count = spec->time_count;
    bl->time = (double *)malloc(sizeof(double) * spec->time_count);
    for (i = 0; i < spec->time_count; i++) {
        bl->time[i] = spec->time_start + spec->time_step * i;
    }
    bl->uvw_m = (double *)malloc(sizeof(double) * spec->time_count * 3);
    for (i = 0; i < spec->time_count; i++) {
        ha_to_uvw(spec->cfg, a1, a2, bl->time[i] * M_PI / 12, spec->dec,
                  bl->uvw_m + i*3);
    }
    bl->freq_count = spec->freq_count;
    bl->freq = (double *)malloc(sizeof(double) * spec->freq_count);
    for (i = 0; i < spec->freq_count; i++) {
        bl->freq[i] = spec->freq_start + spec->freq_step * i;
    }
    bl->antenna1 = a1;
    bl->antenna2 = a2;

}

bool create_bl_groups(hid_t vis_group, struct work_config *work_cfg, int worker)
{
    struct vis_spec *spec = &work_cfg->spec;
    struct ant_config *cfg = spec->cfg;

    // Map baselines to work
    struct subgrid_work **bl_work = (struct subgrid_work **)
        calloc(sizeof(struct subgrid_work *), cfg->ant_count * cfg->ant_count);
    struct subgrid_work *work = work_cfg->subgrid_work + worker*work_cfg->subgrid_max_work;
    int iwork;
    for (iwork = 0; iwork < work_cfg->subgrid_max_work; iwork++) {
        if (work[iwork].nbl == 0) continue;
        struct subgrid_work_bl *bl;
        for (bl = work[iwork].bls; bl; bl = bl->next) {
            // Note this might overlap (here: overwrite). We are just
            // interested in an example below.
            bl_work[bl->a1 * cfg->ant_count + bl->a2] = &work[iwork];
        }
    }

    int a1, a2;
    int ncreated = 0;
    int nvis = 0;
    for (a1 = 0; a1 < cfg->ant_count; a1++) {
        // Progress message
        if (a1 % 32 == 0) { printf("%d ", a1); fflush(stdout); }

        hid_t a1_g = 0;
        for (a2 = a1+1; a2 < cfg->ant_count; a2++) {
            struct subgrid_work *bw = bl_work[a1 * cfg->ant_count + a2];
            if (!bw) continue;

            // Create outer antenna group, if not already done so
            if (!a1_g) {
                char a1name[12];
                sprintf(a1name, "%d", a1);
                a1_g = H5Gcreate(vis_group, a1name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (a1_g < 0) {
                    fprintf(stderr, "Could not open '%s' antenna group!\n", a1name);
                    return false;
                }
            }

            // Create inner antenna group
            char a2name[12];
            sprintf(a2name, "%d", a2);
            hid_t a2_g = H5Gcreate(a1_g, a2name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (a2_g < 0) {
                fprintf(stderr, "Could not open '%s' antenna group!\n", a2name);
                H5Gclose(a1_g);
                return false;
            }

            // Create baseline structure (TODO: baseline-dependent averaging...)
            struct bl_data bl;
            vis_spec_to_bl_data(&bl, spec, a1, a2);

            // Write to visibility group
            if (!create_vis_group(a2_g, spec->freq_chunk, spec->time_chunk, &bl)) {
                H5Gclose(a2_g); H5Gclose(a1_g);
                return 1;
            }

            // Statistics & cleanups
            ncreated++;
            nvis += bl.time_count * bl.freq_count;
            free(bl.time); free(bl.uvw_m); free(bl.freq);
            H5Gclose(a2_g);
        }
        if (a1_g) H5Gclose(a1_g);
    }

    printf("\ndone, %d groups for %d visibilities (~%.3g GB) created\n", ncreated, nvis, 16. * nvis / 1000000000);

    return true;
}
