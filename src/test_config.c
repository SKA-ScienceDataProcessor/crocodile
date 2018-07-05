
#include "grid.h"

#include <hdf5.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

// Convert hour angle / declination to UVW and back for a certain baseline
void ha_to_uvw(struct ant_config *cfg, int a1, int a2,
				   double ha, double dec,
				   double *uvw_m)
{
    double x = cfg->xyz[3*a2+0] - cfg->xyz[3*a1+0];
    double y = cfg->xyz[3*a2+1] - cfg->xyz[3*a1+1];
    double z = cfg->xyz[3*a2+2] - cfg->xyz[3*a1+2];
    // Rotate around z axis (hour angle)
    uvw_m[0] = x * cos(ha) - y * sin(ha);
    double v0 = x * sin(ha) + y * cos(ha);
    // Rotate around x axis (declination)
    uvw_m[2] = z * sin(dec) - v0 * cos(dec);
    uvw_m[1] = z * cos(dec) + v0 * sin(dec);
}

double uv_to_ha(struct ant_config *cfg, int a1, int a2,
				double dec, double *uv_m)
{
    double x = cfg->xyz[3*a2+0] - cfg->xyz[3*a1+0];
    double y = cfg->xyz[3*a2+1] - cfg->xyz[3*a1+1];
    double z = cfg->xyz[3*a2+2] - cfg->xyz[3*a1+2];

	assert(dec != 0 && y*y + x*x != 0);
	double v0  = (uv_m[1] - z*cos(dec)) / sin(dec);
	return asin((x*v0 - y*uv_m[0]) / (y*y + x*x));
}

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

double get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + (double)ts.tv_nsec / 1000000000;
}

bool create_vis_file(char *filename, struct ant_config *cfg,
                     double time_start, int time_count, int time_chunk, double time_step,
                     double freq_start, int freq_count, int freq_chunk, double freq_step)
{

    double pi = atan(1) * 4;

    // Create baseline structure
    struct bl_data bl;
    bl.time_count = time_count;
    bl.time = (double *)malloc(sizeof(double) * time_count);
    bl.uvw_m = (double *)malloc(sizeof(double) * time_count * 3);
    int i;
    for (i = 0; i < time_count; i++) {
        bl.time[i] = time_start + time_step * i;
    }
    bl.freq_count = freq_count;
    bl.freq = (double *)malloc(sizeof(double) * freq_count);
    for (i = 0; i < freq_count; i++) {
        bl.freq[i] = freq_start + freq_step * i;
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
    for (a1 = 0; a1 < cfg->ant_count; a1++) {
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

        for (a2 = a1+1; a2 < cfg->ant_count; a2++) {

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
			double threshold = 10;
            bl.antenna1 = a1;
            bl.antenna2 = a2;
			bl_bounding_box(cfg, &bl, dec, uvw_l_min, uvw_l_max);
			//printf("min: %g %g %g\n", uvw_l_min[0], uvw_l_min[1], uvw_l_min[2]);
			//printf("max: %g %g %g\n", uvw_l_max[0], uvw_l_max[1], uvw_l_max[2]);

            // Populate baseline structure (no data)
            for (i = 0; i < time_count; i++) {
                ha_to_uvw(cfg, a1, a2, bl.time[i] * pi / 12, dec,
						  bl.uvw_m + i*3);

				//printf("t=%.4f: %g %g %g / %g %g %g\n", bl.time[i] * pi / 12,
				//	   uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[0]),
				//	   uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[0]),
				//	   uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[0]),
				//	   uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[bl.freq_count-1]),
				//	   uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[bl.freq_count-1]),
				//	   uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[bl.freq_count-1]));
				assert(uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[0]) >= uvw_l_min[0]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[0]) >= uvw_l_min[1]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[0]) >= uvw_l_min[2]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[bl.freq_count-1]) >= uvw_l_min[0]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[bl.freq_count-1]) >= uvw_l_min[1]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[bl.freq_count-1]) >= uvw_l_min[2]-threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[0]) <= uvw_l_max[0]+threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[0]) <= uvw_l_max[1]+threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[0]) <= uvw_l_max[2]+threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+0],bl.freq[bl.freq_count-1]) <= uvw_l_max[0]+threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+1],bl.freq[bl.freq_count-1]) <= uvw_l_max[1]+threshold);
				assert(uvw_m_to_l(bl.uvw_m[i*3+2],bl.freq[bl.freq_count-1]) <= uvw_l_max[2]+threshold);

            }

            if (!create_vis_group(a2_g, freq_chunk, time_chunk, &bl)) {
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
    load_ant_config("../data/grid/LOWBD2_north_cfg.h5", &cfg);

    create_vis_file("/scratch/vis.h5", &cfg,
                    -45 / 3600, 100, 16, 0.9 / 3600, // h
                    200e6, 100, 16, 300.e6 / 650.); // Hz

    return 0;
}
