
from crocodile.synthesis import coordinateBounds

import numpy
import matplotlib.pyplot as pl
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

def show_image(img, title, theta, **kwargs):
    """Visualise quadratic image in the (L,M) plane (directional
    cosines). We assume (0,0) to be at the image center.

    :param img: Data to visualise as a two-dimensional numpy array
    :param name: Function name to show in the visualisation header
    :param theta: Size of the image in radians. We will assume the
       image to spans coordinates [theta/2;theta/2[ in both L and M.
    :param extra_dep: Extra functiona parameters to add to the
       title. Purely cosmetic.
    :param subplots: Sub-plot to generate graphs into
    """

    # Determine size of image.
    size = img.shape[0]
    lm_lower, lm_upper = coordinateBounds(size)
    lm_lower = (lm_lower-1./size/2)*theta
    lm_upper = (lm_upper+1./size/2)*theta
    extent = (lm_lower, lm_upper, lm_lower, lm_upper)

    return imshow_helper(img, "%s(l,m)" % title, theta, extent, ("l [1]", "m [1]"), **kwargs)


def show_grid(grid, name, theta, **kwargs):

    # Determine size of image. See above.
    size = grid.shape[0]
    lam = size / theta
    uv_lower, uv_upper = coordinateBounds(size)
    uv_lower = (uv_lower-1./size/2)*lam
    uv_upper = (uv_upper+1./size/2)*lam
    extent = (uv_lower, uv_upper, uv_lower, uv_upper)

    return imshow_helper(grid, "%s(u,v)" % name, theta, extent, ("u [$\lambda$]", "v [$\lambda$]"), **kwargs)


def imshow_helper(img, title, theta, extent, xylabels,
                  norm=None, xlim=None, ylim=None, axes=None):
    """Visualise quadratic image in the (L,M) plane (directional
    cosines). We assume (0,0) to be at the image center.

    :param img: Data to visualise as a two-dimensional numpy array
    :param name: Function name to show in the visualisation header
    :param theta: Size of the image in radians. We will assume the
       image to spans coordinates [theta/2;theta/2[ in both L and M.
    :param extra_dep: Extra functiona parameters to add to the
       title. Purely cosmetic.
    :param axes: Sub-plot to generate graphs into
    """

    # Determine normalisation for image.
    if norm is not None:
        if isinstance(norm, tuple):
            norm = colors.Normalize(vmin=norm[0], vmax=norm[1], clip=True)
        else:
            norm = colors.Normalize(vmin=-norm, vmax=norm, clip=True)
    else:
        mi = numpy.min([img.real, img.imag])
        ma = numpy.max([img.real, img.imag])
        norm = colors.Normalize(vmin=mi, vmax=ma)

    # Create subplots if not specified
    if axes is None:
        fig = pl.figure()
        if numpy.any(numpy.iscomplex(img)):
            ax_r = fig.add_subplot(121)
            ax_i = fig.add_subplot(122)
        else:
            ax_r = fig.add_subplot(121)
            ax_i = None
    elif isinstance(axes, tuple):
        ax_r, ax_i = axes
    else:
        ax_r = axes
        ax_i = None

    cax_r = ax_r.imshow(img.real, extent=extent, norm=norm, origin='lower')
    ax_r.set_title(r"$Re$(%s)" % title)
    ax_r.set_xlabel(xylabels[0]); ax_r.set_ylabel(xylabels[1])
    #    if ax_i is None:
    ax_r.figure.colorbar(cax_r,ax=ax_r,shrink=.4,pad=0.025)

    # Limit to particular part of the image
    if xlim is not None:
        ax_r.set_xlim(xlim)
    if ylim is not None:
        ax_r.set_ylim(ylim)

    # Generate graph for imaginary part
    if ax_i is not None:
        cax_i = ax_i.imshow(img.imag, extent=extent, norm=norm, origin='lower')
        ax_i.set_title(r"$Im$(%s)" % title)
        ax_i.set_xlabel(xylabels[0]); ax_i.set_ylabel(xylabels[1])
        ax_i.figure.colorbar(cax_i,ax=ax_i,shrink=.4,pad=0.025)
        if xlim is not None:
            ax_i.set_xlim(xlim)
        if ylim is not None:
            ax_i.set_ylim(ylim)

    if axes is None:
        pl.show()

# from http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
# by CT Zhu
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def make_arrow(ax, source, target, color, name=None, scale=20, lw=3):
    xs, ys, zs = numpy.transpose((source, target))
    ax.add_artist(Arrow3D(xs, ys, zs, mutation_scale=scale, lw=lw, arrowstyle="-|>", color=color))
    if name is not None:
        ax.text(target[0]+0.03, target[1], target[2], name, color=color)


def circular_to_xyz(lon, lat):
    """Circular coordinate transformation appropriate for visualisation"""
    return numpy.array((numpy.sin(lon) * numpy.cos(lat),
                        numpy.cos(lon) * numpy.cos(lat),
                        numpy.sin(lat)))

def visualise_uvw(latitude, hour_angle, declination):
    """Shows a visualisation for the UVW coordinate system for an earth
    observer's UVW coordinate system pointing towards a certain local
    celestial coordinate.

    :param latitude: Latitude of the observer. Should be 0-90 degrees.
    :param hour_angle: Hour angle of the source. Should be -90-90 degrees.
    :param declination: Declination of the source. Should be -90-90 degrees.
    """

    def draw():
        make_arrow(ax, [0,0,0],[0,0,1.1], "black", "Earth axis (towards celestial north)")
        lons = numpy.linspace(-numpy.pi/4, numpy.pi/4, 10)
        lats = numpy.linspace(0, numpy.pi/2, 10)
        x, y, z = circular_to_xyz(numpy.outer(lons, numpy.ones(len(lats))),
                                  numpy.outer(numpy.ones(len(lons)), lats))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, alpha=0.4)
        obs_x, obs_y, obs_z = obs = circular_to_xyz(0, numpy.radians(latitude))
        ax.plot([0, obs_x], [0, obs_y], [0, obs_z], color="black", lw=3)
        ax.text(obs_x+0.03, obs_y, obs_z, "Observer", color="black")
        wdir = circular_to_xyz(numpy.radians(hour_angle), numpy.radians(declination))
        vdir = circular_to_xyz(numpy.radians(hour_angle), numpy.radians(declination+90))
        udir = numpy.cross(vdir, wdir)
        make_arrow(ax, obs, obs+wdir/3, "red", "w (towards phase centre)")
        make_arrow(ax, obs, obs+vdir/3, "red", "v")
        make_arrow(ax, obs, obs+udir/3, "red", "u")
        make_arrow(ax, obs, obs+[0,0,0.2], "black", "")

    fig = pl.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(elev=20., azim=-20.)
    ax.set_title("earth view")
    draw()
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=declination, azim=90.-hour_angle)
    ax.set_title("view from phase centre")
    draw()

def visualise_lmn(hour_angle, declination):
    # Swap X and Y to get a right-handed coordinate system
    def trans(coo):
        x,y,z = coo
        return y, x, z
    def draw(ax):
        ax.set_xlabel('y [$1$]'); ax.set_ylabel('x [$1$]'); ax.set_zlabel('z [$1$]')
        make_arrow(ax, trans([0,0,0]),trans([0,0,1.1]), "black", "Celestial north")
        make_arrow(ax, trans([0,0,0]),trans([1.1,0,0]), "black", "Geographical east")
        lons = numpy.linspace(0, numpy.pi/2, 20)
        lats = numpy.linspace(0, numpy.pi/2, 20)
        x, y, z = circular_to_xyz(numpy.outer(lons, numpy.ones(len(lats))),
                                  numpy.outer(numpy.ones(len(lons)), lats))
        ax.plot_surface(y, x, z, rstride=1, cstride=1, linewidth=0, alpha=0.4, color='white')
        t_x, t_y, t_z = ndir = circular_to_xyz(numpy.radians(-hour_angle), numpy.radians(declination))
        make_arrow(ax, trans((0,0,0)),     trans((t_x,0,0)),     color="gray", lw=3)
        make_arrow(ax, trans((t_x,0,0)),   trans((t_x,t_y,0)),   color="gray", lw=3)
        make_arrow(ax, trans((t_x,t_y,0)), trans((t_x,t_y,t_z)), color="gray", lw=3)
        make_arrow(ax, trans((0,0,0)),     trans((t_x,t_y,t_z)), color="black", name="phase centre", lw=3)
        mdir = circular_to_xyz(numpy.radians(-hour_angle), numpy.radians(declination+90))
        ldir = numpy.cross(ndir, mdir)
        make_arrow(ax, trans(ndir), trans(ndir+ndir/3), "red", "n")
        make_arrow(ax, trans(ndir), trans(ndir+ldir/3), "red", "l")
        make_arrow(ax, trans(ndir), trans(ndir+mdir/3), "red", "m")
        ax.set_title("Phase centre at $(%.2f,%.2f,%.2f)$" % (t_x, t_y, t_z))
    fig = pl.figure()
    ax = fig.add_subplot(121, projection='3d')
    draw(ax)
    ax.view_init(elev=35, azim=25)
    pl.show()
