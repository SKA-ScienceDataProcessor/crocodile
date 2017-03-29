
import datetime
from itertools import product, combinations
import numpy
import random
from simanneal import Annealer
import sys
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Bin:

    def __init__(self, iu0, iu1, iv0, iv1, iw0, iw1, bins,
                 wplanes=1, uchunks=1, vchunks=1):
        """
        Constructs a bin. In the process the bin will be reduced as much
        as possible, and visibility and cost statistics will be cached.
        """

        self.iu0 = iu0
        self.iu1 = iu1
        self.iv0 = iv0
        self.iv1 = iv1
        self.iw0 = iw0
        self.iw1 = iw1
        self.wplanes =  wplanes
        self.uchunks = uchunks
        self.vchunks = vchunks

        # Automatically reduce and determine statistics
        if not self._reduce(bins.density):
            self.iu1 = self.iu0
            self.nvis = 0
            self.cost = 0
            self.cost_direct = 0
        else:
            uvw = bins.bin_to_uvw(numpy.arange(self.iw0, self.iw1), coords=2)
            wsum = self._calc_nvis(bins.density,(1,2))
            wcounts = numpy.transpose([uvw, wsum])

            self.nvis = self._calc_nvis(bins.density)
            self.cost = self._calc_cost(bins, wcounts)

    def __repr__(self):
        return "Bin(%d, %d, %d, %d, %d, %d, %d)" % \
            (self.iu0, self.iu1, self.iv0, self.iv1, self.iw0, self.iw1, self.wplanes)

    def is_zero(self):
        """ Is this bin empty? """
        return self.nvis == 0 or self.iu0 >= self.iu1 or self.iv0 >= self.iv1 or self.iw0 >= self.iw1

    def _reduce(self, density):
        """Reduce bin size as much as possible without reducing the number of
        visibilities contained."""

        while self.iu0 < self.iu1 and numpy.sum(density[self.iw0:self.iw1, self.iv0:self.iv1, self.iu0]) == 0:
            self.iu0 += 1
        while self.iu0 < self.iu1 and numpy.sum(density[self.iw0:self.iw1, self.iv0:self.iv1, self.iu1-1]) == 0:
            self.iu1 -= 1
        while self.iv0 < self.iv1 and numpy.sum(density[self.iw0:self.iw1, self.iv0, self.iu0:self.iu1]) == 0:
            self.iv0 += 1
        while self.iv0 < self.iv1 and numpy.sum(density[self.iw0:self.iw1, self.iv1-1, self.iu0:self.iu1]) == 0:
            self.iv1 -= 1
        while self.iw0 < self.iw1 and numpy.sum(density[self.iw0, self.iv0:self.iv1, self.iu0:self.iu1]) == 0:
            self.iw0 += 1
        while self.iw0 < self.iw1 and numpy.sum(density[self.iw1-1, self.iv0:self.iv1, self.iu0:self.iu1]) == 0:
            self.iw1 -= 1

        # Empty?
        return not (self.iu0 >= self.iu1 or self.iv0 >= self.iv1 or self.iw0 >= self.iw1)

    def _calc_nvis(self, density, axis=None):
        """ Determine how many visibilities lie in this bin """
        return numpy.sum(density[self.iw0:self.iw1, self.iv0:self.iv1, self.iu0:self.iu1], axis=axis)

    def _calc_cost(self, bins, wcounts):
        """Calculate cost for gridding this bin. Automatically determines
        optimal number of w-planes. """

        if self.is_zero():
            return 0

        wplanes = max(1, self.wplanes)
        uchunks = self.uchunks
        vchunks = self.vchunks
        cost = self._calc_cost_wplane(bins, wcounts, wplanes, uchunks, vchunks)

        # Determine number of u/v chunks
        cost0 = cost
        while uchunks > 1:
            ncost = self._calc_cost_wplane(bins, wcounts, wplanes,
                                           uchunks-1, vchunks)
            if ncost > cost: break
            uchunks -= 1
            cost = ncost
        if cost == cost0:
            while True:
                ncost = self._calc_cost_wplane(bins, wcounts, wplanes,
                                               uchunks+1, vchunks)
                if ncost > cost: break
                uchunks += 1
                cost = ncost
        cost0 = cost
        while vchunks > 1:
            ncost = self._calc_cost_wplane(bins, wcounts, wplanes,
                                           uchunks, vchunks-1)
            if ncost > cost: break
            vchunks -= 1
            cost = ncost
        if cost == cost0:
            while True:
                ncost = self._calc_cost_wplane(bins, wcounts, wplanes,
                                               uchunks, vchunks+1)
                if ncost > cost: break
                vchunks += 1
                cost = ncost

        # Decrease benificial? Always try up to steps of 2
        while wplanes > 1:
            ncost = self._calc_cost_wplane(bins, wcounts, wplanes-1,
                                           uchunks, vchunks)
            if ncost > cost:
                if wplanes <= 2: break
                ncost = self._calc_cost_wplane(bins, wcounts, wplanes-2,
                                               uchunks, vchunks)
                if ncost > cost: break
                wplanes -= 1
            wplanes -= 1
            cost = ncost

        # Increase benificial? Test quite some way out if we have a
        # lot of w-planes already
        if wplanes == max(1, self.wplanes):
            success = True
            while success:
                success = False
                for i in range(2+wplanes//10):
                    ncost = self._calc_cost_wplane(bins, wcounts, wplanes+1+i,
                                                   uchunks, vchunks)
                    if ncost <= cost:
                        success = True
                        wplanes += 1+i
                        cost = ncost
                        break

        # Check whether it is better to not do w-stacking
        self.cost_direct = self._calc_cost_direct(bins, wcounts)
        if self.cost_direct < cost:
            self.wplanes = 0
            self.uchunks = 1
            self.vchunks = 1
            return self.cost_direct
        else:
            self.wplanes = wplanes
            self.uchunks = uchunks
            self.vchunks = vchunks
            return cost


    def _calc_cost_direct(self, bins, wcounts):
        """Calculate cost for gridding this bin directly (without w-plane
        trickery)"""

        if self.is_zero():
            return 0

        # Determine grid coordinates
        u0, v0, w0 = bins.bin_to_uvw(numpy.array([self.iu0, self.iv0, self.iw0]))
        u1, v1, w1 = bins.bin_to_uvw(numpy.array([self.iu1, self.iv1, self.iw1]))

        #print("w: %.f - %.f" % (w0, w1))

        # Determine w-kernel size for w-planes and for grid transfer
        args = bins.args
        ws, nvis = numpy.transpose(wcounts)
        # w = max(numpy.abs(w0), numpy.abs(w1))
        u_w = numpy.sqrt( (numpy.abs(ws) * args.theta/2)**2 +
                          (numpy.sqrt(numpy.abs(ws))**3 * args.theta / 2 / numpy.pi / args.epsw) )

        # Be rather pessimistic about required w-size
        nw = 1 + 2 * numpy.ceil(u_w / args.ustep)

        #print("u_w = %f (%d px)" % (u_w, numpy.ceil(u_w / args.wstep)))

        # Kernel pixel sizes - we need to account for the w-kernel
        # (dependent on w) and the A-kernel (assumed constant)
        #print("nwkernel = %d" % numpy.sqrt(numpy.ceil(u_w / args.ustep)**2))
        nkernel2 = numpy.ceil(numpy.sqrt(nw**2 + args.asize**2))**2
        #print("direct:")
        #print(numpy.transpose([ws, nvis, nw, 8 * nkernel2 * nvis]))
        #print("nkernel = %d" % numpy.sqrt(nkernel2))

        # Determine cost
        c_Grid = numpy.sum(8 * nvis * nkernel2)
        #print("c_Grid = %.1f kflop" % (c_Grid / 1000))
        return c_Grid

    def _calc_cost_wplane(self, bins, wcounts, wplanes, uchunks, vchunks):
        """Calculate cost for gridding this bin using a given number of
        w-planes and u/v chunks"""

        if self.is_zero():
            return 0

        #print("wplanes = %d:" % wplanes)

        # Determine grid coordinates
        u0, v0, w0 = bins.bin_to_uvw(numpy.array([self.iu0, self.iv0, self.iw0]))
        u1, v1, w1 = bins.bin_to_uvw(numpy.array([self.iu1, self.iv1, self.iw1]))

        # Bin coordinate
        #print("iu: %.f - %.f" % (self.iu0, self.iu1), end=' ')
        #print("iv: %.f - %.f" % (self.iv0, self.iv1), end=' ')
        #print("iw: %.f - %.f" % (self.iw0, self.iw1))

        # Real coordinates
        #print("u: %.f - %.f" % (u0, u1), end=' ')
        #print("v: %.f - %.f" % (v0, v1), end=' ')
        #print("w: %.f - %.f" % (w0, w1))

        # Grid cell coordinates
        args = bins.args
        #print("cu: %.f - %.f" % (u0/args.ustep, u1/args.ustep), end=' ')
        #print("cv: %.f - %.f" % (v0/args.ustep, v1/args.ustep), end=' ')
        #print("cw: %.f - %.f" % (w0/args.wstep, w1/args.wstep))

        # Determine w-kernel size for w-planes and for grid transfer
        args = bins.args
        ws, nvis = numpy.transpose(wcounts)
        d_iw = self.iw1 - self.iw0
        wp = numpy.arange(0, wplanes * d_iw, wplanes)
        diwp = (wp - d_iw/2 - numpy.floor(wp/d_iw)*d_iw) / wplanes
        dwp = diwp / d_iw * (w1 - w0)

        # Determine (grid) size of kernel to bring w-plane to w=0 plane
        w = max(numpy.abs(w0), numpy.abs(w1))
        u_w = numpy.sqrt( (w * args.theta/2)**2 +
                          (numpy.sqrt(w)**3 * args.theta / 2 / numpy.pi / args.epsw) )
        nw = 1 + 2 * numpy.ceil(u_w / args.ustep)
        #print("w = %.1f, u_w = %.1f, nw = %d" % (w, u_w, nw))

        # Determine size of kernel to w-project visibilities to their w-plane
        d_w = numpy.abs(w1 - w0) / 2 / wplanes
        u_d_w = numpy.sqrt( (numpy.abs(dwp) * args.theta/2)**2 +
                            (numpy.sqrt(numpy.abs(dwp))**3 * args.theta / 2 / numpy.pi / args.epsw) )
        ndw = 1 + 2 * numpy.ceil(u_d_w / args.ustep)

        #print("u_w = %f (%d px)" % (u_w, numpy.ceil(u_w / args.ustep)))
        #print("u_d_w = %f (%d px)" % (u_d_w, numpy.ceil(u_d_w / args.ustep)))

        # Kernel pixel sizes - we need to account for the w-kernel
        # (dependent on w) and the A-kernel (assumed constant)
        nkernel2 = ndw**2 + args.asize**2
        nsb_kernel = numpy.sqrt(nw**2 + args.asize**2)
        nsubgrid2 = numpy.ceil((u1-u0) / args.ustep / uchunks + nsb_kernel)* \
                    numpy.ceil((v1-v0) / args.ustep / vchunks + nsb_kernel)
        # (not quite the same as with IDG!)
        #print("nwkernel = %d" % numpy.sqrt(numpy.ceil(u_d_w / args.ustep)**2))
        #print("nkernel = %d" % numpy.sqrt(nkernel2))
        #print("nsubgrid = %d" % numpy.sqrt(nsubgrid2))

        #print(numpy.transpose([dwp, nvis, ndw, 8 * nvis * nkernel2]))

        # Determine cost
        c_Grid = numpy.sum(8 * nvis * nkernel2)
        c_FFT = uchunks * vchunks * \
                5 * numpy.ceil(numpy.log(nsubgrid2)/numpy.log(2))*nsubgrid2 * (wplanes+1)
        c_Add = uchunks * vchunks * \
                8 * nsubgrid2 * wplanes
        #print("nvis = %d" % self.nvis)
        # print("wplanes=%d, uchunks=%d, vchunks=%d" % (wplanes, uchunks, vchunks))
        # print("nw=%d" % (nw))
        # print("c_Grid = %.1f kflop" % (c_Grid / 1000))
        # print("c_FFT  = %.1f kflop" % (c_FFT / 1000))
        # print("c_Add  = %.1f kflop" % (c_Add / 1000))
        return c_Grid + c_FFT + c_Add

    def split_u(self, at_u, bins):
        """ Split the bin along a u-plane. Returns two new bins. """
        assert(at_u >= self.iu0 and at_u < self.iu1)
        bin1 = Bin(at_u, self.iu1, self.iv0, self.iv1, self.iw0, self.iw1, bins, self.wplanes)
        bin2 = Bin(self.iu0, at_u, self.iv0, self.iv1, self.iw0, self.iw1, bins, self.wplanes)
        return bin1, bin2

    def split_v(self, at_v, bins):
        """ Split the bin along a v-plane. Returns two new bins. """
        assert(at_v >= self.iv0 and at_v < self.iv1)
        bin1 = Bin(self.iu0, self.iu1, at_v, self.iv1, self.iw0, self.iw1, bins, self.wplanes)
        bin2 = Bin(self.iu0, self.iu1, self.iv0, at_v, self.iw0, self.iw1, bins, self.wplanes)
        return bin1, bin2

    def split_w(self, at_w, bins):
        """ Split the bin along a w-plane. Returns two new bins. """
        assert(at_w >= self.iw0 and at_w < self.iw1)
        bin1 = Bin(self.iu0, self.iu1, self.iv0, self.iv1, at_w, self.iw1, bins, self.wplanes)
        bin2 = Bin(self.iu0, self.iu1, self.iv0, self.iv1, self.iw0, at_w, bins, self.wplanes)
        return bin1, bin2

    def merge(self, other, bins):
        """
        Merge two bins. Note that the new bin might contain more
        visibilities than both original bins combined.
        """
        return Bin(min(self.iu0, other.iu0),
                   max(self.iu1, other.iu1),
                   min(self.iv0, other.iv0),
                   max(self.iv1, other.iv1),
                   min(self.iw0, other.iw0),
                   max(self.iw1, other.iw1),
                   bins,
                   self.wplanes)

    def overlaps(self, other):
        """ Checks whether two bins overlap """
        return \
            max(self.iu0, other.iu0) < min(self.iu1, other.iu1) and \
            max(self.iv0, other.iv0) < min(self.iv1, other.iv1) and \
            max(self.iw0, other.iw0) < min(self.iw1, other.iw1)

    def distance(self, other):
        du = max(0, max(self.iu0, other.iu0) - min(self.iu1, other.iu1))
        dv = max(0, max(self.iv0, other.iv0) - min(self.iv1, other.iv1))
        dw = max(0, max(self.iw0, other.iw0) - min(self.iw1, other.iw1))
        return du*du + dv*dv + dw*dw

class BinSet(Annealer):

    def __init__(self, bin_to_uvw, args, density,
                 initial_state=None,
                 name = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
                 add_cost=0,
                 pop_method='random', merge_prop=4,
                 max_merge_distance=100, max_bins=100,
                 progress_image=None,
                 **kwargs):
        """Initialises a bin set for optimisation

        :param bin_to_uvw: Coordinate conversation rule
        :param args: Gridding parameters
        :param density: Array with grid visibility density
        :param pop_method: How bins are selected for modification
          ('random' or 'biased')
        :param merge_prop: Probability modifier for merges. Higher
          values means that the optimisation prefers less bins.
        :param progress_image: Write a visualisation of the progress
          to this file name for every update (slow!)
        """

        self.bin_to_uvw = bin_to_uvw
        self.args = args
        self.density = density
        self.name = name

        # Parameters
        self.add_cost = add_cost
        self.pop_method = pop_method
        self.merge_prop = merge_prop
        self.max_merge_distance = max_merge_distance
        self.max_bins = max_bins
        self.progress_image = progress_image
        self.extra_energy_per_bin = 0.001

        # Make bins
        if initial_state is not None:
            initial_state = [Bin(*coords, bins=self) for coords in initial_state]
            initial_state = list(filter(lambda b: b.nvis > 0, initial_state))
        super(BinSet, self).__init__(initial_state=initial_state, **kwargs)
        self.copy_strategy = 'method'

        # Find initial statistics
        self.cost0 = sum(b.cost for b in self.state)
        self.nvis0 = sum(b.nvis for b in self.state)

    @property
    def bins(self):
        return self.state

    def pop_bin(self):
        """ Remove a random bin from the bin set. """

        if self.pop_method == 'random':
            # Select entirely randomly - performs okay
            return self.state.pop(int(random.uniform(0, len(self.state))))
        elif self.pop_method == 'biased':
            # Select in a biased fashion. Likely not worth the extra cost
            totals = numpy.array([b.nvis for b in self.state])
            totals = numpy.log(totals)
            totals = numpy.cumsum(totals)
            r = random.uniform(0, totals[-1])
            i = numpy.searchsorted(totals, r, side='right')
            return self.state.pop(i)
        assert False, "unknown pop method %s" % self.pop_method

    def push_bin(self,b):
        """ Add a new bin to our bin set. Skips if empty """
        if not b.is_zero():
            assert b.nvis > 0
            self.state.append(b)

    def move(self):
        """ Make a random move """

        op = random.randint(0,6)
        e = self.energy()
        if op == 0:
            b = self.pop_bin()
            b0, b1 = b.split_u(random.randint(b.iu0, b.iu1-1), self)
            assert b0.nvis + b1.nvis == b.nvis
            self.push_bin(b0)
            self.push_bin(b1)
        elif op == 1:
            b = self.pop_bin()
            b0, b1 = b.split_v(random.randint(b.iv0, b.iv1-1), self)
            assert b0.nvis + b1.nvis == b.nvis
            self.push_bin(b0)
            self.push_bin(b1)
        elif op == 2:
            b = self.pop_bin()
            b0, b1 = b.split_w(random.randint(b.iw0, b.iw1-1), self)
            assert b0.nvis + b1.nvis == b.nvis
            self.push_bin(b0)
            self.push_bin(b1)
        elif op >= 3 and len(self.state) >= 2:
            b0 = self.pop_bin()
            # Compile list of merge candidates
            candidates = list(filter(lambda b: b0.distance(b) < self.max_merge_distance, self.state))
            random.shuffle(candidates)
            success = False
            for b1 in candidates:
                self.state.remove(b1)
                b_n = b0.merge(b1, self)
                # Visibility sum works out? Then we can proceed
                nvis = b0.nvis + b1.nvis
                if b_n.nvis == nvis:
                    self.push_bin(b_n)
                    success = True
                    break
                # Put bins back
                self.push_bin(b1)
                success = False
            if not success:
                self.push_bin(b0)

    def energy(self):
        return (self.add_cost + sum(b.cost for b in self.state)
                ) / self.nvis0 + self.extra_energy_per_bin * len(self.state)

    def update(self, step, T, E, acceptance, improvement):

        # Check consistency
        assert sum(b.nvis for b in self.state) == self.nvis0

        if acceptance is not None:

            # Copied
            def time_string(seconds):
                """Returns time in seconds as a string formatted HHHH:MM:SS."""
                s = int(round(seconds))  # round to nearest second
                h, s = divmod(s, 3600)   # get hours and remainder
                m, s = divmod(s, 60)     # split remainder into minutes and seconds
                return '%i:%02i:%02i' % (h, m, s)

            # Compose message
            elapsed = time.time() - self.start
            remain = (self.steps - step) * (elapsed / step)
            title = ('T=%.5f E=%.2f Bins=%.d Acc=%.2f%% Imp=%.2f%% Time=%s/%s' %
                     (T, E-self.extra_energy_per_bin * len(self.state),
                      len(self.state), 100.0 * acceptance, 100.0 * improvement,
                      time_string(elapsed), time_string(elapsed+remain)))

            print('\r'+title, file=sys.stderr, end='\r')

            if self.progress_image is not None:
                self.visualize(title, save=self.progress_image)

    def visualize(self, title='', save=None):

        # Make figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('w')
        ax.view_init(elev=90., azim=-0)

        # Find non-zero coordinates, convert to uvw
        iw,iv,iu = self.density.nonzero()
        u,v,w = numpy.transpose(self.bin_to_uvw(numpy.transpose([iu,iv,iw])))
        ax.scatter(u, v, w, c= 'red',s=self.density[iw,iv,iu]/self.nvis0*10000,lw=0)

        # Helper for drawing a box
        def draw_box(x0, x1, y0, y1, z0, z1, *args):
            r = [0, 1]
            for s, e in combinations(numpy.array(list(product(r, r, r))), 2):
                if numpy.sum(numpy.abs(s-e)) == 1:
                    xyz = numpy.array([s, e])
                    xyz *= numpy.array([x1-x0, y1-y0, z1-z0])
                    xyz += numpy.array([x0, y0, z0])
                    uvw = numpy.transpose(self.bin_to_uvw(xyz))
                    ax.plot3D(*uvw, color="b", lw=0.5)

        for b in self.state:
            if b.wplanes > 0:
                draw_box(b.iu0, b.iu1, b.iv0, b.iv1, b.iw0, b.iw1)
        fig.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=300)
        else:
            plt.show()
        plt.close()

    def save_state(self, fname=None, *args, **kwargs):
        if fname is None:
            true_energy = self.energy() - self.extra_energy_per_bin * len(self.state)
            fname = self.name + "_E%.1f.state" % true_energy
        super(BinSet, self).save_state(fname, *args, **kwargs)
