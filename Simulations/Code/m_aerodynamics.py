# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:11:49 2015

@author: isaac
"""

from __future__ import division

import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.io import loadmat


DATA_FILE = '../Data/Aerodynamics/holden_lift_drag.mat'


def aero_interp_funcs(Res, aoar, lift, drag):
    """Aerodynamic force coefficient interpolation function. We approximate
    the surfaces of lift and drag coefficinet with Reynolds number
    and angle of attack with bivariate splines of first order both directions
    and no smoothing.

    Parameters
    ----------
    Res : array of size (nRe)
        Reynolds numbers
    aoar : array of size (naoa)
        angles of attack in radians
    lift : array of size (nRe, naoa)
        lift coefficients
    drag : array of size (nRe, naoa)
        drag coefficients

    Returns
    -------
    aero_interp : function
        This function takes the angle of attack, velocity, chord lenth
        vector, and an option kinematic viscoisty (default is =1.568e-5),
        and returns lift coefficients, drag coeffients, lift-to-drag
        ratio, and Reynolds number (u * c / nu).
    """

    cl_fun = RectBivariateSpline(aoar, Res, lift.T, kx=1, ky=1, s=0)
    cd_fun = RectBivariateSpline(aoar, Res, drag.T, kx=1, ky=1, s=0)

    aoar_min = aoar.min()
    aoar_max = aoar.max()
    Res_min = Res.min()
    Res_max = Res.max()

    def aero_interp(aoa, U, c, nu=1.568e-5):
        """Calculate the aerodynamic force coefficients.

        Parameters
        ----------
        aoa : array, size=(nbody)
            angle of attack in radians
        U : array, size=(nbody)
            velocity in m/s
        c : array, size=(nbody)
            chord length in m
        nu : float, optional
            kinematic viscosity in m^2/s (Pa-s), default=1.568e-5

        Returns
        -------
        cl : array
            lift coefficient
        cd : array
            drag coefficient
        clcd : array
            lift-to-drag ratio = cl / cd
        Re0 : array
            original Reynolds number

        Notes
        -----
        We return Re0, the original Reynolds number, because if the
        current value is outside the measured range of [3000, 15000],
        we truncate it to that value.
        """

        # aoa should always be in range < 90 deg (by how I define
        # aoa during the simulaion), but maybe if the snake flips aoa < 10 deg
        aoa[aoa < aoar_min] = aoar_min
        aoa[aoa > aoar_max] = aoar_max

        Re0 = U * c / nu
        Re = Re0
        Re[Re < Res_min] = Res_min
        Re[Re > Res_max] = Res_max

        cl = cl_fun.ev(aoa, Re).flatten()
        cd = cd_fun.ev(aoa, Re).flatten()
        clcd = cl / cd

        return cl, cd, clcd, Re0

    return aero_interp


def extend_wind_tunnel_data(bl=1, bd=1, plot=False):
    """Extend the measured lift and drag coefficients from [-10, 60]
    to [-10, 90].

    Parameters
    ----------
    bl : float, default=1
        Force multiplier on the lift the lift force
    bd : float, default=1
        Force multiplier on the drag force
    plot : bool, default=False
        To make plots of lift and drag coefficients (measured and extrapoled),
        like in Holden (2014).

    Returns
    -------
    aero_interp : function which calculates the aerodynamic coefficients.

    Notes
    -----
    We extrapolate the force curves by fitting a parabola to drag, enforcing
    the vertex to be at (90, 2.0-2.1) and matching the experimental value a
    aoa=60. For lift, we fit a 3rd order polynomial (four coefficients) with
    the constraints that Cl(90) = 0, Cl(60) and dCl/daoa(60) match experiments
    with dCl/daoa from 2nd order accurate backward differences. Since we have
    three equations and four parameters, we use a least-squres solution
    technique.

    aero_interp Parameters
    ----------------------
    aoa : float
        angle of attack in radians
    U : float
        velocity in m/s
    c : float
        chord length in m
    nu : float, optional
        kinematic viscosity in m^2/s (Pa-s), default=1.568e-5

    aero_interp Returns
    -------------------
    cl : float
        lift coefficient
    cd : float
        drag coefficient
    clcd : float
        lift-to-drag ratio = cl / cd
    Re0 : float
        original Reynolds number

    aero_interp Notes
    -----------------
    We return Re0, the original Reynolds number, because if the
    current value is outside the measured range of [3000, 15000],
    we truncate it to that value.
    """

    airfoil = loadmat(DATA_FILE)
    Re = np.r_[3000:15001:2000]
    aoad = airfoil['aoa'].flatten().astype(np.int)
    # aoar = np.deg2rad(aoad)
    lift = airfoil['lift'].T
    drag = airfoil['drag'].T

    # extrapolate the curves to 90 deg
    aoae = np.r_[aoad[-1] + 5:91:5]
    aoade = np.r_[aoad, aoae]
    aoare = np.deg2rad(aoade)
    lifte = np.zeros((len(Re), len(aoade)))
    drage = np.zeros((len(Re), len(aoade)))

    lifte[:len(Re), :len(aoad)] = lift
    drage[:len(Re), :len(aoad)] = drag

    Cd = drag[:, -1]
    Cl = lift[:, -1]
    dCl = 1.5 * lift[:, -1] - 2 * lift[:, -2] + .5 * lift[:, -3]

    A_lift = np.array([[90**3, 90**2, 90, 1],  # lift at 90
                       [60**3, 60**2, 60, 1],  # lift at 60
                       [3 * 60**2, 2 * 60, 0, 0]])  # dlift/daoa at 60

    Cd90 = np.linspace(2, 2.1, len(Re))

    # perform the extrapolation
    for i in np.arange(len(Re)):
        # components of matrix equations
        a =(Cd[i] - Cd90[i]) / (60**2 - 180 * 60 + 180**2 / 4)
        b = -180 * a
        c = Cd90[i] + b**2 / (4 * a)
        drag_coeffs = np.array([a, b, c])

        lift_lhs = np.array([0, Cl[i], dCl[i]])
        lift_coeffs = np.linalg.lstsq(A_lift, lift_lhs)[0]

        # extrapolate the lift and drag curves
        drage[i, -len(aoae):] = np.polyval(drag_coeffs, aoae)
        lifte[i, -len(aoae):] = np.polyval(lift_coeffs, aoae)

    # actually perform the interpolation
    aero_interp = aero_interp_funcs(Re, aoare, bl * lifte, bd * drage)

    # make some figures and save them
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm'}
        sns.set('notebook', 'ticks', font='Helvetica',
                font_scale=13/11, color_codes=True, rc=rc)

        from matplotlib.ticker import FuncFormatter

        # http://stackoverflow.com/a/8555837
        def _formatter_remove_zeros(x, pos):
            """Format 1 as 1, 0 as 0, and all values whose absolute values is between
            0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
            formatted as -.4)."""
            val_str = '{:g}'.format(x)
            if np.abs(x) > 0 and np.abs(x) < 1:
                return val_str.replace("0", "", 1)
            else:
                return val_str

        decimal_formatter = FuncFormatter(_formatter_remove_zeros)

        FIG = '../Figures/m_aerodynamics/{}.pdf'
        FIGOPT = {'transparent': True}

        bmap = sns.color_palette("cubehelix_r", len(Re))

        # FOUR PLOT SIMILAR TO HOLDEN (2014)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.axvline(0, color='gray', lw=.5)
        ax2.axvline(0, color='gray', lw=.5)
        ax3.axvline(0, color='gray', lw=.5)
        ax1.axhline(0, color='gray', lw=.5)
        ax2.axhline(2.1, color='gray', lw=.5)
        ax3.axhline(0, color='gray', lw=.5)
        ax4.axhline(0, color='gray', lw=.5)
        ax1.axvline(aoad[-1], color='gray', lw=.5, ls='--')
        ax2.axvline(aoad[-1], color='gray', lw=.5, ls='--')
        ax3.axvline(aoad[-1], color='gray', lw=.5, ls='--')
        for i in range(len(Re)):
            ax1.plot(aoade, lifte[i], c=bmap[i])
            ax2.plot(aoade, drage[i], c=bmap[i], label='{:,}'.format(Re[i]))
            ax3.plot(aoade, lifte[i] / drage[i], c=bmap[i])
            ax4.plot(drage[i], lifte[i], c=bmap[i])

        ax1.set_xlim(-10, 90)
        ax2.set_xlim(-10, 90)
        ax3.set_xlim(-10, 90)
        ax3.set_ylim(-2.5, 3)
        # ax2.set_ylim(0, 3)
        # ax4.set_xlim(0, 3)
        ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax3.set_xticks([0, 15, 30, 45, 60, 75, 90])
        # ax2.legend(loc='best', title=r'$Re = \frac{Uc}{\nu}$', ncol=2)
        ax2.legend(loc='best', title=r'$Re = Uc/\nu$', ncol=2,
                   fontsize='x-small')
        ax1.set_xlabel(r'$\alpha$')
        ax1.set_ylabel(r'$C_L$')
        ax2.set_xlabel(r'$\alpha$')
        ax2.set_ylabel(r'$C_D$')
        ax3.set_xlabel(r'$\alpha$')
        ax3.set_ylabel(r'$C_L/C_D$')
        ax4.set_xlabel(r'$C_D$')
        ax4.set_ylabel(r'$C_L$')

        sns.despine()
        fig.set_tight_layout(True)
        fig.canvas.draw()

        # add degree symbol to angles
        for ax in [ax1, ax2, ax3]:
            ticks = ax.get_xticklabels()
            newticks = []
            for tick in ticks:
                text = tick.get_text()
                newticks.append(text + u'\u00B0')
            ax.set_xticklabels(newticks)

        fig.savefig(FIG.format('four_plot'), **FIGOPT)


        # PLOT WITH TWO AXES CL AND CD ON LEFT, CL/CD ON RIGHT
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True,
                                       sharey=False, figsize=(9.5, 3.75))
        ax1.axvline(0, color='gray', lw=.5)
        ax2.axvline(0, color='gray', lw=.5)
        ax1.axhline(0, color='gray', lw=.5)
        ax2.axhline(0, color='gray', lw=.5)
        for i in range(len(Re)):
            ax1.plot(aoade, lifte[i], c=bmap[i])
            ax1.plot(aoade, drage[i], c=bmap[i])
            ax2.plot(aoade, lifte[i] / drage[i], c=bmap[i],
                     label='{:,}'.format(Re[i]))

        # plot the extended region as a bar on the bottom right
        ax1.plot([60, 90], [-1, -1], c='gray', lw=6,
                 solid_capstyle='butt')
        ax2.plot([60, 90], [-2.5, -2.5], c='gray', lw=6,
                 solid_capstyle='butt')

        ax1.set_xlim(-10, 90)
        ax2.set_xlim(-10, 90)

        ax1.set_ylim(-1, 2.5)
        #ax1.set_yticks([-1, 0, 1, 2])
        #ax1.yaxis.set_major_formatter(decimal_formatter)
        ax2.set_ylim(-2.5, 3)
        ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])

        ax2.legend(loc='best',
                   fontsize='x-small', frameon=False, ncol=2,
                   borderaxespad=0, handletextpad=.25, handlelength=1.1)
        ax1.set_xlabel(r'$\alpha$')
        ax1.set_ylabel('force coefficients')#, fontsize='small')
        ax2.set_xlabel(r'$\alpha$')
        #ax2.set_ylabel('lift-to-drag ratio')#, fontsize='small')
        ax2.text(2, 2, r'$C_L/C_D$')
        ax1.text(14, 1.3, r'$C_L$')
        ax1.text(60, 2.05, r'$C_D$')

        sns.despine()
        fig.set_tight_layout(True)
        fig.canvas.draw()

        # add degree symbol to angles
        for ax in [ax1, ax2]:
            ticks = ax.get_xticklabels()
            newticks = []
            for tick in ticks:
                text = tick.get_text()
                newticks.append(text + u'\u00B0')
            ax.set_xticklabels(newticks)

        fig.savefig(FIG.format('two_plot'), **FIGOPT)


        # PLOT CL AND CD ONE ONE AXIS
        fig, ax1 = plt.subplots(1, 1, sharex=True,
                                sharey=False, figsize=(5, 3.7))
        ax1.axvline(0, color='gray', lw=.5)
        ax1.axhline(0, color='gray', lw=.5)
        for i in range(len(Re)):
            ax1.plot(aoade, lifte[i], c=bmap[i])
            ax1.plot(aoade, drage[i], c=bmap[i], label='{:,}'.format(Re[i]))

        # plot the extended region as a bar on the bottom right
        ax1.plot([60, 90], [-1, -1], c='gray', lw=6,
                 solid_capstyle='butt')

        ax1.set_xlim(-10, 90)
        ax1.set_ylim(-1, 2.3)
        ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax1.legend(loc='best', frameon=False, ncol=3,
                   borderaxespad=0, handletextpad=.25, handlelength=1)
        ax1.set_xlabel(r'$\alpha$')
        ax1.set_ylabel('force coefficients')#, fontsize='small')
        # ax1.text(14, 1.3, r'$\mathsf{C_L}$')
        # ax1.text(60, 2.05, r'$\mathsf{C_D}$')
        ax1.text(14, 1.3, r'$C_L$')
        ax1.text(60, 2.05, r'$C_D$')

        sns.despine()
        fig.set_tight_layout(True)
        fig.canvas.draw()

        # add degree symbol to angles
        for ax in [ax1]:
            ticks = ax.get_xticklabels()
            newticks = []
            for tick in ticks:
                text = tick.get_text()
                newticks.append(text + u'\u00B0')
            ax.set_xticklabels(newticks)

        fig.savefig(FIG.format('one_plot'), **FIGOPT)


        # PLOT THE EXTRAPOLATED AND INTERPOLATED REGIONS
        import matplotlib.patches as patches

        bottom = patches.Rectangle((-10, 0), 10 + 90, 3000, color='gray',
                                   alpha=.25, linewidth=0)

        Re_measured = np.r_[3000, 5000, 7000, 9000, 11000, 13000, 15000]
        aoa_measured = np.arange(-10, 61, 5)
        aoa_extrap = np.arange(65, 91, 5)

        fig, ax = plt.subplots(figsize=(5, 5))

        for i, Re in enumerate(Re_measured):
            for aoa in aoa_measured:
                ax.plot(aoa, Re, 'o', c=bmap[i])

            for aoa in aoa_extrap:
                ax.plot(aoa, Re, 's', c='gray')

        ax.add_patch(bottom)

        ax.grid(True)

        ax.set_xlabel(r'angle of attack, $\alpha$')
        ax.set_ylabel('Reynolds number, Re')

        ax.set_xlim(-12.5, 92.5)
        ax.set_ylim(0, 15500)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_yticks([0, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])

        fig.canvas.draw()

        ticks = ax.get_xticklabels()
        newticks = []
        for tick in ticks:
            text = tick.get_text()
            newticks.append(text + u'\u00B0')
        ax.set_xticklabels(newticks)

        ticks = ax.get_yticklabels()
        newticks = []
        for tick in ticks:
            text = tick.get_text()
            newticks.append('{:,}'.format(int(text)))
        ax.set_yticklabels(newticks)

        sns.despine()

        fig.set_tight_layout(True)

        fig.savefig(FIG.format('extrap_summary'), **FIGOPT)

    return aero_interp

