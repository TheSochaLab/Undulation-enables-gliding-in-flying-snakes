# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:11:49 2015

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from scipy.io import loadmat

import seaborn as sns

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42}
sns.set('notebook', 'ticks', font_scale=1.35, rc=rc)
bmap = sns.color_palette()

COEFFICIENT_FILE = '../Data/Aerodynamics/holden_lift_drag.mat'


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


def extend_wind_tunnel_data(plot=True):
    """Extend the measured lift and drag coefficients from [-10, 60]
    to [-10, 90].

    Parameters
    ----------
    plot : bool, default=True
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

    airfoil = loadmat(COEFFICIENT_FILE)
    Re = np.r_[3000:15001:2000]
    aoad = airfoil['aoa'].flatten().astype(np.int)
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
    aero_interp = aero_interp_funcs(Re, aoare, lifte, drage)

    # plot similar to Holden (2014)
    if plot:
        with sns.color_palette("cubehelix_r", len(Re)):
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
                ax1.plot(aoade, lifte[i])
                ax2.plot(aoade, drage[i], label='{:,}'.format(Re[i]))
                ax3.plot(aoade, lifte[i] / drage[i])
                ax4.plot(drage[i], lifte[i])

            ax1.set_xlim(-10, 90)
            ax2.set_xlim(-10, 90)
            ax3.set_xlim(-10, 90)
            ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
            ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
            ax3.set_xticks([0, 15, 30, 45, 60, 75, 90])
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

            # plt.draw()
            fig.canvas.draw()

            # add degree symbol to angles
            for ax in [ax1, ax2, ax3]:
                ticks = ax.get_xticklabels()
                newticks = []
                for tick in ticks:
                    text = tick.get_text()
                    newticks.append(text + u'\u00B0')
                ax.set_xticklabels(newticks)


        # plot the original Holden data
        with sns.color_palette("cubehelix_r", len(Re)):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.axvline(0, color='gray', lw=.5)
            ax2.axvline(0, color='gray', lw=.5)
            ax3.axvline(0, color='gray', lw=.5)
            ax1.axhline(0, color='gray', lw=.5)
            ax3.axhline(0, color='gray', lw=.5)
            ax4.axhline(0, color='gray', lw=.5)
            for i in range(len(Re)):
                ax1.plot(aoad, lift[i])
                ax2.plot(aoad, drag[i], label='{:,}'.format(Re[i]))
                ax3.plot(aoad, lift[i] / drag[i])
                ax4.plot(drag[i], lift[i])

            ax1.set_xlim(-10, 60)
            ax2.set_xlim(-10, 60)
            ax3.set_xlim(-10, 60)
            ax3.set_ylim(-2, 3)
            ax2.set_ylim(0, 2)
            ax4.set_xlim(0, 2)
            ax1.set_xticks([-10, 0, 10, 20, 30, 40, 50, 60])
            ax2.set_xticks([-10, 0, 10, 20, 30, 40, 50, 60])
            ax3.set_xticks([-10, 0, 10, 20, 30, 40, 50, 60])
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

            # plt.draw()
            fig.canvas.draw()

            # add degree symbol to angles
            for ax in [ax1, ax2, ax3]:
                ticks = ax.get_xticklabels()
                newticks = []
                for tick in ticks:
                    text = tick.get_text()
                    newticks.append(text + u'\u00B0')
                ax.set_xticklabels(newticks)

    return aero_interp


def plot_extrapolated_clcd(aero_interp):
    """Given the interpolationg function, plot the snake coefficients. This
    is a check that the extrapolation function return value works correctly.

    Parameters
    ----------
    aero_interp : function
        Output from extend_wind_tunnel_data

    Returns
    -------
    A "four plot" which is similar to extend_wind_tunnel_data(plot=True),
    except over a larger Re range.
    """

    Re = 1000 * np.r_[1, 3, 5, 7, 9, 11, 13, 15, 17]
    aoa = np.arange(-10, 91, 5)
    nu = 1.568e-5
    c = .022
    U = Re * nu / c

    nre, nang = len(Re), len(aoa)
    lift = np.zeros((nre, nang))
    drag = np.zeros((nre, nang))
    liftdrag = np.zeros((nre, nang))
    Re0s = np.zeros((nre, nang))

    for i in np.arange(nre):
        for j in np.arange(nang):
            cl, cd, clcd, Re0 = aero_interp(np.deg2rad(aoa[j]), U[i], c)
            lift[i, j] = cl
            drag[i, j] = cd
            liftdrag[i, j] = clcd
            Re0s[i, j] = Re0

    with sns.color_palette("cubehelix_r", len(Re)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.axvline(0, color='gray', lw=.5)
        ax2.axvline(0, color='gray', lw=.5)
        ax3.axvline(0, color='gray', lw=.5)
        ax1.axhline(0, color='gray', lw=.5)
        ax2.axhline(2.1, color='gray', lw=.5)
        ax3.axhline(0, color='gray', lw=.5)
        ax4.axhline(0, color='gray', lw=.5)
        ax1.axvline(60, color='gray', lw=.5, ls='--')
        ax2.axvline(60, color='gray', lw=.5, ls='--')
        ax3.axvline(60, color='gray', lw=.5, ls='--')
        for i in range(len(Re)):
            ax1.plot(aoa, lift[i])
            ax2.plot(aoa, drag[i], label='{:,}'.format(int(Re0s[i, 0])))
            ax3.plot(aoa, liftdrag[i])
            ax4.plot(drag[i], lift[i])

        ax1.set_xlim(-10, 90)
        ax2.set_xlim(-10, 90)
        ax3.set_xlim(-10, 90)
        ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax3.set_xticks([0, 15, 30, 45, 60, 75, 90])
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

        plt.draw()

        # add degree symbol to angles
        for ax in [ax1, ax2, ax3]:
            ticks = ax.get_xticklabels()
            newticks = []
            for tick in ticks:
                text = tick.get_text()
                newticks.append(text + u'\u00B0')
            ax.set_xticklabels(newticks)


def extend_wind_tunnel_data_old(plot=True):
    """Extend the measured lift and drag coefficients from [-10, 60]
    to [-10, 90].

    Parameters
    ----------
    plot : bool, default=True
        To make plots of lift and drag coefficients (measured and extrapoled),
        like in Holden (2014).

    Returns
    -------
    aero_interp : function which calculates the aerodynamic coefficients.

    Notes
    -----
    We extrapolate the force curves by fitting a linear line to the drag
    curve and lift-to-drag ratio from aoa = [50, 60] out to 90 degrees. We
    then force cl/cd[90] = 0 by modifying the slope. We then multiply the
    linear fit coeffienents of cd and clcd to get the linear fit coeffiencts
    of cl (it is parabolic). Drag seems to be too high at the end, probably
    when aoa > 75 degrees.

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

    airfoil = loadmat(COEFFICIENT_FILE)
    Re = np.r_[3000:15001:2000]
    aoad = airfoil['aoa'].flatten().astype(np.int)
    aoar = np.deg2rad(aoad)
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

    ix = np.where(aoad >= 50)[0]

    for i in np.arange(len(Re)):
        # make Cd = 2 at alpha = 90 deg
        pfd = np.polyfit(aoar[ix], drag[i, ix], 1)
        pfld = np.polyfit(aoar[ix], lift[i, ix] / drag[i, ix], 1)
        pfld[0] = -2 * 1 * pfld[1] / np.pi  # make cl/cd(aoa=90) = 0
        pfl = np.array([pfd[0] * pfld[0],
                        pfld[0] * pfd[1] + pfd[0] * pfld[1],
                        pfld[1] * pfd[1]])

        lifte[i, -len(aoae):] = np.polyval(pfl, np.deg2rad(aoae))
        drage[i, -len(aoae):] = np.polyval(pfd, np.deg2rad(aoae))

    # actually perform the interpolation
    aero_interp = aero_interp_funcs(Re, aoare, lifte, drage)

    # plot similar to Holden (2014)
    if plot:
        with sns.color_palette("cubehelix_r", 8):
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
                ax1.plot(aoade, lifte[i])
                ax2.plot(aoade, drage[i], label='{:,}'.format(Re[i]))
                ax3.plot(aoade, lifte[i] / drage[i])
                ax4.plot(drage[i], lifte[i])

            ax1.set_xlim(-10, 90)
            ax2.set_xlim(-10, 90)
            ax3.set_xlim(-10, 90)
            ax2.set_ylim(0, 3)
            ax1.set_xticks([0, 15, 30, 45, 60, 75, 90])
            ax2.set_xticks([0, 15, 30, 45, 60, 75, 90])
            ax3.set_xticks([0, 15, 30, 45, 60, 75, 90])
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

            plt.draw()

            # add degree symbol to angles
            for ax in [ax1, ax2, ax3]:
                ticks = ax.get_xticklabels()
                newticks = []
                for tick in ticks:
                    text = tick.get_text()
                    newticks.append(text + u'\u00B0')
                ax.set_xticklabels(newticks)

    return aero_interp


def plot_aero_data_360():
    """This is the single Reynolds number data that was extended over
    a 360 degree range. Cd(90) was set to 2, but I am not sure how they
    extrapolated the other values. Maybe Cl(90) = 0, and then spline
    fitting?
    """

    data = loadmat(COEFFICIENT_FILE)

    # get out the data
    Clall = data['C_lift'].flatten()
    Cdall = data['C_drag'].flatten()
    alphar = data['alpha'].flatten()
    alphad = np.rad2deg(alphar)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(alphad, Clall)
    ax2.plot(alphad, Cdall)
    ax3.plot(alphad, Clall / Cdall)
    ax4.plot(Cdall, Clall)

    ax1.set_ylim(-1, 2)
    ax2.set_ylim(0, 2)
    ax3.set_ylim(-2.5, 3)
    ax4.set_xlim(0, 2)
    ax4.set_ylim(-1, 2)
    ax1.set_xlabel(r'$\alpha$ (deg)')
    ax1.set_ylabel(r'$C_L$')
    ax2.set_xlabel(r'$\alpha$ (deg)')
    ax2.set_ylabel(r'$C_D$')
    ax3.set_xlabel(r'$\alpha$ (deg)')
    ax3.set_ylabel(r'$C_L/C_D$')
    ax4.set_xlabel(r'$C_D$')
    ax4.set_ylabel(r'$C_L$')
    sns.despine()
    fig.set_tight_layout(True)
