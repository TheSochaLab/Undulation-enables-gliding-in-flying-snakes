#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:57:05 2017

%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

cd /Volumes/Yeaton_HD6/Code for Manuscripts/Undulation_confers_stability/Simulations/Code

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import cumtrapz
from scipy.optimize import fmin

np.set_printoptions(suppress=True)

import seaborn as sns
rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

from mayavi import mlab

import time

import m_sim as sim
import m_aerodynamics as aerodynamics
import m_morph as morph

FIG = '../Figures/s_serp3d_nu_theta_full/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}
FIGPNG = '../Figures/s_serp3d_nu_theta_full/{}.png'


# %% To format the VPD

from m_plotting import add_arrow_to_line2D

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

def _formatter_degree(x, pos):
    """Add a degree symbol.
    """

    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x) + u'\u00B0'
    return val_str


decimal_formatter = FuncFormatter(_formatter_remove_zeros)
degree_formatter = FuncFormatter(_formatter_degree)


def setup_body(L=.7, ds=.01, theta_max=90, nu_theta=1.1, f_theta=1.4,
               phi_theta=0, psi_max=10, frac_theta_max=0, d_theta=0, d_psi=0,
               display_ho=1, ho_shift=True, bl=1, bd=1):
    """Setup the body_dict for simulations.
    """

    s = ds / 2 + np.arange(0, L, ds)  # m
    nbody = len(s)
    dt = .010  # sec
    neck_length = .075 * L  # 5% SVL % .05 m on a .7 m snake is 7.14%
    n_neck = np.floor(neck_length / ds).astype(np.int)

    cmax, mtot, Ws_fit, rho_bar = morph.morph_from_svl(L)
    c = morph.chord_dist(s, L)
    m = morph.mass_dist(s, ds, L, mtot)
    g = 9.81
    weight = mtot * g  # N
    darea = ds * c  # m^2, area of each segment
    area = darea.sum()
    Ws = weight / area

    # density of air
    rho = 1.165  # 30 C

    # convert non-dim to dim units
    tscale = np.sqrt(2 * Ws / (rho * g**2))
    pscale = 2 * Ws / (rho * g)
    vscale = np.sqrt(2 * Ws / rho)  # multi to non to get in dim
    ascale = g
    fscale = weight  # mtot * g
    mscale = fscale * pscale  # = 2 * mtot * Ws / rho

    # initial condition of 1.7 m/s, Ws = 29 N/m^2 (Socha 2005)
    v0_non = 1.7 / np.sqrt(2 * Ws / rho)  # ~.2409

    # aerodynamics
    aero_interp = aerodynamics.extend_wind_tunnel_data(bl=bl, bd=bd)

    nu_psi = 2 * nu_theta
    f_psi = 2 * f_theta
    phi_psi = 2 * (phi_theta - np.pi / 2)

    theta_max = np.deg2rad(theta_max)
    amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
    amp_theta = theta_max * amp_theta_fun
    damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
    d_theta = np.deg2rad(d_theta)

    psi_max = np.deg2rad(psi_max)
    frac_psi_max = 0
    amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
    amp_psi = psi_max * amp_psi_fun
    damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
    d_psi = np.deg2rad(d_psi)

    theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                      amp_theta=amp_theta, damp_theta=damp_theta,
                      d_theta=d_theta, L=L, theta_max=theta_max,
                      frac_theta_max=frac_theta_max,
                      amp_theta_fun=amp_theta_fun)
    psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                    amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                    psi_max=psi_max, frac_psi_max=frac_psi_max,
                    amp_psi_fun=amp_psi_fun)

    # phase shift the serpenoid curve for a near zero angular momentum
    t = 0
    ho_args = (s, t, m, n_neck, theta_dict, psi_dict)
    phi_theta = fmin(sim.func_ho_to_min, phi_theta, args=ho_args,
                     ftol=1e-7, xtol=1e-7, disp=display_ho)
    phi_theta = float(phi_theta)
    phi_psi = 2 * (phi_theta - np.pi / 2)

    if ho_shift:
        theta_dict['phi_theta'] = phi_theta
        psi_dict['phi_psi'] = phi_psi

    # dictionary with all of the simulation parameters in it
    body_dict = dict(L=L, ds=ds, s=s, nbody=nbody, neck_length=neck_length,
                     n_neck=n_neck, cmax=cmax, mtot=mtot, rho_bar=rho_bar,
                     c=c, m=m, weight=weight, darea=darea, area=area, Ws=Ws,
                     theta_dict=theta_dict, psi_dict=psi_dict,
                     tscale=tscale, pscale=pscale, vscale=vscale,
                     ascale=ascale, fscale=fscale, mscale=mscale,
                     dt=dt, g=g, rho=rho, aero_interp=aero_interp,
                     head_control=False, v0_non=v0_non)

    return body_dict


def rotational_dynamics_eom(t, state, vel_body_dict):
    """
    """

    # unpack the arguments
    dRo, body_dict = vel_body_dict

    # current angles and angular velocity
    omg, ang = np.split(state, 2)
    yaw, pitch, roll = ang

    # unpack needed kinematics variables
    s, m, n_neck, g = body_dict['s'], body_dict['m'], body_dict['n_neck'], body_dict['g']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    vscale, weight, rho = body_dict['vscale'], body_dict['weight'], body_dict['rho']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']

    # body kinematics
    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)
    p, dp, ddp = out['p'], out['dp'], out['ddp']
    tv, cv, bv = out['tv'], out['cv'], out['bv']

    # rotation matrix from inertial to body
    C = sim.euler2C(yaw, pitch, roll)

    # control tv, cv, bv based on head orientation
    head_control = body_dict['head_control']
    if head_control:
        tv, cv, bv, _ = sim.serp3d_tcb_head_control(out['dpds'], C)

    # positions, velocities, and accelerations
    r, dr, ddr = sim.rotate(C.T, p), sim.rotate(C.T, dp), sim.rotate(C.T, ddp)
    dR = dRo + dr + np.cross(omg, r)

    # gravitational force in inertial frame
    F = np.zeros(r.shape)
    F[:, 2] = -m * g

    # aerodynamic forces
    if aero_interp is not None:
        Fa = sim.aero_forces(tv, cv, bv, C, dR, ds, c, rho, aero_interp)
        F += Fa

    # form the dynamic equations
    M, N, _, _ = sim.dynamics_submatrices(r, dr, ddr, omg, m, F)

    # extract only rotational dynamics
    M = M[3:, 3:]  # lower right
    N = N[3:]

    # solve for domg
    domg = np.linalg.solve(M, -N)

    # solve for change in Euler angles (kinematic differential equations)
    omg_body = sim.rotate(C, omg)
    dang = np.dot(sim.euler2kde(yaw, pitch, roll), omg_body)

    # combine our derivatives as the return parameter
    return np.r_[domg, dang]


def find_rotational_ic(body_dict, print_time=False):
    """
    """

    now = time.time()

    from scipy.integrate import ode

    # turn off aerodynamic forces
    body_dict = body_dict.copy()  # just to be safe
    body_dict['aero_interp'] = None

    ang0 = np.r_[0, 0, 0]  # horizontal flight
    omg0 = np.r_[0, 0, 0]  # no angular velocity

    # v0_non = 1.7 / np.sqrt(2 * 29 / rho)  # .2409, Ws=29
    soln0 = np.r_[omg0, ang0]

    # pick a velocity in the VPD (ignored in rotation function)
    dRo0 = np.r_[0, 0, 0]

    # arguments to the rotation dynamics function
    vel_body_dict = (dRo0, body_dict)

    # phases in cycle to simulate rotational dynamics
    ntime_rot = 200
    tend = 1 / body_dict['theta_dict']['f_theta']
    ts_rot = np.linspace(0, tend, ntime_rot + 1)[:-1]

    solver = ode(rotational_dynamics_eom)
    solver.set_integrator('dopri5')
    solver.set_initial_value(soln0, ts_rot[0])  # x0, t0
    solver.set_f_params(vel_body_dict)

    # integrate over one undulation cycle
    i = 1
    soln_rot = [soln0]
    while i < ntime_rot:
        solver.integrate(ts_rot[i])
        out = solver.y

        soln_rot.append(out)
        i = i + 1

    if print_time:
        print('Find rotational IC: {0:.3f} sec'.format(time.time() - now))

    # unpack the solution
    soln_rot = np.array(soln_rot)
    wx, wy, wz, yaw, pitch, roll = soln_rot.T
    ang = np.c_[yaw, pitch, roll]
    # omg = np.c_[wx, wy, wz]

    # initial yaw, pitch, and roll angles in radians
    return -ang.mean(axis=0)


def simulation_recon(body_dict, ts, Ro, dRo, omg, ang):
    """Reconstruct the body and forces from the simulation.
    """

    # state vector
    yaw, pitch, roll = ang.T

    # unpack additional arguments
    s, m, n_neck = body_dict['s'], body_dict['m'], body_dict['n_neck']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    rho, g = body_dict['rho'], body_dict['g']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']
    nbody = body_dict['nbody']

    ntime = len(ts)

    # kinematics simulation
    theta = np.zeros((ntime, nbody))
    psi = np.zeros((ntime, nbody))
    dthetads = np.zeros((ntime, nbody))
    dpsids = np.zeros((ntime, nbody))
    p = np.zeros((ntime, nbody, 3))
    dp = np.zeros((ntime, nbody, 3))
    ddp = np.zeros((ntime, nbody, 3))
    dpds = np.zeros((ntime, nbody, 3))
    ddpdds = np.zeros((ntime, nbody, 3))

    tv = np.zeros((ntime, nbody, 3))
    cv = np.zeros((ntime, nbody, 3))
    bv = np.zeros((ntime, nbody, 3))
    Tv = np.zeros((ntime, nbody, 3))
    Cv = np.zeros((ntime, nbody, 3))
    Bv = np.zeros((ntime, nbody, 3))
    Crs = np.zeros((ntime, nbody, 3, 3))
    Crs_I = np.zeros((ntime, nbody, 3, 3))

    C = np.zeros((ntime, 3, 3))
    Fl = np.zeros((ntime, nbody, 3))
    Fd = np.zeros((ntime, nbody, 3))
    Fa = np.zeros((ntime, nbody, 3))
    dR_BC = np.zeros((ntime, nbody, 3))
    dR_T = np.zeros((ntime, nbody, 3))
    aoa = np.zeros((ntime, nbody))
    Re = np.zeros((ntime, nbody))
    Ml = np.zeros((ntime, nbody, 3))
    Md = np.zeros((ntime, nbody, 3))
    Ma = np.zeros((ntime, nbody, 3))
    dynP = np.zeros((ntime, nbody))
    U_BC = np.zeros((ntime, nbody))
    U_tot = np.zeros((ntime, nbody))
    Dh = np.zeros((ntime, nbody, 3))
    Lh = np.zeros((ntime, nbody, 3))
    cl = np.zeros((ntime, nbody))
    cd = np.zeros((ntime, nbody))
    clcd = np.zeros((ntime, nbody))
    dR_B = np.zeros((ntime, nbody, 3))
    dR_TC = np.zeros((ntime, nbody, 3))
    U_TC = np.zeros((ntime, nbody))
    beta = np.zeros((ntime, nbody))
    dynP_frac = np.zeros((ntime, nbody))

    Fl_B = np.zeros((ntime, nbody, 3))
    Fd_B = np.zeros((ntime, nbody, 3))
    Fa_B = np.zeros((ntime, nbody, 3))
    Ml_B = np.zeros((ntime, nbody, 3))
    Md_B = np.zeros((ntime, nbody, 3))
    Ma_B = np.zeros((ntime, nbody, 3))

    # rotational power = tau \dot \omega
    power = np.zeros((ntime, nbody))

    r = np.zeros((ntime, nbody, 3))
    dr = np.zeros((ntime, nbody, 3))
    ddr = np.zeros((ntime, nbody, 3))
    dR = np.zeros((ntime, nbody, 3))

    Nframe = np.eye(3)
    nframe = np.zeros((ntime, 3, 3))

    for i in np.arange(ntime):
        t = ts[i]

        out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

        # store the values
        theta[i] = out['theta']
        psi[i] = out['psi']
        dthetads[i] = out['dthetads']
        dpsids[i] = out['dpsids']

        # position, velocity, acceleration
        p[i] = out['p']
        dp[i] = out['dp']
        ddp[i] = out['ddp']

        # derivatives along spine
        dpds[i] = out['dpds']
        ddpdds[i] = out['ddpdds']

        # body coordinate system
        tv[i] = out['tv']
        cv[i] = out['cv']
        bv[i] = out['bv']
        Crs[i] = out['Crs']

        C[i] = sim.euler2C(yaw[i], pitch[i], roll[i])

        r[i] = sim.rotate(C[i].T, p[i])
        dr[i] = sim.rotate(C[i].T, dp[i])
        ddr[i] = sim.rotate(C[i].T, ddp[i])
        dR[i] = dRo[i] + dr[i] + sim.cross(omg[i], r[i])

        Tv[i] = sim.rotate(C[i].T, tv[i])
        Cv[i] = sim.rotate(C[i].T, cv[i])
        Bv[i] = sim.rotate(C[i].T, bv[i])
        for j in np.arange(nbody):
            Crs_I[i, j] = np.dot(C[i].T, Crs[i, j])

        out_aero = sim.aero_forces(tv[i], cv[i], bv[i], C[i], dR[i], ds, c, rho,
                                   aero_interp, full_out=True)

        # forces in the inertial frame
        Fl[i] = out_aero['Fl']
        Fd[i] = out_aero['Fd']
        Fa[i] = out_aero['Fa']
        aoa[i] = out_aero['aoa']
        Re[i] = out_aero['Re']
        dR_T[i] = out_aero['dR_T']
        dR_BC[i] = out_aero['dR_BC']

        dynP[i] = out_aero['dynP']
        U_BC[i] = out_aero['U_BC']
        U_tot[i] = out_aero['U_tot']
        Dh[i] = out_aero['Dh']
        Lh[i] = out_aero['Lh']
        cl[i] = out_aero['cl']
        cd[i] = out_aero['cd']
        clcd[i] = out_aero['clcd']
        dR_B[i] = out_aero['dR_B']
        dR_TC[i] = out_aero['dR_TC']
        U_TC[i] = out_aero['U_TC']
        beta[i] = out_aero['beta']
        dynP_frac[i] = out_aero['dynP_frac']

        # aero moments in the inertial frame
        Ml[i] = np.cross(r[i], Fl[i])
        Md[i] = np.cross(r[i], Fd[i])
        Ma[i] = np.cross(r[i], Fa[i])

        # aero forces and moments in the body frame
        Fl_B[i] = sim.rotate(C[i], Fl[i])
        Fd_B[i] = sim.rotate(C[i], Fd[i])
        Fa_B[i] = sim.rotate(C[i], Fa[i])
        Ml_B[i] = np.cross(p[i], Fl_B[i])
        Md_B[i] = np.cross(p[i], Fd_B[i])
        Ma_B[i] = np.cross(p[i], Fa_B[i])

        power[i] = np.dot(Ma[i], omg[i])

        nframe[i] = sim.rotate(C[i].T, Nframe)

    # airfoil in inertial frame
    foils_I, foil_color = sim.apply_airfoil_shape(r, c, Crs_I)
    foils_B, foil_color = sim.apply_airfoil_shape(r, c, Crs)

    # put everything into a dictionary
    out = dict(theta=theta, psi=psi, dthetads=dthetads, dpsids=dpsids,
               p=p, dp=dp, ddp=ddp, dpds=dpds, ddpdds=ddpdds,
               tv=tv, cv=cv, bv=bv, Tv=Tv, Cv=Cv, Bv=Bv, Crs=Crs, Crs_I=Crs_I,
               C=C, Fl=Fl, Fd=Fd, Fa=Fa, dR_BC=dR_BC, dR_T=dR_T,
               aoa=aoa, Re=Re, Ml=Ml, Md=Md, Ma=Ma,
               dynP=dynP, U_BC=U_BC, U_tot=U_tot, Dh=Dh, Lh=Lh, cl=cl, cd=cd,
               clcd=clcd, dR_B=dR_B, dR_TC=dR_TC, U_TC=U_TC, beta=beta,
               dynP_frac=dynP_frac, Fl_B=Fl_B, Fd_B=Fd_B, Fa_B=Fa_B,
               Ml_B=Ml_B, Md_B=Md_B, Ma_B=Ma_B,
               power=power, r=r, dr=dr, ddr=ddr, dR=dR,
               Nframe=Nframe, nframe=nframe,
               foils_I=foils_I, foils_B=foils_B, foil_color=foil_color)

    return out


def parallel_simulation(args):
    """Run the simulations in parallel.
    """

    # unpack the arguments
    fname, params = args
    nu_theta, theta_max, f_theta, d_psi, psi_max, L = params

    # if undulation is turned off, set parameters for f = 1.2 Hz
    f_theta_orig = f_theta
    if f_theta_orig == 0:
        f_theta = 1.2

    # setup the body
    body_dict = setup_body(L=L, ds=.01, theta_max=theta_max,
                       nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                       psi_max=psi_max, frac_theta_max=0,
                       d_theta=0, d_psi=d_psi, display_ho=0)
    vscale = body_dict['vscale']
    dt = float(body_dict['dt'])

    # find the initial Euler angle offsets so minimize inertial effects at the start
    ang0 = find_rotational_ic(body_dict, print_time=False)

    # now turn undulation back off --- for both waves
    if f_theta_orig != f_theta:
        body_dict['theta_dict']['f_theta'] = 0
        body_dict['psi_dict']['f_psi'] = 0

    # velocity initial condition
    v0_non = .2409  # 1.7 m/s for a 29 N/m^2 snake
    v0_dim = v0_non * vscale  # near 1.7 m/s

    # initial conditions
    tend = None
    Ro0 = np.r_[0, 0, 10]
    dRo0 = np.r_[0, v0_dim, 0]
    dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

    C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
    omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
    omg0 = np.dot(C0.T, omg0_body)
    soln0 = np.r_[Ro0, dRo0, omg0, ang0]

    # run the dynamics simulation
    out = sim.integrate(soln0, body_dict, dt, tend=tend, print_time=False)

    # extract values
    ts, Ro, dRo, omg, ang = out
    yaw, pitch, roll = np.rad2deg(ang.T)

    # reconstruct all values from the simulation
    out_recon = simulation_recon(body_dict, ts, Ro, dRo, omg, ang)

    # save the values
    np.savez(fname,
             vscale=vscale, dt=dt,
             ts=ts, Ro=Ro, dRo=dRo, omg=omg, ang=ang,
             yaw=yaw, pitch=pitch, roll=roll, f_theta=f_theta,
             psi_max=psi_max, d_psi=d_psi, **out_recon)


def parallel_simulation_boosted(args):
    """Run the simulations in parallel.
    """

    # unpack the arguments
    fname, params = args
    nu_theta, theta_max, f_theta, d_psi, psi_max, L = params

    # if undulation is turned off, set parameters for f = 1.2 Hz
    f_theta_orig = f_theta
    if f_theta_orig == 0:
        f_theta = 1.2

    # setup the body
    body_dict = setup_body(L=L, ds=.01, theta_max=theta_max,
                       nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                       psi_max=psi_max, frac_theta_max=0,
                       d_theta=0, d_psi=d_psi, display_ho=0,
                       bl=1.36, bd=.6)  # BOOST THE FORCES
    vscale = body_dict['vscale']
    dt = float(body_dict['dt'])

    # find the initial Euler angle offsets so minimize inertial effects at the start
    ang0 = find_rotational_ic(body_dict, print_time=False)

    # now turn undulation back off --- for both waves
    if f_theta_orig != f_theta:
        body_dict['theta_dict']['f_theta'] = 0
        body_dict['psi_dict']['f_psi'] = 0

    # velocity initial condition
    v0_non = .2409  # 1.7 m/s for a 29 N/m^2 snake
    v0_dim = v0_non * vscale  # near 1.7 m/s

    # initial conditions
    tend = None
    Ro0 = np.r_[0, 0, 10]
    dRo0 = np.r_[0, v0_dim, 0]
    dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

    C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
    omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
    omg0 = np.dot(C0.T, omg0_body)
    soln0 = np.r_[Ro0, dRo0, omg0, ang0]

    # run the dynamics simulation
    out = sim.integrate(soln0, body_dict, dt, tend=tend, print_time=False)

    # extract values
    ts, Ro, dRo, omg, ang = out
    yaw, pitch, roll = np.rad2deg(ang.T)

    # reconstruct all values from the simulation
    out_recon = simulation_recon(body_dict, ts, Ro, dRo, omg, ang)

    # save the values
    np.savez(fname,
             vscale=vscale, dt=dt,
             ts=ts, Ro=Ro, dRo=dRo, omg=omg, ang=ang,
             yaw=yaw, pitch=pitch, roll=roll, f_theta=f_theta,
             psi_max=psi_max, d_psi=d_psi, **out_recon)


# %% Parameter values to sweep over

psi_maxs = np.r_[0, 10, 20]
d_psis = np.r_[-20, -10, 0, 10, 20]

f_theta = 1.2
L = .7

# use slope and intercept from nu_thetas to keep sims consistent,
# although the effect on theta_m is very small
m = -56.6005  # -58.129 with snake 86 in
b = 175.72823  # 177.95 with snake 86 in

nu_thetas = np.arange(1, 1.51, .05)
theta_maxs = m * nu_thetas + b

n_nu = len(nu_thetas)
n_theta = len(theta_maxs)

# standard force coefficients
BASE = '../Output/s_serp3d_nu_theta_full/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

# boosted Cl/Cd output
# BASE = '../Output/s_serp3d_nu_theta_full_b/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

args = []
for nu_theta, theta_max in zip(nu_thetas, theta_maxs):
    for d_psi in d_psis:
        for psi_max in psi_maxs:
            params = (nu_theta, theta_max, f_theta, d_psi, psi_max, L)
            fname = BASE.format(*params)
            arg = (fname, params)
            args.append(arg)


# %% Run the parallel simulations

from multiprocessing import Pool

pool = Pool(processes=4)

now = time.time()

# uncomment to run the simulations
#pool.map(parallel_simulation, args)
#pool.map(parallel_simulation_boosted, args)

runtime = time.time() - now
print('Elapsed time: {0:.1f} min'.format(runtime / 60))

# Elapsed time: 108.7 min - cyc
# Elapsed time: 35.6 min - full
# Elapsed time: 38.4 min - full boosted

# %%


# %% Analyze the trials

def ret_fnames_cyc(nu, theta, d_psi=10, psi_max=20, f_theta=1.2, L=.7):

    from glob import glob

    BASE = '../Output/s_serp3d_nu_theta_cyc/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

    searchname = BASE.format(nu, theta, f_theta, d_psi, psi_max, L)

    return sorted(glob(searchname))


def ret_fnames_full(nu, theta, d_psi=10, psi_max=20, f_theta=1.2, L=.7):

    from glob import glob

    BASE = '../Output/s_serp3d_nu_theta_full/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

    searchname = BASE.format(nu, theta, f_theta, d_psi, psi_max, L)

    return sorted(glob(searchname))


def ret_fnames_full_boosted(nu, theta, d_psi=10, psi_max=20, f_theta=1.2, L=.7):

    from glob import glob

    BASE = '../Output/s_serp3d_nu_theta_full_b/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

    searchname = BASE.format(nu, theta, f_theta, d_psi, psi_max, L)

    return sorted(glob(searchname))


psi_maxs = np.r_[0, 10, 20]
d_psis = np.r_[-20, -10, 0, 10, 20]

npsi = len(psi_maxs)
n_dpsi = len(d_psis)

f_theta = 1.2
L = .7

# use slope and intercept from nu_thetas to keep sims consistent,
# although the effect on theta_m is very small
m = -56.6005  # -58.129 with snake 86 in
b = 175.72823  # 177.95 with snake 86 in

nu_thetas = np.arange(1, 1.51, .05)
theta_maxs = m * nu_thetas + b

n_nu = len(nu_thetas)
n_theta = len(theta_maxs)


# %% Body dict for mass and chord

nu_theta, theta_max = nu_thetas[0], theta_maxs[0]
d_psi, psi_max = d_psis[0], psi_maxs[0]


# setup the body
bd = setup_body(L=L, ds=.01, theta_max=theta_max,
                nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                psi_max=psi_max, frac_theta_max=0,
                d_theta=0, d_psi=d_psi)

# test that ret_fnames works (returns 165 names)
names_cyc, names_full = [], []
for nu_theta, theta_max in zip(nu_thetas, theta_maxs):
    for d_psi in d_psis:
        for psi_max in psi_maxs:
            names_cyc.append(ret_fnames_cyc(nu_theta, theta_max, d_psi, psi_max))
            names_full.append(ret_fnames_full(nu_theta, theta_max, d_psi, psi_max))


# %% FIGURE 5B,C,D: Combine position, pitch, and moments in one
#    3 x 5 plot, psi_max=20

psi_max = psi_maxs[1]  # 20 for paper

colors = sns.color_palette("husl", n_nu)

with sns.axes_style('white'):
    fig, axs = plt.subplots(3, 5, sharex=False, sharey=False,
                            figsize=(8, 8))

    for i, ax in enumerate(axs[0]):
        ax.set_aspect('equal')
        ax.set_ylim(-.5, 10.5)
        ax.set_xlim(-1, 6.5)
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([0, 5, 10])
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    for i, ax in enumerate(axs[1]):
        ax.set_ylim(-90, 90)
        ax.set_yticks([-90, 0, 90])
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.axhline(0, color='gray', lw=1)
        ax.yaxis.set_major_formatter(degree_formatter)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    for i, ax in enumerate(axs[2]):
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1, 6.5)
        ax.set_yticks([0, 3, 6])
        ax.axvline(0, color='gray', lw=1)
        ax.axhline(0, color='gray', lw=1)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    axs[0, 0].set_xlabel('Y (m)', fontsize='x-small')
    axs[0, 0].set_ylabel('Z (m)', fontsize='x-small')

    axs[1, 0].set_ylabel('Pitch (deg)', fontsize='x-small')
    axs[1, 0].set_xlabel('Time', fontsize='x-small')

    axs[2, 0].set_ylabel(r'Y$_\mathsf{landing}$', fontsize='x-small')
    axs[2, 0].set_xlabel('Pitch moment', fontsize='x-small')

    for j in np.arange(n_dpsi):
        axs[0, j].set_title(str(d_psis[j]) + u'\u00B0', fontsize='x-small')

    sns.despine()

    for j in np.arange(n_dpsi):
        d_psi = d_psis[j]

        M_avg = np.zeros((n_nu, 3))
        path = np.zeros(n_nu)

        for k in np.arange(n_nu):
            nu_theta, theta_max = nu_thetas[k], theta_maxs[k]

            fname_cyc = ret_fnames_cyc(nu_theta, theta_max, d_psi, psi_max)[0]
            d_cyc = np.load(fname_cyc)

            # STANDARD
            fname_full = ret_fnames_full(nu_theta, theta_max, d_psi, psi_max)[0]
            d_full = np.load(fname_full)

            # # BOOSTED (for SI)
            # fname_full_boosted = ret_fnames_full_boosted(nu_theta, theta_max,
            #                                              d_psi, psi_max)[0]
            # d_full = np.load(fname_full_boosted)

            M_B_tot = 1000 * d_cyc['M_B_tot']
            M_B_avg = M_B_tot.mean(axis=0)[-1]
            M_avg[k] = M_B_avg

            x, y, z = d_full['Ro'].T
            dx = np.gradient(x, edge_order=2)
            dy = np.gradient(y, edge_order=2)
            path[k] = np.cumsum(np.sqrt(dx**2 + dy**2))[-1]
            path[k] = y[-1]

            pitch = d_full['pitch']
            ts_non = d_full['ts'] * 1.2

            X, Y, Z = d_full['Ro'].T

            # Y vs. Z
            axs[0, j].plot(Y, Z, c=colors[k])

            # pitch vs. time
            axs[1, j].plot(ts_non, pitch, c=colors[k])

            # glide distance vs. pitch moment
            axs[2, j].plot(M_avg[k, 0], path[k], 'o', c=colors[k])

        # average pitch moment
        axs[2, j].plot(M_avg[:, 0].mean(), path.mean(), 'o', mfc='none',
                       mec='k', mew=2)


#    fig.savefig(FIG.format('3 by 5 psi_max={0}'.format(psi_max)), **FIGOPT)


# %% FIGURE SI: BOOSTED LIMS Combine position, pitch, and moments in one
#    3 x 5 plot, psi_max=20

psi_max = psi_maxs[2]  # 20

colors = sns.color_palette("husl", n_nu)
with sns.axes_style('white'):
    fig, axs = plt.subplots(2, 5, sharex=False, sharey=False,
                            figsize=(10, 5))

    for i, ax in enumerate(axs[0]):
        ax.set_aspect('equal')
        ax.set_ylim(-.5, 10.5)
        ax.set_xlim(-.5, 10.5)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])


    for i, ax in enumerate(axs[1]):
        ax.set_ylim(-90, 90)
        ax.set_yticks([-90, 0, 90])
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.axhline(0, color='gray', lw=1)
        ax.yaxis.set_major_formatter(degree_formatter)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])


    axs[0, 0].set_xlabel('Y (m)', fontsize='x-small')
    axs[0, 0].set_ylabel('Z (m)', fontsize='x-small')

    axs[1, 0].set_ylabel('Pitch (deg)', fontsize='x-small')
    axs[1, 0].set_xlabel('Time', fontsize='x-small')

    sns.despine()

    for j in np.arange(n_dpsi):
        d_psi = d_psis[j]

        M_avg = np.zeros((n_nu, 3))
        path = np.zeros(n_nu)

        for k in np.arange(n_nu):
            nu_theta, theta_max = nu_thetas[k], theta_maxs[k]

            fname_full = ret_fnames_full(nu_theta, theta_max, d_psi, psi_max)[0]
            d_full = np.load(fname_full)

            x, y, z = d_full['Ro'].T
            dx = np.gradient(x, edge_order=2)
            dy = np.gradient(y, edge_order=2)
            path[k] = np.cumsum(np.sqrt(dx**2 + dy**2))[-1]
            path[k] = y[-1]

            pitch = d_full['pitch']
            ts_non = d_full['ts'] * 1.2

            X, Y, Z = d_full['Ro'].T

            # Y vs. Z
            axs[0, j].plot(Y, Z, c=colors[k])

            # pitch vs. time
            axs[1, j].plot(ts_non, pitch, c=colors[k])


#    fig.savefig(FIG.format('3 by 5 psi_max={0} boosted lims'.format(psi_max)), **FIGOPT)


# %% FIGURE SI: BOOSTED Combine position, pitch, and moments in
#    one 3 x 5 plot, psi_max=20

psi_max = psi_maxs[2]  # 20

colors = sns.color_palette("husl", n_nu)

with sns.axes_style('white'):
    fig, axs = plt.subplots(2, 5, sharex=False, sharey=False,
                            figsize=(10, 5))


    for i, ax in enumerate(axs[0]):
        ax.set_aspect('equal')
        ax.set_ylim(-.5, 10.5)
        ax.set_xlim(-.5, 10.5)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])


    for i, ax in enumerate(axs[1]):
        ax.set_ylim(-90, 90)
        ax.set_yticks([-90, 0, 90])
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.axhline(0, color='gray', lw=1)
        ax.yaxis.set_major_formatter(degree_formatter)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    axs[0, 0].set_xlabel('Y (m)', fontsize='x-small')
    axs[0, 0].set_ylabel('Z (m)', fontsize='x-small')

    axs[1, 0].set_ylabel('Pitch (deg)', fontsize='x-small')
    axs[1, 0].set_xlabel('Time', fontsize='x-small')

    for j in np.arange(n_dpsi):
        axs[0, j].set_title(str(d_psis[j]) + u'\u00B0', fontsize='x-small')

    sns.despine()

    for j in np.arange(n_dpsi):
        d_psi = d_psis[j]

        M_avg = np.zeros((n_nu, 3))
        path = np.zeros(n_nu)

        for k in np.arange(n_nu):
            nu_theta, theta_max = nu_thetas[k], theta_maxs[k]

            fname_full = ret_fnames_full(nu_theta, theta_max, d_psi, psi_max)[0]
            d_full = np.load(fname_full)

            # BOOSTED
            # fname_full_boosted = ret_fnames_full_boosted(nu_theta, theta_max, d_psi, psi_max)[0]
            # d_full = np.load(fname_full_boosted)

            x, y, z = d_full['Ro'].T
            dx = np.gradient(x, edge_order=2)
            dy = np.gradient(y, edge_order=2)
            path[k] = np.cumsum(np.sqrt(dx**2 + dy**2))[-1]
            path[k] = y[-1]

            pitch = d_full['pitch']
            ts_non = d_full['ts'] * 1.2

            X, Y, Z = d_full['Ro'].T

            # Y vs. Z
            axs[0, j].plot(Y, Z, c=colors[k])

            # pitch vs. time
            axs[1, j].plot(ts_non, pitch, c=colors[k])


#    fig.savefig(FIG.format('3 by 5 psi_max={0} boosted'.format(psi_max)), **FIGOPT)


# %% FIGURE 5E: CYC nu vs. Pitch moment for psi_max = 20 - new colors

psi_max = psi_maxs[2]

d_psi_colors = sns.color_palette("Greys", n_dpsi * 2)[-n_dpsi:]
d_psi_colors = sns.color_palette("Greys", n_dpsi)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                        figsize=(4.6, 3.25))

ax.axhline(0, color='gray', lw=1)
ax.set_xticks(nu_thetas[::2])

ax.set_ylim(-1.1, 1.1)

for k in np.arange(n_nu):
    ax.plot(nu_thetas[k], -1, 'o', ms=8, c=colors[k])

for j in np.arange(n_dpsi):
    d_psi = d_psis[j]
    M_avg = np.zeros((n_nu, 3))

    for k in np.arange(n_nu):
        nu_theta, theta_max = nu_thetas[k], theta_maxs[k]

        fname = ret_fnames_cyc(nu_theta, theta_max, d_psi, psi_max)[0]
        d = np.load(fname)

        M_B_tot = 1000 * d['M_B_tot']
        M_B_avg = M_B_tot.mean(axis=0)[-1]
        M_avg[k] = M_B_avg

    if d_psi == 10:
        lw = 3
    else:
        lw = 2

    label = '{}'.format(d_psi) + u'\u00B0'
    line, = ax.plot(nu_thetas, M_avg[:, 0], '-', lw=lw, label=label, zorder=0)

leg = ax.legend(loc='upper left', ncol=5, # title=r'$d_\psi$',
                    fontsize='xx-small', handlelength=1, handletextpad=.25,
                    columnspacing=1)
leg.get_title().set_fontsize('x-small')

ax.set_xlabel('Number of spatial periods', fontsize='x-small')
ax.set_ylabel('Pitch moment (Nmm)', fontsize='x-small')

sns.despine()

#fig.savefig(FIG.format('d_psi nu_theata M_pitch no dots'), **FIGOPT)


# %% FIGURE 5F: CYC nu vs. Pitch moment for psi_max = 20 different d_psi

psi_max = psi_maxs[2]

d_psi_colors = sns.color_palette("Greys", n_dpsi * 2)[-n_dpsi:]
d_psi_colors = sns.color_palette("Greys", n_dpsi)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                        figsize=(4.6, 3.25))

ax.axhline(0, color='gray', lw=1)

ax.set_ylim(-1.1, 1.1)

ax.xaxis.set_major_formatter(degree_formatter)

for k in np.arange(n_nu):
    nu_theta, theta_max = nu_thetas[k], theta_maxs[k]


    M_avg = np.zeros((n_dpsi, 3))

    for j in np.arange(n_dpsi):
        d_psi = d_psis[j]


        fname = ret_fnames_cyc(nu_theta, theta_max, d_psi, psi_max)[0]
        d = np.load(fname)

        M_B_tot = 1000 * d['M_B_tot']
        M_B_avg = M_B_tot.mean(axis=0)[-1]
        M_avg[j] = M_B_avg

    ax.plot(d_psis, M_avg[:, 0], '-', c=colors[k])

ax.set_xlabel('Dorsoventral flexion', fontsize='x-small')
ax.set_ylabel('Pitch moment (Nmm)', fontsize='x-small')

sns.despine()

#fig.savefig(FIG.format('d_psi Mpitch psi_max={0}'.format(psi_max)), **FIGOPT)


# %% Preprocess data for cycle averaged moments figure

nu, theta = 1.40, 96  # for paper

fname_cyc = ret_fnames_cyc(nu, theta, d_psi=10, psi_max=20)[0]
fname_full = ret_fnames_full(nu, theta, d_psi=10, psi_max=20)[0]

d_cyc = np.load(fname_cyc)
d_full = np.load(fname_full)

ts_phs_non = d_cyc['ts_phs_non']

ts_cyc_non = d_cyc['ts_cyc_non']
Z_cyc = d_cyc['soln_cyc'][:, 1]
Z_phs = np.interp(ts_phs_non, ts_cyc_non, Z_cyc)

M_B_tot = d_cyc['M_B_tot']
dho_B_tot = d_cyc['dho_B_tot']

M_B_avg = M_B_tot.mean(axis=0)
dho_B_avg = dho_B_tot.mean(axis=0)

F_B_tot = d_cyc['F_B_tot']

frame_c = [bmap[2], bmap[1], bmap[0]]


Ma_B_tot = d_full['Ma_B'].sum(axis=1)
Ma_tot = d_full['Ma'].sum(axis=1)
yaw, pitch, roll = d_full['yaw'], d_full['pitch'], d_full['roll']

ts_full = d_full['ts']
ts_full_non = ts_full * 1.2

yaw_params = np.polyfit(ts_full_non, yaw, 1)
pitch_params = np.polyfit(ts_full_non, pitch, 1)
roll_params = np.polyfit(ts_full_non, roll, 1)

yaw_fit = np.polyval(yaw_params, ts_full_non)
pitch_fit = np.polyval(pitch_params, ts_full_non)
roll_fit = np.polyval(roll_params, ts_full_non)


# %% FIGURE 6A,B: Aerodynamic moments through the trajectory --- vs. ts_phs_non
# Long plot

with sns.axes_style('white'):
    figsize = (5.75, 3.75)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=figsize)

    ax1, ax2, ax3 = axs[0]
    ax4, ax5, ax6 = axs[1]

    for ax in axs.flatten():
        ax.axhline(0, color='gray', lw=1)

    ax1.fill_between(ts_phs_non,
                     1000 * M_B_tot[:, :, 0].min(axis=0),
                     1000 * M_B_tot[:, :, 0].max(axis=0),
                     alpha=.25, color='gray')
    ax2.fill_between(ts_phs_non,
                     1000 * M_B_tot[:, :, 1].min(axis=0),
                     1000 * M_B_tot[:, :, 1].max(axis=0),
                     alpha=.25, color='gray')
    ax3.fill_between(ts_phs_non,
                     1000 * M_B_tot[:, :, 2].min(axis=0),
                     1000 * M_B_tot[:, :, 2].max(axis=0),
                     alpha=.25, color='gray')
    ax4.fill_between(ts_phs_non,
                     1000 * dho_B_tot[:, :, 0].min(axis=0),
                     1000 * dho_B_tot[:, :, 0].max(axis=0),
                     alpha=.25, color='gray')
    ax5.fill_between(ts_phs_non,
                     1000 * dho_B_tot[:, :, 1].min(axis=0),
                     1000 * dho_B_tot[:, :, 1].max(axis=0),
                     alpha=.25, color='gray')
    ax6.fill_between(ts_phs_non,
                     1000 * dho_B_tot[:, :, 2].min(axis=0),
                     1000 * dho_B_tot[:, :, 2].max(axis=0),
                     alpha=.25, color='gray')

    idx = 0
    ax1.plot(ts_phs_non, 1000 * M_B_tot[idx, :, 0], c=frame_c[0])
    ax2.plot(ts_phs_non, 1000 * M_B_tot[idx, :, 1], c=frame_c[1])
    ax3.plot(ts_phs_non, 1000 * M_B_tot[idx, :, 2], c=frame_c[2])

    ax4.plot(ts_phs_non, 1000 * dho_B_tot[0, :, 0], c=frame_c[0])
    ax5.plot(ts_phs_non, 1000 * dho_B_tot[0, :, 1], c=frame_c[1])
    ax6.plot(ts_phs_non, 1000 * dho_B_tot[0, :, 2], c=frame_c[2])

    ax1.plot(ts_phs_non, 1000 * M_B_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')

    ax1.set_xlim(0, ts_phs_non[-1])
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([-2, 0, 2])

    ax4.set_xlabel('Normalized time', fontsize='x-small')
    ax1.set_title('Pitch moment', fontsize='x-small')
    ax2.set_title('Roll moment', fontsize='x-small')
    ax3.set_title('Yaw moment', fontsize='x-small')
    ax1.set_ylabel('Aerodynamic', fontsize='x-small')
    ax4.set_ylabel('Inertial', fontsize='x-small')

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize='x-small')
        ax.tick_params(axis='both', which='minor', labelsize='x-small')

    sns.despine()
    fig.set_tight_layout(True)

#    fig.savefig(FIG.format('long cyc avg terms nu={} theta={}'.format(nu, theta)), **FIGOPT)


# %% FIGURE 6C: Fr for the 'full' trials - long plot

with sns.axes_style('white'):
    frame_c = [bmap[2], bmap[1], bmap[0]]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(5.75, 3.75 / 1.25))

    for ax in axs:
        ax.axhline(1, color='gray', lw=1)

    for j in np.arange(n_dpsi):
        d_psi = d_psis[j]
        psi_max = 20

        for k in np.arange(n_nu):
            nu_theta, theta_max = nu_thetas[k], theta_maxs[k]
            col = colors[k]

            fname = ret_fnames_cyc(nu_theta, theta_max, d_psi, psi_max)[0]
            d = np.load(fname)

            ts_phs_non = d['ts_phs_non']
            ts_cyc_non = d['ts_cyc_non']
            Z_cyc = d['soln_cyc'][:, 1]
            Z_phs = np.interp(ts_phs_non, ts_cyc_non, Z_cyc)

            M_ptp = 1000 * d['M_B_tot'].ptp(axis=0)
            dho_ptp = 1000 * d['dho_B_tot'].ptp(axis=0)
            Fr = dho_ptp / M_ptp

            axs[0].semilogy(ts_phs_non, Fr[:, 0], c=col, lw=1, label='Pitch')
            axs[1].semilogy(ts_phs_non, Fr[:, 1], c=col, lw=1, label='Roll')
            axs[2].semilogy(ts_phs_non, Fr[:, 2], c=col, lw=1, label='Yaw')

            # highlight a particular trajectory
            if d_psi == 10 and nu_theta == nu_thetas[8]:
                col = 'k'
                axs[0].semilogy(ts_phs_non, Fr[:, 0], c=col, lw=1.5, zorder=100)
                axs[1].semilogy(ts_phs_non, Fr[:, 1], c=col, lw=1.5, zorder=100)
                axs[2].semilogy(ts_phs_non, Fr[:, 2], c=col, lw=1.5, zorder=100)

    axs[0].set_title('Pitch', fontsize='x-small')
    axs[1].set_title('Roll', fontsize='x-small')
    axs[2].set_title('Yaw', fontsize='x-small')
    axs[0].set_xlabel('Normalized time', fontsize='x-small')
    axs[2].set_ylim(10**-1, 10**3)
    axs[2].set_xticks([0, 1, 2])

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize='x-small')
        ax.tick_params(axis='both', which='minor', labelsize='x-small')

    sns.despine()

    # fig.savefig(FIG.format('long Fr psi_max=20'), **FIGOPT)