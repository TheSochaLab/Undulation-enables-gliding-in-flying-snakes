#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:15:08 2017

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
#rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
#      'font.sans-serif': 'Helvetica',
#      'axes.titlesize': 16, 'axes.labelsize': 16,
#      'xtick.labelsize': 15, 'ytick.labelsize': 15,
#      'legend.fontsize': 15}
#sns.set('paper', 'ticks', font='Helvetica',
#        font_scale=1.6, color_codes=True, rc=rc)

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

#from mayavi import mlab

import time

import m_sim as sim
import m_aerodynamics as aerodynamics
import m_morph as morph

FIG = '../Figures/s_serp3d_nu_theta_cyc/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %%

def setup_body(L=.7, ds=.01, theta_max=90, nu_theta=1.1, f_theta=1.4,
               phi_theta=np.pi / 4, psi_max=10, frac_theta_max=0, d_theta=0, d_psi=0,
               nu_ratio=2, f_ratio=2, A_phi=2, B_phi=-np.pi / 2):
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
    v0_non = 1.7 / np.sqrt(2 * Ws / rho)  # .2409

    # aerodynamics
    aero_interp = aerodynamics.extend_wind_tunnel_data()

    # wave parameters
    nu_psi = nu_ratio * nu_theta
    f_psi = f_ratio * f_theta
    phi_psi = A_phi * (phi_theta + B_phi)

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

    # dictionary with all of the simulation parameters in it
    body_dict = dict(L=L, ds=ds, s=s, nbody=nbody, neck_length=neck_length,
                     n_neck=n_neck, cmax=cmax, mtot=mtot, rho_bar=rho_bar,
                     c=c, m=m, weight=weight, darea=darea, area=area, Ws=Ws,
                     theta_dict=theta_dict, psi_dict=psi_dict,
                     tscale=tscale, pscale=pscale, vscale=vscale,
                     ascale=ascale, fscale=fscale, mscale=mscale,
                     dt=dt, g=g, rho=rho, aero_interp=aero_interp,
                     head_control=False, v0_non=v0_non,
                     nu_ratio=nu_ratio, f_ratio=f_ratio,
                     A_phi=A_phi, B_phi=B_phi)

    return body_dict


def cycle_avg_dynamics_eom(tintegrator, state, body_dict):
    """Called by fixed_point.
    """

    # non-dimensional velocities
    y, z, vy, vz = state

    # unpack needed variables
    s, m, n_neck, g = body_dict['s'], body_dict['m'], body_dict['n_neck'], body_dict['g']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    rho = body_dict['rho']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']

    # phase coupling parameters
    A_phi, B_phi = body_dict['A_phi'], body_dict['B_phi']

    # phases of the wave to evaluate at
    nphase = 10  # every 2 pi / 10 = pi / 5 radians
    phi_theta_phases = np.linspace(0, 2 * np.pi, nphase + 1)[:-1]
    phi_psi_phases = A_phi * (phi_theta_phases + B_phi)

    # trivial dynamics variables
    C = np.eye(3)
    omg = np.r_[0, 0, 0]

    # CoM velocity
    dRo = np.r_[0, vy, vz]

    F = np.zeros((nphase, 3))
    for i in np.arange(nphase):
        t = 0

        # phase in the undulation cycle
        theta_dict['phi_theta'] = phi_theta_phases[i]
        psi_dict['phi_psi'] = phi_psi_phases[i]

        out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck,
                                             theta_dict, psi_dict)

        p, dp = out['p'], out['dp']
        tv, cv, bv = out['tv'], out['cv'], out['bv']

        # positions, velocities, and accelerations
        r, dr = sim.rotate(C.T, p), sim.rotate(C.T, dp)
        dR = dRo + dr + np.cross(omg, r)

        Fi = np.zeros(r.shape)
        Fi[:, 2] = -m * g

        # aerodynamic forces
        aout = sim.aero_forces(tv, cv, bv, C, dR, ds, c, rho,
                               aero_interp, full_out=True)
        Fi += aout['Fa']  # Fl + Fd

        F[i] = Fi.sum(axis=0)

    # average force
    F_avg = F.mean(axis=0)
    accel = F_avg / m.sum()
    ay, az = accel[1], accel[2]  # don't itegrate the x component

    return np.r_[vy, vz, ay, az]


def cycle_avg_dynamics(body_dict, dt=.025, print_time=False):

    from scipy.integrate import ode

    now = time.time()

    # initial position and velocity
    soln0 = np.r_[0, 10, 1.7, 0]

    # setup the integrator
    solver = ode(cycle_avg_dynamics_eom)
    solver.set_integrator('dopri5')
    solver.set_initial_value(soln0, 0)  # x0, t0
    solver.set_f_params(body_dict)

    # integrate the cycle averaged EoM
    soln, ts = [soln0], [0]
    at_equil = False
    while not at_equil:
        solver.integrate(solver.t + dt)
        out = solver.y

        y, z, vy, vz = out[0], out[1], out[2], out[3]

        at_equil = z < 0

        soln.append(out)
        ts.append(solver.t)

    if print_time:
        print('Cycle avg dynamics time: {0:.3f} sec'.format(time.time() - now))

    soln_cyc = np.array(soln)
    ts_cyc = np.array(ts)

    return soln_cyc, ts_cyc


def cyc_avg_moments(body_dict, ts_cyc, vy_cyc, vz_cyc, print_time=False):

    ntime_orig = len(ts_cyc)

    # unpack needed variables
    s, m, n_neck, g = body_dict['s'], body_dict['m'], body_dict['n_neck'], body_dict['g']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    vscale, rho = body_dict['vscale'], body_dict['rho']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']
    nbody = body_dict['nbody']


    # phases of the wave to evaluate at
    nphase = 60
    phi_theta_phases = np.linspace(0, 2 * np.pi, nphase + 1)[:-1]
    phi_psi_phases = 2 * (phi_theta_phases - np.pi / 2)

    # make the velocity dimensional
    dRo = np.c_[np.zeros(ntime_orig), vy_cyc, vz_cyc] * vscale
    C = np.eye(3)
    omg = np.r_[0, 0, 0]

    ntime = 5 * ntime_orig
    ts_phs = np.linspace(ts_cyc[0], ts_cyc[-1], ntime)

    # interpolate the velocity
    dRo_x_many = np.interp(ts_phs, ts_cyc, dRo[:, 0])
    dRo_y_many = np.interp(ts_phs, ts_cyc, dRo[:, 1])
    dRo_z_many = np.interp(ts_phs, ts_cyc, dRo[:, 2])
    dRo_many = np.c_[dRo_x_many, dRo_y_many, dRo_z_many]

    dRo = dRo_many

    npntnb3 = np.zeros((nphase, ntime, nbody, 3))

    M_B = npntnb3.copy()
    F_B = npntnb3.copy()
    ho_B = npntnb3.copy()
    dho_B = npntnb3.copy()

    p = npntnb3.copy()
    dp = npntnb3.copy()
    ddp = npntnb3.copy()

    tv = npntnb3.copy()
    cv = npntnb3.copy()
    bv = npntnb3.copy()

    if print_time:
        now = time.time()

    # i is index through phases
    for i in np.arange(nphase):

        # phase in the undulation cycle
        theta_dict['phi_theta'] = phi_theta_phases[i]
        psi_dict['phi_psi'] = phi_psi_phases[i]

        # j is index through cycle
        for j in np.arange(ntime):

            # time through cycle
            t = ts_phs[j]

            out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict,
                                                 psi_dict)

            pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
            tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

            p[i, j] = pi
            dp[i, j] = dpi
            ddp[i, j] = ddpi

            tv[i, j] = tvi
            cv[i, j] = cvi
            bv[i, j] = bvi

            # positions, velocities, and accelerations
            ri, dri = sim.rotate(C.T, pi), sim.rotate(C.T, dpi)
            dRi = dRo[j] + dri + np.cross(omg, ri)

            # ho
            ho_B[i, j] = np.cross((m * pi.T).T, dpi)
            dho_B[i, j] = np.cross((m * pi.T).T, ddpi)

            # aerodynamic forces
            aout = sim.aero_forces(tvi, cvi, bvi, C, dRi, ds, c, rho,
                                   aero_interp, full_out=True)

            # aerodynamic force
            Fi_iner = aout['Fa']
            F_B[i, j] = sim.rotate(C, Fi_iner)

            # aerodynamic moments
            Mi_iner = sim.cross(ri, Fi_iner)
            M_B[i, j] = sim.rotate(C, Mi_iner)

    if print_time:
        print('Cycle avg moment time: {0:.3f} sec'.format(time.time() - now))

    return ts_phs, dRo, F_B, M_B, ho_B, dho_B, p, dp, ddp, tv, cv, bv, phi_theta_phases, phi_psi_phases


def parallel_simulation(args):
    """Run the simulations in parallel.
    """

    # unpack the arguments
    fname, params = args
    nu_theta, theta_max, f_theta, d_psi, psi_max, L = params

    # setup the body
    body_dict = setup_body(L=L, ds=.01, theta_max=theta_max,
                           nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                           psi_max=psi_max, frac_theta_max=0,
                           d_theta=0, d_psi=d_psi)
    vscale = body_dict['vscale']
    dt = .025
    soln_cyc, ts_cyc = cycle_avg_dynamics(body_dict, dt=dt, print_time=True)

    # non-dim velocities
    vy_cyc, vz_cyc = soln_cyc[:, 2:].T / vscale

    # non-dimensional time by undulation frequency
    ts_cyc_non = ts_cyc * body_dict['theta_dict']['f_theta']

    ts_phs, dRo_phs, F_B, M_B, ho_B, dho_B, p, dp, ddp, tv, cv, bv, phi_theta_phases, phi_psi_phases = cyc_avg_moments(body_dict, ts_cyc, vy_cyc, vz_cyc, print_time=True)

    F_B_tot = F_B.sum(axis=2)
    M_B_tot = M_B.sum(axis=2)
    ho_B_tot = ho_B.sum(axis=2)
    dho_B_tot = dho_B.sum(axis=2)

    ts_phs_non = ts_phs * body_dict['theta_dict']['f_theta']

    np.savez(fname,
             vscale=vscale,
             soln_cyc=soln_cyc, ts_cyc=ts_cyc,
             vy_cyc=vy_cyc, vz_cyc=vz_cyc,
             ts_cyc_non=ts_cyc_non,
             ts_phs=ts_phs, dRo_phs=dRo_phs,
             F_B=F_B, M_B=M_B, ho_B=ho_B, dho_B=dho_B,
             p=p, dp=dp, ddp=ddp,
             tv=tv, cv=cv, bv=bv,
             phi_theta_phases=phi_theta_phases,
             phi_psi_phases=phi_psi_phases,
             F_B_tot=F_B_tot, M_B_tot=M_B_tot,
             ho_B_tot=ho_B_tot, dho_B_tot=dho_B_tot,
             ts_phs_non=ts_phs_non, dt=dt)


# %% Setup simulation runs

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


BASE = '../Output/s_serp3d_nu_theta_cyc/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

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

# uncomment to run simulation
# pool.map(parallel_simulation, args)

runtime = time.time() - now
print('Elapsed time: {0:.1f} min'.format(runtime / 60))

# Elapsed time: 108.7 min


# %% Analyze the trials - See s_serp3d_nu_theta_full.py