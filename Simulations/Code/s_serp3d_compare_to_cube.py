#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on the script s_serp3d_nu_theta.py

% %reset -f
% %pylab
% %clear
%load_ext autoreload
%autoreload 2

cd /Volumes/Yeaton_HD6/Code for Manuscripts/squiggle-snake/Code

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
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

from mayavi import mlab

import time

import m_sim as sim
import m_aerodynamics as aerodynamics
import m_morph as morph

FIG = '../Figures/s_serp3d_compare_to_cube/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}
FIGPNG = '../Figures/s_serp3d_compare_to_cube/{}.png'

CUBE_DIR = '../../snake-cube/'

SAVEFIG = False


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
#    if np.abs(x) > 0 and np.abs(x) < 1:
#        return val_str.replace("0", "", 1)
#    else:
#        return val_str
    return val_str


decimal_formatter = FuncFormatter(_formatter_remove_zeros)
degree_formatter = FuncFormatter(_formatter_degree)


def setup_body(L=.7, ds=.01, theta_max=90, nu_theta=1.1, f_theta=1.4,
               phi_theta=0, psi_max=10, frac_theta_max=0, d_theta=0, d_psi=0,
               display_ho=1, ho_shift=True, bl=1, bd=1, known_mass=None):
    """Setup the body_dict for simulations.
    """

    s = ds / 2 + np.arange(0, L, ds)  # m
    nbody = len(s)
    dt = .010  # sec
    neck_length = .075 * L  # 5% SVL % .05 m on a .7 m snake is 7.14%
    n_neck = np.floor(neck_length / ds).astype(np.int)

    cmax, mtot, Ws_fit, rho_bar = morph.morph_from_svl(L)
    
    # adjust the mass, since we know the value and the cube snakes are
    # generally heavier
    if known_mass:
        mtot = known_mass
    
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
    ntime_rot = 200  # 200
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
        t1 = time.time()
        if print_time and i % 10 == 0:
            print('Phase {} of {} in find_rotational_ic'.format(i, ntime_rot))
        solver.integrate(ts_rot[i])
        out = solver.y

        soln_rot.append(out)
        i = i + 1
        t2 = time.time()
        if print_time and i % 10 == 0:
            print('\t{0:.3f} sec'.format(t2 - t1))

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
    # s, m, n_neck, theta_dict, psi_dict = args[0]
    # ds, c, g, rho, aero_interp = args[1]
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

    #    out = chrysoserp(s, t,  mi, n_neck, theta_dict, psi_dict)
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


def cube_parallel_simulation(trial):
    """Run the simulations in parallel.
    """
    
    save_fname = '../Output/s_serp3d_compare_to_cube/{}_{}.npz'

    # load the saved trial information
    snake_id = int(trial['Snake ID'])
    trial_id = int(trial['Trial ID'])    
    data_fname = ret_fnames(snake=snake_id, trial=trial_id)[0]
    d = np.load(data_fname)
    
    # setup simulation parameters
    body_dict = setup_body(L=trial['SVL'] / 100,
                           ds=.01,
                           theta_max=trial['theta_max'],
                           nu_theta=trial['nu_theta'],
                           f_theta=trial['f_theta'],
                           phi_theta=0,
                           psi_max=trial['psi_max'],
                           frac_theta_max=0,
                           d_theta=0,
                           d_psi=trial['d_psi_avg'],
                           display_ho=0,
                           ho_shift=True,
                           bl=1,
                           bd=1,
                           known_mass=trial['mass'] / 1000)
    
    # setup the simulation
    dt = float(body_dict['dt'])
    
    # find the initial Euler angle offsets so minimize inertial
    # effects at the start
    ang0 = find_rotational_ic(body_dict, print_time=False)
    
    # initial conditions
    tend = None
    Ro0 = d['Ro_I'][0] / 1000
    dRo0 = d['dRo_I'][0] / 1000
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
    np.savez(save_fname.format(trial_id, snake_id),
             Ro_I=d['Ro_I'] / 1000,  # trial position
             dRo_I=d['dRo_I'] / 1000,  # trial velocity
             ddRo_I=d['ddRo_I'] / 1000,  # trial acceleration
             dt=dt, ts=ts, Ro=Ro, dRo=dRo, omg=omg, ang=ang,
             yaw=yaw, pitch=pitch, roll=roll,
             **out_recon)


#################
#
# Compare simulations to recorded glides
#
#

# %% Functions take from snake-cube/Code/s_all_proc_plots.py

def ret_fnames(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = CUBE_DIR + 'Data/Processed Qualisys output/'
    fn_search = fn_proc + fn_trial

    return sorted(glob(fn_search))

def trial_info(fname):
    trial_id = fname.split('/')[-1][:3]
    snake_id = fname.split('/')[-1][4:6]

    return int(snake_id), int(trial_id)

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

# %% Information about the trials

fn_names = ret_fnames()
snakes = []
for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    snakes.append(int(snake_id))

snakes = np.array(sorted(snakes))

snake_ids = np.unique(snakes)
#snake_ids =  np.array([30, 31, 32, 33, 35, 81, 88, 90, 91, 95])

ntrials = len(snakes)
nsnakes = len(snake_ids)

colors = sns.husl_palette(n_colors=nsnakes)

total_counts = 0
snake_counts = {}
snake_colors = {}
for i, snake_id in enumerate(snake_ids):
    snake_counts[snake_id] = np.sum(snakes == snake_id)
    snake_colors[snake_id] = colors[i]
    total_counts += snake_counts[snake_id]


# average start in X and Y of com as the new
X0, Y0, Z0 = np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials)
for i, fname in enumerate(ret_fnames()):
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    X, Y, Z = d['Ro_I'].T / 1000  # m
    X0[i] = X[0]
    Y0[i] = Y[0]
    Z0[i] = Z[0]
Xo = X0.mean()  # 0.4507950812699839
Yo = Y0.mean()  # -4.706641162498193
Zo = Z0.mean()  # 8.020337351313595

# for fource calculations
grav = 9.81  # m/s^2 (gravitational acceleration)
rho = 1.17  # kg/m^3 (air density)


# %% Import serpenoid curve parameters from Cube analysis

df_cod = pd.read_csv('../Data/From Cube Analysis/COD_36_paper_trial.csv',
                     index_col=0)

# %% Verify - reproduce figure in manuscript
# Figure 2f,g,h from the manuscript
_figsize = (12.24, 4.8)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=_figsize)
ax1.plot(df_cod['nu_theta'], df_cod['theta_max'], 'o')
ax2.plot(df_cod['nu_theta'], df_cod['psi_max'], 'o')
ax3.plot(df_cod['nu_theta'], df_cod['d_psi_avg'], 'o')
ax1.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
sns.despine()
fig.set_tight_layout(True)


# %% Select one trial

i = 20
trial = df_cod.iloc[i]

body_dict = setup_body(L=trial['SVL'] / 100,
                       ds=.01,
                       theta_max=trial['theta_max'],
                       nu_theta=trial['nu_theta'],
                       f_theta=trial['f_theta'],
                       phi_theta=0,
                       psi_max=trial['psi_max'],
                       frac_theta_max=0,
                       d_theta=0,
                       d_psi=trial['d_psi_avg'],
                       display_ho=0,
                       ho_shift=True,
                       bl=1,
                       bd=1,
                       known_mass=trial['mass'] / 1000)

# directory of snake runs
fname = ret_fnames(snake=int(trial['Snake ID']),
                   trial=int(trial['Trial ID']))[0]
d = np.load(fname)

# %% Simulate the glide

now = time.time()

vscale = body_dict['vscale']
dt = float(body_dict['dt'])

# find the initial Euler angle offsets so minimize inertial effects at the start
ang0 = find_rotational_ic(body_dict, print_time=True)

# initial conditions
tend = None
Ro0 = d['Ro_I'][0] / 1000
dRo0 = d['dRo_I'][0] / 1000
dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
omg0 = np.dot(C0.T, omg0_body)
soln0 = np.r_[Ro0, dRo0, omg0, ang0]

# run the dynamics simulation
out = sim.integrate(soln0, body_dict, dt, tend=tend, print_time=True)

print('{0:.3f} sec'.format(time.time() - now))

# extract values
ts, Ro, dRo, omg, ang = out
yaw, pitch, roll = np.rad2deg(ang.T)

# reconstruct all values from the simulation
out_recon = simulation_recon(body_dict, ts, Ro, dRo, omg, ang)

# %% Compare the trial to the simulation

fig, ax = plt.subplots()
ax.plot(Ro[:, 1] - Yo, Ro[:, 2])
ax.plot(d['Ro_I'][:, 1] / 1000 - Yo, d['Ro_I'][:, 2] / 1000)
ax.set_aspect('equal')
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(Ro[:, 1] - Yo, Ro[:, 0] - Xo)
ax.plot(d['Ro_I'][:, 1] / 1000 - Yo, d['Ro_I'][:, 0] / 1000 - Xo)
ax.set_aspect('equal')
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('Y (m)')
ax.set_ylabel('X (m)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(dRo[:, 1], dRo[:, 2])
ax.plot(d['dRo_I'][:, 1] / 1000, d['dRo_I'][:, 2] / 1000)
ax.set_aspect('equal')
ax.set_xlabel('Vy (m/s)')
ax.set_ylabel('Vz (m/s)')
sns.despine()
fig.set_tight_layout(True)

# %% Run Cube simulations in parallel

from multiprocessing import Pool

pool = Pool(processes=3)

now = time.time()

pool.map(cube_parallel_simulation, trials)

runtime = time.time() - now
print('Elapsed time: {0:.1f} min'.format(runtime / 60))

# Elapsed time: 11.3 min for Cube trials

# %% Prcoess how the Cube data and simulations compare

def ret_fnames_sims(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = '../Output/s_serp3d_compare_to_cube/'
    fn_search = fn_proc + fn_trial

    return sorted(glob(fn_search))


fn_names = ret_fnames_sims()
snakes = []
for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    snakes.append(int(snake_id))

snakes = np.array(sorted(snakes))

snake_ids = np.unique(snakes)  # np.array([81, 88, 90, 91, 95])

ntrials = len(snakes)
nsnakes = len(snake_ids)

colors = sns.husl_palette(n_colors=nsnakes)

# %% FIGURE: CoM positions

hratios = 10 / 4


fig, axs = plt.subplots(2, 2, sharex=True,
                        gridspec_kw = {'height_ratios':1*[hratios, 1]},
                        figsize=[5.3, 7.75])
fig.set_tight_layout(True)
sns.despine()

((ax1_yz, ax2_yz), (ax1_yx, ax2_yx)) = axs
ax2_yz.set_yticklabels([])
ax2_yx.set_yticklabels([])
ax1_yz.set_title('Measured')
ax2_yz.set_title('Simulated')
n_unstable = 0
for i, snake_id in enumerate(snake_ids):
    trials = ret_fnames_sims(snake=snake_id)
    color = colors[i]
    for trial in trials:
        d = np.load(trial)
        Xs, Ys, Zs = d['Ro'].T  # simulation
        Xc, Yc, Zc = d['Ro_I'].T  # Cube
        # gamma = np.rad2deg(np.arctan2(-d['dRo'][:, 2], d['dRo'][:, 1]))
        if Ys[-1] < Ys.max():
            color_sim = 'r'
            zorder = 1
            n_unstable += 1
        else:
            color_sim = 'b'
            zorder = 1000
        idx = np.where(Zs >= Zc[-1])[0]  # crop on ending height
        ax1_yz.plot(Yc - Yo, Zc, c='b')
        ax2_yz.plot(Ys[idx] - Yo, Zs[idx], c=color_sim, zorder=zorder)
        ax1_yx.plot(Yc - Yo, Xc - Xo, c='b')
        ax2_yx.plot(Ys[idx] - Yo, Xs[idx] - Xo, c=color_sim, zorder=zorder)

for ax in axs.flatten():
    ax.set_aspect(1, share=True)

for ax in axs[0, :]:
    ax.set_xlim(-.5, 4.75)
    ax.set_ylim(-.3, 8.75)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xticks([0, 1, 2, 3, 4])

for ax in axs[1, :]:
    ax.set_xlim(-.5, 4.75)
    ax.set_ylim(-3, 2)
    ax.set_yticks([-3, -2, -1, 0, 1, 2])
    ax.set_xticks([0, 1, 2, 3, 4])

ax1_yz.set_ylabel('Z (m)')
ax1_yx.set_xlabel('Y (m)')
ax1_yx.set_ylabel('X (m)')

if SAVEFIG:
    fig.savefig(FIG.format('sim_vs_cube_2x2_colors', **FIGOPT))

# %% FIGURE: Euler angles for simulation

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8.3, 4))
(ax1, ax2, ax3) = axs
for i, snake_id in enumerate(snake_ids):
    trials = ret_fnames_sims(snake=snake_id)
    color = colors[i]
    for trial in trials:
        d = np.load(trial)
        ax1.plot(d['Ro'][:, 1] - Yo, d['pitch'], 'r')
        ax2.plot(d['Ro'][:, 1] - Yo, d['roll'], 'g')
        ax3.plot(d['Ro'][:, 1] - Yo, d['yaw'], 'b')

for ax in axs:
    ax.yaxis.set_major_formatter(degree_formatter)
    ax.set_xlim(-.5, 4.75)
    ax.set_ylim(-90, 90)
    ax.set_yticks(np.r_[-90:91:30]) 
    ax.set_xticks([0, 1, 2, 3, 4])

ax1.set_title('Pitch')
ax2.set_title('Roll')
ax3.set_title('Yaw')
ax1.set_xlabel('Y (m)')
ax1.set_ylabel('Euler angles (deg)')
sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('sim_vs_cube_Euler_angles', **FIGOPT))
