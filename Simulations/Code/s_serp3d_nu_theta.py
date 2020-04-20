# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:37:40 2016

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
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

from mayavi import mlab

import time

import m_sim as sim
import m_aerodynamics as aerodynamics
import m_morph as morph

FIG = '../Figures/s_serp3d_nu_theta/{}.pdf'
FIGPNG = '../Figures/s_serp3d_nu_theta/{}.png'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


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


# %% Function definitions


def setup_body(L=.7, ds=.01, theta_max=90, nu_theta=1.1, f_theta=1.4,
               phi_theta=0, psi_max=10, frac_theta_max=0, d_theta=0, d_psi=0,
               display_ho=1, bl=1, bd=1):
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


def parallel_simulation(args):
    """Run the simulations in parallel.
    """

    # unpack the arguments
    fname, params = args
    nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0 = params

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
    Ro0 = np.r_[0, 0, z0]
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
    nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0 = params

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
    Ro0 = np.r_[0, 0, z0]
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

# nt_reg
# LinregressResult(slope=-56.600508336133117, intercept=175.72823421502233, rvalue=-0.88231660914492116, pvalue=2.4520681352353897e-12, stderr=5.2558484122712361)
m = -56.6005
b = 175.72823

#nu_thetas = [1, 1.25, 1.5]
#theta_maxs = [120, 105, 90]
#nu_thetas = np.r_[1, 1.125, 1.25, 1.375, 1.5]

nu_thetas = np.arange(1, 1.51, .05)
theta_maxs = m * nu_thetas + b

n_nu = len(nu_thetas)
n_theta = len(theta_maxs)


# %% 10 m glides

z0 = 10

BASE = '../Output/s_serp3d_nu_theta_z10/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

args = []

for nu_theta in nu_thetas:
    for theta_max in theta_maxs:
        for f_theta in np.r_[0, 1.2]:
            for d_psi in np.r_[10]:
                for psi_max in np.r_[20]:
                    for L in [.7]:
                        if f_theta == 0:
                            params = (nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0)
                            fname = BASE.format(*params)
                            arg = (fname, params)
                            args.append(arg)


# %% 75 m glides

z0 = 75

BASE = '../Output/s_serp3d_nu_theta_z75/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

args = []

for nu_theta in nu_thetas:
    for theta_max in theta_maxs:
        for f_theta in np.r_[0, 1.2]:
            for d_psi in np.r_[10]:
                for psi_max in np.r_[20]:
                    for L in [.7]:
                        if f_theta == 0:
                            params = (nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0)
                            fname = BASE.format(*params)
                            arg = (fname, params)
                            args.append(arg)


# %% 10 m glides --- boosted force

z0 = 10

BASE = '../Output/s_serp3d_nu_theta_z10_b/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

args = []

for nu_theta in nu_thetas:
    for theta_max in theta_maxs:
        for f_theta in np.r_[0, 1.2]:
            for d_psi in np.r_[10]:
                for psi_max in np.r_[20]:
                    for L in [.7]:
                        params = (nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0)
                        fname = BASE.format(*params)
                        arg = (fname, params)
                        args.append(arg)


# %% 75 m glides --- boosted force

z0 = 75

BASE = '../Output/s_serp3d_nu_theta_z75_b/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

args = []

for nu_theta in nu_thetas:
    for theta_max in theta_maxs:
        for f_theta in np.r_[0, 1.2]:
            for d_psi in np.r_[10]:
                for psi_max in np.r_[20]:
                    for L in [.7]:
                        params = (nu_theta, theta_max, f_theta, d_psi, psi_max, L, z0)
                        fname = BASE.format(*params)
                        arg = (fname, params)
                        args.append(arg)



# %% Run the parallel simulations

from multiprocessing import Pool

pool = Pool(processes=4)

now = time.time()

# uncomment to run simulations
#pool.map(parallel_simulation, args)
# pool.map(parallel_simulation_boosted, args)

runtime = time.time() - now
print('Elapsed time: {0:.1f} min'.format(runtime / 60))

# 40.5 min for z = 10 m
# 57.7 min for z = 75 m

# Elapsed time: 43.3 min for z = 10 m, boosted
# Elapsed time: 57.7 min for z = 750 m, boosted


# %% ANALYZE SIMULATION OUTPUT

def ret_fnames(z, nu, theta, f_theta, d_psi=10, psi_max=20, L=.7):

    from glob import glob

    BASE = '../Output/s_serp3d_nu_theta_z{}/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

    searchname = BASE.format(z, nu, theta, f_theta, d_psi, psi_max, L)

    return sorted(glob(searchname))


def ret_fnames_boosted(z, nu, theta, f_theta, d_psi=10, psi_max=20, L=.7):

    from glob import glob

    BASE = '../Output/s_serp3d_nu_theta_z{}_b/nu_{:.2f}_th_{:.0f}_fs_{:.1f}_dpsi_{}_psim_{}_L_{:.1f}.npz'

    searchname = BASE.format(z, nu, theta, f_theta, d_psi, psi_max, L)

    return sorted(glob(searchname))


m = -56.6005
b = 175.72823

nu_thetas = np.arange(1, 1.51, .05)
theta_maxs = m * nu_thetas + b

n_nu = len(nu_thetas)
n_theta = len(theta_maxs)


# %% FIGURE 4A: BODY SHAPES --- full 11 x 11 grid

fig, ax = plt.subplots(figsize=(7.5, 6.5))

for i in np.arange(n_theta):
    theta_max = theta_maxs[::-1][i]

    for j in np.arange(n_nu):
        nu_theta = nu_thetas[j]

        offx = 3 * 16 * j
        offy = 3 * 15 * i

        body_dict = setup_body(L=.7, ds=.01, theta_max=theta_max,
                               nu_theta=nu_theta, f_theta=1.4,
                               psi_max=0, frac_theta_max=0, display_ho=0)

        s, m, n_neck = body_dict['s'], body_dict['m'], body_dict['n_neck']
        theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
        out = sim.aerialserp_pos_vel_acc_tcb(s, 0, m, n_neck, theta_dict, psi_dict)

        p = 100 * out['p']  # cm

        ax.plot(offx + p[:, 0], offy + p[:, 1], '-', lw=2, ms=9, c='g',
                markevery=100)

ax.set_aspect('equal')
ax.margins(.03)
ax.axis('off')

#fig.savefig(FIG.format('body shapes all - Cube data'), **FIGOPT)


# %% Bending angles for nu_theta = 1.4, theta_m = 96 deg

nu_j = nu_thetas[8]  # = 1.5
theta_i = theta_maxs[8]  # = 90.82747

fname = ret_fnames(z=10, nu=nu_j, theta=theta_i, f_theta=0)[0]
d_f0 = np.load(fname)

fname = ret_fnames(z=10, nu=nu_j, theta=theta_i, f_theta=1.2)[0]
d = np.load(fname)

theta = np.rad2deg(d['theta'])
theta_f0 = np.rad2deg(d_f0['theta'])

psi = np.rad2deg(d['psi'])
psi_f0 = np.rad2deg(d_f0['psi'])


# %% FIGURE SI 3A: CoM position

ntime = len(d['Ro'])
ntime_f0 = len(d_f0['Ro'])

ii = np.r_[ntime // 3, ntime // 3 * 2, ntime - 1]
ii_f0 = np.r_[ntime_f0 // 3, ntime_f0 // 3 * 2, ntime_f0 - 1]

fig, ax = plt.subplots()
ax.plot(d['Ro'][:, 1], d['Ro'][:, 2], 'r', lw=3)
ax.plot(d_f0['Ro'][:, 1], d_f0['Ro'][:, 2], c='gray', lw=3)

ax.plot(d['Ro'][ii, 1], d['Ro'][ii, 2], 'ro', ms=8, mfc='none', mec='r', mew=2)
ax.plot(d_f0['Ro'][ii_f0, 1], d_f0['Ro'][ii_f0, 2], 'o', c='gray',
        ms=8, mfc='none', mec='gray', mew=2)

#ax.plot(d['Ro'][ii, 1], d['Ro'][ii, 2], 'ro', ms=9)
#ax.plot(d_f0['Ro'][ii_f0, 1], d_f0['Ro'][ii_f0, 2], 'o', c='gray', ms=9)

ax.set_aspect('equal')

ax.set_xticks([0, 6])
ax.set_xticklabels(['0 m', '6 m'])
ax.set_yticks([0, 5, 10])
ax.set_yticklabels(['0 m', '5 m', '10 m'])
ax.set_xlim(-.2, 6)
ax.set_ylim(-.2, 10.2)
ax.set_aspect('equal', adjustable='box')
sns.despine()

#fig.savefig(FIG.format('1p4_94 YZ'), **FIGOPT)


# %% FIGURE SI S3C: Demonstration of bending angles

L = .7
theta_max = theta_i
nu_theta = nu_j
f_theta = 1.2
d_psi = 10
psi_max = 20


body_dict = setup_body(L=L, ds=.01, theta_max=theta_max,
                       nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                       psi_max=psi_max, frac_theta_max=0,
                       d_theta=0, d_psi=d_psi, display_ho=0)


snon = body_dict['s'] / body_dict['L']


i = 55


figsize = (5.5, 4)
fig, ax = plt.subplots(figsize=figsize)
ax.axhline(0, color='gray', lw=1)

ax.plot(100 * snon, theta[i], 'b', lw=3)
ax.plot(100 * snon, psi[i], 'y', lw=3)

ax.set_ylim(-120, 120)
ax.set_yticks([-120, -80, -40, 0, 40, 80, 120])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])

ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Bending angles')

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('1p4_94 bending angles'), **FIGOPT)


# %% For figure S3A: Body with forces overlaid

# with undulation
foils, foil_color = sim.apply_airfoil_shape(d['r'], body_dict['c'], d['Crs_I'])
nframe = d['nframe']
ntime = len(d['ts'])
r = d['r']
ntime, nbody = r.shape[0], r.shape[1]
Fl, Fd, Fa = d['Fl'], d['Fd'], d['Fa']
dR, dR_BC, dR_T = d['dR'], d['dR_BC'], d['dR_T']
dRo = d['dRo']


# no undulation
foils, foil_color = sim.apply_airfoil_shape(d_f0['r'], body_dict['c'], d_f0['Crs_I'])
nframe = d_f0['nframe']
ntime = len(d_f0['ts'])
r = d_f0['r']
ntime, nbody = r.shape[0], r.shape[1]
Fl, Fd, Fa = d_f0['Fl'], d_f0['Fd'], d_f0['Fa']
dR, dR_BC, dR_T = d_f0['dR'], d_f0['dR_BC'], d_f0['dR_T']
dRo = d_f0['dRo']


# %% For force sheets from body

L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 10

for i in np.arange(ntime):
    for j in np.arange(nbody):
        # in inertial frame
        L[i, j, :, 0] = r[i, j]
        L[i, j, :, 1] = r[i, j] + scale_forces * Fl[i, j]
        D[i, j, :, 0] = r[i, j]
        D[i, j, :, 1] = r[i, j] + scale_forces * Fd[i, j]
        A[i, j, :, 0] = r[i, j]
        A[i, j, :, 1] = r[i, j] + scale_forces * Fa[i, j]

# %% Select different times, plot the body and forces

istart = 0
i0 = ntime // 3
i1 = ntime // 3 * 2
i2 = ntime - 1

#i = istart
i = i0
#i = i1
#i = i2

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# CoM and body axes
frame_c = [bmap[2], bmap[1], bmap[0]]
bframe = []
for ii in np.arange(3):
    frm = mlab.quiver3d(nframe[i, ii, 0], nframe[i, ii, 1], nframe[i, ii, 2],
                        scale_factor=.05, color=frame_c[ii], mode='arrow',
                        opacity=1, resolution=64)
    bframe.append(frm)

# inertial axies
_args = dict(opacity=.75, tube_radius=.001)
mlab.plot3d([-.2, .2], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-.2, .2], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-.075, .075], color=frame_c[2], **_args)

body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

# CoM velocity
vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
                     color=(0, 0, 0), mode='arrow', resolution=64)

op = .6
ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)

fig.scene.isometric_view()
fig.scene.parallel_projection = True

# %% Figure S3A: Save the 3D visualization with forces

# with undulation
name = '1p4_94 f=12 i{}'.format(i)

# no undulation
name = '1p4_94 f=0 i{}'.format(i)

if False:
    mlab.savefig(FIGPNG.format(name),
                 size=(3 * 750, 3 * 708), figure=fig)


# %% Figure S3: Save movies for paper (no undulation)

# no undulation
foils, foil_color = sim.apply_airfoil_shape(d_f0['r'], body_dict['c'], d_f0['Crs_I'])
nframe = d_f0['nframe']
ntime = len(d_f0['ts'])
r = d_f0['r']
ntime, nbody = r.shape[0], r.shape[1]
Fl, Fd, Fa = d_f0['Fl'], d_f0['Fd'], d_f0['Fa']
dR, dR_BC, dR_T = d_f0['dR'], d_f0['dR_BC'], d_f0['dR_T']
dRo = d_f0['dRo']

L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 10

for i in np.arange(ntime):
    for j in np.arange(nbody):
        L[i, j, :, 0] = r[i, j]
        L[i, j, :, 1] = r[i, j] + scale_forces * Fl[i, j]
        D[i, j, :, 0] = r[i, j]
        D[i, j, :, 1] = r[i, j] + scale_forces * Fd[i, j]
        A[i, j, :, 0] = r[i, j]
        A[i, j, :, 1] = r[i, j] + scale_forces * Fa[i, j]

view = (45.0, 54.735610317245346, 1.1313471393462222,
     np.array([ 0., 0.,  0.00076376]))

savename = '../Movies/0_s_nu_theta_f0/f0_{0:03d}.jpg'

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

sx, sy = fig.scene.get_size()

for i in np.arange(ntime):

    print('Image {:3d} of {:3d}'.format(i, ntime))

    mlab.clf()

    # CoM and body axes
    frame_c = [bmap[2], bmap[1], bmap[0]]
    bframe = []
    for ii in np.arange(3):
        frm = mlab.quiver3d(nframe[i, ii, 0], nframe[i, ii, 1], nframe[i, ii, 2],
                            scale_factor=.05, color=frame_c[ii], mode='arrow',
                            opacity=1, resolution=64)
        bframe.append(frm)

    # inertial axies
    _args = dict(opacity=.75, tube_radius=.001)
    mlab.plot3d([-.2, .2], [0, 0], [0, 0], color=frame_c[0], **_args)
    mlab.plot3d([0, 0], [-.2, .2], [0, 0], color=frame_c[1],**_args)
    mlab.plot3d([0, 0], [0, 0], [-.075, .075], color=frame_c[2], **_args)

    body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                     scalars=foil_color[i], colormap='YlGn', opacity=1,
                     vmin=0, vmax=1)

    # CoM velocity
    vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
                         color=(0, 0, 0), mode='arrow', resolution=64)

    op = .6
    ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
    md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)


    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    mlab.view(*view)
    mlab.draw()

    mlab.savefig(savename.format(i), size=(8*sx, 8*sy))

print('Image save time: {0:.3f} sec'.format(time.time() - now))

# 15536.142 = 4.3 hr


# %% Figure S3: with undulation

# load in the data
foils, foil_color = sim.apply_airfoil_shape(d['r'], body_dict['c'], d['Crs_I'])
nframe = d['nframe']
ntime = len(d['ts'])
r = d['r']
ntime, nbody = r.shape[0], r.shape[1]
Fl, Fd, Fa = d['Fl'], d['Fd'], d['Fa']
dR, dR_BC, dR_T = d['dR'], d['dR_BC'], d['dR_T']
dRo = d['dRo']

# construct force sheets
L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 10

for i in np.arange(ntime):
    for j in np.arange(nbody):
        # in inertial frame
        L[i, j, :, 0] = r[i, j]
        L[i, j, :, 1] = r[i, j] + scale_forces * Fl[i, j]
        D[i, j, :, 0] = r[i, j]
        D[i, j, :, 1] = r[i, j] + scale_forces * Fd[i, j]
        A[i, j, :, 0] = r[i, j]
        A[i, j, :, 1] = r[i, j] + scale_forces * Fa[i, j]

# consistent view
view = (45.0, 54.735610317245346, 1.1313471393462222,
     np.array([ 0., 0.,  0.00076376]))

savename = '../Movies/0_s_nu_theta_f12/f12_{0:03d}.jpg'

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

sx, sy = fig.scene.get_size()

for i in np.arange(ntime):

    mlab.clf()

    # CoM and body axes
    frame_c = [bmap[2], bmap[1], bmap[0]]
    bframe = []
    for ii in np.arange(3):
        frm = mlab.quiver3d(nframe[i, ii, 0], nframe[i, ii, 1], nframe[i, ii, 2],
                            scale_factor=.05, color=frame_c[ii], mode='arrow',
                            opacity=1, resolution=64)
        bframe.append(frm)

    # inertial axies
    _args = dict(opacity=.75, tube_radius=.001)
    mlab.plot3d([-.2, .2], [0, 0], [0, 0], color=frame_c[0], **_args)
    mlab.plot3d([0, 0], [-.2, .2], [0, 0], color=frame_c[1],**_args)
    mlab.plot3d([0, 0], [0, 0], [-.075, .075], color=frame_c[2], **_args)

    body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                     scalars=foil_color[i], colormap='YlGn', opacity=1,
                     vmin=0, vmax=1)

    # CoM velocity
    vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
                         color=(0, 0, 0), mode='arrow', resolution=64)

    op = .6
    ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
    md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)


    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    mlab.view(*view)
    mlab.draw()

#    mlab.savefig(savename.format(i), size=(8*sx, 8*sy))

print('Image save time: {0:.3f} sec'.format(time.time() - now))
# 22594.229 = 6.27


# %% FIGURE S3B: Euler angles vs z --- z = 10 m

z0 = 10
f_theta = 1.2
with_f = True
with_f0 = True

labels = ['yaw', 'pitch', 'roll']
cols = [bmap[0], bmap[2], bmap[1]]

with sns.axes_style('white'):

    fig, axs = plt.subplots(6, 6, sharex=False, sharey=False,
                                        figsize=(15, 10.5))

sns.despine()


for i in np.arange(6):
    for j in np.arange(6):
        ax = axs[i, j]

        ax.set_xlim(10, 0)
        ax.set_ylim(-90, 90)
        ax.set_yticks(np.arange(-90, 91, 90))
        ax.set_yticklabels([])

        [tick.label.set_fontsize('x-small') for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize('x-small') for tick in ax.yaxis.get_major_ticks()]
        for axis in ['left', 'bottom']:
            ax.spines[axis].set_linewidth(0.5)

        ax.spines['bottom'].set_position('zero')
        ax.set_xticks([0])
        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(length=0, width=0)


for i in np.arange(n_theta / 2, dtype=np.int):  # 120 at top, 85 at bottom
    theta_i = theta_maxs[2 * i]

    for j in np.arange(n_nu / 2, dtype=np.int):  # 1 on left, 1.5 on right
        nu_j = nu_thetas[2 * j]

        for k in np.arange(3):
            if i == 0 and j == 0:
                label = labels[k]
            else:
                label = None

            # with undulation
            if with_f:
                fname = ret_fnames(z0, nu_j, theta_i, f_theta)[0]
                d = np.load(fname)
                ts_non = d['ts'] * f_theta
                ang = np.rad2deg(d['ang'][:, k])
                axs[i, j].plot(d['Ro'][:, 2], ang, c=cols[k], label=label)
                if np.abs(ang[-1]) >= 85:
                    axs[i, j].plot(d['Ro'][-1, 2], ang[-1], 'o', c=cols[k])

            # without undulation
            if with_f0:
                fname = ret_fnames(z0, nu_j, theta_i, 0)[0]
                d = np.load(fname)
                ts_non = d['ts'] * f_theta
                ang = np.rad2deg(d['ang'][:, k])
                axs[i, j].plot(d['Ro'][:, 2], ang, '--', c=cols[k])
                if np.abs(ang[-1]) >= 85:
                    axs[i, j].plot(d['Ro'][-1, 2], ang[-1], '^', c=cols[k])

            if j == 0:  # if column is zero
                label = '{:.0f}'.format(theta_i) + u'\u00B0'
                axs[i, j].set_ylabel(label, fontsize='small')
            if i == 0:  # if row is zero
                axs[i, j].set_title('{:.1f}'.format(nu_j), fontsize='small')

ax = axs[0, 0]
ax.legend(fontsize='xx-small', frameon=False, loc='upper left',
          handlelength=1, columnspacing=1, borderaxespad=.3,
          ncol=3, handletextpad=.4)

ax.set_yticklabels([-90, 0, 90])
[tick.label.set_fontsize('x-small') for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize('x-small') for tick in ax.yaxis.get_major_ticks()]

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    tick.set_rotation(0)
    text = tick.get_text()  # remove float
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

#fig.savefig(FIG.format('Euler angles - f and f0 - z10'), **FIGOPT)


# %% Summary of landing locations and angles

z0s = np.r_[10, 75]

T = np.zeros((n_theta, n_nu), dtype=np.int)
N = np.zeros((n_theta, n_nu))

tend = np.zeros((2, n_theta, n_nu))
pos_end = np.zeros((2, n_theta, n_nu, 3))
ang_end = np.zeros((2, n_theta, n_nu, 3))
ang_max = np.zeros((2, n_theta, n_nu, 3))
pos_end_f0 = np.zeros((2, n_theta, n_nu, 3))

height_score = np.zeros((2, n_theta, n_nu))
distance_score = np.zeros((2, n_theta, n_nu))

height_score_dim = np.zeros((2, n_theta, n_nu))
distance_score_dim = np.zeros((2, n_theta, n_nu))

dist_f = np.zeros((2, n_theta, n_nu))
dist_f0 = np.zeros((2, n_theta, n_nu))
height_f = np.zeros((2, n_theta, n_nu))
height_f0 = np.zeros((2, n_theta, n_nu))

angle_cond = np.zeros((2, n_theta, n_nu))
angle_cond_f0 = np.zeros((2, n_theta, n_nu))

for m in np.arange(2):
    z0 = z0s[m]

    for i in np.arange(n_theta):  # 120 at top, 85 at bottom
        theta_i = theta_maxs[i]

        for j in np.arange(n_nu):  # 1 on left, 1.5 on right
            nu_j = nu_thetas[j]

            # arrays of theta and nu for plotting
            T[i, j] = int(theta_i)
            N[i, j] = nu_j

            # simulation with undulation on
            fname = ret_fnames(z=z0, nu=nu_j, theta=theta_i, f_theta=1.2)[0]
            # fname = ret_fnames_boosted(z=z0, nu=nu_j, theta=theta_i, f_theta=1.2)[0]
            d = np.load(fname)

            # simulation with undulation off
            fname = ret_fnames(z=z0, nu=nu_j, theta=theta_i, f_theta=0)[0]
            # fname = ret_fnames_boosted(z=z0, nu=nu_j, theta=theta_i, f_theta=0)[0]
            d_f0 = np.load(fname)

            # performance metrics with undulation
            tend[m, i, j] = d['ts'][-1] * d['f_theta']
            pos_end[m, i, j] = d['Ro'][-1]
            pos_end[m, i, j, 2] = z0 - pos_end[m, i, j, 2]  # height lost
            ang_end[m, i, j] = np.rad2deg(d['ang'][-1])

            # end position without undulation
            pos_end_f0[m, i, j] = d_f0['Ro'][-1]
            pos_end_f0[m, i, j, 2] = z0 - pos_end_f0[m, i, j, 2]  # height lost

            # maximum angles with no undulation
            ang_max_idx = np.abs(d['ang']).argmax(axis=0)
            angs = np.rad2deg(d['ang'])
            ang_max[m, i, j, 0] = angs[ang_max_idx[0], 0]
            ang_max[m, i, j, 1] = angs[ang_max_idx[1], 1]
            ang_max[m, i, j, 2] = angs[ang_max_idx[2], 2]

            # percent greater height lost due to undulation
            zf = z0 - d['Ro'][-1, 2]
            zf_f0 = z0 - d_f0['Ro'][-1, 2]
            height_score[m, i, j] = (zf - zf_f0) / zf_f0 * 100
            height_score_dim[m, i, j] = zf - zf_f0

            # percent greater glide distance along the ground due to undulation
            x, y, z = d['Ro'].T
            dx = np.gradient(x, edge_order=2)
            dy = np.gradient(y, edge_order=2)
            path = np.cumsum(np.sqrt(dx**2 + dy**2))
            x, y, z = d_f0['Ro'].T
            dx = np.gradient(x, edge_order=2)
            dy = np.gradient(y, edge_order=2)
            path_f0 = np.cumsum(np.sqrt(dx**2 + dy**2))
            distance_score[m, i, j] = (path[-1] - path_f0[-1]) / path_f0[-1] * 100
            distance_score_dim[m, i, j] = path[-1] - path_f0[-1]

            # scores for the paper (in physical units)
            dist_f[m, i, j] = path[-1]
            dist_f0[m, i, j] = path_f0[-1]
            height_f[m, i, j] = zf
            height_f0[m, i, j] = zf_f0

            # what angle ultimately caused the instability
            for k in np.arange(3):
                if np.abs(angs[-1, k]) >= 85:
                    angle_cond[m, i, j] = k + 1

            # what angle ultimately caused the instability - f = 0
            angs_f0 = np.rad2deg(d_f0['ang'])
            for k in np.arange(3):
                if np.abs(angs_f0[-1, k]) >= 85:
                    angle_cond_f0[m, i, j] = k + 1


# %% DataFrames for summary plots

m = 0
t_z10 = pd.DataFrame(data=tend[m], index=T[:, 0], columns=N[0])
x_z10 = pd.DataFrame(data=pos_end[m, :, :, 0], index=T[:, 0], columns=N[0])
y_z10 = pd.DataFrame(data=pos_end[m, :, :, 1], index=T[:, 0], columns=N[0])
z_z10 = pd.DataFrame(data=pos_end[m, :, :, 2], index=T[:, 0], columns=N[0])

x_z10_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 0], index=T[:, 0], columns=N[0])
y_z10_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 1], index=T[:, 0], columns=N[0])
z_z10_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 2], index=T[:, 0], columns=N[0])

#yaw_z10 = pd.DataFrame(data=ang_end[m, :, :, 0], index=T[:, 0], columns=N[0])
#pitch_z10 = pd.DataFrame(data=ang_end[m, :, :, 1], index=T[:, 0], columns=N[0])
#roll_z10 = pd.DataFrame(data=ang_end[m, :, :, 2], index=T[:, 0], columns=N[0])

yaw_z10 = pd.DataFrame(data=ang_max[m, :, :, 0], index=T[:, 0], columns=N[0])
pitch_z10 = pd.DataFrame(data=ang_max[m, :, :, 1], index=T[:, 0], columns=N[0])
roll_z10 = pd.DataFrame(data=ang_max[m, :, :, 2], index=T[:, 0], columns=N[0])

height_score_z10 = pd.DataFrame(data=height_score[m], index=T[:, 0], columns=N[0])
distance_score_z10 = pd.DataFrame(data=distance_score[m], index=T[:, 0], columns=N[0])

height_score_dim_z10 = pd.DataFrame(data=height_score_dim[m], index=T[:, 0], columns=N[0])
distance_score_dim_z10 = pd.DataFrame(data=distance_score_dim[m], index=T[:, 0], columns=N[0])

angle_cond_z10 = pd.DataFrame(data=angle_cond[m], index=T[:, 0], columns=N[0])
angle_cond_z10_f0 = pd.DataFrame(data=angle_cond_f0[m], index=T[:, 0], columns=N[0])

dist_z10 = pd.DataFrame(data=dist_f[m], index=T[:, 0], columns=N[0])
dist_z10_f0 = pd.DataFrame(data=dist_f0[m], index=T[:, 0], columns=N[0])
height_z10 = pd.DataFrame(data=height_f[m], index=T[:, 0], columns=N[0])
height_z10_f0 = pd.DataFrame(data=height_f0[m], index=T[:, 0], columns=N[0])


m = 1
t_z75 = pd.DataFrame(data=tend[m], index=T[:, 0], columns=N[0])
x_z75 = pd.DataFrame(data=pos_end[m, :, :, 0], index=T[:, 0], columns=N[0])
y_z75 = pd.DataFrame(data=pos_end[m, :, :, 1], index=T[:, 0], columns=N[0])
z_z75 = pd.DataFrame(data=pos_end[m, :, :, 2], index=T[:, 0], columns=N[0])

x_z75_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 0], index=T[:, 0], columns=N[0])
y_z75_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 1], index=T[:, 0], columns=N[0])
z_z75_f0 = pd.DataFrame(data=pos_end_f0[m, :, :, 2], index=T[:, 0], columns=N[0])

#yaw_z75 = pd.DataFrame(data=ang_end[m, :, :, 0], index=T[:, 0], columns=N[0])
#pitch_z75 = pd.DataFrame(data=ang_end[m, :, :, 1], index=T[:, 0], columns=N[0])
#roll_z75 = pd.DataFrame(data=ang_end[m, :, :, 2], index=T[:, 0], columns=N[0])

yaw_z75 = pd.DataFrame(data=ang_max[m, :, :, 0], index=T[:, 0], columns=N[0])
pitch_z75 = pd.DataFrame(data=ang_max[m, :, :, 1], index=T[:, 0], columns=N[0])
roll_z75 = pd.DataFrame(data=ang_max[m, :, :, 2], index=T[:, 0], columns=N[0])

height_score_z75 = pd.DataFrame(data=height_score[m], index=T[:, 0], columns=N[0])
distance_score_z75 = pd.DataFrame(data=distance_score[m], index=T[:, 0], columns=N[0])

height_score_dim_z75 = pd.DataFrame(data=height_score_dim[m], index=T[:, 0], columns=N[0])
distance_score_dim_z75 = pd.DataFrame(data=distance_score_dim[m], index=T[:, 0], columns=N[0])

angle_cond_z75 = pd.DataFrame(data=angle_cond[m], index=T[:, 0], columns=N[0])
angle_cond_z75_f0 = pd.DataFrame(data=angle_cond_f0[m], index=T[:, 0], columns=N[0])

dist_z75 = pd.DataFrame(data=dist_f[m], index=T[:, 0], columns=N[0])
dist_z75_f0 = pd.DataFrame(data=dist_f0[m], index=T[:, 0], columns=N[0])
height_z75 = pd.DataFrame(data=height_f[m], index=T[:, 0], columns=N[0])
height_z75_f0 = pd.DataFrame(data=height_f0[m], index=T[:, 0], columns=N[0])


# %% SI TABLE for PAPER: Statistics for paper

# All UNSTABLE - height - percent unstable
# 5.785123966942149
# 50.413223140495866
#
# Diagonal
# 0.0
# 65.3061224489796
#
# Lower left
# 0.0
# 16.666666666666664
#
# Upper right
# 19.444444444444446
# 63.888888888888886

h_z10 = np.array(height_z10)
h_z10_f0 = np.array(height_z10_f0)
d_z10 = np.array(dist_z10)
d_z10_f0 = np.array(dist_z10_f0)
ngrid = h_z10.size

mask_l = np.tri(11, 11, 2, dtype=np.bool)
mask_r = mask_l.T
mask =  mask_l * mask_r
mask_ll = ~mask_r
mask_ur = ~mask_l
nmask = np.where(mask > 0)[0].size  # 49
nmask_ll = np.where(mask_ll > 0)[0].size  # 36
nmask_ur = np.where(mask_ur > 0)[0].size  # 36

mask_flat = mask.flatten()
mask_ll_flat = mask_ll.flatten()
mask_ur_flat = mask_ur.flatten()

# height statistics

# statistics for all simulations
print('All UNSTABLE - height - percent unstable')
print((h_z10 < 10).sum() / ngrid * 100)
print((h_z10_f0 < 10).sum() / ngrid * 100)

# diagonal statistics
print()
print('Diagonal')
print(((h_z10 < 10) * mask).sum() / nmask * 100)
print(((h_z10_f0 < 10) * mask).sum() / nmask * 100)

# lower left statistics
print()
print('Lower left')
print(((h_z10 < 10) * mask_ll).sum() / nmask_ll * 100)
print(((h_z10_f0 < 10) * mask_ll).sum() / nmask_ll * 100)

# upper right statistics
print()
print('Upper right')
print(((h_z10 < 10) * mask_ur).sum() / nmask_ur * 100)
print(((h_z10_f0 < 10) * mask_ur).sum() / nmask_ur * 100)


# %% SI TABLE for PAPER: Statistics for paper

# All STABLE - height - percent unstable
# 94.21487603305785
# 49.586776859504134
#
# Diagonal
# 100.0
# 34.69387755102041
#
# Lower left
# 100.0
# 83.33333333333334
#
# Upper right
# 80.55555555555556
# 36.11111111111111

# statistics for all simulations
print('All STABLE - height - percent unstable')
print((h_z10 >= 10).sum() / ngrid * 100)
print((h_z10_f0 >= 10).sum() / ngrid * 100)

# diagonal statistics
print()
print('Diagonal')
print(((h_z10 >= 10) * mask).sum() / nmask * 100)
print(((h_z10_f0 >= 10) * mask).sum() / nmask * 100)

# lower left statistics
print()
print('Lower left')
print(((h_z10 >= 10) * mask_ll).sum() / nmask_ll * 100)
print(((h_z10_f0 >= 10) * mask_ll).sum() / nmask_ll * 100)

# upper right statistics
print()
print('Upper right')
print(((h_z10 >= 10) * mask_ur).sum() / nmask_ur * 100)
print(((h_z10_f0 >= 10) * mask_ur).sum() / nmask_ur * 100)


# %% SI TABLE for PAPER: Glide distance statistics

# All - distance
# 4.2510146430858935
# 3.9642939629474507
#
# Diagonal
# 4.912981344132635
# 3.9790832460240764
#
# Diagonal std
# 0.7009747457114743
# 0.7975536074665346
#
# Lower left
# 3.3614042089986462
# 4.24203655825348
#
# Upper right
# 4.239614845192853
# 3.6664215101204607

# statistics for all simulations
print()
print('All - distance')
print(d_z10.mean())
print(d_z10_f0.mean())

# diagonal statistics
print()
print('Diagonal')
print(d_z10.flatten()[mask_flat].mean())
print(d_z10_f0.flatten()[mask_flat].mean())
print()
print('Diagonal std')
print(d_z10.flatten()[mask_flat].std())
print(d_z10_f0.flatten()[mask_flat].std())

# lower left statistics
print()
print('Lower left')
print(d_z10.flatten()[mask_ll_flat].mean())
print(d_z10_f0.flatten()[mask_ll_flat].mean())

# upper right statistics
print()
print('Upper right')
print(d_z10.flatten()[mask_ur_flat].mean())
print(d_z10_f0.flatten()[mask_ur_flat].mean())


# %% SI TABLE for PAPER: Height and glide scores (dimenional) statistics

# All - scores - height/dist
# 10.4811879885056
# 5.0445643417470745
#
# Diagonal
# 12.460630379202083
# 6.858845905084482
#
# Diagonal std
# 7.565492194357119
# 5.551765939495469
#
# Lower left
# 13.809317058990578
# 5.868512938134028
#
# Upper right
# 4.458817886239299
# 1.7511769508175385

hs_dim_z75 = np.array(height_score_dim_z75)
ds_dim_z75 = np.array(distance_score_dim_z75)

# statistics for all simulations
print()
print('All - scores - height/dist')
print(hs_dim_z75.mean())
print(ds_dim_z75.mean())

# diagonal statistics
print()
print('Diagonal')
print(hs_dim_z75.flatten()[mask_flat].mean())
print(ds_dim_z75.flatten()[mask_flat].mean())
print()
print('Diagonal std')
print(hs_dim_z75.flatten()[mask_flat].std())
print(ds_dim_z75.flatten()[mask_flat].std())

# lower left
print()
print('Lower left')
print(hs_dim_z75.flatten()[mask_ll_flat].mean())
print(ds_dim_z75.flatten()[mask_ll_flat].mean())


# upper right
print()
print('Upper right')
print(hs_dim_z75.flatten()[mask_ur_flat].mean())
print(ds_dim_z75.flatten()[mask_ur_flat].mean())


# %% SI TABLE for PAPER: How does undulation improve the long term glide behavior

# Undulation boost glide distance
# 85.9504132231405
# along diagonal
# 100.0
#
# Undulation boost height fallen
# 85.9504132231405
# along diagonal:
# 91.83673469387756
# avg increase where no undulation performed better:
# 0.1665972511453561

print()
print('Undulation boost glide distance')
print((hs_dim_z75 > 0).sum() / ngrid * 100)
print('along diagonal')
print((hs_dim_z75 > 0).flatten()[mask_flat].sum() / nmask * 100)


print()
print('Undulation boost height fallen')
print((ds_dim_z75 > 0).sum() / ngrid * 100)
print('along diagonal:')
print((ds_dim_z75 > 0).flatten()[mask_flat].sum() / nmask * 100)
print('avg increase where no undulation performed better:')
print((ds_dim_z75 < 0).flatten()[mask_flat].mean() / nmask * 100)


# %% FIGURE S2B: Masks used for the analysis

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(11.75, 4.75))
ax1.matshow(mask)
ax2.matshow(mask_ll)
ax3.matshow(mask_ur)

ax1.set_title('Mask', fontsize='small')
ax2.set_title('Mask lower left', fontsize='small')
ax3.set_title('Mask upper right', fontsize='small')

for ax in (ax1, ax2, ax3):
    ax.axis('off')

plt.setp(axs, aspect=1.0, adjustable='box')

#fig.savefig(FIG.format('masks for nu theta band'), **FIGOPT)


# %% FIGURE 4B-D and FIGURE S2: Glide distance and height fallen

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12.5, 9))

dfs_height = [height_z10_f0, height_z10, height_score_dim_z75]
dfs_dist = [dist_z10_f0, dist_z10, distance_score_dim_z75]

vlims_height = [[0, 10], [0, 10], [-57, 57]]
vlims_dist = [[2, 6], [2, 6], [-22, 22]]

cmap_height = [plt.cm.Purples_r, plt.cm.Purples_r, plt.cm.coolwarm]
cmap_dist = [plt.cm.Greens_r, plt.cm.Greens_r, plt.cm.coolwarm]

center = [None, None, 0]
alpha = [.5, .5, .5]

xx = np.r_[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
yy = np.r_[3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]

# height
for i in np.arange(3):
    ax = axs[0, i]
    sns.heatmap(dfs_height[i], ax=ax, cmap=cmap_height[i],
                vmin=vlims_height[i][0], vmax=vlims_height[i][1],
                annot=True, fmt='.0f', annot_kws={'fontsize': 'xx-small'},
                cbar=False, cbar_kws={'shrink': .75},
                rasterized=True,
                center=center[i])

    ax.plot(xx, yy, 'k', lw=1.5)
    ax.plot(yy, xx, 'k', lw=1.5)

    ax.xaxis.set_tick_params(length=5, width=.75)
    ax.yaxis.set_tick_params(length=5, width=.75)
    [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

# glide distance
for i in np.arange(3):
    ax = axs[1, i]
    sns.heatmap(dfs_dist[i], ax=ax, cmap=cmap_dist[i],
                vmin=vlims_dist[i][0], vmax=vlims_dist[i][1],
                annot=True, fmt='.0f', annot_kws={'fontsize': 'xx-small'},
                cbar=False, cbar_kws={'shrink': .75},
                rasterized=True,
                center=center[i])

    ax.plot(xx, yy, 'k', lw=1.5)
    ax.plot(yy, xx, 'k', lw=1.5)

    ax.xaxis.set_tick_params(length=5, width=.75)
    ax.yaxis.set_tick_params(length=5, width=.75)
    [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

fontsize = 'x-small'
axs[0, 0].set_title('Height fallen (m), z$_0$ = 10 m, f = 0 Hz', fontsize=fontsize)
axs[0, 1].set_title('Height fallen (m), z$_0$ = 10 m, f = 1.2 Hz',
                    fontsize=fontsize)
axs[0, 2].set_title('Increase in height fallen (m), z$_0$ = 75 m', fontsize=fontsize)
axs[1, 0].set_title('Glide distance (m), z$_0$ = 10 m, f = 0 Hz', fontsize=fontsize)
axs[1, 1].set_title('Glide distance (m), z$_0$ = 10 m, f = 1.2 Hz',
                    fontsize=fontsize)
axs[1, 2].set_title('Increase in glide distance (m), z$_0$ = 75 m',
                    fontsize=fontsize)

# add degree symbol to angles
fig.canvas.draw()
for ax in [axs[0, 0], axs[1, 0]]:
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        tick.set_rotation(0)
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    ax.set_yticklabels(newticks)

# doesn't work with heatmaps
#for ax in axs.flatten():
#    ax.yaxis.set_major_formatter(degree_formatter)

# turn everything off for the paper
for ax in axs.flatten():
    ax.axis('off')

fig.set_tight_layout(True)

#fig.savefig(FIG.format('0 performance map - z75 scores - no alpha - paper - new colors - coolwarm'), **FIGOPT)
#fig.savefig(FIG.format('0 performance map - z75 scores - no alpha - paper - new colors - coolwarm - boosted'), **FIGOPT)


# %% Figures below are interesting, but not included in the manuscript


# %% What angle caused the simulation to stop

# how to annotate the angles: http://stackoverflow.com/a/37799642

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 9))

dfs_z10 = [angle_cond_z10_f0, angle_cond_z10]
dfs_z75 = [angle_cond_z75_f0, angle_cond_z75]

# z = 10 m
for i in np.arange(2):
    ax = axs[0, i]
    df_orig = dfs_z10[i].astype(np.int)
    df_label = df_orig.copy()
    df_label = df_label.replace(np.arange(4), ['', 'Y', 'P', 'R'])
    sns.heatmap(df_orig, ax=ax, cmap=plt.cm.magma,
                vmin=0, vmax=3,
                annot=df_label, fmt='', annot_kws={'fontsize': 'xx-small'},
                cbar=i == 2, cbar_kws={'shrink': .75},
                rasterized=True)

    ax.xaxis.set_tick_params(length=5, width=.75)
    ax.yaxis.set_tick_params(length=5, width=.75)
    [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

# z = 75 m
for i in np.arange(2):
    ax = axs[1, i]
    df_orig = dfs_z75[i].astype(np.int)
    df_label = df_orig.copy()
    df_label = df_label.replace(np.arange(4), ['', 'Y', 'P', 'R'])
    sns.heatmap(df_orig, ax=ax, cmap=plt.cm.magma,
                vmin=0, vmax=3,
                annot=df_label, fmt='', annot_kws={'fontsize': 'xx-small'},
                cbar=i == 2, cbar_kws={'shrink': .75})

    ax.xaxis.set_tick_params(length=5, width=.75)
    ax.yaxis.set_tick_params(length=5, width=.75)
    [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

fontsize = 'x-small'
axs[0, 0].set_title('Rotational stability, z$_0$ = 10 m, f = 0 Hz',
                    fontsize=fontsize)
axs[0, 1].set_title('Rotational stability, z$_0$ = 10 m, f = 1.2 Hz',
                    fontsize=fontsize)
axs[1, 0].set_title('Rotational stability, z$_0$ = 75 m, f = 0 Hz',
                    fontsize=fontsize)
axs[1, 1].set_title('Rotational stability, z$_0$ = 75 m, f = 1.2 Hz',
                    fontsize=fontsize)

# add degree symbol to angles
fig.canvas.draw()
for ax in [axs[0, 0], axs[1, 0]]:
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        tick.set_rotation(0)
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    ax.set_yticklabels(newticks)

#fig.savefig(FIG.format('angle condition - 4 plot'), **FIGOPT)

# %% What angle caused the simulation to stop


def angle_plotter(df, title):
    figsize = (6.5, 5)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap=plt.cm.magma, vmin=0, vmax=3,
                annot=False, fmt='.0f',
                annot_kws={'fontsize': 'x-small'},
                cbar=False, rasterized=True)

    cbar = fig.colorbar(ax.collections[0], shrink=.4)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Stable', 'Yaw', 'Pitch', 'Roll'])
    #cbar.set_yticklabels(['Stable', 'Yaw', 'Pitch', 'Roll'], fontsize='small')
    [tick.label.set_fontsize('small') for tick in cbar.ax.yaxis.get_major_ticks()]
    cbar.outline.set_linewidth(0)

    ax.xaxis.set_major_formatter(decimal_formatter)
    ax.set_xticklabels(N[0])
    ax.set_title(title, fontsize='small')
    [tick.label.set_fontsize('x-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('x-small') for tick in ax.yaxis.get_major_ticks()]

    fig.canvas.draw()
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        tick.set_rotation(0)
        text = tick.get_text()  # remove float
        newticks.append(text + u'\u00B0')
    ax.set_yticklabels(newticks)

    return fig


fig = angle_plotter(angle_cond_z10, 'Unstable angle, z = 10 m')
#fig.savefig(FIG.format('angle condition - z10'), **FIGOPT)
#
fig = angle_plotter(angle_cond_z75, 'Unstable angle, z = 75 m')
#fig.savefig(FIG.format('angle condition - z75'), **FIGOPT)