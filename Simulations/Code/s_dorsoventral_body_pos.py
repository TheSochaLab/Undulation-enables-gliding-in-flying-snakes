#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:39:00 2017

Script extracted from s_serp3d_full.py

%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

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

FIG = '../Figures/s_dorsoventral_body_pos/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}
FIGPNG = '../Figures/s_dorsoventral_body_pos/{}.png'


# %% Function definitions

def setup_body(L=.7, ds=.01, theta_max=90, nu_theta=1.1, f_theta=1.4,
               phi_theta=0, psi_max=10, frac_theta_max=0, d_theta=0, d_psi=0,
               display_ho=1, ho_shift=True):
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
    aero_interp = aerodynamics.extend_wind_tunnel_data()

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


# %% FIGURE 5A: Create 3D snake shapes showing d_psi effect

# useful parameters
nu_thetas = [1, 1.25, 1.5]
theta_maxs = [120, 105, 90]
f_thetas = [0, .8, 1, 1.2, 1.4, 1.6]
d_psis = [-20, -10, 0, 10, 20]
psi_maxs = [0, 10, 20]
Ls = [.3, .5, .7, .9]

# setup body
L = .7
nu_theta = nu_thetas[1]
theta_max = theta_maxs[1]
f_theta = 1.2
psi_max = 20

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

for d_psi in d_psis:
    bd = setup_body(L=L, ds=.01, theta_max=theta_max,
                    nu_theta=nu_theta, f_theta=f_theta, phi_theta=0,
                    psi_max=psi_max, frac_theta_max=0,
                    d_theta=0, d_psi=d_psi, display_ho=0, ho_shift=False)

    snon = bd['s'] / bd['L']

    t = 0
    out = sim.aerialserp_pos_vel_acc_tcb(bd['s'], t, bd['m'], bd['n_neck'],
                                         bd['theta_dict'], bd['psi_dict'])


    p = np.zeros((1, out['p'].shape[0], out['p'].shape[1]))
    p[0] = out['p']

    Crs = np.zeros((1, out['Crs'].shape[0], out['Crs'].shape[1],
                   out['Crs'].shape[2]))
    Crs[0] = out['Crs']

    foils, foil_color = sim.apply_airfoil_shape(p, bd['c'], Crs)

    p, cv, bv = out['p'], out['cv'], out['bv']


    i = 0

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

    frame_c = [bmap[2], bmap[1], bmap[0]]

    # inertial axies
    _args = dict(opacity=.75, tube_radius=.001)
    mlab.plot3d([-.1, .1], [0, 0], [0, 0], color=frame_c[0], **_args)
    mlab.plot3d([0, 0], [-.1, .1], [0, 0], color=frame_c[1],**_args)
    mlab.plot3d([0, 0], [0, 0], [-.075, .075], color=frame_c[2], **_args)

    body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                     scalars=foil_color[i], colormap='YlGn', opacity=1,
                     vmin=0, vmax=1)

    # uncomment to plot both side and back views
    # off = .21
    # # inertial axies
    # _args = dict(opacity=.75, tube_radius=.001)
    # mlab.plot3d([-.1, .1], [-off, -off], [0, 0], color=frame_c[1], **_args)
    # mlab.plot3d([0, 0], [-.1 - off, .1 - off], [0, 0], color=frame_c[0],**_args)
    # mlab.plot3d([0, 0], [-off, -off], [-.075, .075], color=frame_c[2], **_args)
    #
    # body = mlab.mesh(-foils[i, :, :, 1], foils[i, :, :, 0] - off, foils[i, :, :, 2],
    #             scalars=foil_color[i], colormap='YlGn', opacity=1,
    #             vmin=0, vmax=1)


    fig.scene.isometric_view()
    fig.scene.parallel_projection = True

    if False:
        mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
        mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
        mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
        mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)

    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    # mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)

    # save the images
    name = 'side d_psi={0}'.format(d_psi)
    # name = 'back d_psi={0}'.format(d_psi)
    # name = 'both d_psi={0}'.format(d_psi)

    # mlab.savefig(FIGPNG.format(name),
    #              size=(3 * 750, 3 * 708), figure=fig)