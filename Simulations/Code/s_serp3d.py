# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:37:40 2016

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

from mayavi import mlab

import time

import m_sim as sim
import m_aerodynamics as aerodynamics
import m_morph as morph

FIG = '../Figures/s_serp3d/{}.pdf'
FIGOPT = {'transparent': True}


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

decimal_formatter = FuncFormatter(_formatter_remove_zeros)


# %% Run a sample simulation

L = .7  # .686  # m
ds = .01  # m
#ds = .001  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .075 * L  # 5% SVL % .05 m on a .7 m snake is 7.14%
#neck_length = .065
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
v0_non = 1.7 / np.sqrt(2 * 29 / rho)  # .2409
vo_non_rng = np.r_[1.7, 1.1, 2.3, .8, 2.8] / vscale  # jump parameters

# aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data()

# base configuration
nu_theta = 1  # .8, 1, 1.2
nu_theta = 1.1
#nu_theta = 3
#nu_theta = 1.05
#nu_theta = -1  # waves propogate forward, not backwards!
#f_theta = 1.4  # Hz
f_theta = 1.3

phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * (phi_theta - np.pi / 2)

theta_max = np.deg2rad(90)  # 70, 90, 110
#frac_theta_max = .1  # "opens up" the anterior portion
frac_theta_max = .1
frac_theta_max = 0  # pure aerial serpenoid curve
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(15)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
#d_psi = np.deg2rad(3)
d_psi = np.deg2rad(-20)
#d_psi = np.deg2rad(10)
d_psi = np.deg2rad(-4)

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
phi_theta = fmin(sim.func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
phi_theta = float(phi_theta)
phi_psi = 2 * (phi_theta - np.pi / 2)

theta_dict['phi_theta'] = phi_theta
psi_dict['phi_psi'] = phi_psi


## TODO turn undulation off
theta_dict['f_theta'] = 0
psi_dict['f_psi'] = 0


out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

p = out['p']
tv, cv, bv = out['tv'], out['cv'], out['bv']

# dictionary with all of the simulation parameters in it
body_dict = dict(L=L, ds=ds, s=s, nbody=nbody, neck_length=neck_length,
                 n_neck=n_neck, cmax=cmax, mtot=mtot, rho_bar=rho_bar,
                 c=c, m=m, weight=weight, darea=darea, area=area, Ws=Ws,
                 theta_dict=theta_dict, psi_dict=psi_dict,
                 tscale=tscale, pscale=pscale, vscale=vscale,
                 ascale=ascale, fscale=fscale, mscale=mscale,
                 dt=dt, g=g, rho=rho, aero_interp=aero_interp,
                 head_control=False)

# with head control to shift the body CS
#tv, cv, bv, Crs = sim.serp3d_tcb_head_control(out['dpds'], np.eye(3))


# %%

skip = 2

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.plot(p[:, 0], p[:, 1], '-o', c='g', markevery=skip, lw=3)

ax.quiver(p[::skip, 0], p[::skip, 1], tv[::skip, 0], tv[::skip, 1], color='r',
          units='xy', width=.0015, zorder=10, label='tv')
ax.quiver(p[::skip, 0], p[::skip, 1], cv[::skip, 0], cv[::skip, 1], color='b',
          units='xy', width=.0015, zorder=10, label='cv')

ax.set_title('Body shape')

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Lateral direction')
ax.set_ylabel('Transverse direction')
ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.plot(p[:, 0], p[:, 2], '-o', c='g', markevery=skip, lw=3)
ax.plot(p[0, 0], p[0, 2], 'ko')

ax.quiver(p[::skip, 0], p[::skip, 2], bv[::skip, 0], bv[::skip, 2], color='r',
          units='xy', width=.0015, zorder=10, label='bv')
ax.quiver(p[::skip, 0], p[::skip, 2], cv[::skip, 0], cv[::skip, 2], color='b',
          units='xy', width=.0015, zorder=10, label='cv')

ax.set_title('Body shape')

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Lateral direction')
ax.set_ylabel('Transverse direction')
ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


# %% Find initial conditions so that dho is zero, angles over one cycle zero

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


ang0 = find_rotational_ic(body_dict, print_time=True)


# %% One undulation cycle, make movie

#yaw0, pitch0, roll0 = np.deg2rad(np.r_[0, 0, 0])
#C0 = sim.euler2C(yaw0, pitch0, roll0)

C0 = np.eye(3)

ntime = 101
ntime = 200
ts = np.linspace(0, 2 / f_theta, ntime + 1)[:-1]

ntnb3 = np.zeros((ntime, nbody, 3))
p = ntnb3.copy()
r = ntnb3.copy()
Crs = np.zeros((ntime, nbody, 3, 3))
Crs_iner = np.zeros((ntime, nbody, 3, 3))
tv, cv, bv = ntnb3.copy(), ntnb3.copy(), ntnb3.copy()
Tv, Cv, Bv = ntnb3.copy(), ntnb3.copy(), ntnb3.copy()
theta = np.zeros((ntime, nbody))
psi = np.zeros((ntime, nbody))

for i in np.arange(ntime):
    t = ts[i]
    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck,
                                         theta_dict, psi_dict)

    theta[i] = np.rad2deg(out['theta'])
    psi[i] = np.rad2deg(out['psi'])

    p[i] = out['p']
    Crs[i] = out['Crs']
    tv[i], cv[i], bv[i] = out['tv'], out['cv'], out['bv']

    # apply head control
    dpds = out['dpds']
    # tv[i], cv[i], bv[i], Crs[i] = sim.serp3d_tcb_head_control(dpds, C0)
    tv[i], cv[i], bv[i], Crs[i] = sim.serp3d_tcb(dpds)

    # rotate into interial frame
    r[i] = sim.rotate(C0.T, out['p'])
    Tv[i] = sim.rotate(C0.T, tv[i])
    Cv[i] = sim.rotate(C0.T, cv[i])
    Bv[i] = sim.rotate(C0.T, bv[i])

    # convert rotation matrices for body into inertial frame
    for j in np.arange(nbody):
        Crs_iner[i, j] = np.dot(C0.T, Crs[i, j])

foils_body, foil_color_body = sim.apply_airfoil_shape(p, c, Crs)
foils_iner, foil_color_iner = sim.apply_airfoil_shape(r, c, Crs_iner)

# for top view
foil_color_theta = foil_color_body.copy()
foil_color_psi = foil_color_body.copy()

for i in np.arange(ntime):
    for j in np.arange(nbody):
        foil_color_theta[i, j + 1] = theta[i, j]
        foil_color_theta[i, 0] = foil_color_theta[i, 1]
        foil_color_theta[i, -1] = foil_color_theta[i, -2]

        foil_color_psi[i, j + 1] = psi[i, j]
        foil_color_psi[i, 0] = foil_color_psi[i, 1]
        foil_color_psi[i, -1] = foil_color_psi[i, -2]


# %% Snake in body frame

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 725))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]

# top view
for ii in np.arange(3):
    mlab.quiver3d([0], [0], [0],
                  [Nframe[ii, 0]], [Nframe[ii, 1]], [Nframe[ii, 2]], scale_factor=.035,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

# front view
for ii in np.arange(3):
    mlab.quiver3d([0], [.18], [0],
                  [Nframe[ii, 0]], [Nframe[ii, 2]], [Nframe[ii, 1]], scale_factor=.035,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

# side view
for ii in np.arange(3):
    mlab.quiver3d([.2], [0], [0], [-Nframe[ii, 2]], [Nframe[ii, 1]], [Nframe[ii, 0]], scale_factor=.035,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

# top view
sxy = mlab.mesh(foils_body[i, :, :, 0],
                foils_body[i, :, :, 1],
                foils_body[i, :, :, 2],
                scalars=foil_color_theta[i], colormap='RdBu', opacity=1,
                vmin=-90, vmax=90)
sxy.module_manager.scalar_lut_manager.reverse_lut = True

# front view
sxz = mlab.mesh(foils_body[i, :, :, 0],
                .18 + foils_body[i, :, :, 2],
                foils_body[i, :, :, 1],
                scalars=foil_color_psi[i], colormap='PuOr', opacity=1,
                vmin=-10, vmax=10)
sxz.module_manager.scalar_lut_manager.reverse_lut = True

# side view
szy = mlab.mesh(.2 - foils_body[i, :, :, 2],
                foils_body[i, :, :, 1],
                foils_body[i, :, :, 0],
                scalars=foil_color_psi[i], colormap='PuOr', opacity=1,
                vmin=-10, vmax=10)
szy.module_manager.scalar_lut_manager.reverse_lut = True

#sxy = ax2.scatter(pi[:, 0], pi[:, 1], **kwargs_theta)  # top view
#sxz = ax2.scatter(pi[:, 0], 20 + pi[:, 2], **kwargs_psi)  # front view
#szy = ax2.scatter(20 + pi[:, 2], pi[:, 1], **kwargs_psi)  # side view


fig.scene.isometric_view()
fig.scene.parallel_projection = True
#mlab.orientation_axes()

mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)

mlab.view(*(0.0, 0.0, 0.9769115034306598,
            np.array([ 0.03948808,  0.03851232,  0.00519992])))

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)
    mlab.view(azimuth=-90, elevation=0, distance='auto')  # top, head to R


# %% SAVE A MOVIE OF THE KINEMATICS

#@mlab.animate(delay=100)
#def anim():
#    for i in np.arange(ntime):
#        sxy.mlab_source.set(x=foils_body[i, :, :, 0],
#                            y=foils_body[i, :, :, 1],
#                            z=foils_body[i, :, :, 2],
#                            scalars=foil_color_theta[i])
#        sxz.mlab_source.set(x=foils_body[i, :, :, 0],
#                            y=.18 + foils_body[i, :, :, 2],
#                            z=foils_body[i, :, :, 1],
#                            scalars=foil_color_psi[i])
#        szy.mlab_source.set(x=.2 - foils_body[i, :, :, 2],
#                            y=foils_body[i, :, :, 1],
#                            z=foils_body[i, :, :, 0],
#                            scalars=foil_color_psi[i])
#        mlab.savefig('../Movies/s_serp3d/3views/anim_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 666), figure=fig)
#        yield
#manim = anim()
#mlab.show()


@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            print(i)
            sxy.mlab_source.set(x=foils_body[i, :, :, 0],
                                y=foils_body[i, :, :, 1],
                                z=foils_body[i, :, :, 2],
                                scalars=foil_color_theta[i])
            sxz.mlab_source.set(x=foils_body[i, :, :, 0],
                                y=.18 + foils_body[i, :, :, 2],
                                z=foils_body[i, :, :, 1],
                                scalars=foil_color_psi[i])
            szy.mlab_source.set(x=.2 - foils_body[i, :, :, 2],
                                y=foils_body[i, :, :, 1],
                                z=foils_body[i, :, :, 0],
                                scalars=foil_color_psi[i])
            yield
manim = anim()
mlab.show()


# %% SNAKE IN BODY FRAME WITH NO AXES, TOP VIEW

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# top view
for ii in np.arange(3):
    mlab.quiver3d([0], [0], [0],
                  [Nframe[ii, 0]], [Nframe[ii, 1]], [Nframe[ii, 2]], scale_factor=.035,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

body = mlab.mesh(foils_body[i, :, :, 0], foils_body[i, :, :, 1], foils_body[i, :, :, 2],
                 scalars=foil_color_body[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

fig.scene.isometric_view()
fig.scene.parallel_projection = True
#mlab.orientation_axes()

mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

#@mlab.animate(delay=100)
#def anim():
#    for i in np.arange(ntime):
#        body.mlab_source.set(x=foils_body[i, :, :, 0],
#                             y=foils_body[i, :, :, 1],
#                             z=foils_body[i, :, :, 2],
#                             scalars=foil_color[i])
#        mlab.savefig('../Movies/s_serp3d/overhead_view/z_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
#
#        yield
#manim = anim()
#mlab.show()

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            print(i)
            print(foils_body[i].shape)
            body.mlab_source.set(x=foils_body[i, :, :, 0],
                                 y=foils_body[i, :, :, 1],
                                 z=foils_body[i, :, :, 2],
                                 scalars=foil_color_body[i])
            yield
manim = anim()
mlab.show()


# %% In rotated frame

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = sim.rotate(C0.T, np.eye(3))
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

body = mlab.mesh(foils_iner[i, :, :, 0], foils_iner[i, :, :, 1], foils_iner[i, :, :, 2],
                 scalars=foil_color_iner[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

# in rotated frame
cv_quiv = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
                        Cv[i, :, 0], Cv[i, :, 1], Cv[i, :, 2],
                        color=bmap[0], scale_factor=.025,
                        resolution=64, mode='arrow')
bv_quiv = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
                        Bv[i, :, 0], Bv[i, :, 1], Bv[i, :, 2],
                        color=bmap[3], scale_factor=.025,
                        resolution=64, mode='arrow')

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=foils_iner[i, :, :, 0],
                                 y=foils_iner[i, :, :, 1],
                                 z=foils_iner[i, :, :, 2],
                                 scalars=foil_color_iner[i])
            cv_quiv.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
                                    u=Cv[i, :, 0], v=Cv[i, :, 1], w=Cv[i, :, 2])
            bv_quiv.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
                                    u=Bv[i, :, 0], v=Bv[i, :, 1], w=Bv[i, :, 2])
            yield
manim = anim()
mlab.show()


# %% MOVIE OF TANGENT ANGLE AND KINEMATICS

kwargs_theta = dict(c=theta[0], s=60, cmap=plt.cm.RdBu_r,
                      linewidths=0, zorder=2)

kwargs_psi = dict(c=psi[0], s=60, cmap=plt.cm.PuOr_r,
                      linewidths=0, zorder=1)

#scatter_kwargs = dict(c=s / L, s=60, cmap=plt.cm.viridis,
#                      linewidths=0)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
ax1.axhline(0, color='gray', lw=1, zorder=0)
#ax2.axvline(0, color='gray', lw=1, zorder=0)
#ax2.axhline(0, color='gray', lw=1, zorder=0)
#ax2.axhline(20, color='gray', lw=1, zorder=0)
#ax2.axvline(20, color='gray', lw=1, zorder=0)

gray_kwargs = dict(c='gray', lw=1, zorder=0)
ax2.plot([0, 0], [-15, 15], **gray_kwargs)
ax2.plot([-15, 15], [0, 0], **gray_kwargs)

ax2.plot([0, 0], [17.5, 22.5], **gray_kwargs)
ax2.plot([-15, 15], [20, 20], **gray_kwargs)

ax2.plot([20, 20], [-15, 15], **gray_kwargs)
ax2.plot([17.5, 22.5], [0, 0], **gray_kwargs)

stheta = ax1.scatter(100 * s / L, theta[0], **kwargs_theta)
#ax1.plot(s / L, np.rad2deg(amp_theta), c='gray', lw=1)
#ax1.plot(s / L, -np.rad2deg(amp_theta), c='gray', lw=1)

spsi = ax1.scatter(100 * s / L, psi[0], **kwargs_psi)
#ax1.plot(s / L, np.rad2deg(amp_psi), c='gray', lw=1)
#ax1.plot(s / L, -np.rad2deg(amp_psi), c='gray', lw=1)

ax1.set_xlim(0, 100)
ax1.set_xticks([0, 25, 50, 75, 100])
ax1.xaxis.set_major_formatter(decimal_formatter)
ax1.set_ylim(-105, 105)
ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90])

pi = 100 * p[0]  # convert to cm
sxy = ax2.scatter(pi[:, 0], pi[:, 1], **kwargs_theta)  # top view
sxz = ax2.scatter(pi[:, 0], 20 + pi[:, 2], **kwargs_psi)  # front view
szy = ax2.scatter(20 - pi[:, 2], pi[:, 1], **kwargs_psi)  # side view
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-20, 25)
ax2.set_ylim(-15, 25)
#ax2.set_xticks([])
#ax2.set_yticks([])
ax2.axis('off')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

ax1.set_xlabel('Distance along body (%SVL)')
ax1.set_ylabel('Body angles')
#ax2.set_xlabel('Fore-aft excursion (cm)')
#ax2.set_ylabel('Lateral and Vertical excursions (cm)')

sns.despine()
fig.set_tight_layout(True)

def init():
    stheta.set_offsets(np.c_[[], []])
    stheta.set_array([])

    spsi.set_offsets(np.c_[[], []])
    spsi.set_array([])

    sxy.set_offsets(np.c_[[], []])
    sxy.set_array([])

    sxz.set_offsets(np.c_[[], []])
    sxz.set_array([])

    szy.set_offsets(np.c_[[], []])
    szy.set_array([])

    return stheta, spsi, sxy, sxz, szy


def animate(i):
    pi = 100 * p[i]
    thetai = theta[i]
    psii = psi[i]

    stheta.set_offsets(np.c_[100 * s / L, thetai])
    stheta.set_array(thetai)

    spsi.set_offsets(np.c_[100 * s / L, psii])
    spsi.set_array(psii)

    sxy.set_offsets(np.c_[pi[:, 0], pi[:, 1]])
    sxy.set_array(thetai)

    sxz.set_offsets(np.c_[pi[:, 0], 20 + pi[:, 2]])
    sxz.set_array(psii)

    szy.set_offsets(np.c_[20 - pi[:, 2], pi[:, 1]])
    szy.set_array(psii)

    return stheta, spsi, sxy, sxz, szy


dt_kin_movie = np.diff(ts).mean()

from matplotlib.animation import FuncAnimation

slowed = 5
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt_kin_movie * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False, init_func=init)

save_movie = False
if save_movie:
    #ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

    ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])


# %% Angle sheets

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

surf_psi = mlab.surf(psi, colormap='PuOr', vmin=-10, vmax=10)
surf_theta = mlab.surf(theta, colormap='RdBu', vmin=-90, vmax=90)

surf_psi.module_manager.scalar_lut_manager.reverse_lut = True
surf_theta.module_manager.scalar_lut_manager.reverse_lut = True


# %%

# %% Run a dynamics simulation

#rho = 1.165  # 30 C
##rho = 1.18  # standard 25 C
#g = 9.81
##g = 0
#
## aerodynamics
#aero_interp = m_aerodynamics.extend_wind_tunnel_data(plot=False)
## aero_interp = None
##aero_interp = None
#
tend = None
##tend = 10
#
#
## arguments
##args = (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)
##
##params = (dt, L, nbody, mass_total, rho_body, neck_length, Stot, \
##            wing_loading, wave_length_m, freq_undulation_hz)
#
#args1 = (s, m, n_neck, theta_dict, psi_dict)
#args2 = (ds, c, g, rho, aero_interp)
#args = (args1, args2)

# initial conditions
Ro0 = np.r_[0, 0, 10]
#dRo0 = np.r_[-1.7, 0, 0]
#dRo0 = np.r_[1.7, 0, 0]
dRo0 = np.r_[0, 1.7, 0]
#dRo0 = np.r_[0, 2, -6]
#ang0 = np.deg2rad(np.r_[0, 0, 0])  # yaw, pitch, roll
ang0 = np.deg2rad(np.array([ 17.09433418,  -0.32207885,   1.2865577 ]))  # d_psi=0
#ang0 = np.deg2rad(np.array([ 13.6856538 ,   8.35388744, -12.63686864]))  # d_psi=-10
#ang0 = np.deg2rad(array([ 11.66870808,  -9.06973921,  15.33712671]))
#ang0 = find_rotational_ic(body_dict, print_time=True)
dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
omg0 = np.dot(C0.T, omg0_body)
# omg0 = np.deg2rad(np.r_[0, 0, 0])
soln0 = np.r_[Ro0, dRo0, omg0, ang0]

# implement vestibular control
body_dict['head_control'] = False

#body_dict['aero_interp'] = None
body_dict['aero_interp'] = aero_interp


# %% Run a dynamics simulation

import time

# perform the integration
#out = sim.integrate(soln0, args, dt, tend=tend, print_time=True)
out = sim.integrate(soln0, body_dict, dt, tend=tend, print_time=True)

# extract values
ts, Ro, dRo, omg, ang = out
yaw, pitch, roll = ang.T
ntime = len(ts)


# %%

i = 0

# height ratio: 10 m to 4 m = 2.5 to 1
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                               gridspec_kw = {'height_ratios':[2.5, 1]},
                               figsize=(4.5, 8.9))

ax2.axhline(0, color='gray', lw=1)

ax1.plot(Ro[:, 1], Ro[:, 2])
ax2.plot(Ro[:, 1], Ro[:, 0])

yz_cur, = ax1.plot(Ro[i, 1], Ro[i, 2], 'bo')
yx_cur, = ax2.plot(Ro[i, 1], Ro[i, 0], 'bo')

yz_past, = ax1.plot(Ro[:i, 1], Ro[:i, 2], 'b', lw=3)
yx_past, = ax2.plot(Ro[:i, 1], Ro[:i, 0], 'b', lw=3)

ax1.set_ylim(0, 10)
ax1.set_xlim(0, 6)
ax2.set_ylim(2, -2)  # switch so that x points down, y forward
ax2.set_yticks([2, 0, -2])
ax2.set_xticks([0, 2, 4, 6])

plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')

ax1.set_ylabel('Vertical (m)')
ax2.set_xlabel('Forward (m)')
ax2.set_ylabel('Lateral (m)')

sns.despine()
fig.set_tight_layout(True)


def init():
    yz_cur.set_data(([], []))
    yx_cur.set_data(([], []))

    yz_past.set_data(([], []))
    yx_past.set_data(([], []))

    return yz_cur, yx_cur, yz_past, yx_past


def animate(i):
    yz_cur.set_data((Ro[i, 1], Ro[i, 2]))
    yx_cur.set_data((Ro[i, 1], Ro[i, 0]))

    yz_past.set_data((Ro[:i, 1], Ro[:i, 2]))
    yx_past.set_data((Ro[:i, 1], Ro[:i, 0]))

    return yz_cur, yx_cur, yz_past, yx_past


from matplotlib.animation import FuncAnimation

slowed = 10
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False, init_func=init)

save_movie = False
if save_movie:
    #ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

#    ani.save('../Movies/s_serp3d/sample_glide_projs_10X.mp4',
#             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

#    ani.save('../Movies/s_serp3d/sample_glide_projs_10X_f=0.mp4',
#             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

#    ani.save('../Movies/s_serp3d/sample_glide_projs_10X_f=1.4_new_cdir.mp4',
#             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

    ani.save('../Movies/s_serp3d/sample_glide_projs_10X_f=0_new_cdir.mp4',
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])


# %% Euler angles in non-dimensional time --- Movie

i = 0

# colors for x, y, z (pitch, roll, yaw)
frame_c = [bmap[2], bmap[1], bmap[0]]

ts_non = ts * f_theta

#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4.5))
fig, ax1 = plt.subplots(figsize=(5.5, 6.5))
#fig, ax1 = plt.subplots()

ax1.axhline(0, color='gray', lw=1)

ax1.plot(ts_non, np.rad2deg(pitch), c=frame_c[0])
ax1.plot(ts_non, np.rad2deg(roll), c=frame_c[1])
ax1.plot(ts_non, np.rad2deg(yaw), c=frame_c[2])

# current time
pit_c, = ax1.plot(ts_non[i], np.rad2deg(pitch)[i], 'o', c=frame_c[0])
rol_c, = ax1.plot(ts_non[i], np.rad2deg(roll)[i], 'o', c=frame_c[1])
yaw_c, = ax1.plot(ts_non[i], np.rad2deg(yaw)[i], 'o', c=frame_c[2])

# past history
pit_past, = ax1.plot(ts_non[:i], np.rad2deg(pitch)[:i], lw=4, c=frame_c[0], label='pitch')
rol_past, = ax1.plot(ts_non[:i], np.rad2deg(roll)[:i], lw=4, c=frame_c[1], label='roll')
yaw_past, = ax1.plot(ts_non[:i], np.rad2deg(yaw)[:i], lw=4, c=frame_c[2], label='yaw')


ax1.set_ylim(-90, 90)
ax1.set_yticks(np.arange(-90, 91, 30))
ax1.set_xlim(ts_non[0], ts_non[-1])
ax1.legend(loc='upper left')

ax1.xaxis.set_major_formatter(decimal_formatter)

ax1.set_ylabel('Euler angles (deg)')
ax1.set_xlabel(r'Time (T$_\mathsf{lateral}$)')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)


def animate(i):
    pit_c.set_data((ts_non[i], np.rad2deg(pitch)[i]))
    rol_c.set_data((ts_non[i], np.rad2deg(roll)[i]))
    yaw_c.set_data((ts_non[i], np.rad2deg(yaw)[i]))

    pit_past.set_data((ts_non[:i], np.rad2deg(pitch)[:i]))
    rol_past.set_data((ts_non[:i], np.rad2deg(roll)[:i]))
    yaw_past.set_data((ts_non[:i], np.rad2deg(yaw)[:i]))

    return pit_c, rol_c, yaw_c, pit_past, rol_past, yaw_past


slowed = 5
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False)

save_movie = False
if save_movie:
    #ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

    ani.save('../Movies/s_serp3d/sample_glide_Euler_5X.mp4',
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])


# %%

fig, ax = plt.subplots()
ax.plot(Ro[:, 1], Ro[:, 2])
ax.set_aspect('equal', adjustable='box')
#ax.set_ylim(0, Ro[:, 2].max())
ax.set_ylim(0, 10)
ax.set_xlim(0, 6)
ax.set_xlabel('Forward')
ax.set_ylabel('Vertical')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.plot(Ro[:, 1], Ro[:, 0])
ax.set_aspect('equal', adjustable='box')
ax.set_ylabel('Lateral (m)')
ax.set_xlabel('Forward (m)')
ax.set_ylim(-2, 2)
ax.set_yticks([-2, -1, 0, 1, 2])
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots(figsize=(5.75, 5.5))
ax.plot(dRo[:, 1], dRo[:, 2], '+-', mew=1, markevery=5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %%

# colors for x, y, z (pitch, roll, yaw)
frame_c = [bmap[2], bmap[1], bmap[0]]

ts_non = ts * f_theta

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4.5))

ax1.axhline(0, color='gray', lw=1)
ax2.axhline(0, color='gray', lw=1)

ax1.plot(ts_non, np.rad2deg(pitch), c=frame_c[0], label='pitch')
ax1.plot(ts_non, np.rad2deg(roll), c=frame_c[1], label='roll')
ax1.plot(ts_non, np.rad2deg(yaw), c=frame_c[2], label='yaw')

ax2.plot(ts, np.rad2deg(pitch), c=frame_c[0], label='pitch')
ax2.plot(ts, np.rad2deg(roll), c=frame_c[1], label='roll')
ax2.plot(ts, np.rad2deg(yaw), c=frame_c[2], label='yaw')

ax1.set_ylim(-90, 90)
ax1.set_yticks(np.arange(-90, 91, 30))
ax1.set_xlim(ts_non[0], ts_non[-1])
ax2.set_xlim(ts[0], ts[-1])
ax1.legend(loc='upper left')

ax1.xaxis.set_major_formatter(decimal_formatter)
ax2.xaxis.set_major_formatter(decimal_formatter)

ax1.set_ylabel('Euler angles (deg)')
ax1.set_xlabel(r'Time (t/T$_\mathsf{lateral}$)')
ax2.set_xlabel('Time (sec)')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('sample glide Euler angles'), **FIGOPT)


# %% Euler angles in non-dimensional time

# colors for x, y, z (pitch, roll, yaw)
frame_c = [bmap[2], bmap[1], bmap[0]]

ts_non = ts * f_theta

#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4.5))
fig, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.axvline(1, color='gray', lw=1.5)
ax1.axvline(2, color='gray', lw=1.5)

ax1.axhline(0, color='gray', lw=1)

ax1.plot(ts_non, np.rad2deg(pitch), c=frame_c[0], lw=3, label='pitch')
ax1.plot(ts_non, np.rad2deg(roll), c=frame_c[1], lw=3, label='roll')
ax1.plot(ts_non, np.rad2deg(yaw), c=frame_c[2], lw=3, label='yaw')

ax1.set_ylim(-90, 90)
ax1.set_yticks(np.arange(-90, 91, 30))
#ax1.set_xlim(ts_non[0], ts_non[-1])
ax1.set_xlim(0, 2.786)
ax1.legend(loc='upper left')

ax1.xaxis.set_major_formatter(decimal_formatter)

ax1.set_ylabel('Euler angles (deg)')
ax1.set_xlabel(r'Time (t/T$_\mathsf{lateral}$)')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('sample glide Euler angles Tund f=0 new cdir'), **FIGOPT)


# %%

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
    dR[i] = dRo[i] + dr[i] + cross(omg[i], r[i])

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
    dR_BC[i] = out_aero['dR_BC']
    aoa[i] = out_aero['aoa']
    Re[i] = out_aero['Re']
    dR_T[i] = out_aero['dR_T']

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
foils, foil_color = sim.apply_airfoil_shape(r, c, Crs_I)


# %%

power_tot = power.sum(axis=1)

fig, ax = plt.subplots()
ax.plot(ts_non, 1000 * power_tot)
sns.despine()
fig.set_tight_layout(True)


# %%

Ma_B_tot = 1000 * Ma_B.sum(axis=1)
Ml_B_tot = 1000 * Ml_B.sum(axis=1)
Md_B_tot = 1000 * Md_B.sum(axis=1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(5, 8))
ax1.plot(ts_non, Ml_B_tot[:, 0])
ax2.plot(ts_non, Ml_B_tot[:, 1])
ax3.plot(ts_non, Ml_B_tot[:, 2])

ax1.plot(ts_non, Md_B_tot[:, 0])
ax2.plot(ts_non, Md_B_tot[:, 1])
ax3.plot(ts_non, Md_B_tot[:, 2])

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(5, 8))

ax1.plot(ts_non, Ml_B_tot[:, 0].cumsum())
ax2.plot(ts_non, Ml_B_tot[:, 1].cumsum())
ax3.plot(ts_non, Ml_B_tot[:, 2].cumsum())

ax1.plot(ts_non, Md_B_tot[:, 0].cumsum())
ax2.plot(ts_non, Md_B_tot[:, 1].cumsum())
ax3.plot(ts_non, Md_B_tot[:, 2].cumsum())

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(5, 8))

ax1.plot(ts_non, Ma_B_tot[:, 0])
ax2.plot(ts_non, Ma_B_tot[:, 1])
ax3.plot(ts_non, Ma_B_tot[:, 2])

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

sns.despine()
fig.set_tight_layout(True)


# %%

L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))

Utot = np.zeros((ntime, nbody, 3, 2))
U_BC = np.zeros((ntime, nbody, 3, 2))
U_T = np.zeros((ntime, nbody, 3, 2))
#u = np.zeros((ntime, nbody, 3, 2))
#bc = np.zeros((ntime, nbody, 3, 2))
#tc = np.zeros((ntime, nbody, 3, 2))

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

        Utot[i, j, :, 0] = r[i, j]
        Utot[i, j, :, 1] = r[i, j] + scale_velocities * dR[i, j]

        U_BC[i, j, :, 0] = r[i, j]
        U_BC[i, j, :, 1] = r[i, j] + scale_velocities * dR_BC[i, j]

        U_T[i, j, :, 0] = r[i, j]
        U_T[i, j, :, 1] = r[i, j] + scale_velocities * dR_T[i, j]


# %% MOVIE OF FULL GLIDE

i = 0

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


#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

fc_minmax = foil_color[i].copy()
fc_minmax[-1, -1] = 0
fc_minmax[-1, -2] = 1
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=fc_minmax, colormap='YlGn')

#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1)

# CoM velocity
vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
                     color=(0, 0, 0), mode='arrow', resolution=64)


#sk = 2
#mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
#          Cv[i, ::sk, 0], Cv[i, ::sk, 1], Cv[i, ::sk, 2],
#          color=bmap[1], mode='arrow', resolution=64, scale_factor=.025)
## bhat
#mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
#          Bv[i, ::sk, 0], Bv[i, ::sk, 1], Bv[i, ::sk, 2],
#          color=bmap[0], mode='arrow', resolution=64, scale_factor=.025)

op = .6
ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)

#U_Bt = np.zeros_like(Utot)
#U_Bt[:, :, :, 0] = Utot[:, :, :, 0]
#assert(np.allclose(U_Bt[:, :, :, 0], U_BC[:, :, :, 0]))
#U_Bt[:, :, :, 1] =U_Bt[:, :, :, 0] + ( Utot[:, :, :, 1] - U_BC[:, :, :, 1])
#mlab.mesh(Utot[i, :, 0], Utot[i, :, 1], Utot[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(U_BC[i, :, 0], U_BC[i, :, 1], U_BC[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_Bt[i, :, 0], U_Bt[i, :, 1], U_Bt[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_T[i, :, 0], U_T[i, :, 1], U_T[i, :, 2], color=bmap[0], opacity=.8)

fig.scene.isometric_view()
fig.scene.parallel_projection = True
#mlab.orientation_axes()


if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)
    mlab.view(azimuth=-90, elevation=0, distance='auto')  # top, head to R


# %%

#ffmpeg -f image2 -r 10 -i iso_forces_%03d.png -pix_fmt yuv420p iso_forces_slowed10x.mp4

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils[i, :, :, 0],
                             y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2]),
#                             scalars=foil_color[i])
        ml.mlab_source.set(x=L[i, :, 0],
                           y=L[i, :, 1],
                           z=L[i, :, 2])
        md.mlab_source.set(x=D[i, :, 0],
                           y=D[i, :, 1],
                           z=D[i, :, 2])
        vcom.mlab_source.set(u=[dRo[i, 0]],
                             v=[dRo[i, 1]],
                             w=[dRo[i, 2]])
        for ii in np.arange(3):
            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
                                       v=nframe[i, ii, 1],
                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
#        mlab.savefig('../Movies/s_serp3d/sample_glide_f=0/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
#        mlab.savefig('../Movies/s_serp3d/sample_glide_f=1.4_new_cdir/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        mlab.savefig('../Movies/s_serp3d/sample_glide_f=0_new_cdir/iso_forces_{0:03d}.png'.format(i),
                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()


##@mlab.animate(delay=100)
#def anim():
#    while True:
#        for i in np.arange(ntime):
#            print i
#            body.mlab_source.set(x=foils[i, :, :, 0],
#                                 y=foils[i, :, :, 1],
#                                 z=foils[i, :, :, 2],
#                                 scalars=foil_color[i])
#            ml.mlab_source.set(x=L[i, :, 0],
#                               y=L[i, :, 1],
#                               z=L[i, :, 2])
#            md.mlab_source.set(x=D[i, :, 0],
#                               y=D[i, :, 1],
#                               z=D[i, :, 2])
#            vcom.mlab_source.set(u=[dRo[i, 0]],
#                                 v=[dRo[i, 1]],
#                                 w=[dRo[i, 2]])
#            for ii in np.arange(3):
#                bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
#                                           v=nframe[i, ii, 1],
#                                           w=nframe[i, ii, 2])
##            cv_quiv.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
##                                    u=Cv[i, :, 0], v=Cv[i, :, 1], w=Cv[i, :, 2])
##            bv_quiv.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
##                                    u=Bv[i, :, 0], v=Bv[i, :, 1], w=Bv[i, :, 2])
#            yield
#manim = anim()
#mlab.show()


# %% BODY FOR EULER ANGLES

i = 0

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

fig.scene.isometric_view()
fig.scene.parallel_projection = True

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)
    mlab.view(azimuth=-90, elevation=0, distance='auto')  # top, head to R


# %%

#ffmpeg -f image2 -r 10 -i iso_forces_%03d.png -pix_fmt yuv420p iso_forces_slowed5x.mp4

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils[i, :, :, 0],
                             y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2],
                             scalars=foil_color[i])
        for ii in np.arange(3):
            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
                                       v=nframe[i, ii, 1],
                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_clean_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()


# %%

#    dang[i] = angi
#    ddang[i] = sim.ddang_ddt(ang[i], dang[i], omg[i], domg[i], C[i])


# %% BODY AND FORCES FOR DYNAMICS SCHEMATIC

i = 46

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# CoM and body axes
#frame_c = [bmap[2], bmap[1], bmap[0]]
#bframe = []
#for ii in np.arange(3):
#    frm = mlab.quiver3d(nframe[i, ii, 0], nframe[i, ii, 1], nframe[i, ii, 2],
#                        scale_factor=.05, color=frame_c[ii], mode='arrow',
#                        opacity=1, resolution=64)
#    bframe.append(frm)

#for ii in np.arange(3):
#    frm = mlab.quiver3d([.1], [-.05], [-.1],
#                        [Nframe[ii, 0]], [Nframe[ii, 1]], [Nframe[ii, 2]],
#                        scale_factor=.05, color=frame_c[ii], mode='arrow',
#                        opacity=1, resolution=64)

#mlab.quiver3d([.1], [-.05], [-.1],
#              [0], [0], [0],
#              scale_factor=.05, color=(.7, .7, .7), mode='arrow',
#              opacity=1, resolution=64)

#              [-.1, 0, 0], [0, .05, 0], [0, 0, .1],

## inertial axies
#_args = dict(opacity=.75, tube_radius=.001)
#mlab.plot3d([-.2, .2], [0, 0], [0, 0], color=frame_c[0], **_args)
#mlab.plot3d([0, 0], [-.2, .2], [0, 0], color=frame_c[1],**_args)
#mlab.plot3d([0, 0], [0, 0], [-.075, .075], color=frame_c[2], **_args)


body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

# CoM velocity
#vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
#                     color=(0, 0, 0), mode='arrow', resolution=64)

# angular velocity
#vcom = mlab.quiver3d([omg[i, 0]], [omg[i, 1]], [omg[i, 2]], scale_factor=.02,
#                     color=bmap[3], mode='arrow', resolution=64)

sk = 1
mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
          Cv[i, ::sk, 0], Cv[i, ::sk, 1], Cv[i, ::sk, 2],
          color=frame_c[0], mode='arrow', resolution=64, scale_factor=.025)
# bhat
mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
          Bv[i, ::sk, 0], Bv[i, ::sk, 1], Bv[i, ::sk, 2],
          color=frame_c[2], mode='arrow', resolution=64, scale_factor=.025)

op = .6
#ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
#md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)

#U_Bt = np.zeros_like(Utot)
#U_Bt[:, :, :, 0] = Utot[:, :, :, 0]
#assert(np.allclose(U_Bt[:, :, :, 0], U_BC[:, :, :, 0]))
#U_Bt[:, :, :, 1] =U_Bt[:, :, :, 0] + ( Utot[:, :, :, 1] - U_BC[:, :, :, 1])
#mlab.mesh(Utot[i, :, 0], Utot[i, :, 1], Utot[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(U_BC[i, :, 0], U_BC[i, :, 1], U_BC[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_Bt[i, :, 0], U_Bt[i, :, 1], U_Bt[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_T[i, :, 0], U_T[i, :, 1], U_T[i, :, 2], color=bmap[0], opacity=.8)

fig.scene.isometric_view()
fig.scene.parallel_projection = True
#mlab.orientation_axes()


# %%

#mlab.savefig('../Figures/s_serp3d/sample_glide_iso_body.png'.format(i),
#                     size=(8 * 750, 8 * 708), figure=fig)

#mlab.savefig('../Figures/s_serp3d/sample_glide_iso_forces.png'.format(i),
#                     size=(8 * 750, 8 * 708), figure=fig)

#mlab.savefig('../Figures/s_serp3d/sample_glide_iso_clean.png'.format(i),
#                     size=(4 * 750, 4 * 546), figure=fig)

mlab.savefig('../Figures/s_serp3d/sample_glide_iso_tcb.png'.format(i),
                     size=(4 * 795, 4 * 479), figure=fig)


# %%

#%load_ext line_profiler

#%lprun -f integrate -f dynamics -f aerialserp_pos_vel_acc_tcb -f serp3d_tcb  -f aerialserp -f sim.aero_forces -f aero_interp -s -T profile_2016-07-05.txt integrate(soln0, args, dt, tend=.1, print_time=True)




# %%

# %% Find fixed point in the velocity field

def find_equil_amag_to_zero(vguess, args):
    """Called by fixed_point.
    """

    arg, one_step = args

    # non-dimensional velocities
    vy, vz = vguess

    # unpack needed variables
    s, m, n_neck, g = arg['s'], arg['m'], arg['n_neck'], arg['g']
    theta_dict, psi_dict = arg['theta_dict'], arg['psi_dict']
    vscale, weight, rho = arg['vscale'], arg['weight'], arg['rho']
    ds, c, aero_interp = arg['ds'], arg['c'], arg['aero_interp']

    # undulation frequency
    f_theta = theta_dict['f_theta']

    # where to evaluate the undulation at
    ntime = 10
    ts = np.linspace(0, 1 / f_theta, ntime + 1)[:-1]

    # speed-up the calculation by evaluation forces at one time step
    if one_step:
        ntime = 1
        ts = np.linspace(0, 0, ntime)

    # trivial dynamics variables
    C = np.eye(3)
    omg = np.r_[0, 0, 0]

    # convert non-dim to dimensional for simulation
    dRo_unitless = np.r_[0, vy, vz]
    dRo = vscale * dRo_unitless

    F = np.zeros((ntime, 3))
    for i in np.arange(ntime):
        t = ts[i]

        # kinematics
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

        F[i] = Fi.sum(axis=0)  # integrate forces on the body

    # calculate forces
    F_avg = F.mean(axis=0) / weight  # normalized 1 cycle average force
    F_mag = np.sqrt(np.sum(F_avg**2))

    return F_mag


def find_equilibrium(body_dict, print_time=False):
    """Find the equilibrium points.
    """

    from scipy.optimize import root, minimize, fmin

    # perform minimization at one body position
    vguess_non_init = np.r_[1, -.4]
    one_step = True
    args = (body_dict, one_step)

    now = time.time()
    sol_init = minimize(find_equil_amag_to_zero, vguess_non_init, args=(args,))
    #print sol_init.x
    if print_time:
        print('Elapsed time one_step: {0:.3f} sec'.format(time.time() - now))

    # now perform the cycle average
    vguess_non = sol_init.x
    one_step = False
    args = (body_dict, one_step)

    now = time.time()
    sol = minimize(find_equil_amag_to_zero, vguess_non, args=(args,))
    #print sol.x
    if print_time:
        print('Elapsed time cycle avg: {0:.3f} sec'.format(time.time() - now))

    # velocity and glide angle at equilibrium
    veq = np.r_[0, sol.x[0], sol.x[1]]
    vmag_eq = np.sqrt(np.sum(veq**2))
    gam_eq = -np.arctan(veq[2] / veq[1])

    return veq, vmag_eq, gam_eq


# %%

veq, vmag_eq, gam_eq = find_equilibrium(body_dict, print_time=True)



# %% Integrate one trajectory through average VPD

def cycle_avg_dynamics_eom(tintegrator, state, body_dict, one_step=False):
    """Called by fixed_point.
    """

    # non-dimensional velocities
    y, z, vy, vz = state

    # unpack needed variables
    s, m, n_neck, g = body_dict['s'], body_dict['m'], body_dict['n_neck'], body_dict['g']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    vscale, weight, rho = body_dict['vscale'], body_dict['weight'], body_dict['rho']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']

    # undulation frequency
    f_theta = theta_dict['f_theta']

    # where to evaluate the undulation at
    ntime = 10
    ts = np.linspace(0, 1 / f_theta, ntime + 1)[:-1]

    if one_step:
        ntime = 1
        ts = np.linspace(0, 0, ntime)

    # trivial dynamics variables
    C = np.eye(3)
    omg = np.r_[0, 0, 0]

    # CoM velocity
    dRo = np.r_[0, vy, vz]

    F = np.zeros((ntime, 3))
    for i in np.arange(ntime):
        t = ts[i]

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



def cycle_avg_dynamics(body_dict, veq, dt=.05, one_step=False, print_time=False):

    from scipy.integrate import ode

    vscale = body_dict['vscale']

    now = time.time()

    #v0_non = 1.7 / np.sqrt(2 * 29 / rho)  # .2409, Ws=29
    #soln0 = np.r_[0, 0, vscale * v0_non, 0]

    #TODO
    # maybe we just want to set all snake sizes to 1.7 m/s
    soln0 = np.r_[0, 0, 1.7, 0]

    # equilibrium velocity in physical units
    vy_eq, vz_eq = vscale * veq[1], vscale * veq[2]

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

        vy, vz = out[2], out[3]

        # within 1% of the equilibrium configuration
        at_equil_y = np.abs((vy - vy_eq) / vy_eq) * 100 < .1
        at_equil_z = np.abs((vz - vz_eq) / vz_eq) * 100 < .1

        at_equil = at_equil_y and at_equil_z

        #print solver.t

        soln.append(out)
        ts.append(solver.t)

    if print_time:
        print('Cycle avg dynamics time: {0:.3f} sec'.format(time.time() - now))

    soln_cyc = np.array(soln)
    ts_cyc = np.array(ts)

    return soln_cyc, ts_cyc


# %% Trajectory through VPD, ignoring rotational dynamics

soln_cyc, ts_cyc = cycle_avg_dynamics(body_dict, veq, dt=.05, one_step=True, print_time=True)

# non-dimensional time by undulation frequency
ts_cyc_non = ts_cyc * f_theta


# %% Extract interrogation points along the trajectory

ntime_cyc = len(ts_cyc)

vy_cyc, vz_cyc = soln_cyc[:, 2:].T / vscale
y_cyc, z_cyc = soln_cyc[:, :2].T
vmag_cyc = np.sqrt(vy_cyc**2 + vz_cyc**2)
gam_cyc = -np.arctan(vz_cyc / vy_cyc)

idx_trans = vz_cyc.argmin()
idx_trans2 = gam_cyc.argmax()
idx_ball = np.argmin(np.abs(vz_cyc - vz_cyc[idx_trans] / 2))
idx_z10 = np.where(z_cyc > -10)[0][-1]

#vy_shallow = np.mean([vy[idx_trans], veq[1]])
#vz_shallow = np.mean([vz[idx_trans], veq[2]])
#v_shallow = np.sqrt(vy_shallow**2 + vz_shallow**2)
#idx_shallow = np.abs(v_shallow - vmag).argmin()

idx_vz = np.where(vz_cyc < vz_cyc[-1])[0]
gam_shallow = (gam_eq + gam_cyc[idx_trans]) / 2
idx_shallow0 = np.argmin(np.abs(gam_cyc[idx_vz] - gam_shallow))
idx_shallow = idx_vz[0] + idx_shallow0

# rescaled velocities at the interrogation points
vstart = np.r_[0, vy_cyc[0], vz_cyc[0]]
vball = np.r_[0, vy_cyc[idx_ball], vz_cyc[idx_ball]]
vtrans = np.r_[0, vy_cyc[idx_trans], vz_cyc[idx_trans]]
vlanding = np.r_[0, vy_cyc[idx_z10], vz_cyc[idx_z10]]
vshallow = np.r_[0, vy_cyc[idx_shallow], vz_cyc[idx_shallow]]

# velocities of interest
vinter = np.c_[vstart, vball, vtrans, vshallow, veq, vlanding].T
idx_inter = np.r_[0, idx_ball, idx_trans, idx_shallow, -1, idx_z10]


# %% Plot the cycle averaged dynamics

fig, ax = plt.subplots()
glide_traj = ax.plot(y_cyc, z_cyc)
add_arrow_to_line2D(ax, glide_traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
ax.plot(y_cyc[idx_z10], z_cyc[idx_z10], 'ro')
ax.set_aspect('equal')
sns.despine()
fig.set_tight_layout(True)
fig.savefig(FIG.format('y_cyc vs. z_cyc'), **FIGOPT)


fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(y_cyc, z_cyc)
ax.plot(y_cyc[idx_ball], z_cyc[idx_ball], 'bo', ms=10)
ax.plot(y_cyc[idx_trans], z_cyc[idx_trans], 'go', ms=10)
ax.plot(y_cyc[idx_z10], z_cyc[idx_z10], 'ms', ms=10)
ax.plot(y_cyc[idx_shallow], z_cyc[idx_shallow], 'ro', ms=10)
ax.plot(y_cyc[-1], z_cyc[-1], 'ko', ms=10)
ax.set_aspect('equal')
ax.set_ylim(z_cyc.min(), z_cyc.max())
sns.despine()
fig.set_tight_layout(True)
fig.savefig(FIG.format('y_cyc vs. z_cyc markers'), **FIGOPT)


# %% Y vs. Z until landing

#fig, ax = plt.subplots(figsize=(6, 4.5))
fig, ax = plt.subplots(figsize=(3.4, 3.84))
ax.plot(y_cyc[:idx_z10+2], 10 + z_cyc[:idx_z10+2])
ax.set_aspect('equal')
ax.set_xlim(0, 7.5)
ax.set_ylim(0, 10)
ax.set_xlabel('Forward (m)')
ax.set_ylabel('Vertical (m)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig(FIG.format('y_cyc vs. z_cyc landing'), **FIGOPT)

#fig, ax = plt.subplots(figsize=(6, 4.5))
fig, ax = plt.subplots(figsize=(3.4, 3.84))
ax.plot(y_cyc[:idx_z10+2], 10 + z_cyc[:idx_z10+2])
ax.plot(y_cyc[idx_ball], 10 + z_cyc[idx_ball], 'ko', ms=10)
ax.plot(y_cyc[idx_trans], 10 + z_cyc[idx_trans], 'mo', ms=10)
ax.plot(y_cyc[idx_z10], 10 + z_cyc[idx_z10], 'yo', ms=10)
ax.plot(y_cyc[idx_shallow], 10 + z_cyc[idx_shallow], 'co', ms=10)
#ax.plot(y_cyc[-1], z_cyc[-1], 'ko', ms=10)
ax.set_aspect('equal')
ax.set_xlim(0, 7.5)
ax.set_ylim(0, 10)
ax.set_xlabel('Forward (m)')
ax.set_ylabel('Vertical (m)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig(FIG.format('y_cyc vs. z_cyc markers landing'), **FIGOPT)


# %%


fig, ax = plt.subplots(figsize=(6, 5.5))
traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5, markevery=10)
add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'ko', ms=10)
ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'mo', ms=10)
ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'yo', ms=10)
ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'co', ms=10)
ax.plot(veq[1], veq[2], 'o', c='gray', ms=10)

ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)
fig.savefig(FIG.format('vy_cyc vs. vz_cyc'), **FIGOPT)


# %%

fig, ax = plt.subplots(figsize=(6, 5.5))
traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5, markevery=10)
add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
#ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'ko', ms=10)
#ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'mo', ms=10)
#ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'yo', ms=10)
#ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'co', ms=10)
#ax.plot(veq[1], veq[2], 'o', c='gray', ms=10)

ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)
fig.savefig(FIG.format('vy_cyc vs. vz_cyc no markers'), **FIGOPT)



# %%

fig, ax = plt.subplots(figsize=(6, 5.5))
traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5)
add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'ko', ms=10)
ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'mo', ms=10)
ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'yo', ms=10)
ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'co', ms=10)
ax.plot(vy_cyc, vz_cyc, 'x', ms=10, mew=2, mec='k', markevery=10)

ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)
fig.savefig(FIG.format('vy_cyc vs. vz_cyc markers'), **FIGOPT)


# %% Cycle averaged moments along the trajectory

# phases to evaluate the rotational moments
ntime_cyc_mom = 50
ts_cyc_mom = np.linspace(0, 1 / f_theta, ntime_cyc_mom + 1)[:-1]

# non-dimensional time through cycle
ts_cyc_mom_non = ts_cyc_mom * f_theta

dRo = np.c_[np.zeros_like(vy_cyc), vy_cyc, vz_cyc] * vscale
C = np.eye(3)
omg = np.r_[0, 0, 0]

npnt3 = np.zeros((ntime_cyc, ntime_cyc_mom, 3))
npntnb3 = np.zeros((ntime_cyc, ntime_cyc_mom, nbody, 3))

M_traj_iner = npntnb3.copy()
M_traj_body = npntnb3.copy()
M_traj_iner_tot = npnt3.copy()
M_traj_body_tot = npnt3.copy()

F_traj_iner = npntnb3.copy()
F_traj_body = npntnb3.copy()
F_traj_iner_tot = npnt3.copy()
F_traj_body_tot = npnt3.copy()

dho_traj_iner = npntnb3.copy()
dho_traj_body = npntnb3.copy()
dho_traj_iner_tot = npnt3.copy()
dho_traj_body_tot = npnt3.copy()

p = npntnb3.copy()
dp = npntnb3.copy()
ddp = npntnb3.copy()

now = time.time()

# j is index through trajectory
for j in np.arange(ntime_cyc):

    # i is index through cycle
    for i in np.arange(ntime_cyc_mom):

        # time through cycle
        t = ts_cyc_mom[i]

        out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

        pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
        tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

        p[j, i] = pi
        dp[j, i] = dpi
        ddp[j, i] = ddpi

        # positions, velocities, and accelerations
        ri, dri, ddri = sim.rotate(C.T, pi), sim.rotate(C.T, dpi), sim.rotate(C.T, ddpi)
        dRi = dRo[j] + dri + np.cross(omg, ri)

        # ho_dot
        dhoi_iner = np.cross((m * ri.T).T, ddri)
        dhoi_body = np.cross((m * pi.T).T, ddpi)

        dho_traj_iner[j, i] = dhoi_iner
        dho_traj_body[j, i] = dhoi_body
        dho_traj_iner_tot[j, i] = dhoi_iner.sum(axis=0)
        dho_traj_body_tot[j, i] = dhoi_body.sum(axis=0)

        # aerodynamic forces
        aout = sim.aero_forces(tvi, cvi, bvi, C, dRi, ds, c, rho,
                               aero_interp, full_out=True)

        # aerodynamic force
        Fi_iner = aout['Fa']
        Fi_body = sim.rotate(C, Fi_iner)

        F_traj_iner[j, i] = Fi_iner
        F_traj_body[j, i] = Fi_body
        F_traj_iner_tot[j, i] = Fi_iner.sum(axis=0)
        F_traj_body_tot[j, i] = Fi_body.sum(axis=0)

        # aerodynamic moments
        Mi_iner = sim.cross(ri, Fi_iner)
        Mi_body = sim.rotate(C, Mi_iner)

        M_traj_iner[j, i] = Mi_iner
        M_traj_body[j, i] = Mi_body
        M_traj_iner_tot[j, i] = Mi_iner.sum(axis=0)
        M_traj_body_tot[j, i] = Mi_body.sum(axis=0)

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

# convert to Nmm
M_traj_body_tot *= 1000
dho_traj_body_tot *= 1000

# average over the cycle
M_traj_iner_avg = M_traj_iner_tot.mean(axis=1)
M_traj_body_avg = M_traj_body_tot.mean(axis=1)
F_traj_iner_avg = F_traj_iner_tot.mean(axis=1)
F_traj_body_avg = F_traj_body_tot.mean(axis=1)
dho_traj_iner_avg = dho_traj_iner_tot.mean(axis=1)
dho_traj_body_avg = dho_traj_body_tot.mean(axis=1)

M_traj_body_max = M_traj_body_tot.max(axis=1)
M_traj_body_min = M_traj_body_tot.min(axis=1)
M_traj_body_ptp = M_traj_body_tot.ptp(axis=1)
M_traj_body_std = M_traj_body_tot.std(axis=1)
M_traj_body_rms = np.sqrt(np.mean(M_traj_body_tot**2, axis=1))

dho_traj_body_max = dho_traj_body_tot.max(axis=1)
dho_traj_body_min = dho_traj_body_tot.min(axis=1)
dho_traj_body_ptp = dho_traj_body_tot.ptp(axis=1)
dho_traj_body_std = dho_traj_body_tot.std(axis=1)
dho_traj_body_rms = np.sqrt(np.mean(dho_traj_body_tot**2, axis=1))


# %% EVALUATE THE MOMENTS AT ONE PHASE

# time points to evaluate the moments
ntime_cyc_many = 1000
ts_cyc_many = np.linspace(ts_cyc[0], ts_cyc[-1], ntime_cyc_many)

# non-dimensionalize the time
ts_cyc_many_non = ts_cyc_many * f_theta

M_traj_body_cyc = np.zeros((ntime_cyc_many, nbody, 3))
dho_traj_body_cyc = np.zeros((ntime_cyc_many, nbody, 3))
M_traj_body_cyc_tot = np.zeros((ntime_cyc_many, 3))
dho_traj_body_cyc_tot = np.zeros((ntime_cyc_many, 3))

#p = np.zeros((ntime_mom, nbody, 3))
#dp = np.zeros((ntime_mom, nbody, 3))
#ddp = np.zeros((ntime_mom, nbody, 3))

# interpolate the velocity
dRo_x_many = np.interp(ts_cyc_many, ts_cyc, dRo[:, 0])
dRo_y_many = np.interp(ts_cyc_many, ts_cyc, dRo[:, 1])
dRo_z_many = np.interp(ts_cyc_many, ts_cyc, dRo[:, 2])
dRo_many = np.c_[dRo_x_many, dRo_y_many, dRo_z_many]

now = time.time()

# j is index through trajectory
for j in np.arange(ntime_cyc_many):

    t = ts_cyc_many[j]

    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
    tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

    # positions, velocities, and accelerations
    ri, dri, ddri = sim.rotate(C.T, pi), sim.rotate(C.T, dpi), sim.rotate(C.T, ddpi)
    dRi = dRo_many[j] + dri + np.cross(omg, ri)

    # ho_dot
    dhoi_body = np.cross((m * pi.T).T, ddpi)
    dho_traj_body_cyc[j] = dhoi_body
    dho_traj_body_cyc_tot[j] = dhoi_body.sum(axis=0)

    # aerodynamic forces
    aout = sim.aero_forces(tvi, cvi, bvi, C, dRi, ds, c, rho,
                           aero_interp, full_out=True)

    Mi_body = sim.cross(pi, sim.rotate(C.T, aout['Fa']))
    M_traj_body_cyc[j] = Mi_body
    M_traj_body_cyc_tot[j] = Mi_body.sum(axis=0)

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %% Convert to Nmm by multiplying by 1000

dho_traj_body_cyc_tot *= 1000
M_traj_body_cyc_tot *= 1000


# %% Aerodynamic moments through the trajectory

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_cyc, M_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.plot(ts_cyc, M_traj_body_avg[:, 0] + M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1] + M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2] + M_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc, M_traj_body_avg[:, 0] - M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1] - M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2] - M_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc, M_traj_body_max[:, 0], 'k')
ax2.plot(ts_cyc, M_traj_body_max[:, 1], 'k')
ax3.plot(ts_cyc, M_traj_body_max[:, 2], 'k')

ax1.plot(ts_cyc, M_traj_body_min[:, 0], 'k')
ax2.plot(ts_cyc, M_traj_body_min[:, 1], 'k')
ax3.plot(ts_cyc, M_traj_body_min[:, 2], 'k')

ax1.plot(ts_cyc_many, M_traj_body_cyc_tot[:, 0])
ax2.plot(ts_cyc_many, M_traj_body_cyc_tot[:, 1])
ax3.plot(ts_cyc_many, M_traj_body_cyc_tot[:, 2])

sns.despine()
fig.set_tight_layout(True)


# %% Aerodynamic moments through the trajectory

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_cyc_non, M_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(ts_cyc_non, M_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(ts_cyc_non, M_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.plot(ts_cyc_non, M_traj_body_avg[:, 0] + M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc_non, M_traj_body_avg[:, 1] + M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc_non, M_traj_body_avg[:, 2] + M_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc_non, M_traj_body_avg[:, 0] - M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc_non, M_traj_body_avg[:, 1] - M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc_non, M_traj_body_avg[:, 2] - M_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc_non, M_traj_body_max[:, 0], 'k')
ax2.plot(ts_cyc_non, M_traj_body_max[:, 1], 'k')
ax3.plot(ts_cyc_non, M_traj_body_max[:, 2], 'k')

ax1.plot(ts_cyc_non, M_traj_body_min[:, 0], 'k')
ax2.plot(ts_cyc_non, M_traj_body_min[:, 1], 'k')
ax3.plot(ts_cyc_non, M_traj_body_min[:, 2], 'k')

ax1.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 0])
ax2.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 1])
ax3.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 2])

sns.despine()
fig.set_tight_layout(True)


# %% Aerodynamic moments through the trajectory

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8.625, 8))
#                                    figsize=(12, 9))

(ax1, ax2, ax3) = axs[:, 0]
(ax4, ax5, ax6) = axs[:, 1]

for ax in axs.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(ts_cyc_non[idx_ball], color='k', lw=1.5)
    ax.axvline(ts_cyc_non[idx_trans], color='m', lw=1.5)
    ax.axvline(ts_cyc_non[idx_z10], color='y', lw=1.5)
    ax.axvline(ts_cyc_non[idx_shallow], color='c', lw=1.5)

ax1.fill_between(ts_cyc_non, M_traj_body_max[:, 0], M_traj_body_min[:, 0],
                 alpha=.25, color='gray')
ax2.fill_between(ts_cyc_non, M_traj_body_max[:, 1], M_traj_body_min[:, 1],
                 alpha=.25, color='gray')
ax3.fill_between(ts_cyc_non, M_traj_body_max[:, 2], M_traj_body_min[:, 2],
                 alpha=.25, color='gray')

ax4.fill_between(ts_cyc_non, dho_traj_body_max[:, 0], dho_traj_body_min[:, 0],
                 alpha=.25, color='gray')
ax5.fill_between(ts_cyc_non, dho_traj_body_max[:, 1], dho_traj_body_min[:, 1],
                 alpha=.25, color='gray')
ax6.fill_between(ts_cyc_non, dho_traj_body_max[:, 2], dho_traj_body_min[:, 2],
                 alpha=.25, color='gray')

ax1.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 0], c=frame_c[0])
ax2.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 1], c=frame_c[1])
ax3.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 2], c=frame_c[2])

ax4.plot(ts_cyc_many_non, dho_traj_body_cyc_tot[:, 0], c=frame_c[0])
ax5.plot(ts_cyc_many_non, dho_traj_body_cyc_tot[:, 1], c=frame_c[1])
ax6.plot(ts_cyc_many_non, dho_traj_body_cyc_tot[:, 2], c=frame_c[2])

ax1.plot(ts_cyc_non, M_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(ts_cyc_non, M_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(ts_cyc_non, M_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.set_xlim(0, ts_cyc_non[idx_z10])
ax1.set_ylim(-11, 11)
ax1.set_xticks([0, 1, 2, 3])
ax3.set_xlabel('Time (t/T$_\mathsf{lateral}$)')
ax6.set_xlabel('Time (t/T$_\mathsf{lateral}$)')
ax1.set_ylabel('Pitch moment')
ax2.set_ylabel('Roll moment')
ax3.set_ylabel('Yaw moment')

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('Mo dho to landing'), **FIGOPT)


# %% Aerodynamic moments through the trajectory

frame_c = [bmap[2], bmap[1], bmap[0]]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(5, 8))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(ts_cyc_non[idx_ball], color='k', lw=1.5)
    ax.axvline(ts_cyc_non[idx_trans], color='m', lw=1.5)
    ax.axvline(ts_cyc_non[idx_z10], color='y', lw=1.5)
    ax.axvline(ts_cyc_non[idx_shallow], color='c', lw=1.5)

#ax1.fill_between(ts_cyc_non, M_traj_body_max[:, 0], M_traj_body_min[:, 0],
#                 alpha=.25, color='gray')
#ax2.fill_between(ts_cyc_non, M_traj_body_max[:, 1], M_traj_body_min[:, 1],
#                 alpha=.25, color='gray')
#ax3.fill_between(ts_cyc_non, M_traj_body_max[:, 2], M_traj_body_min[:, 2],
#                 alpha=.25, color='gray')

#ax1.plot(ts_cyc_non, M_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
#ax2.plot(ts_cyc_non, M_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
#ax3.plot(ts_cyc_non, M_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

#ax1.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 0], c=frame_c[0])
#ax2.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 1], c=frame_c[1])
#ax3.plot(ts_cyc_many_non, M_traj_body_cyc_tot[:, 2], c=frame_c[2])

ax1.set_xlim(0, ts_cyc_non[idx_z10])
ax1.set_ylim(-5, 5)
ax1.set_xticks([0, 1, 2, 3])
ax3.set_xlabel('Time (t/T$_\mathsf{lateral}$)')
ax1.set_ylabel('Pitch moment')
ax2.set_ylabel('Roll moment')
ax3.set_ylabel('Yaw moment')

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('Mo to landing - 1'), **FIGOPT)
#fig.savefig(FIG.format('Mo to landing - 2'), **FIGOPT)
#fig.savefig(FIG.format('Mo to landing - 3'), **FIGOPT)
#fig.savefig(FIG.format('Mo to landing'), **FIGOPT)


# %% Froude number TODO

Fr = dho_traj_body_ptp / M_traj_body_ptp

fig, ax = plt.subplots(figsize=(5.5, 4))

ax.axhline(1, color='gray', lw=1)
ax.axvline(ts_cyc_non[idx_ball], color='k', lw=1.5)
ax.axvline(ts_cyc_non[idx_trans], color='m', lw=1.5)
ax.axvline(ts_cyc_non[idx_z10], color='y', lw=1.5)
ax.axvline(ts_cyc_non[idx_shallow], color='c', lw=1.5)

# ax.semilogy(ts_cyc_non, dho_traj_body_std / M_traj_body_std)
ax.semilogy(ts_cyc_non, Fr[:, 0], c=frame_c[0], lw=3, label='pitch axis')
ax.semilogy(ts_cyc_non, Fr[:, 1], c=frame_c[1], lw=3, label='roll axis')
ax.semilogy(ts_cyc_non, Fr[:, 2], c=frame_c[2], lw=3, label='yaw axis')
ax.legend(loc='upper right', frameon=True)
ax.set_xlabel('Time (t/T$_\mathsf{lateral}$)')
ax.set_ylabel('Froude number')
sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('Froude number full'), **FIGOPT)


# %%

#fig, ax = plt.subplots(figsize=(6.5, 5))
fig, ax = plt.subplots(figsize=(5.5, 4))

ax.axhline(1, color='gray', lw=1)
ax.axvline(ts_cyc_non[idx_ball], color='k', lw=1.5)
ax.axvline(ts_cyc_non[idx_trans], color='m', lw=1.5)
ax.axvline(ts_cyc_non[idx_z10], color='y', lw=1.5)
ax.axvline(ts_cyc_non[idx_shallow], color='c', lw=1.5)
# ax.semilogy(ts_cyc_non, dho_traj_body_std / M_traj_body_std)
ax.semilogy(ts_cyc_non, Fr[:, 0], c=frame_c[0], lw=3, label='pitch axis')
ax.semilogy(ts_cyc_non, Fr[:, 1], c=frame_c[1], lw=3, label='roll axis')
ax.semilogy(ts_cyc_non, Fr[:, 2], c=frame_c[2], lw=3, label='yaw axis')
ax.legend(loc='upper right', frameon=True)
ax.set_xlim(0, ts_cyc_non[idx_z10])
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel('Time (t/T$_\mathsf{lateral}$)')
ax.set_ylabel('Froude number')
sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('Froude number'), **FIGOPT)


# %%

fig, ax = plt.subplots()
ax.axhline(1, color='gray', lw=1)
for ii in idx_inter:
#    ax.axvline(ts_cyc[ii], color='gray', lw=1)
    ax.axvline(ts_cyc_non[ii], color='gray', lw=1)

ax.semilogy(ts_cyc_non, dho_traj_body_ptp / M_traj_body_ptp)
#ax.plot(ts_cyc_many_non, dho_traj_body_cyc_tot / M_traj_body_cyc_tot)
#ax.semilogy(ts_cyc_many_non, np.abs(dho_traj_body_cyc_tot / M_traj_body_cyc_tot))

sns.despine()
fig.set_tight_layout(True)

# %% Aerodynamic moments through the trajectory

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_cyc, M_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.plot(ts_cyc, M_traj_body_avg[:, 0] + M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1] + M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2] + M_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc, M_traj_body_avg[:, 0] - M_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, M_traj_body_avg[:, 1] - M_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, M_traj_body_avg[:, 2] - M_traj_body_std[:, 2], 'gray')

for i in np.arange(ntime_cyc):
    ts_plot = np.ones(ntime_cyc_mom) * ts_cyc[i]
    ax1.scatter(ts_plot, M_traj_body_tot[i, :, 0],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(ts_plot, M_traj_body_tot[i, :, 1],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(ts_plot, M_traj_body_tot[i, :, 2],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %% Inertial moments through the trajectory

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_cyc, dho_traj_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(ts_cyc, dho_traj_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(ts_cyc, dho_traj_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.plot(ts_cyc, dho_traj_body_avg[:, 0] + dho_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, dho_traj_body_avg[:, 1] + dho_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, dho_traj_body_avg[:, 2] + dho_traj_body_std[:, 2], 'gray')

ax1.plot(ts_cyc, dho_traj_body_avg[:, 0] - dho_traj_body_std[:, 0], 'gray')
ax2.plot(ts_cyc, dho_traj_body_avg[:, 1] - dho_traj_body_std[:, 1], 'gray')
ax3.plot(ts_cyc, dho_traj_body_avg[:, 2] - dho_traj_body_std[:, 2], 'gray')

for i in np.arange(ntime_cyc):
    ts_plot = np.ones(ntime_cyc_mom) * ts_cyc[i]
    ax1.scatter(ts_plot, dho_traj_body_tot[i, :, 0],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(ts_plot, dho_traj_body_tot[i, :, 1],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(ts_plot, dho_traj_body_tot[i, :, 2],
               c=ts_cyc_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)

# %% dho vs. time

ts_non = ts_mom * f_theta

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_non, dho_body_tot[0, :, 0], 'k', label=r'$h_\mathsf{o,x}$')
ax2.plot(ts_non, dho_body_tot[0, :, 1], 'k', label=r'$h_\mathsf{o,y}$')
ax3.plot(ts_non, dho_body_tot[0, :, 2], 'k', label=r'$h_\mathsf{o,z}$')

sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
for ii in idx_inter:
    ax.axvline(ts_cyc[ii], color='gray', lw=1)

ax.plot(ts_cyc, M_traj_body_avg)
ax.plot(ts_cyc, dho_traj_body_avg)
#ax.plot(ts_cyc[idx_ball], M_traj_body_avg[idx_ball], 'bo', ms=10)
#ax.plot(ts_cyc[idx_trans], M_traj_body_avg[idx_trans], 'go')
#ax.plot(ts_cyc[idx_z10], M_traj_body_avg[idx_z10], 'mo')
#ax.plot(ts_cyc[idx_shallow], M_traj_body_avg[idx_shallow], 'ro', ms=10)
#ax.plot(ts_cyc[-1], M_traj_body_avg[-1], 'ko', ms=10)

sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
for ii in idx_inter:
    ax.axvline(ts_cyc[ii], color='gray', lw=1)

ax.plot(ts_cyc, M_traj_body_std)
ax.plot(ts_cyc, dho_traj_body_std, ls='--')
#ax.plot(ts_cyc[idx_ball], M_traj_body_avg[idx_ball], 'bo', ms=10)
#ax.plot(ts_cyc[idx_trans], M_traj_body_avg[idx_trans], 'go')
#ax.plot(ts_cyc[idx_z10], M_traj_body_avg[idx_z10], 'mo')
#ax.plot(ts_cyc[idx_shallow], M_traj_body_avg[idx_shallow], 'ro', ms=10)
#ax.plot(ts_cyc[-1], M_traj_body_avg[-1], 'ko', ms=10)

sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.axhline(1, color='gray', lw=1)
for ii in idx_inter:
#    ax.axvline(ts_cyc[ii], color='gray', lw=1)
    ax.axvline(ts_cyc_non[ii], color='gray', lw=1)

#ax.semilogy(ts_cyc, dho_traj_body_std / M_traj_body_std)
#ax.semilogy(ts_cyc, dho_traj_body_ptp / M_traj_body_ptp)

ax.semilogy(ts_cyc_non, dho_traj_body_std / M_traj_body_std)
ax.semilogy(ts_cyc_non, dho_traj_body_ptp / M_traj_body_ptp)

sns.despine()
fig.set_tight_layout(True)


# %% plot one phase through the cycle

# first time point in cycle
i = 40

fig, ax = plt.subplots()
ax.plot(ts_cyc, M_traj_body_tot[:, i])
ax.plot(ts_cyc, dho_traj_body_tot[:, i])


# %% Plot just one component of the moments

k = 2

fig, ax = plt.subplots()
ax.plot(ts_cyc, M_traj_body_tot[:, :, k])
ax.plot(ts_cyc, dho_traj_body_tot[:, :, k])


# %% CYCLE AVERAGED VELOCITY POLAR DIAGRAM

# remove last point, don't duplicate the t=0 point
ntime_vpd = 50
ts_vpd = np.linspace(0, 1 / f_theta, ntime + 1)[:-1]

dRo = np.r_[0, 0, 0]
C = eye(3)
omg = np.zeros(3)

# velocity initial conditions (unitless)
#ngrid = 26
#vy0 = np.linspace(0, 1.25, ngrid)
#vz0 = np.linspace(0, -1.25, ngrid)
#ngrid = 21
#ngrid = 41
#vy0 = np.linspace(0, 1, ngrid)
#vz0 = np.linspace(0, -1, ngrid)
#VZ0, VY0 = np.meshgrid(vz0, vy0)

ngrid = 51
vy0 = np.linspace(0, 1.25, ngrid)
vz0 = np.linspace(0, -1.25, ngrid)
VZ0, VY0 = np.meshgrid(vz0, vy0)

Fx = np.zeros((ngrid, ngrid, ntime, nbody))
Fy = np.zeros((ngrid, ngrid, ntime, nbody))
Fz = np.zeros((ngrid, ngrid, ntime, nbody))

Fl = np.zeros((ngrid, ngrid, ntime, nbody, 3))
Fd = np.zeros((ngrid, ngrid, ntime, nbody, 3))

p = np.zeros((ntime, nbody, 3))
dp = np.zeros((ntime, nbody, 3))
ddp = np.zeros((ntime, nbody, 3))

now = time.time()
for i in np.arange(ntime):
    t = ts[i]

    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
    tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

    p[i] = pi
    dp[i] = dpi
    ddp[i] = ddpi

    # vy
    for j in np.arange(ngrid):

        # vz
        for k in np.arange(ngrid):
            # select out velocities and make sure we are indexing correctly
            #assert(np.allclose(vy0[j], VY0[j, k]))
            #assert(np.allclose(vz0[k], VZ0[j, k]))

            # convert non-dim to dimensional for simulation
            dRo_unitless = np.r_[0, vy0[j], vz0[k]]
            dRo = vscale * dRo_unitless

            # positions, velocities, and accelerations
            ri, dri, ddri = sim.rotate(C.T, pi), sim.rotate(C.T, dpi), sim.rotate(C.T, ddpi)
            dRi = dRo + dri + np.cross(omg, ri)

            Fi = np.zeros(ri.shape)
            Fi[:, 2] = -m * g

            # aerodynamic forces
            aout = sim.aero_forces(tvi, cvi, bvi, C, dRi, ds, c, rho,
                                   aero_interp, full_out=True)
            Fi += aout['Fa']  # Fl + Fd
            Fli, Fdi = aout['Fl'], aout['Fd']

            # store the force values
            Fx[j, k, i] = Fi[:, 0]
            Fy[j, k, i] = Fi[:, 1]
            Fz[j, k, i] = Fi[:, 2]

            Fl[j, k, i] = Fli
            Fd[j, k, i] = Fdi

# cycle average forces, normalized by the weight
# sum aong the body, then average over the cycle
Fx_tot = Fx.sum(axis=-1)
Fy_tot = Fy.sum(axis=-1)
Fz_tot = Fz.sum(axis=-1)
Fx_avg = Fx_tot.mean(axis=-1) / weight
Fy_avg = Fy_tot.mean(axis=-1) / weight
Fz_avg = Fz_tot.mean(axis=-1) / weight
F_tot = np.sqrt(Fx_tot**2 + Fy_tot**2 + Fz_tot**2)
F_avg = np.sqrt(Fx_avg**2 + Fy_avg**2 + Fz_avg**2)

print('Cycle avg. VPD: {0:.3f} sec'.format(time.time() - now))



# %% Plot streamlines

fig, ax = plt.subplots(figsize=(6, 5.5))

ax.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
#ax.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)

vpd = ax.streamplot(VY0[:, 0], VZ0[0], Fy_avg.T, Fz_avg.T, color='gray',
              arrowstyle='-|>', linewidth=1)

# plot the trajectory
traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5)
add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'bo', ms=10)
ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'go', ms=10)
ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'ms', ms=10)
ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'ro', ms=10)
ax.plot(veq[1], veq[2], 'ko', ms=10)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')  # these need to be square
ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)

sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)
#fig.savefig(FIG.format('VPD cycle avg'), **FIGOPT)


# %%

fig, ax = plt.subplots(figsize=(6, 5.5))

ax.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
#ax.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)

vpd = ax.streamplot(VY0[:, 0], VZ0[0], Fy_avg.T, Fz_avg.T, color='gray',
              arrowstyle='-|>', linewidth=1)

# plot the trajectory
traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5)
add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'ko', ms=10)
ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'mo', ms=10)
ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'yo', ms=10)
ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'co', ms=10)
ax.plot(veq[1], veq[2], 'o', color='gray', ms=10)
# ax.plot(vy_cyc, vz_cyc, 'x', ms=9, mew=2, mec='k', markevery=10)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')  # these need to be square
ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)

sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)
fig.savefig(FIG.format('VPD cycle avg markers'), **FIGOPT)


# %% Contour plots of force

#fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
#                               figsize=(9, 4))

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               figsize=(12, 5.5))

cax1 = ax1.pcolormesh(VY0, VZ0, Fz_avg, cmap=plt.cm.coolwarm,
                      vmin=-1, vmax=1)
cax1.set_edgecolor('face')
ax1.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
ax1.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax1.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar1 = fig.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.55,
                     fraction=.025)
cbar1.ax.set_title(r'F$_\mathsf{z}$')
cbar1.solids.set_edgecolor('face')
cbar1.set_ticks([-1, 0, 1])

cax2 = ax2.pcolormesh(VY0, VZ0, Fy_avg, cmap=plt.cm.coolwarm,
                  vmin=-.2, vmax=.2)
cax2.set_edgecolor('face')
ax2.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
ax2.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax2.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar2 = fig.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.55,
                     fraction=.025)
cbar2.ax.set_title(r'F$_\mathsf{y}$')
cbar2.solids.set_edgecolor("face")
cbar2.set_ticks([-.2, 0, .2])

plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')


for ax in [ax1, ax2]:

    # plot the trajectory
    traj = ax.plot(vy_cyc, vz_cyc, 'g-', lw=2.5)
    add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)
    ax.plot(vy_cyc[idx_ball], vz_cyc[idx_ball], 'ko', ms=10)
    ax.plot(vy_cyc[idx_trans], vz_cyc[idx_trans], 'mo', ms=10)
    ax.plot(vy_cyc[idx_z10], vz_cyc[idx_z10], 'yo', ms=10)
    ax.plot(vy_cyc[idx_shallow], vz_cyc[idx_shallow], 'co', ms=10)
    ax.plot(veq[1], veq[2], 'o', color='gray', ms=10)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-1.25, 0)
    ax.set_xticks([0, .25, .5, .75, 1, 1.25])
    ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
    ax.xaxis.set_major_formatter(decimal_formatter)
    ax.yaxis.set_major_formatter(decimal_formatter)

#    # http://stackoverflow.com/a/13583251
#    xticks = ax.xaxis.get_major_ticks()
#    yticks = ax.yaxis.get_major_ticks()
#    for ii in np.arange(len(xticks))[1:-1]:
#        xticks[ii].label1.set_visible(False)
#        xticks[ii].label2.set_visible(False)
#        yticks[ii].label1.set_visible(False)
#        yticks[ii].label2.set_visible(False)

    sns.despine(ax=ax, top=False, bottom=True)

fig.set_tight_layout(True)
fig.savefig(FIG.format('VPD Fy and Fz'), **FIGOPT)


# %%

i = 10

fig, ax = plt.subplots()
#cax = ax.contourf(VY0, VZ0, Fz_avg, 26, cmap=plt.cm.coolwarm,
#                  vmin=-1, vmax=1)
#cax = ax.pcolormesh(VY0, VZ0, F_avg, cmap=plt.cm.viridis,
#                  vmin=0, vmax=1)
vz_null_contour = ax.contour(VY0, VZ0, Fz_avg, [0], colors='k', lw=2)
vy_null_contour = ax.contour(VY0, VZ0, Fy_avg, [0], colors='k', lw=2, linestyles='--')
ax_region = ax.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.2)

ax.contour(VY0, VZ0, Fz_tot[:, :, i], [0], colors='r', lw=2)
ax.contour(VY0, VZ0, Fy_tot[:, :, i], [0], colors='r', lw=2, linestyles='--')
#ax.contourf(VY0, VZ0, F_tot[:, :, i], [0, 0.1], colors=['y'], alpha=.2)

ax.streamplot(VY0[:, 0], VZ0[0], Fy_tot[:, :, i].T, Fz_tot[:, :, i].T,
              color='gray')

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlim(0, 1.25)
ax.set_ylim(-1.25, 0)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
ax.xaxis.set_major_formatter(decimal_formatter)
ax.yaxis.set_major_formatter(decimal_formatter)
sns.despine(ax=ax, top=False, bottom=True)

fig.savefig(FIG.format('VPD indiv'), **FIGOPT)


# %% Contour plots of force - colorbar is horizontal

#fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
#                               figsize=(9, 4))

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               figsize=(10, 6.5))

cax1 = ax1.pcolormesh(VY0, VZ0, Fz_avg, cmap=plt.cm.coolwarm,
                      vmin=-1, vmax=1)
cax1.set_edgecolor('face')
ax1.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
ax1.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax1.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar1 = fig.colorbar(cax1, ax=ax1, orientation='horizontal', shrink=.55,
                     fraction=.025)
#cbar1.ax.set_title(r'F$_\mathsf{z}$')
cbar1.ax.set_ylabel(r'F$_\mathsf{z}$  ', rotation=0)
cbar1.solids.set_edgecolor('face')
cbar1.set_ticks([-1, 0, 1])

#divider = make_axes_locatable(ax1)
#cax = divider.append_axes("right", size="2.5%", pad=0.05)
#cbar1 = fig.colorbar(cax1, cax=cax, ax=ax)

cax2 = ax2.pcolormesh(VY0, VZ0, Fy_avg, cmap=plt.cm.coolwarm,
                  vmin=-.2, vmax=.2, alpha=1, linewidth=0)
cax2.set_edgecolor('face')
ax2.contour(VY0, VZ0, Fz_avg, [0], colors='k', linewidths=1.5)
ax2.contour(VY0, VZ0, Fy_avg, [0], colors='k', linewidths=1, linestyles='--')
ax2.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar2 = fig.colorbar(cax2, ax=ax2, orientation='horizontal', shrink=.55,
                     fraction=.025)
cbar2.ax.set_ylabel(r'F$_\mathsf{y}$  ', rotation=0)
cbar2.solids.set_edgecolor("face")
cbar2.set_ticks([-.2, 0, .2])

plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')


for ax in [ax1, ax2]:

    # plot the trajectory
    traj = ax.plot(vy, vz, 'g', lw=2.5)
    add_arrow_to_line2D(ax, traj, arrow_locs=[.2], arrowstyle='->', arrowsize=3)

    #ax.plot(vy[0], vz[0], 'go')
    ax.plot(vy[idx_ball], vz[idx_ball], 'bo', ms=10)
    ax.plot(vy[idx_trans], vz[idx_trans], 'go')
    #ax.plot(vy[idx_trans2], vz[idx_trans2], 'gs')
    ax.plot(vy[idx_z10], vz[idx_z10], 'mo')
    ax.plot(vy[idx_shallow], vz[idx_shallow], 'ro', ms=10)
    ax.plot(veq[1], veq[2], 'ko', ms=10)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticks([0, .25, .5, .75, 1, 1.25])
    ax.set_yticks([0, -.25, -.5, -.75, -1, -1.25])
    ax.xaxis.set_major_formatter(decimal_formatter)
    ax.yaxis.set_major_formatter(decimal_formatter)

#    # http://stackoverflow.com/a/13583251
#    xticks = ax.xaxis.get_major_ticks()
#    yticks = ax.yaxis.get_major_ticks()
#    for ii in np.arange(len(xticks))[1:-1]:
#        xticks[ii].label1.set_visible(False)
#        xticks[ii].label2.set_visible(False)
#        yticks[ii].label1.set_visible(False)
#        yticks[ii].label2.set_visible(False)

    sns.despine(ax=ax, top=False, bottom=True)

fig.set_tight_layout(True)


# %%
## trajectory with 1.7 m/s initial condition
##traj = ax.streamplot(VY0[:, 0], VZ0[0], Fy_avg.T, Fz_avg.T, color='g',
##              arrowstyle='-', start_points=np.array([[va[0], va[1]]]),
##              zorder=100, linewidth=2)
#
#traj = ax.streamplot(VY0[:, 0], VZ0[0], Fy_avg.T, Fz_avg.T, color='g',
#              arrowstyle='-', start_points=np.array([va]),
#              zorder=100, linewidth=2)
#
## plot a single line, not with the edges
#x, y = np.array([]), np.array([])
#lines = traj.lines.get_paths()
#for line in lines:
#    xx, yy = line.vertices[0]
#    x = np.r_[x, xx]
#    y = np.r_[y, yy]
#
#traj.lines.set_linewidth(0)
#ax.plot(x, y, 'g', lw=3)
#
#gam = -np.arctan(y / x)
##idx_trans = gam.argmax()
#idx_trans = y.argmin()
#ax.plot(x[idx_trans], y[idx_trans], 'go')
#
#ax.plot(x[-1], y[-1], 'go')
#
#gam_shallow = (gam[-1] + gam[idx_trans]) / 2
#idx_shallow = np.argmin(np.abs(gam[y < y[-1]] - gam_shallow))
#ax.plot(x[y < y[-1]][idx_shallow], y[y < y[-1]][idx_shallow], 'go')
#
#idx_ball = np.argmin(np.abs(y - y[idx_trans] / 2))
#ax.plot(x[idx_ball], y[idx_ball], 'go')
#
##x, y = np.array([]), np.array([])
##lines = vpd.lines.get_paths()
##for line in lines:
##    xx, yy = line.vertices[0]
##    x = np.r_[x, xx]
##    y = np.r_[y, yy]
##
##ax.plot(x, y, 'gray', lw=1)
#
#ax.xaxis.set_label_position('top')
#ax.xaxis.tick_top()
#ax.set_aspect('equal', adjustable='box')  # these need to be square
#ax.set_xticks([0, .5, 1])
#ax.set_yticks([0, -.5, -1])
#
#sns.despine(ax=ax, top=False, bottom=True)


# %% With the equilibrium point, pitch the snake

#vstart = np.r_[0, vy[0], vz[0]]
#vball = np.r_[0, vy[idx_ball], vz[idx_ball]]
#vtrans = np.r_[0, vy[idx_trans], vz[idx_trans]]
#vlanding = np.r_[0, vy[idx_z10], vz[idx_z10]]
#vshallow = np.r_[0, vy[idx_shallow], vz[idx_shallow]]
#
#vinter = np.c_[vstart, vball, vtrans, vlanding, vshallow, veq].T

ntime_mom = 50
ts_mom = np.linspace(0, 1 / f_theta, ntime_mom + 1)[:-1]
# velocity at equilibrium in physical units
#dRo_non = vstart
dRo_non = vball
#dRo_non = vtrans
#dRo_non = vlanding
#dRo_non = vshallow
#dRo_non = veq
dRo = dRo_non * vscale
omg = np.r_[0, 0, 0]

pitches = np.arange(-25, 26, 1)
rolls = np.arange(-25, 26, 1)
yaws = np.arange(-25, 26, 1)

npitch = len(pitches)
nroll = len(rolls)
nyaw = len(yaws)

npnt3 = np.zeros((npitch, ntime_mom, 3))
npntnb3 = np.zeros((npitch, ntime_mom, nbody, 3))
M_iner = npntnb3.copy()
M_body = npntnb3.copy()
M_iner_tot = npnt3.copy()
M_body_tot = npnt3.copy()

F_iner = npntnb3.copy()
F_body = npntnb3.copy()
F_iner_tot = npnt3.copy()
F_body_tot = npnt3.copy()

dHo_iner = npntnb3.copy()
dHo_body = npntnb3.copy()
dHo_iner_tot = npnt3.copy()
dHo_body_tot = npnt3.copy()

p = np.zeros((ntime_mom, nbody, 3))
dp = np.zeros((ntime_mom, nbody, 3))
ddp = np.zeros((ntime_mom, nbody, 3))

now = time.time()
for i in np.arange(ntime_mom):
    t = ts_mom[i]

    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
    tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

    p[i] = pi
    dp[i] = dpi
    ddp[i] = ddpi

    for j in np.arange(npitch):

        pitch = pitches[j]
        ang = np.deg2rad(np.r_[0, pitch, 0])
        C = sim.euler2C(ang[0], ang[1], ang[2])

#        roll = rolls[j]
#        ang = np.deg2rad(np.r_[0, 0, roll])
#        C = sim.euler2C(ang[0], ang[1], ang[2])

#        yaw = yaws[j]
#        ang = np.deg2rad(np.r_[yaw, 0, 0])
#        C = sim.euler2C(ang[0], ang[1], ang[2])

        # positions, velocities, and accelerations
        ri, dri, ddri = sim.rotate(C.T, pi), sim.rotate(C.T, dpi), sim.rotate(C.T, ddpi)
        dRi = dRo + dri + np.cross(omg, ri)

        # ho_dot
        dHoi_iner = np.cross((m * ri.T).T, ddri)
        dHoi_body = np.cross((m * pi.T).T, ddpi)

        dHo_iner[j, i] = dHoi_iner
        dHo_body[j, i] = dHoi_body
        dHo_iner_tot[j, i] = dHoi_iner.sum(axis=0)
        dHo_body_tot[j, i] = dHoi_body.sum(axis=0)

        # aerodynamic forces
        aout = sim.aero_forces(tvi, cvi, bvi, C, dRi, ds, c, rho,
                               aero_interp, full_out=True)

        # aerodynamic force
        Fi_iner = aout['Fa']
        Fi_body = sim.rotate(C, Fi_iner)

        F_iner[j, i] = Fi_iner
        F_body[j, i] = Fi_body
        F_iner_tot[j, i] = Fi_iner.sum(axis=0)
        F_body_tot[j, i] = Fi_body.sum(axis=0)

        # aerodynamic moments
        Mi_iner = sim.cross(ri, Fi_iner)
        Mi_body = sim.rotate(C, Mi_iner)

        M_iner[j, i] = Mi_iner
        M_body[j, i] = Mi_body
        M_iner_tot[j, i] = Mi_iner.sum(axis=0)
        M_body_tot[j, i] = Mi_body.sum(axis=0)


# average over the cycle
M_iner_avg = M_iner_tot.mean(axis=1)
M_body_avg = M_body_tot.mean(axis=1)
F_iner_avg = F_iner_tot.mean(axis=1)
F_body_avg = F_body_tot.mean(axis=1)
dHo_iner_avg = dHo_iner_tot.mean(axis=1)
dHo_body_avg = dHo_body_tot.mean(axis=1)

# normalize forces and moments
m_iner_avg = M_iner_avg / mscale
m_body_avg = M_body_avg / mscale
f_iner_avg = F_iner_avg / fscale
f_body_avg = F_body_avg / fscale
dho_iner_avg = dHo_iner_avg #/ mscale
dho_body_avg = dHo_body_avg #/ mscale

m_iner = M_iner / mscale
m_body = M_body / mscale

m_iner_tot = M_iner_tot / mscale
m_body_tot = M_body_tot / mscale

dho_iner = dHo_iner #/ mscale
dho_body = dHo_body #/ mscale

dho_iner_tot = dHo_iner_tot #/ mscale
dho_body_tot = dHo_body_tot #/ mscale

M_body_max = M_body_tot.max(axis=1)
M_body_min = M_body_tot.min(axis=1)
M_body_ptp = M_body_tot.ptp(axis=1)
M_body_std = M_body_tot.std(axis=1)
M_body_rms = np.sqrt(np.mean(M_body_tot**2, axis=1))

dho_body_max = dho_body_tot.max(axis=1)
dho_body_min = dho_body_tot.min(axis=1)
dho_body_ptp = dho_body_tot.ptp(axis=1)
dho_body_std = dho_body_tot.std(axis=1)
dho_body_rms = np.sqrt(np.mean(dho_body_tot**2, axis=1))

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %% Aerodynamic moments with pitch, physical units

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, M_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, M_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, M_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

#ax1.plot(pitches, M_body_max[:, 0], 'k')
#ax2.plot(pitches, M_body_max[:, 1], 'k')
#ax3.plot(pitches, M_body_max[:, 2], 'k')
#
#ax1.plot(pitches, M_body_min[:, 0], 'k')
#ax2.plot(pitches, M_body_min[:, 1], 'k')
#ax3.plot(pitches, M_body_min[:, 2], 'k')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime_mom) * pitches[j]
    ax1.scatter(pitch_plot, M_body_tot[j, :, 0],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, M_body_tot[j, :, 1],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, M_body_tot[j, :, 2],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %% Aerodynamic moments with pitch, physical units

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, M_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, M_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, M_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

ax1.plot(pitches, M_body_avg[:, 0] + M_body_std[:, 0], 'k')
ax2.plot(pitches, M_body_avg[:, 1] + M_body_std[:, 1], 'k')
ax3.plot(pitches, M_body_avg[:, 2] + M_body_std[:, 2], 'k')

ax1.plot(pitches, M_body_avg[:, 0] - M_body_std[:, 0], 'k')
ax2.plot(pitches, M_body_avg[:, 1] - M_body_std[:, 1], 'k')
ax3.plot(pitches, M_body_avg[:, 2] - M_body_std[:, 2], 'k')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime_mom) * pitches[j]
    ax1.scatter(pitch_plot, M_body_tot[j, :, 0],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, M_body_tot[j, :, 1],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, M_body_tot[j, :, 2],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %% dho vs. time

ts_non = ts_mom * f_theta

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_non, dho_body_tot[0, :, 0], 'k', label=r'$h_\mathsf{o,x}$')
ax2.plot(ts_non, dho_body_tot[0, :, 1], 'k', label=r'$h_\mathsf{o,y}$')
ax3.plot(ts_non, dho_body_tot[0, :, 2], 'k', label=r'$h_\mathsf{o,z}$')

sns.despine()
fig.set_tight_layout(True)


# %% dho with pitch (not interesting, as in body frame)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, dho_body_avg[:, 0], 'k', label=r'$h_\mathsf{o,x}$')
ax2.plot(pitches, dho_body_avg[:, 1], 'k', label=r'$h_\mathsf{o,y}$')
ax3.plot(pitches, dho_body_avg[:, 2], 'k', label=r'$h_\mathsf{o,z}$')

ax1.plot(pitches, dho_body_avg[:, 0] + dho_body_std[:, 0], 'k')
ax2.plot(pitches, dho_body_avg[:, 1] + dho_body_std[:, 1], 'k')
ax3.plot(pitches, dho_body_avg[:, 2] + dho_body_std[:, 2], 'k')

ax1.plot(pitches, dho_body_avg[:, 0] - dho_body_std[:, 0], 'k')
ax2.plot(pitches, dho_body_avg[:, 1] - dho_body_std[:, 1], 'k')
ax3.plot(pitches, dho_body_avg[:, 2] - dho_body_std[:, 2], 'k')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime_mom) * pitches[j]
    ax1.scatter(pitch_plot, dho_body_tot[j, :, 0],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, dho_body_tot[j, :, 1],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, dho_body_tot[j, :, 2],
               c=ts_mom, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %%

Fr_body_ptp = dho_body_ptp / M_body_ptp

fig, ax = plt.subplots()
ax.plot(pitches, Fr_body_ptp[:, 0])
ax.plot(pitches, Fr_body_ptp[:, 1])
ax.plot(pitches, Fr_body_ptp[:, 2])
sns.despine()


Fr_body_std = dho_body_std / M_body_std

fig, ax = plt.subplots()
ax.plot(pitches, Fr_body_std[:, 0])
ax.plot(pitches, Fr_body_std[:, 1])
ax.plot(pitches, Fr_body_std[:, 2])
sns.despine()


Fr_body_rms = dho_body_rms / M_body_rms

fig, ax = plt.subplots()
ax.plot(pitches, Fr_body_rms[:, 0])
ax.plot(pitches, Fr_body_rms[:, 1])
ax.plot(pitches, Fr_body_rms[:, 2])
sns.despine()


# %% Plot time series of moments in body frame

ts_non = ts * f_theta
#colors = [plt.cm.viridis(x) for x in linspace(0, 1, npitch)]
#colors = [plt.cm.coolwarm(x) for x in linspace(0, 1, npitch)]
colors = [plt.cm.RdGy_r(x) for x in linspace(0, 1, npitch)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    # ax.axvline(0, color='gray', lw=1)

#ax1.plot(ts_non, M_body_tot[:, :, 0].T)
#ax1.plot(ts_non, dho_body_tot[:, :, 0].T)
#
#ax2.plot(ts_non, M_body_tot[:, :, 1].T)
#ax2.plot(ts_non, dho_body_tot[:, :, 1].T)
#
#ax3.plot(ts_non, M_body_tot[:, :, 2].T)
#ax3.plot(ts_non, dho_body_tot[:, :, 2].T)

# this doesn't change in the body frame
ax1.plot(ts_non, dho_body_tot[0, :, 0], c='b')
ax2.plot(ts_non, dho_body_tot[0, :, 1], c='b')
ax3.plot(ts_non, dho_body_tot[0, :, 2], c='b')

for j in np.arange(npitch):
    ax1.plot(ts_non, M_body_tot[j, :, 0], c=colors[j])
#    ax1.plot(ts_non, dho_body_tot[j, :, 0], c=colors[j])

    ax2.plot(ts_non, M_body_tot[j, :, 1], c=colors[j])
#    ax2.plot(ts_non, dho_body_tot[j, :, 1], c=colors[j])

    ax3.plot(ts_non, M_body_tot[j, :, 2], c=colors[j])
#    ax3.plot(ts_non, dho_body_tot[j, :, 2], c=colors[j])

sns.despine()
fig.set_tight_layout(True)


# %% Aerodynamic moments with pitch, non-dimensional units

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, m_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, m_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, m_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

m_body_max = m_body_tot.max(axis=1)
m_body_min = m_body_tot.min(axis=1)
m_body_ptp = m_body_tot.ptp(axis=1)
dho_body_ptp = dho_body_tot.ptp(axis=1)

ax1.plot(pitches, m_body_max[:, 0], 'k')
ax2.plot(pitches, m_body_max[:, 1], 'k')
ax3.plot(pitches, m_body_max[:, 2], 'k')

ax1.plot(pitches, m_body_min[:, 0], 'k')
ax2.plot(pitches, m_body_min[:, 1], 'k')
ax3.plot(pitches, m_body_min[:, 2], 'k')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime) * pitches[j]
    ax1.scatter(pitch_plot, m_body_tot[j, :, 0],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, m_body_tot[j, :, 1],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, m_body_tot[j, :, 2],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %%


# %%

# phase shift the serpenoid curve for a near zero angular momentum
t = 0
ho_args = (s, t, m, n_neck, theta_dict, psi_dict)
phi_theta_opt = fmin(sim.func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
phi_theta = float(phi_theta_opt)
phi_psi = 2 * phi_theta - np.deg2rad(180)

theta_dict['phi_theta'] = phi_theta
psi_dict['phi_psi'] = phi_psi

nho = 100
ho = np.zeros((nho, 3))
dho = np.zeros((nho, 3))
ts_ho = np.linspace(0, 1 / f_theta, nho + 1)[:-1]

for i in np.arange(nho):
    t = ts_ho[i]
    out = sim.aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)
    p, dp, ddp = out['p'], out['dp'], out['ddp']
    ho[i] = (m * np.cross(p, dp).T).T.sum(axis=0)
    dho[i] = (m * np.cross(p, ddp).T).T.sum(axis=0)

ho_mean = ho.mean(axis=0)
dho_mean = dho.mean(axis=0)


# %%

ts_ho_non = ts_ho * f_theta
c = ['b', 'g', 'r']
label = ['pitch', 'yaw', 'roll']

fig, ax = plt.subplots()
for i in np.arange(3):
    ax.axhline(ho_mean[i], c=c[i], lw=1)
    ax.plot(ts_ho_non, ho[:, i], c=c[i], label=label[i])
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for i in np.arange(3):
    ax.axhline(dho_mean[i], c=c[i], lw=1)
    ax.plot(ts_ho_non, dho[:, i], c=c[i], label=label[i])
sns.despine()
fig.set_tight_layout(True)


# %% Integrate one trajectory through average VPD

def rotational_dynamics(t, state, vel_body_dict):
    """Called by fixed_point.
    """

    # turn off undulation
#    t = 0

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
#    dp *= 0
#    ddp *= 0
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


from scipy.integrate import ode

now = time.time()

body_dict['aero_interp'] = None
body_dict['aero_interp'] = aero_interp
body_dict['head_control'] = False

ang0 = np.r_[0, 0, 0]  # horizontal flight
#ang0 = -np.deg2rad(np.array([-17.09433418,   0.32207885,  -1.2865577 ]))
omg0 = np.r_[0, 0, 0]  # no angular velocity

# v0_non = 1.7 / np.sqrt(2 * 29 / rho)  # .2409, Ws=29
soln0 = np.r_[omg0, ang0]

# pick a velocity in the VPD
# vstart, vball , vtrans, vlanding, vshallow, veq
dRo0 = vscale * vball
dRo0 = vscale * vshallow
dRo0 = np.r_[0, 0, 0]
body_dict['aero_interp'] = None

# arguments to the rotation dynamics function
vel_body_dict = (dRo0, body_dict)

# setup the integrator
# https://docs.scipy.org/doc/scipy-0.17.1/
# reference/generated/scipy.integrate.ode.html

# integrate over one undulation cycle

# phases in cycle to simulate rotational dynamics
ntime_rot = 20


trot0 = 0
tend = 2 * 1 / f_theta + trot0

ntime_rot = 200
ts_rot = np.linspace(0, 2 / f_theta, ntime_rot + 1)[:-1]

solver = ode(rotational_dynamics)
solver.set_integrator('dopri5')
solver.set_initial_value(soln0, trot0)  # x0, t0
solver.set_f_params(vel_body_dict)

#solver._integrator.iwork[2] = -1  # suppress Fortran-printed warning

#soln_rot, ts_rot = [soln0], [solver.t]
soln_rot = [soln0]
at_equil = False
i = 1
while i < ntime_rot:
    #solver.integrate(solver.t + .01)
    solver.integrate(ts_rot[i])
    out = solver.y

    print solver.t

    soln_rot.append(out)
    i = i + 1
    #ts_rot.append(solver.t)

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

#soln_rot, ts_rot = np.array(soln_rot), np.array(ts_rot)
soln_rot = np.array(soln_rot)
ts_rot_non = ts_rot * f_theta

wx, wy, wz, yaw, pitch, roll = soln_rot.T
ang = np.c_[yaw, pitch, roll]
omg = np.c_[wx, wy, wz]

yaw_d, pitch_d, roll_d = np.rad2deg(ang).T

# Euler angle rates
ntime_rot = len(ts_rot)
domg = np.zeros((ntime_rot, 3))
dang = np.zeros((ntime_rot, 3))
ddang = np.zeros((ntime_rot, 3))
C = np.zeros((ntime_rot, 3, 3))
for i in np.arange(ntime_rot):
    C[i] = sim.euler2C(yaw[i], pitch[i], roll[i])
    out = rotational_dynamics(ts_rot[i], soln_rot[i], vel_body_dict)
    omgi, angi = np.split(out, 2)
    domg[i] = omgi
    dang[i] = angi
    ddang[i] = sim.ddang_ddt(ang[i], dang[i], omg[i], domg[i], C[i])

dyaw_d, dpitch_d, droll_d = np.rad2deg(dang).T
ddyaw_d, ddpitch_d, ddroll_d = np.rad2deg(ddang).T


# %%

fig, ax = plt.subplots()
#ax.axhline(0, color='gray', lw=1)
ax.plot(ts_rot_non, pitch_d, label='pitch')
ax.plot(ts_rot_non, roll_d, label='roll')
ax.plot(ts_rot_non, yaw_d, label='yaw')
ax.set_xlim(ts_rot_non[0], ts_rot_non[-1])
ax.legend(loc='best')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 9))
ax1.plot(ts_rot_non, pitch_d, label='pitch')
ax1.plot(ts_rot_non, roll_d, label='roll')
ax1.plot(ts_rot_non, yaw_d, label='yaw')
ax2.plot(ts_rot_non, dpitch_d, label='pitch')
ax2.plot(ts_rot_non, droll_d, label='roll')
ax2.plot(ts_rot_non, dyaw_d, label='yaw')
ax3.plot(ts_rot_non, ddpitch_d, label='pitch')
ax3.plot(ts_rot_non, ddroll_d, label='roll')
ax3.plot(ts_rot_non, ddyaw_d, label='yaw')
ax1.set_xlim(ts_rot_non[0], ts_rot_non[-1])
ax.legend(loc='best')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
#ax.axhline(0, color='gray', lw=1)
ax.plot(ts_rot, np.rad2deg(omg))#, label='pitch')
#ax.plot(ts_rot, roll, label='roll')
#ax.plot(ts_rot, yaw, label='yaw')
ax.legend(loc='best')
sns.despine()
fig.set_tight_layout(True)


# %% Histograms of moment at a particular time

j = 0

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

sns.distplot(M_body_tot[j, :, 0], bins=20, ax=ax1, vertical=True)
sns.distplot(M_body_tot[j, :, 1], bins=20, ax=ax2, vertical=True)
sns.distplot(M_body_tot[j, :, 2], bins=20, ax=ax3, vertical=True)
sns.despine()
fig.set_tight_layout(True)


# %% Plot time series of moments in inertial frame

ts_non = ts * f_theta

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    # ax.axvline(0, color='gray', lw=1)

ax1.plot(ts_non, M_iner_tot[:, :, 0].T)
ax1.plot(ts_non, dho_iner_tot[:, :, 0].T)

ax2.plot(ts_non, M_iner_tot[:, :, 1].T)
ax2.plot(ts_non, dho_iner_tot[:, :, 1].T)

ax3.plot(ts_non, M_iner_tot[:, :, 2].T)
ax3.plot(ts_non, dho_iner_tot[:, :, 2].T)

sns.despine()
fig.set_tight_layout(True)


# %% Plot circles of moment in 3D

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3) / 100
frame_c = [bmap[2], bmap[1], bmap[0]]

for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

j = 5
mlab.plot3d(M_body_tot[j, :, 0], M_body_tot[j, :, 1], M_body_tot[j, :, 2],
            tube_radius=.00003, color=bmap[0])

mlab.plot3d(dho_body_tot[j, :, 0], dho_body_tot[j, :, 1], dho_body_tot[j, :, 2],
            tube_radius=.00003, color=bmap[1])


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, M_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, M_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, M_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

M_body_max = M_body_tot.max(axis=1)
M_body_min = M_body_tot.min(axis=1)
M_body_ptp = M_body_tot.ptp(axis=1)

ax1.plot(pitches, M_body_max[:, 0], 'k')
ax2.plot(pitches, M_body_max[:, 1], 'k')
ax3.plot(pitches, M_body_max[:, 2], 'k')

ax1.plot(pitches, M_body_min[:, 0], 'k')
ax2.plot(pitches, M_body_min[:, 1], 'k')
ax3.plot(pitches, M_body_min[:, 2], 'k')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime) * pitches[j]
    ax1.scatter(pitch_plot, M_body_tot[j, :, 0],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, M_body_tot[j, :, 1],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, M_body_tot[j, :, 2],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)

colors = [plt.cm.coolwarm(x) for x in linspace(0, 1, ntime)]

idx4, idx2 = len(ts)//4, len(ts)//2

#for i in np.r_[0, idx4, idx2, idx2 + idx4]:
#for i in np.r_[idx4, idx2 + idx4]:
#    ax1.plot(pitches, M_body_tot[:, i, 0], c='gray')
#    ax2.plot(pitches, M_body_tot[:, i, 1], c='gray')
#    ax3.plot(pitches, M_body_tot[:, i, 2], c='gray')
#    ax1.plot(pitches, M_body_tot[:, i, 0], c=colors[i])
#    ax2.plot(pitches, M_body_tot[:, i, 1], c=colors[i])
#    ax3.plot(pitches, M_body_tot[:, i, 2], c=colors[i])

sns.despine()
fig.set_tight_layout(True)


# %%

#colors = [plt.cm.coolwarm(x) for x in linspace(0, 1, ntime)]
colors = [plt.cm.coolwarm(x) for x in linspace(0, 1, ntime)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

for i in np.arange(ntime):
    ax1.plot(pitches, M_body_tot[:, i, 0], c=colors[i])
    ax2.plot(pitches, M_body_tot[:, i, 1], c=colors[i])
    ax3.plot(pitches, M_body_tot[:, i, 2], c=colors[i])


ax1.plot(pitches, M_body_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, M_body_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, M_body_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

M_body_max = M_body_tot.max(axis=1)
M_body_min = M_body_tot.min(axis=1)

ax1.plot(pitches, M_body_max[:, 0], 'gray')
ax2.plot(pitches, M_body_max[:, 1], 'gray')
ax3.plot(pitches, M_body_max[:, 2], 'gray')

ax1.plot(pitches, M_body_min[:, 0], 'gray')
ax2.plot(pitches, M_body_min[:, 1], 'gray')
ax3.plot(pitches, M_body_min[:, 2], 'gray')

#for j in np.arange(npitch):
#    pitch_plot = np.ones(ntime) * pitches[j]
#    ax1.scatter(pitch_plot, M_body_tot[j, :, 0],
#               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
#    ax2.scatter(pitch_plot, M_body_tot[j, :, 1],
#               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
#    ax3.scatter(pitch_plot, M_body_tot[j, :, 2],
#               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)


# %%



# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

for ax in [ax1, ax2, ax3]:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pitches, M_iner_avg[:, 0], 'k', label=r'$M_\mathsf{x}$')
ax2.plot(pitches, M_iner_avg[:, 1], 'k', label=r'$M_\mathsf{y}$')
ax3.plot(pitches, M_iner_avg[:, 2], 'k', label=r'$M_\mathsf{z}$')

for j in np.arange(npitch):
    pitch_plot = np.ones(ntime) * pitches[j]
    ax1.scatter(pitch_plot, M_iner_tot[j, :, 0],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax2.scatter(pitch_plot, M_iner_tot[j, :, 1],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)
    ax3.scatter(pitch_plot, M_iner_tot[j, :, 2],
               c=ts, cmap=plt.cm.coolwarm, s=30, linewidth=0)

sns.despine()
fig.set_tight_layout(True)






# %%





def integrate(soln0, args, dt, tend=None, print_time=False, ):
    """Integrate the equations of motion until the centroid drops
    below z = 0, or until the current time is greater than tend.

    Parameters
    ----------
    soln0 : array, size (12)
        initial conditions: Ro0, dRo0, omg0, ang0
    args : tuple
        arguments to the function:
        (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)
    dt : float
        time step for integration

    Returns
    -------
    ts : array, size (ntime)
        time points of simulation, separated by dt
    """




# %% Contour plots of force and streamlines in one figure

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                               figsize=(12, 4))

cax1 = ax1.pcolormesh(VY0, VZ0, Fz_avg, cmap=plt.cm.coolwarm,
                      vmin=-1, vmax=1)
cax1.set_edgecolor('face')
ax1.contour(VY0, VZ0, Fz_avg, [0], colors='k', lw=1)
ax1.contour(VY0, VZ0, Fy_avg, [0], colors='k', lw=1, linestyles='--')
#ax1.contour(VY0, VZ0, F_avg, [0.1], colors='m', lw=1.5)
ax1.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar1 = fig.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.55,
                     fraction=.025)
#cbar1.set_label(r'Fz')
cbar1.ax.set_title(r'$\mathsf{F_z}$')
cbar1.solids.set_edgecolor('face')
cbar1.set_ticks([-1, 0, 1])

# ax1.contour(VY0, VZ0, Re_dim, Re_contours, colors='gray')

cax2 = ax2.pcolormesh(VY0, VZ0, Fy_avg, cmap=plt.cm.coolwarm,
                  vmin=-.2, vmax=.2)
cax2.set_edgecolor('face')
ax2.contour(VY0, VZ0, Fz_avg, [0], colors='k', lw=1)
ax2.contour(VY0, VZ0, Fy_avg, [0], colors='k', lw=1, linestyles='--')
ax2.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)
cbar2 = fig.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.55,
                     fraction=.025)
#cbar2.set_label(r'Fy')

cbar2.ax.set_title(r'$\mathsf{F_y}$')

cbar2.solids.set_edgecolor("face")
cbar2.set_ticks([-.2, 0, .2])

ax3.contour(VY0, VZ0, Fz_avg, [0], colors='k', lw=1)
ax3.contour(VY0, VZ0, Fy_avg, [0], colors='k', lw=1, linestyles='--')
#ax1.contour(VY0, VZ0, F_avg, [0.1], colors='m', lw=1.5)
ax3.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.4)

ax3.streamplot(VY0[:, 0], VZ0[0], Fy_avg.T, Fz_avg.T, color='gray')


plt.setp([ax1, ax2, ax3], aspect=1.0, adjustable='box-forced')

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
#    ax.set_xlabel(r"$v_y$", fontsize=20)
#    ax.set_ylabel(r"$v_z    $", fontsize=20, rotation=0)
    # ax.set_aspect('equal', adjustable='box')  # these need to be square
    sns.despine(ax=ax, top=False, bottom=True)

fig.set_tight_layout(True)


# %% Plot streamlines for instantaneous body position

i = 20

fig, ax = plt.subplots()
#cax = ax.contourf(VY0, VZ0, Fz_avg, 26, cmap=plt.cm.coolwarm,
#                  vmin=-1, vmax=1)
#cax = ax.pcolormesh(VY0, VZ0, F_avg, cmap=plt.cm.viridis,
#                  vmin=0, vmax=1)
vz_null_contour = ax.contour(VY0, VZ0, Fz_avg, [0], colors='k', lw=2)
vy_null_contour = ax.contour(VY0, VZ0, Fy_avg, [0], colors='k', lw=2)
ax_region = ax.contourf(VY0, VZ0, F_avg, [0, 0.1], colors=['m'], alpha=.2)

ax.contour(VY0, VZ0, Fz_tot[:, :, i], [0], colors='b', lw=2)
ax.contour(VY0, VZ0, Fy_tot[:, :, i], [0], colors='r', lw=2)
ax.contourf(VY0, VZ0, F_tot[:, :, i], [0, 0.1], colors=['y'], alpha=.2)

ax.streamplot(VY0[:, 0], VZ0[0], Fy_tot[:, :, i].T, Fz_tot[:, :, i].T,
              color='gray')

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r"$v_y$", fontsize=20)
ax.set_ylabel(r"$v_z    $", fontsize=20, rotation=0)
ax.set_aspect('equal', adjustable='box')  # these need to be square
#ax.set_xticks([0, .25, .5, .75, 1])
#ax.set_yticks([0, -.25, -.5, -.75, -1])

#cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.75)
#cbar.set_label(r'Fz', fontsize='medium')
#cbar.solids.set_edgecolor("face")
##cbar.set_ticks([-1, -.5, 0, .5, 1])
#cbar.set_ticks([-1, 0, 1])

sns.despine(ax=ax, top=False, bottom=True)


# %% Reynolds number contours on VPD

VY0_dim, VZ0_dim = VY0 * vscale, VZ0 * vscale
Umag = np.sqrt(VY0_dim**2 + VZ0_dim**2)

nu = 1.568e-5  # Pa-s = m^2/s
Re_dim = Umag * cmax / nu

Re_contours = np.arange(3000, 15001, 2000)

fig, ax = plt.subplots()
#ax.contour(VY0, VZ0, Re_dim, Re_contours, colors='k')
ax.contourf(VY0, VZ0, Re_dim, Re_contours)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax, top=False, bottom=True)


# %% Exctract forces along the nullclines

# http://stackoverflow.com/a/5666461
vz_null = vz_null_contour.collections[0].get_paths()[0].vertices
vy_null = vy_null_contour.collections[0].get_paths()[0].vertices

fig, ax = plt.subplots()
ax.plot(vz_null[:, 0], vz_null[:, 1], 'b')
ax.plot(vy_null[:, 0], vy_null[:, 1], 'g')

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r"$v_x$", fontsize=20)
ax.set_ylabel(r"$v_z    $", fontsize=20, rotation=0)
ax.set_aspect('equal', adjustable='box')  # these need to be square
ax.set_xlim(0, 1)
ax.set_ylim(-1, 0)
ax.grid(True)

sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %% Effect of body shape on force production at equilibrium (dyanmics off)

def aero_effects(args, wing_loading):
    """Effect of kinematics on aerodynamic forces.
    """

    # kinematics parameters
    s, m, n_neck, theta_dict, psi_dict = args[0]
    ds, c, g, rho, aero_interp = args[1]

    nbody = len(s)
    f_theta = theta_dict['f_theta']
    ntime = 100
    ts = np.linspace(0, 1 / f_theta, ntime)


    # initial conditions
    Ro0 = np.r_[0, 0, 0]
    dRo0_non_rescaled = np.r_[.75926, 0, -.41155]
    dRo0 = np.sqrt(2 * wing_loading / rho) * dRo0_non_rescaled
    ang0 = np.deg2rad(np.r_[0, 0, 0])  # yaw, pitch, roll
    dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

    C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
    omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
    omg0 = np.dot(C0.T, omg0_body)
    soln0 = np.r_[Ro0, dRo0, omg0, ang0]

    dRo = dRo0
    C = C0
    omg = omg0

    Fl = np.zeros((ntime, nbody, 3))
    Fd = np.zeros((ntime, nbody, 3))
    Faero = np.zeros((ntime, nbody, 3))
    Ml = np.zeros((ntime, nbody, 3))
    Md = np.zeros((ntime, nbody, 3))
    Maero = np.zeros((ntime, nbody, 3))
    Re = np.zeros((ntime, nbody))
    aoa = np.zeros((ntime, nbody))
    beta = np.zeros((ntime, nbody))
    dR_BC = np.zeros((ntime, nbody, 3))
    dR_TC = np.zeros((ntime, nbody, 3))
    theta = np.zeros((ntime, nbody))
    psi = np.zeros((ntime, nbody))
    dthetads = np.zeros((ntime, nbody))
    dpsids = np.zeros((ntime, nbody))
    p = np.zeros((ntime, nbody, 3))
    dp = np.zeros((ntime, nbody, 3))
    dpds = np.zeros((ntime, nbody, 3))
    ddpdds = np.zeros((ntime, nbody, 3))
    ddp = np.zeros((ntime, nbody, 3))
    kap = np.zeros((ntime, nbody))
    tv = np.zeros((ntime, nbody, 3))
    cv = np.zeros((ntime, nbody, 3))
    bv = np.zeros((ntime, nbody, 3))
    Crs = np.zeros((ntime, nbody, 3, 3))
    lateral_bend = np.zeros((ntime, nbody))
    back_bend = np.zeros((ntime, nbody))

    from sim import aero_forces_more

    for i in np.arange(ntime):
        t = ts[i]

        out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

        pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
        tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

        # positions, velocities, and accelerations
        ri, dri, ddri = rotate(C.T, pi), rotate(C.T, dpi), rotate(C.T, ddpi)
        # Ri = Ro + ri
        dRi = dRo + dri + np.cross(omg, ri)

        # aerodynamic forces
        fout = aero_forces_more(tvi, cvi, bvi, C, dRi, ds, c, rho, aero_interp)

        Fli, Fdi, dRiBC, aoai, Rei, betai, dRiTC = fout

        # store the values
        Fl[i] = Fli
        Fd[i] = Fdi
        Faero[i] = Fli + Fdi
        Ml[i] = np.cross(pi, Fli)
        Md[i] = np.cross(pi, Fdi)
        Maero[i] = np.cross(pi, Faero[i])
        Re[i] = Rei
        aoa[i] = np.rad2deg(aoai)
        beta[i] = np.rad2deg(betai)
        dR_BC[i] = dRiBC
        dR_TC[i] = dRiTC

        # store the values
        theta[i] = out['theta']
        psi[i] = out['psi']
        dthetads[i] = out['dthetads']
        dpsids[i] = out['dpsids']
        kap[i] = out['kap']

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
        lateral_bend[i] = out['lateral_bend']
        back_bend[i] = out['back_bend']

    out = dict(Fl=Fl, Fd=Fd, Faero=Faero, Ml=Ml, Md=Md, Maero=Maero,
               Re=Re, aoa=aoa, beta=beta, dR_BC=dR_BC, dR_TC=dR_TC,
               dRo=dRo, ts=ts,
               theta=theta, psi=psi, dthetads=dthetads, dpsids=dpsids,
               kap=kap,
               p=p, dp=dp, ddp=ddp,
               dpds=dpds, ddpdds=ddpdds,
               tv=tv, cv=cv, bv=bv, Crs=Crs,
               lateral_bend=lateral_bend, back_bend=back_bend)

    return out



# %% Check yaw, pitch, and roll  sign convnections directions

ang = np.deg2rad(np.r_[0, 20, 0])  # yaw, pitch, roll
C = sim.euler2C(ang[0], ang[1], ang[2])

P = sim.rotate(C.T, p)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]

for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

nframe = sim.rotate(C.T, Nframe)
for ii in np.arange(3):
    mlab.quiver3d(nframe[ii, 0], nframe[ii, 1], nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)


mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=.004, color=bmap[1],
            opacity=.5)

mlab.plot3d(P[:, 0], P[:, 1], P[:, 2], tube_radius=.004, color=bmap[1],
            opacity=1)

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()







# %%%



# %%

# lks = 1 / np.r_[.4, .5, .6, .7, .8]


L = .7  # .686  # m
ds = .01  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)  # number of points for neck

# chord length
# mean_chord = weight / wing_loading / L
# c = mean_chord * np.ones(nbody)
chord_para, chord_interp = chord_dist(s, ds, L)
c = chord_para
area = np.sum(ds * c)  # m^2

# uniform mass distribution
#mass_total = .0405  # kg
#rho_body = mass_total / L  # linear mass density
#m = np.ones(nbody) * rho_body * L / nbody

# mass distribution from wing loading
wing_loading = 29  # N/m^2 from socha 2005
g = 9.81  # m/s^2
mass_total = wing_loading * area / g
mass_para, mass_interp = mass_dist(s, ds, L, mass_total)
m = mass_para
weight = np.sum(m * g)

ntime = 201
ts = np.linspace(0, 3, ntime)

#lambdak_theta = 0.6
#nu_theta = 5 / 3.  # m
nu_theta = 1  # 1.16667 == 5/3 * .7
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

#lambdak_psi = lambdak_theta / 2
nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = .2  # .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
# damp_theta = -2 * frac_theta_max * theta_max / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(0)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                  amp_theta=amp_theta, damp_theta=damp_theta,
                  d_theta=d_theta, L=L, theta_max=theta_max,
                  frac_theta_max=frac_theta_max,
                  amp_theta_fun=amp_theta_fun)
psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                psi_max=psi_max, frac_psi_max=frac_psi_max,
                amp_psi_fun=amp_psi_fun)


t = 0
ho_args = (s, t, m, n_neck, theta_dict, psi_dict)

phi_theta = fmin(func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
phi_theta = float(phi_theta)
phi_psi = 2 * phi_theta

#phi_theta = phi_theta_newton
#phi_psi = phi_psi_newton

theta_dict['phi_theta'] = phi_theta
psi_dict['phi_psi'] = phi_psi

## turn off undulation
#theta_dict['f_theta'] = 0
#psi_dict['f_psi'] = 0


# %% Mass and chord distributions

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.axhline(100 * chord_para.mean(), color='gray', lw=1)
ax2.axhline(1000 * mass_para.mean(), color='gray', lw=1)
ax1.plot(s / L, 100 * chord_interp, c='gray', lw=3)
ax1.plot(s / L, 100 * chord_para, c='k', lw=3)
ax2.plot(s / L, 1000 * mass_interp, c='gray', lw=3)
ax2.plot(s / L, 1000 * mass_para, c='k', lw=3)
ax1.set_ylabel('Chord length (cm)')
ax2.set_ylabel('Mass (g)')
ax2.set_xlabel('Distance along body (SVL)')
ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0)
sns.despine()
fig.set_tight_layout(True)


# %% Effect of body shape on force production at equilibrium (dyanmics off)

def aero_effects(args, wing_loading):
    """Effect of kinematics on aerodynamic forces.
    """

    # kinematics parameters
    s, m, n_neck, theta_dict, psi_dict = args[0]
    ds, c, g, rho, aero_interp = args[1]

    nbody = len(s)
    f_theta = theta_dict['f_theta']
    ntime = 100
    ts = np.linspace(0, 1 / f_theta, ntime)


    # initial conditions
    Ro0 = np.r_[0, 0, 0]
    dRo0_non_rescaled = np.r_[.75926, 0, -.41155]
    dRo0 = np.sqrt(2 * wing_loading / rho) * dRo0_non_rescaled
    ang0 = np.deg2rad(np.r_[0, 0, 0])  # yaw, pitch, roll
    dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

    C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
    omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
    omg0 = np.dot(C0.T, omg0_body)
    soln0 = np.r_[Ro0, dRo0, omg0, ang0]

    dRo = dRo0
    C = C0
    omg = omg0

    Fl = np.zeros((ntime, nbody, 3))
    Fd = np.zeros((ntime, nbody, 3))
    Faero = np.zeros((ntime, nbody, 3))
    Ml = np.zeros((ntime, nbody, 3))
    Md = np.zeros((ntime, nbody, 3))
    Maero = np.zeros((ntime, nbody, 3))
    Re = np.zeros((ntime, nbody))
    aoa = np.zeros((ntime, nbody))
    beta = np.zeros((ntime, nbody))
    dR_BC = np.zeros((ntime, nbody, 3))
    dR_TC = np.zeros((ntime, nbody, 3))
    theta = np.zeros((ntime, nbody))
    psi = np.zeros((ntime, nbody))
    dthetads = np.zeros((ntime, nbody))
    dpsids = np.zeros((ntime, nbody))
    p = np.zeros((ntime, nbody, 3))
    dp = np.zeros((ntime, nbody, 3))
    dpds = np.zeros((ntime, nbody, 3))
    ddpdds = np.zeros((ntime, nbody, 3))
    ddp = np.zeros((ntime, nbody, 3))
    kap = np.zeros((ntime, nbody))
    tv = np.zeros((ntime, nbody, 3))
    cv = np.zeros((ntime, nbody, 3))
    bv = np.zeros((ntime, nbody, 3))
    Crs = np.zeros((ntime, nbody, 3, 3))
    lateral_bend = np.zeros((ntime, nbody))
    back_bend = np.zeros((ntime, nbody))

    from sim import aero_forces_more

    for i in np.arange(ntime):
        t = ts[i]

        out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

        pi, dpi, ddpi = out['p'], out['dp'], out['ddp']
        tvi, cvi, bvi = out['tv'], out['cv'], out['bv']

        # positions, velocities, and accelerations
        ri, dri, ddri = rotate(C.T, pi), rotate(C.T, dpi), rotate(C.T, ddpi)
        # Ri = Ro + ri
        dRi = dRo + dri + np.cross(omg, ri)

        # aerodynamic forces
        fout = aero_forces_more(tvi, cvi, bvi, C, dRi, ds, c, rho, aero_interp)

        Fli, Fdi, dRiBC, aoai, Rei, betai, dRiTC = fout

        # store the values
        Fl[i] = Fli
        Fd[i] = Fdi
        Faero[i] = Fli + Fdi
        Ml[i] = np.cross(pi, Fli)
        Md[i] = np.cross(pi, Fdi)
        Maero[i] = np.cross(pi, Faero[i])
        Re[i] = Rei
        aoa[i] = np.rad2deg(aoai)
        beta[i] = np.rad2deg(betai)
        dR_BC[i] = dRiBC
        dR_TC[i] = dRiTC

        # store the values
        theta[i] = out['theta']
        psi[i] = out['psi']
        dthetads[i] = out['dthetads']
        dpsids[i] = out['dpsids']
        kap[i] = out['kap']

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
        lateral_bend[i] = out['lateral_bend']
        back_bend[i] = out['back_bend']

    out = dict(Fl=Fl, Fd=Fd, Faero=Faero, Ml=Ml, Md=Md, Maero=Maero,
               Re=Re, aoa=aoa, beta=beta, dR_BC=dR_BC, dR_TC=dR_TC,
               dRo=dRo, ts=ts,
               theta=theta, psi=psi, dthetads=dthetads, dpsids=dpsids,
               kap=kap,
               p=p, dp=dp, ddp=ddp,
               dpds=dpds, ddpdds=ddpdds,
               tv=tv, cv=cv, bv=bv, Crs=Crs,
               lateral_bend=lateral_bend, back_bend=back_bend)

    return out


# %%

L = .7  # .686  # m
ds = .01  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)

# chord length
chord_para, chord_interp = chord_dist(s, ds, L)
c = chord_para
area = np.sum(ds * c)  # m^2

# mass distribution from wing loading
wing_loading = 29  # N/m^2 from socha 2005
g = 9.81  # m/s^2
mass_total = wing_loading * area / g
mass_para, mass_interp = mass_dist(s, ds, L, mass_total)
m = mass_para
weight = np.sum(m * g)

nu_theta = 1
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = .2  # .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(15)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                  amp_theta=amp_theta, damp_theta=damp_theta,
                  d_theta=d_theta, L=L, theta_max=theta_max,
                  frac_theta_max=frac_theta_max,
                  amp_theta_fun=amp_theta_fun)
psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                psi_max=psi_max, frac_psi_max=frac_psi_max,
                amp_psi_fun=amp_psi_fun)


#t = 0
#ho_args = (s, t, m, n_neck, theta_dict, psi_dict)
#
#phi_theta = fmin(func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
#phi_theta = float(phi_theta)
#phi_psi = 2 * phi_theta
#
#theta_dict['phi_theta'] = phi_theta
#psi_dict['phi_psi'] = phi_psi

rho = 1.165  # 30 C
g = 9.81

# aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)

args1 = (s, m, n_neck, theta_dict, psi_dict)
args2 = (ds, c, g, rho, aero_interp)
args = (args1, args2)


phi_rng = np.r_[0, 5, 10, 15, 20, 25, 30, 35]

dd = {}
for phi_max_deg in phi_rng:

    print(phi_max_deg)

    psi_max = np.deg2rad(phi_max_deg)
    frac_psi_max = 0
    amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
    amp_psi = psi_max * amp_psi_fun
    damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
    d_psi = np.deg2rad(-0)

    psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                    amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                    psi_max=psi_max, frac_psi_max=frac_psi_max,
                    amp_psi_fun=amp_psi_fun)

    args1 = (s, m, n_neck, theta_dict, psi_dict)
    args = (args1, args2)

    out = aero_effects(args, wing_loading)

    dd[phi_max_deg] = out


# %%

colors = sns.color_palette('cubehelix', len(phi_rng))
# colors = sns.dark_palette('purple', len(phi_rng))
#colors = sns.dark_palette('navy', len(phi_rng), reverse=True)
#colors = sns.cubehelix_palette(len(phi_rng))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 9.75))
vline_kwargs = dict(color='gray', lw=1.5, ls='--')
ax1.axhline(0, **vline_kwargs)
ax2.axhline(0, **vline_kwargs)
ax3.axhline(1, **vline_kwargs)
ax3.axhline(0, **vline_kwargs)
for i, key in enumerate(phi_rng):
    d = dd[key]
    ts = d['ts'] * f_theta

    Faero = d['Faero'].sum(axis=1) / weight
    ax1.plot(ts, Faero[:, 0], c=colors[i], label=key)
    ax2.plot(ts, Faero[:, 1], c=colors[i])
    ax3.plot(ts, Faero[:, 2], c=colors[i])

ax1.set_ylabel('Forward force')
ax2.set_ylabel('Transver force')
ax3.set_ylabel('Vertical force')
ax3.set_xlabel(r'Time ($\mathrm{T}_\mathrm{undulation}$)')

ax1.legend(loc='upper right', frameon=True, framealpha=.8)
ax1.set_xlim(ts.min(), ts.max())
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10.75, 5.5))

for i, key in enumerate(phi_rng):
    d = dd[key]
    ts = d['ts'] * f_theta
    aoa = d['aoa']

    ax1.plot(s / L, aoa.mean(axis=0), c=colors[i], label=key)
    ax2.plot(ts, aoa.mean(axis=1), c=colors[i])

ax1.legend(loc='lower left', ncol=2, frameon=True)
ax1.set_xlabel('Length (SVL)')
ax2.set_xlabel(r'Time ($\mathrm{T}_\mathrm{undulation}$)')
ax1.set_ylabel('Average angle of attack')

ax1.set_ylim(0, 90)

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)




# %%




# %%

L = .7  # .686  # m
ds = .01  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)  # number of points for neck

# chord length
# mean_chord = weight / wing_loading / L
# c = mean_chord * np.ones(nbody)
chord_para, chord_interp = chord_dist(s, ds, L)
c = chord_para
area = np.sum(ds * c)  # m^2

# uniform mass distribution
#mass_total = .0405  # kg
#rho_body = mass_total / L  # linear mass density
#m = np.ones(nbody) * rho_body * L / nbody

# mass distribution from wing loading
wing_loading = 29  # N/m^2 from socha 2005
g = 9.81  # m/s^2
mass_total = wing_loading * area / g
mass_para, mass_interp = mass_dist(s, ds, L, mass_total)
m = mass_para
weight = np.sum(m * g)

ntime = 201
ts = np.linspace(0, 3, ntime)

#lambdak_theta = 0.6
#nu_theta = 5 / 3.  # m
nu_theta = 1  # 1.16667 == 5/3 * .7
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

#lambdak_psi = lambdak_theta / 2
nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = .2  # .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
# damp_theta = -2 * frac_theta_max * theta_max / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(10)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                  amp_theta=amp_theta, damp_theta=damp_theta,
                  d_theta=d_theta, L=L, theta_max=theta_max,
                  frac_theta_max=frac_theta_max,
                  amp_theta_fun=amp_theta_fun)
psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                psi_max=psi_max, frac_psi_max=frac_psi_max,
                amp_psi_fun=amp_psi_fun)


t = 0
ho_args = (s, t, m, n_neck, theta_dict, psi_dict)

phi_theta = fmin(func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
phi_theta = float(phi_theta)
phi_psi = 2 * phi_theta

#phi_theta = phi_theta_newton
#phi_psi = phi_psi_newton

theta_dict['phi_theta'] = phi_theta
psi_dict['phi_psi'] = phi_psi

## turn off undulation
#theta_dict['f_theta'] = 0
#psi_dict['f_psi'] = 0

ntime = 100
ts = np.linspace(0, 1 / f_theta, ntime)

p = np.zeros((ntime, nbody, 3))
thetas = np.zeros((ntime, nbody))
psis = np.zeros((ntime, nbody))

for i in np.arange(ntime):
    t = ts[i]
    out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)
    p[i] = out['p']
    thetas[i] = out['theta']
    psis[i] = out['psi']


# %%

kwargs_theta = dict(c=np.rad2deg(thetas[0]), s=60, cmap=plt.cm.viridis,
                      linewidths=0, zorder=2)

kwargs_psi = dict(c=np.rad2deg(psis[0]), s=60, cmap=plt.cm.coolwarm,
                      linewidths=0, zorder=1)

#scatter_kwargs = dict(c=s / L, s=60, cmap=plt.cm.viridis,
#                      linewidths=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ax2.axvline(0, color='gray', lw=1, zorder=0)
ax2.axhline(0, color='gray', lw=1, zorder=0)
ax2.axhline(20, color='gray', lw=1, zorder=0)
ax1.axhline(0, color='gray', lw=1, zorder=0)

stheta = ax1.scatter(s / L, np.rad2deg(thetas[0]), **kwargs_theta)
ax1.plot(s / L, np.rad2deg(amp_theta), c='gray', lw=1)
ax1.plot(s / L, -np.rad2deg(amp_theta), c='gray', lw=1)

spsi = ax1.scatter(s / L, np.rad2deg(psis[0]), **kwargs_psi)
ax1.plot(s / L, np.rad2deg(amp_psi), c='gray', lw=1)
ax1.plot(s / L, -np.rad2deg(amp_psi), c='gray', lw=1)

ax1.set_xlim(0, 1)

pi = 100 * p[0]
syx = ax2.scatter(pi[:, 1], pi[:, 0], **kwargs_theta)
syz = ax2.scatter(pi[:, 1], 20 - pi[:, 2], **kwargs_psi)
ax2.set_aspect('equal', adjustable='box')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

ax1.set_xlabel('Distance along body (SVL)')
ax1.set_ylabel('Body angles')
ax2.set_xlabel('Laternal excursion (cm)')
ax2.set_ylabel('Fore-aft and vertical excursions (cm)')

sns.despine()
fig.set_tight_layout(True)

def animate(i):
    pi = 100 * p[i]
    theta = thetas[i]
    psi = psis[i]

    stheta.set_offsets(np.c_[s/L, np.rad2deg(theta)])
    stheta.set_array(np.rad2deg(theta))

    spsi.set_offsets(np.c_[s/L, np.rad2deg(psi)])
    spsi.set_array(np.rad2deg(psi))

    syx.set_offsets(np.c_[pi[:, 1], pi[:, 0]])
    syx.set_array(np.rad2deg(theta))

    syz.set_offsets(np.c_[pi[:, 1], 20 - pi[:, 2]])
    syz.set_array(np.rad2deg(psi))

    return stheta, spsi, syx, syz


dt = np.diff(ts).mean()

from matplotlib.animation import FuncAnimation

slowed = 5
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False)

ani.save('5X aerial serpnoid curve.mp4',
         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


# %%


lbody, = ax.plot([], [], 'o-', c=emerald_green)
phead, = ax.plot([], [], 'o', c=emerald_green, ms=13)

lbody_n, = ax.plot([], [], 'o-', c=emerald_green, alpha=.5)
phead_n, = ax.plot([], [], 'o', c=emerald_green, ms=13, alpha=.5)

ax.set_aspect('equal')
xlim = (-.2, .2)
ylim = (-.135, .135)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

ax.axis('off')
fig.set_tight_layout(True)

def init():
    lbody.set_data([], [])
    phead.set_data([], [])
    return lbody, phead, lbody_n, phead_n


def animate(i):
    t = i * dt

    out = sim.serp_pos_vel_acc_tcb(s, t, A, k, w, phi, n_neck=5)
    p, v, a, tang, kap, tcb = out
    xs, ys, _ = p.T
    lbody.set_data(xs, ys)
    phead.set_data(xs[0], ys[0])

    out = sim.serp_pos_vel_acc_tcb(s, t, A, k, w, phi, n_neck=0)
    p, v, a, tang, kap, tcb = out
    xs, ys, _ = p.T
    lbody_n.set_data(xs, ys)
    phead_n.set_data(xs[0], ys[0])

    return lbody, phead, lbody_n, phead_n

slowed = 5
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=True, init_func=init)



# %%

fig, ax = plt.subplots()
ax.axhline(1, color='gray', ls='--', lw=1.5)
ax.plot(ts, Faero[:, :, 2].sum(axis=1) / weight)
ax.plot(ts, Fl[:, :, 2].sum(axis=1) / weight)
ax.plot(ts, Fd[:, :, 2].sum(axis=1) / weight)
ax.set_ylim(0, 1.2)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', ls='--', lw=1.5)
ax.plot(ts, Faero[:, :, 0].sum(axis=1) / weight)
ax.plot(ts, Fl[:, :, 0].sum(axis=1) / weight)
ax.plot(ts, Fd[:, :, 0].sum(axis=1) / weight)
#ax.set_ylim(-.5, .5)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', ls='--', lw=1.5)
ax.plot(ts, Faero[:, :, 1].sum(axis=1) / weight)
ax.plot(ts, Fl[:, :, 1].sum(axis=1) / weight)
ax.plot(ts, Fd[:, :, 1].sum(axis=1) / weight)
#ax.set_ylim(-.1, .1)
sns.despine()
fig.set_tight_layout(True)


# %%

# pitch momemnt
fig, ax = plt.subplots()
#ax.axhline(1, color='gray', ls='--', lw=1.5)
ax.plot(ts, Maero[:, :, 1].sum(axis=1))
ax.plot(ts, Ml[:, :, 1].sum(axis=1))
ax.plot(ts, Md[:, :, 1].sum(axis=1))
#ax.set_ylim(0, 1.2)
sns.despine()
fig.set_tight_layout(True)


# pitch vs roll moment
fig, ax = plt.subplots()
ax.axvline(0, color='gray', ls='--', lw=1.5)
ax.axhline(0, color='gray', ls='--', lw=1.5)

ax.axvline(Ml[:, :, 1].sum(axis=1).mean(), lw=2, c=bmap[0])
ax.axhline(Ml[:, :, 0].sum(axis=1).mean(), lw=2, c=bmap[0])
ax.plot(Ml[:, :, 1].sum(axis=1), Ml[:, :, 0].sum(axis=1), '-o', c=bmap[0])

ax.axvline(Md[:, :, 1].sum(axis=1).mean(), lw=2, c=bmap[1])
ax.axhline(Md[:, :, 0].sum(axis=1).mean(), lw=2, c=bmap[1])
ax.plot(Md[:, :, 1].sum(axis=1), Md[:, :, 0].sum(axis=1), '-^', c=bmap[1])


ax.axvline(Maero[:, :, 1].sum(axis=1).mean(), lw=2, c=bmap[2])
ax.axhline(Maero[:, :, 0].sum(axis=1).mean(), lw=2, c=bmap[2])
ax.plot(Maero[:, :, 1].sum(axis=1), Maero[:, :, 0].sum(axis=1), '-^', c=bmap[2])

ax.set_aspect('equal', adjustable='box-forced')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(s / L, aoa.mean(axis=0))
ax2.plot(ts * f_theta, aoa.mean(axis=1))
#ax1.set_ylim(0, 45)
ax1.set_xlabel('Length (SVL)')
ax2.set_xlabel(r'Time ($\mathrm{T}_\mathrm{undulation}$)')
ax1.set_ylabel('Average angle of attack')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)


# %% Contour plots


# %%

fig, ax = plt.subplots()
ax.hist(Faero[:, :, 2].sum(axis=1) / weight, bins=30)
sns.despine()
fig.set_tight_layout(True)


# %% Max chord width and wing loading vs. L

# this is probably a carpet plot


# %% Figure of the effects of the different serp parameters

fn_plots = './Figures/serp3d/{}'

L = .7  # .686  # m
ds = .001  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)

# mass distribution
mass_total = .0405  # kg
rho_body = mass_total / L  # linear mass density
m = np.ones(nbody) * rho_body * L / nbody

# base configuration
nu_theta = 1
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = 0  # .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(10)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict_base = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                       amp_theta=amp_theta, damp_theta=damp_theta,
                       d_theta=d_theta, L=L, theta_max=theta_max,
                       frac_theta_max=frac_theta_max,
                       amp_theta_fun=amp_theta_fun)
psi_dict_base = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                     amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                     psi_max=psi_max, frac_psi_max=frac_psi_max,
                     amp_psi_fun=amp_psi_fun)

# serpenoid parameters to sweep over
#nus = np.r_[.8, .9, 1, 1.1, 1.2]
#theta0s = np.r_[70, 80, 90, 100, 100]
#psi0s = np.r_[0, 5, 10, 15, 20]  # mid off
#fracs_theta = np.r_[0, .1, .2, .3, .4]
#fracs_psi = np.r_[0, .1, .2, .3, .4]

nus = np.r_[.9, 1, 1.2]
theta0s = np.r_[70, 90, 110]
psi0s = np.r_[0, 15, 30]  # mid off
fracs_theta = np.r_[0, .2, .4]
fracs_psi = np.r_[0, .2, .4]

sweep_type = ['num_waves', 'theta0', 'psi0', 'frac_theta', 'frac_psi']
var_sweep = (nus, theta0s, psi0s, fracs_theta, fracs_psi)

for i in np.arange(5):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                                   figsize=(10, 5))

    for j in np.arange(3):
        theta_dict = theta_dict_base.copy()
        psi_dict = psi_dict_base.copy()

        if i == 0:
            nu = var_sweep[i][j]
            theta_dict['nu_theta'] = nu
            psi_dict['nu_psi'] = 2 * nu
        elif i == 1:
            theta_max = np.deg2rad(var_sweep[i][j])
            frac_theta_max = 0
            amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
            amp_theta = theta_max * amp_theta_fun
            damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
            theta_dict['amp_theta'] = amp_theta
            theta_dict['damp_theta'] = damp_theta
        elif i == 2:
            psi_max = np.deg2rad(var_sweep[i][j])
            frac_psi_max = 0
            amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
            amp_psi = psi_max * amp_psi_fun
            damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
            psi_dict['amp_psi'] = amp_psi
            psi_dict['damp_psi'] = damp_psi
        elif i == 3:
            theta_max = theta_dict_base['theta_max']
            frac_theta_max = var_sweep[i][j]
            amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
            amp_theta = theta_max * amp_theta_fun
            damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
            theta_dict['amp_theta'] = amp_theta
            theta_dict['damp_theta'] = damp_theta
        elif i == 4:
            psi_max = np.deg2rad(15)  # psi_dict_base['psi_max']
            frac_psi_max = var_sweep[i][j]
            amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
            amp_psi = psi_max * amp_psi_fun
            damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
            psi_dict['amp_psi'] = amp_psi
            psi_dict['damp_psi'] = damp_psi

        out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)
        # out = dict(p=p, dpds=dp, ddpdds=ddp, theta=theta, psi=psi,
        #            dthetads=dtheta, dpsids=dpsi, kap=kap)

        x, y, z = 100 * out['p'].T

        scatter_kwargs = dict(c=s, s=40, cmap=plt.cm.viridis,
                      linewidths=0)
        ax1.scatter(x, y, **scatter_kwargs)
#        ax1.plot(x, y, c='gray')
        ax2.scatter(y[::-1], 2 * z[::-1], **scatter_kwargs)
#        ax2.plot(y, 2 * z, c='gray')

    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    # ax1.set_title(i)
    ax1.axvline(0, color='gray', lw=1, zorder=0)
    ax1.axhline(0, color='gray', lw=1, zorder=0)
    ax2.axvline(0, color='gray', lw=1, zorder=0)
    ax2.axhline(0, color='gray', lw=1, zorder=0)
    # ax1.set_aspect('equal', adjustable='box')

    ax1.set_xlabel('y (cm)')
    ax1.set_ylabel('x (cm)')
    ax2.set_xlabel('y (cm)')
    ax2.set_ylabel('z (cm)')

    plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
    sns.despine()
    fig.set_tight_layout(True)

    savename = fn_plots.format(sweep_type[i] + '.pdf')
    fig.savefig(savename, bbox_inches='tight')


# %%

fn_plots = './Figures/serp3d/{}'

L = .7  # .686  # m
ds = .01  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)

# mass distribution
mass_total = .0405  # kg
rho_body = mass_total / L  # linear mass density
m = np.ones(nbody) * rho_body * L / nbody

# base configuration
nu_theta = 1
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta # + np.pi / 4

theta_max = np.deg2rad(90)
frac_theta_max = 0  # .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(10)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict_base = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                       amp_theta=amp_theta, damp_theta=damp_theta,
                       d_theta=d_theta, L=L, theta_max=theta_max,
                       frac_theta_max=frac_theta_max,
                       amp_theta_fun=amp_theta_fun)
psi_dict_base = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                     amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                     psi_max=psi_max, frac_psi_max=frac_psi_max,
                     amp_psi_fun=amp_psi_fun)

# serpenoid parameters to sweep over
#nus = np.r_[.8, .9, 1, 1.1, 1.2]
#theta0s = np.r_[70, 80, 90, 100, 100]
#psi0s = np.r_[0, 5, 10, 15, 20]  # mid off
#fracs_theta = np.r_[0, .1, .2, .3, .4]
#fracs_psi = np.r_[0, .1, .2, .3, .4]

nus = np.r_[.9, 1, 1.2]
theta0s = np.r_[70, 90, 110]
psi0s = np.r_[0, 15, 30]  # mid off
fracs_theta = np.r_[0, .2, .4]
fracs_psi = np.r_[0, .2, .4]

sweep_type = ['num_waves', 'theta0', 'psi0', 'frac_theta', 'frac_psi']
var_sweep = (nus, theta0s, psi0s, fracs_theta, fracs_psi)

for i in np.arange(5):

    # setup the figure and axes
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 450))
    Nframe = np.eye(3)
    frame_c = [bmap[2], bmap[1], bmap[0]]
    _gray = .4
    gray = (_gray, _gray, _gray)
    frame_c = [gray, gray, gray]
    for ii in np.arange(3):
        mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    # mlab.orientation_axes()

    for j in np.arange(3):
        theta_dict = theta_dict_base.copy()
        psi_dict = psi_dict_base.copy()

        if i == 0:
            nu = var_sweep[i][j]
            theta_dict['nu_theta'] = nu
            psi_dict['nu_psi'] = 2 * nu
        elif i == 1:
            theta_max = np.deg2rad(var_sweep[i][j])
            frac_theta_max = 0
            amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
            amp_theta = theta_max * amp_theta_fun
            damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
            theta_dict['amp_theta'] = amp_theta
            theta_dict['damp_theta'] = damp_theta
        elif i == 2:
            psi_max = np.deg2rad(var_sweep[i][j])
            frac_psi_max = 0
            amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
            amp_psi = psi_max * amp_psi_fun
            damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
            psi_dict['amp_psi'] = amp_psi
            psi_dict['damp_psi'] = damp_psi
        elif i == 3:
            theta_max = theta_dict_base['theta_max']
            frac_theta_max = var_sweep[i][j]
            amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
            amp_theta = theta_max * amp_theta_fun
            damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
            theta_dict['amp_theta'] = amp_theta
            theta_dict['damp_theta'] = damp_theta
        elif i == 4:
            psi_max = np.deg2rad(15)  # psi_dict_base['psi_max']
            frac_psi_max = var_sweep[i][j]
            amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
            amp_psi = psi_max * amp_psi_fun
            damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
            psi_dict['amp_psi'] = amp_psi
            psi_dict['damp_psi'] = damp_psi

        out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

        x, y, z = out['p'].T

        offx = -.2 * np.ones(nbody)
        offy = -.2 * np.ones(nbody)
        offz = -.06 * np.ones(nbody)
        mlab.plot3d(x, y, z, tube_radius=.002, color=bmap[j])
        mlab.plot3d(x, offy, z, tube_radius=.002, color=bmap[j])
        mlab.plot3d(offx, y, z, tube_radius=.002, color=bmap[j])
        # mlab.plot3d(x, y, offz, tube_radius=.002, color=bmap[j])

    mlab.view(*(45., 54.75, 1, np.r_[-0.027, -0.0276,  0.0150]))
    fig.scene.camera.zoom(1.7)

    # savename = fn_plots.format(sweep_type[i] + '.png')
    # mlab.savefig(savename, size=(3*750, 3*450))


# %%

L = .7  # .686  # m
ds = .01  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)

# mass distribution
mass_total = .0405  # kg
rho_body = mass_total / L  # linear mass density
m = np.ones(nbody) * rho_body * L / nbody

# base configuration
nu_theta = 1
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(10)
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                       amp_theta=amp_theta, damp_theta=damp_theta,
                       d_theta=d_theta, L=L, theta_max=theta_max,
                       frac_theta_max=frac_theta_max,
                       amp_theta_fun=amp_theta_fun)
psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                     amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                     psi_max=psi_max, frac_psi_max=frac_psi_max,
                     amp_psi_fun=amp_psi_fun)

out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

fig, ax = plt.subplots()
ax.plot(s / L, np.rad2deg(out['theta']), c=bmap[0])
ax.plot(s / L, np.rad2deg(amp_theta), c=bmap[0], lw=1)
ax.plot(s / L, -np.rad2deg(amp_theta), c=bmap[0], lw=1)
ax.plot(s / L, np.rad2deg(out['psi']), c=bmap[1])
ax.plot(s / L, np.rad2deg(amp_psi), c=bmap[1], lw=1)
ax.plot(s / L, -np.rad2deg(amp_psi), c=bmap[1], lw=1)
sns.despine()
fig.set_tight_layout(True)


kwargs_theta = dict(c=np.rad2deg(out['theta']), s=60, cmap=plt.cm.viridis,
                      linewidths=0, zorder=2)

kwargs_psi = dict(c=np.rad2deg(out['psi']), s=60, cmap=plt.cm.coolwarm,
                      linewidths=0, zorder=1)

#scatter_kwargs = dict(c=s / L, s=60, cmap=plt.cm.viridis,
#                      linewidths=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ax2.axvline(0, color='gray', lw=1, zorder=0)
ax2.axhline(0, color='gray', lw=1, zorder=0)
ax2.axhline(20, color='gray', lw=1, zorder=0)
ax1.axhline(0, color='gray', lw=1, zorder=0)

ax1.scatter(s / L, np.rad2deg(out['theta']), **kwargs_theta)
ax1.plot(s / L, np.rad2deg(amp_theta), c='gray', lw=1)
ax1.plot(s / L, -np.rad2deg(amp_theta), c='gray', lw=1)

ax1.scatter(s / L, np.rad2deg(out['psi']), **kwargs_psi)
ax1.plot(s / L, np.rad2deg(amp_psi), c='gray', lw=1)
ax1.plot(s / L, -np.rad2deg(amp_psi), c='gray', lw=1)

ax1.set_xlim(0, 1)

ax2.scatter(100 * out['p'][:, 1], 100 * out['p'][:, 0], **kwargs_theta)
ax2.scatter(100 * out['p'][:, 1], 20 - 100 * out['p'][:, 2], **kwargs_psi)
ax2.set_aspect('equal', adjustable='box')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax1.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax1.set_yticklabels(newticks)

ax1.set_xlabel('Distance along body (SVL)')
ax1.set_ylabel('Body angles')
ax2.set_xlabel('Laternal excursion (cm)')
ax2.set_ylabel('Fore-aft and vertical excursions (cm)')

sns.despine()
fig.set_tight_layout(True)


# %% Effect of out-of-plane on aerodynamic forces

L = .7  # .686  # m
ds = .01  # m
#s = np.r_[0, ds / 2 + np.arange(0, L, ds)]  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec
neck_length = .05  # m
neck_length = .065
n_neck = np.floor(neck_length / ds).astype(np.int)

# chord length
# mean_chord = weight / wing_loading / L
# c = mean_chord * np.ones(nbody)
chord_para, chord_interp = chord_dist(s, ds, L)
c = chord_para
area = np.sum(ds * c)  # m^2

# uniform mass distribution
#mass_total = .0405  # kg
#rho_body = mass_total / L  # linear mass density
#m = np.ones(nbody) * rho_body * L / nbody

# mass distribution from wing loading
wing_loading = 29  # N/m^2 from socha 2005
g = 9.81  # m/s^2
mass_total = wing_loading * area / g
mass_para, mass_interp = mass_dist(s, ds, L, mass_total)
m = mass_para
weight = np.sum(m * g)

# mass distribution
#mass_total = .0405  # kg
#rho_body = mass_total / L  # linear mass density
#m = np.ones(nbody) * rho_body * L / nbody

# base configuration
nu_theta = 1
f_theta = 1.4  # Hz
phi_theta = np.deg2rad(0)

nu_psi = 2 * nu_theta
f_psi = 2 * f_theta
phi_psi = 2 * phi_theta

theta_max = np.deg2rad(90)
frac_theta_max = .2
amp_theta_fun = np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
amp_theta = theta_max * amp_theta_fun
damp_theta = (amp_theta[-1] - amp_theta[0]) / (s[-1] - s[0])
d_theta = np.deg2rad(0)

psi_max = np.deg2rad(10) * .1  # * 0 makes it planar
frac_psi_max = 0
amp_psi_fun = np.linspace(1 - frac_psi_max, 1 + frac_psi_max, nbody)
amp_psi = psi_max * amp_psi_fun
damp_psi = (amp_psi[-1] - amp_psi[0]) / (s[-1] - s[0])
d_psi = np.deg2rad(-0)

theta_dict = dict(nu_theta=nu_theta, f_theta=f_theta, phi_theta=phi_theta,
                       amp_theta=amp_theta, damp_theta=damp_theta,
                       d_theta=d_theta, L=L, theta_max=theta_max,
                       frac_theta_max=frac_theta_max,
                       amp_theta_fun=amp_theta_fun)
psi_dict = dict(nu_psi=nu_psi, f_psi=f_psi, phi_psi=phi_psi,
                     amp_psi=amp_psi, damp_psi=damp_psi, d_psi=d_psi, L=L,
                     psi_max=psi_max, frac_psi_max=frac_psi_max,
                     amp_psi_fun=amp_psi_fun)

t = 0
ho_args = (s, t, m, n_neck, theta_dict, psi_dict)

phi_theta = fmin(func_ho_to_min, phi_theta, args=ho_args, ftol=1e-7, xtol=1e-7)
phi_theta = float(phi_theta)
phi_psi = 2 * phi_theta

theta_dict['phi_theta'] = phi_theta
psi_dict['phi_psi'] = phi_psi


out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

#print out['p']


## %% Check yaw, pitch, and roll  sign convnections directions
#
#ang = np.deg2rad(np.r_[0, 0, 45])  # yaw, pitch, roll
#C = sim.euler2C(ang[0], ang[1], ang[2])
#
##C = C.T
#
#p = out['p']
#P = sim.rotate(C.T, p)
#
#fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
#
#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]
#
#nframe = sim.rotate(C.T, Nframe)
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
#                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)
#
#for ii in np.arange(3):
#    mlab.quiver3d(nframe[ii, 0], nframe[ii, 1], nframe[ii, 2], scale_factor=.05,
#                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
#
#
#mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=.004, color=bmap[1],
#            opacity=.5)
#
#mlab.plot3d(P[:, 0], P[:, 1], P[:, 2], tube_radius=.004, color=bmap[1],
#            opacity=1)

#mlab.yaw = 90


# %% Run a dynamics simulation

import sim
import aerodynamics

rho = 1.165  # 30 C
g = 9.81
#g = 0

# aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)
# aero_interp = None
#aero_interp = None

tend = None
#tend = 10


# arguments
#args = (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)
#
#params = (dt, L, nbody, mass_total, rho_body, neck_length, Stot, \
#            wing_loading, wave_length_m, freq_undulation_hz)

args1 = (s, m, n_neck, theta_dict, psi_dict)
args2 = (ds, c, g, rho, aero_interp)
args = (args1, args2)

# initial conditions
Ro0 = np.r_[0, 0, 10]
dRo0 = np.r_[1.7, 0, 0]
ang0 = np.deg2rad(np.r_[0, -15, 0])  # yaw, pitch, roll
dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
omg0 = np.dot(C0.T, omg0_body)
# omg0 = np.deg2rad(np.r_[0, 0, 0])
soln0 = np.r_[Ro0, dRo0, omg0, ang0]


# %% Run a dynamics simulation

import time

# perform the integration
out = integrate(soln0, args, dt, tend=tend, print_time=True)

# extract values
ts, Ro, dRo, omg, ang = out
yaw, pitch, roll = ang.T
ntime = len(ts)


# %%

fig, ax = plt.subplots()
ax.plot(Ro[:, 0], Ro[:, 2])
ax.set_aspect('equal', adjustable='box')
ax.set_ylim(0, Ro[:, 2].max())
ax.set_xlabel('x')
ax.set_ylabel('z')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(Ro[:, 1], Ro[:, 0])
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('y')
ax.set_ylabel('x')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(dRo[:, 0], dRo[:, 2])
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
ax.set_xlabel('vx')
ax.set_ylabel('vz')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(ts, np.rad2deg(pitch), label='pitch')
ax.plot(ts, np.rad2deg(roll), label='roll')
ax.plot(ts, np.rad2deg(yaw), label='yaw')
#ax.plot(np.rad2deg(pitch))
#ax.plot(np.rad2deg(roll))
#ax.plot(np.rad2deg(yaw))
ax.legend(loc='upper left')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(ts, np.rad2deg(pitch), label='pitch')
ax.plot(ts, np.rad2deg(roll), label='roll')
ax.plot(ts, 180 - np.rad2deg(yaw), label='yaw')
#ax.plot(np.rad2deg(pitch))
#ax.plot(np.rad2deg(roll))
#ax.plot(np.rad2deg(yaw))
ax.legend(loc='upper left')
sns.despine()
fig.set_tight_layout(True)


# %%

ntime = len(ts)

# kinematics simulation
theta = np.zeros((ntime, nbody))
psi = np.zeros((ntime, nbody))
dthetads = np.zeros((ntime, nbody))
dpsids = np.zeros((ntime, nbody))
p = np.zeros((ntime, nbody, 3))
dp = np.zeros((ntime, nbody, 3))
dpds = np.zeros((ntime, nbody, 3))
ddpdds = np.zeros((ntime, nbody, 3))
ddp = np.zeros((ntime, nbody, 3))
kap = np.zeros((ntime, nbody))
tv = np.zeros((ntime, nbody, 3))
cv = np.zeros((ntime, nbody, 3))
bv = np.zeros((ntime, nbody, 3))
Crs = np.zeros((ntime, nbody, 3, 3))
lateral_bend = np.zeros((ntime, nbody))
back_bend = np.zeros((ntime, nbody))

for i in np.arange(ntime):
    t = ts[i]

#    out = chrysoserp(s, t,  mi, n_neck, theta_dict, psi_dict)
    out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    # store the values
    theta[i] = out['theta']
    psi[i] = out['psi']
    dthetads[i] = out['dthetads']
    dpsids[i] = out['dpsids']
    kap[i] = out['kap']

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
    lateral_bend[i] = out['lateral_bend']
    back_bend[i] = out['back_bend']


# %%

fig, ax = plt.subplots()
ax.plot(s, np.rad2deg(lateral_bend[i]))
ax.plot(s, np.rad2deg(back_bend[i]))
sns.despine()
fig.set_tight_layout(True)


# %%

i = 0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)

for i in np.arange(0, ntime, 40):
    ax1.plot(s, np.rad2deg(lateral_bend[i]), c=bmap[0])
    ax2.plot(s, np.rad2deg(back_bend[i]), c=bmap[1])
sns.despine()
#    plt.cla()
#    plt.draw()
fig.set_tight_layout(True)


SS, TT = np.meshgrid(s, ts)

fig, ax = plt.subplots()
vmax = np.rad2deg(np.max(np.abs(lateral_bend)))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, np.rad2deg(lateral_bend), cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'lateral bending angle (deg)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)


fig, ax = plt.subplots()
vmax = np.rad2deg(np.max(np.abs(back_bend)))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, np.rad2deg(back_bend), cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'back bending angle (deg)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)


# %%

ho = np.zeros((ntime, 3))
dho = np.zeros((ntime, 3))
for i in np.arange(ntime):
    ho[i] = np.sum((m * np.cross(p[i], dp[i]).T).T, axis=0)
#    ho[i] = np.sum((np.cross(p[i], (m * dp[i].T).T).T).T, axis=0)
    dho[i] = np.sum((m * np.cross(p[i], ddp[i]).T).T, axis=0)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', ls='--', lw=1.5)
ax.plot(ts, ho[:, 0], 'o-', label='x')
ax.plot(ts, ho[:, 1], 'o-', label='y')
ax.plot(ts, ho[:, 2], 'o-', label='z')
ax.plot(ts, np.linalg.norm(ho, axis=1), 'o-', label='norm')
ax.legend(loc='best', frameon=True)
sns.despine()
fig.set_tight_layout(True)


# %% Apply airfoil shape

# np.rad2deg(np.c_[yaw, pitch, roll])



# %%


i = -1

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

#body = mlab.plot3d(p[:, 0], p[:, 1], p[:, 2],
#                   tube_radius=.004, color=bmap[1], opacity=.5)

body = mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
                   tube_radius=.004, color=bmap[1], opacity=.5)

mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
              color=(0, 0, 0), mode='arrow', resolution=64)

Ci = sim.euler2C(yaw[i], pitch[i], roll[i])
nframe = sim.rotate(Ci.T, Nframe)
P = sim.rotate(Ci.T, p[i])
#P = sim.rotate(Ci.T, p)

for ii in np.arange(3):
    mlab.quiver3d(nframe[ii, 0], nframe[ii, 1], nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

body = mlab.plot3d(P[:, 0], P[:, 1], P[:, 2],
                   tube_radius=.004, color=bmap[1])

#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)
#
## airfoil orientation (c, b, t) directions
#cv_quiv = mlab.quiver3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
#                        cv[i, :, 0], cv[i, :, 1], cv[i, :, 2],
#                        color=bmap[0], scale_factor=.05,
#                        resolution=64, mode='arrow')
#bv_quiv = mlab.quiver3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
#                        bv[i, :, 0], bv[i, :, 1], bv[i, :, 2],
#                        color=bmap[3], scale_factor=.05,
#                        resolution=64, mode='arrow')
#mlab.quiver3d(p[:, 0], p[:, 1], p[:, 2],
#              tdir[:, 0], tdir[:, 1], tdir[:, 2],
#              color=bmap[2], scale_factor=.05,
#              resolution=64, mode='arrow')

#horz = mlab.plot3d(p[i, :, 0], p[i, :, 1], 0 * p[i, :, 2],
#                   tube_radius=.002, color=bmap[0])
#
#vert = mlab.plot3d(p[i, :, 0], 0 * p[i, :, 1], p[i, :, 2],
#                   tube_radius=.002, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                                 z=foils[i, :, :, 2], scalars=foil_color[i])
            cv_quiv.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2],
                                    u=cv[i, :, 0], v=cv[i, :, 1], w=cv[i, :, 2])
            bv_quiv.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2],
                                    u=bv[i, :, 0], v=bv[i, :, 1], w=bv[i, :, 2])
            yield
manim = anim()
mlab.show()


# %%

#fig, ax = plt.subplots()
#ax.axhline(0, color='gray', ls='--', lw=1.5)
#ax.plot(s, theta_d, c=bmap[0], label='theta')
#ax.plot(s, theta_amp_d, ':', c=bmap[0])
#ax.plot(s, -theta_amp_d, ':', c=bmap[0])
#ax.plot(s, psi_d, c=bmap[2], label='psi')
#ax.plot(s, psi_amp_d, ':', c=bmap[2])
#ax.plot(s, -psi_amp_d, ':', c=bmap[2])
#ax.plot(s, theta_serp_d, c=bmap[3], label='theta serp')
#ax.legend(loc='best', frameon=True)
#ax.set_xlabel('distance along body, s (m)')
#ax.set_ylabel('body angles')
#sns.despine()
#fig.set_tight_layout(True)

i = 11

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.axvline(s[n_neck], color='gray', lw=1)
ax2.axvline(s[n_neck], color='gray', lw=1)
ax1.axhline(0, color='gray', ls='--', lw=1.5)
ax2.axhline(0, color='gray', ls='--', lw=1.5)
ax1.plot(s, theta[i])
ax1.plot(s, psi[i])
ax1.plot(s, amp_theta, c=bmap[0], ls=':', lw=1)
ax1.plot(s, -amp_theta, c=bmap[0], ls=':', lw=1)
ax1.plot(s, amp_psi, c=bmap[1], ls=':', lw=1)
ax1.plot(s, -amp_psi, c=bmap[1], ls=':', lw=1)
ax2.plot(s[:n_neck], dthetads[i, :n_neck], 'o-', c=bmap[0])
ax2.plot(s[n_neck:], dthetads[i, n_neck:],'o-', c=bmap[0])
ax2.plot(s[:n_neck], dpsids[i, :n_neck], 'o-', c=bmap[1])
ax2.plot(s[n_neck:], dpsids[i, n_neck:], 'o-', c=bmap[1])
sns.despine()
fig.set_tight_layout(True)





# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax.axhline(0, color='gray', ls='--', lw=1.5)
ax1.plot(ts, ho[:, 0], '-', label='x')
ax1.plot(ts, ho[:, 1], '-', label='y')
ax1.plot(ts, ho[:, 2], '-', label='z')
ax1.plot(ts, np.linalg.norm(ho, axis=1), '-', label='norm')
ax2.plot(ts, dho[:, 0], '-', label='x')
ax2.plot(ts, dho[:, 1], '-', label='y')
ax2.plot(ts, dho[:, 2], '-', label='z')
ax2.plot(ts, np.linalg.norm(dho, axis=1), '-', label='norm')
#ax.plot(ts, np.linalg.norm(ho, axis=1), 'o-', label='norm')
ax1.legend(loc='best', frameon=True)
sns.despine()
fig.set_tight_layout(True)



# %%

fig, ax = plt.subplots()
ax.axhline(0, color='gray', ls='--', lw=1.5)
ax.plot(ts, np.abs(ho[:, 0]), label='x')
ax.plot(ts, np.abs(ho[:, 1]), label='y')
ax.plot(ts, np.abs(ho[:, 2]), label='z')
ax.plot(ts, np.linalg.norm(ho, axis=1), label='norm')
ax.legend(loc='best', frameon=True)
sns.despine()
fig.set_tight_layout(True)


#fig, ax = plt.subplots()
#ax.axhline(0, color='gray', ls='--', lw=1.5)
#ax.plot(ts, dho[:, 0], label='x')
#ax.plot(ts, dho[:, 1], label='y')
#ax.plot(ts, dho[:, 2], label='z')
#ax.legend(loc='best', frameon=True)
#sns.despine()
#fig.set_tight_layout(True)


# %%

SS, TT = np.meshgrid(s, ts)


fig, ax = plt.subplots()
vmax = np.max(np.abs(p[:, :, 1]))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, p[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'lateral excursion (m)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


fig, ax = plt.subplots()
vmax = np.max(np.abs(dp[:, :, 1]))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, dp[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'lateral velocity (m/s)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Contour plots of angles and curvature

SS, TT = np.meshgrid(s, ts)

fig, ax = plt.subplots()
vmax = np.max(np.abs(kap))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, kap, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/m)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


fig, ax = plt.subplots()
vmax = np.rad2deg(np.max(np.abs(theta)))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, np.rad2deg(theta), cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'in-plane angle', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


fig, ax = plt.subplots()
vmax = np.rad2deg(np.max(np.abs(psi)))
vmin = -vmax
cax = ax.pcolormesh(TT, SS, np.rad2deg(psi), cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'out-of-plane angle', fontsize='medium')
cbar.solids.set_edgecolor("face")
#ax.contour(TS, SSmm, K, [0], colors=emerald_green)
#ax.set_xlim(TS.min(), TS.max())
#ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (m)')
sns.despine(ax=ax)
fig.set_tight_layout(True)



# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.axvline(s[n_neck], color='gray', lw=1)
ax2.axvline(s[n_neck], color='gray', lw=1)
ax1.axhline(0, color='gray', ls='--', lw=1.5)
ax2.axhline(0, color='gray', ls='--', lw=1.5)
ax1.plot(s, dpds[i, :, 0], label='dx')
ax1.plot(s, dpds[i, :, 1], label='dy')
ax1.plot(s, dpds[i, :, 2], label='dz')
ax2.plot(s, ddpdds[i, :, 0], label='dx')
ax2.plot(s, ddpdds[i, :, 1], label='dy')
ax2.plot(s, ddpdds[i, :, 2], label='dz')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=1)
ax.axhline(0, color='k', lw=1)
for i in np.arange(ntime):
    ax.plot(p[i, :, 0], p[i, :, 1], lw=1, c='gray')
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
fig.set_tight_layout(True)


fig = plt.figure(figsize=(8, 9.8375))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313)
axs = [ax1, ax2, ax3]

ax1.axhline(0, color='gray', ls='--', lw=1.5)
ax2.axhline(0, color='gray', ls='--', lw=1.5)
ax3.axvline(0, color='gray', lw=1)
ax3.axhline(0, color='gray', lw=1)
ax1.scatter(s, theta_d, c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax2.scatter(s, kap[i], c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax3.scatter(p[i, :, 0], p[i, :, 1], c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax3.set_aspect('equal', adjustable='box')
ax3.axis('off')
sns.despine(ax=ax1)
sns.despine(ax=ax2)
fig.set_tight_layout(True)



# %%

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

body = mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
                   tube_radius=.004, color=bmap[1])

# airfoil orientation (c, b, t) directions
#cv_quiv = mlab.quiver3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
#                        cv[i, :, 0], cv[i, :, 1], cv[i, :, 2],
#                        color=bmap[0], scale_factor=.05,
#                        resolution=64, mode='arrow')
#bv_quiv = mlab.quiver3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
#                        bv[i, :, 0], bv[i, :, 1], bv[i, :, 2],
#                        color=bmap[3], scale_factor=.05,
#                        resolution=64, mode='arrow')
#mlab.quiver3d(p[:, 0], p[:, 1], p[:, 2],
#              tdir[:, 0], tdir[:, 1], tdir[:, 2],
#              color=bmap[2], scale_factor=.05,
#              resolution=64, mode='arrow')

#horz = mlab.plot3d(p[i, :, 0], p[i, :, 1], 0 * p[i, :, 2],
#                   tube_radius=.002, color=bmap[0])
#
#vert = mlab.plot3d(p[i, :, 0], 0 * p[i, :, 1], p[i, :, 2],
#                   tube_radius=.002, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2])
#            cv_quiv.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2],
#                                    u=cv[i, :, 0], v=cv[i, :, 1], w=cv[i, :, 2])
#            bv_quiv.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2],
#                                    u=bv[i, :, 0], v=bv[i, :, 1], w=bv[i, :, 2])
            yield
manim = anim()
mlab.show()


# %%

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                  color=frame_c[ii], mode='arrow', opacity=.5, resolution=64)

body = mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
                   tube_radius=.004, color=bmap[1])

horz = mlab.plot3d(p[i, :, 0], p[i, :, 1], 0 * p[i, :, 2],
                   tube_radius=.002, color=bmap[0])

vert = mlab.plot3d(p[i, :, 0], 0 * p[i, :, 1], p[i, :, 2],
                   tube_radius=.002, color=bmap[2])

#mlab.plot3d(p_serp[:, 0], p_serp[:, 1], p_serp[:, 2],
#            tube_radius=.004, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2])
            horz.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=0 * p[i, :, 2])
            vert.mlab_source.set(x=p[i, :, 0], y=0 * p[i, :, 1], z=p[i, :, 2])
            yield
manim = anim()
mlab.show()



# %%

lks = 1 / np.r_[.4, .5, .6, .7, .8]

p2 = 2 * np.pi

#lambdak_theta = 0.6
nu_theta = 5 / 3.  # m
f_theta = 1.4  # Hz

#lambdak_psi = lambdak_theta / 2
nu_psi = 2 * nu_theta
f_psi = 2 * f_theta

frac_theta_max = .2
theta_max = np.deg2rad(90)
theta_amp = theta_max * np.linspace(1 + frac_theta_max, 1 - frac_theta_max, nbody)
#theta_amp = theta_max * np.linspace(1 - frac_theta_max, 1 + frac_theta_max, nbody)
#frac_k_theta = 0
#k_theta_max = 2 * np.pi / lambdak_theta
#k_theta = k_theta_max * np.linspace(1 - frac_k_theta, 1 + frac_k_theta, nbody)
#omg_theta = 2 * np.pi * f_theta
phi_theta = 0
d_theta = np.deg2rad(0)

frac_psi = 0
psi_max = np.deg2rad(15)
psi_amp = psi_max * np.linspace(1 - frac_psi, 1 + frac_psi, nbody)
#frac_k_psi = 0
#k_psi_max = 2 * np.pi / lambdak_psi
#k_psi = k_psi_max * np.linspace(1 + frac_k_psi, 1 - frac_k_psi, nbody)
#omg_psi = 2 * np.pi * f_psi
phi_psi = 0
d_psi = np.deg2rad(-0)

thetas = np.zeros((ntime, nbody))
psis = np.zeros((ntime, nbody))
ps = np.zeros((ntime, nbody, 3))
ps_serp = np.zeros((ntime, nbody, 3))
kaps = np.zeros((ntime, nbody))
kaps_serp = np.zeros((ntime, nbody))
for i in np.arange(ntime):
    t = ts[i]

    cos_theta = np.cos(2 * np.pi * nu * s - 2 * np.pi * f * t)
    theta = theta_amp * np.sin(np.pi / 2 * cos_theta) + d_theta * s
    theta_serp = theta_amp * np.cos(k_theta * s - omg_theta * t + phi_theta) + d_theta * s
    psi = psi_amp * np.sin(np.pi / 2 * np.cos(p2 * nu_psi * s - omg_psi * t + phi_psi) + d_psi * s
    psi_serp = psi_amp * np.cos(k_psi * s - omg_psi * t + phi_psi) + d_psi * s

    # if have neck region, make snake look in -x direction
    if n_neck > 0:
        c_theta = theta[n_neck] / s[n_neck]  # tangent angle at end of 'neck'
        theta_to_int = c_theta * np.ones(n_neck)

        c_theta_serp = theta_serp[n_neck] / s[n_neck]
        theta_serp_to_int = c_theta_serp * np.ones(n_neck)

        c_psi = psi[n_neck] / s[n_neck]
        psi_to_int = c_psi * np.ones(n_neck)

        theta[:n_neck] = theta_to_int * s[:n_neck]
        theta_serp[:n_neck] = theta_serp_to_int * s[:n_neck]
        psi[:n_neck] = psi_to_int * s[:n_neck]


    # now integrate to get the spine
    dxds = -np.cos(psi) * np.cos(theta)
    dyds = -np.cos(psi) * np.sin(theta)
    dzds = np.sin(psi)

    dxds_serp = -np.cos(psi) * np.cos(theta_serp)
    dyds_serp = -np.cos(psi) * np.sin(theta_serp)
    dzds_serp = np.sin(psi)

    x = cumtrapz(dxds, s, initial=0)
    y = cumtrapz(dyds, s, initial=0)
    z = cumtrapz(dzds, s, initial=0)
    p = np.c_[x, y, z]

    x_serp = cumtrapz(dxds_serp, s, initial=0)
    y_serp = cumtrapz(dyds_serp, s, initial=0)
    z_serp = cumtrapz(dzds_serp, s, initial=0)
    p_serp = np.c_[x_serp, y_serp, z_serp]

    com = np.sum((p.T * mi).T, axis=0) / mass_total
    com_serp = np.sum((p_serp.T * mi).T, axis=0) / mass_total

    p = p - com
    dp = np.c_[dxds, dyds, dzds]
    ddp = np.gradient(dp, ds, edge_order=2)[0]  # axis=0

    p_serp = p_serp - com_serp
    dp_serp = np.c_[dxds_serp, dyds_serp, dzds_serp]
    ddp_serp = np.gradient(dp_serp, ds, edge_order=2)[0]

    kap = np.cross(dp, ddp).sum(axis=1) / np.linalg.norm(dp, axis=1)**3
    kap_serp = np.cross(dp_serp, ddp_serp).sum(axis=1) / np.linalg.norm(dp_serp, axis=1)**3

    # store the values
    thetas[i] = theta
    psis[i] = psi
    ps[i] = p
    ps_serp[i] = p_serp
    kaps[i] = kap
    kaps_serp[i] = kap_serp

theta = thetas
psi = psis
p = ps
p_serp = ps_serp
kap = kaps
kap_serp = kaps_serp

i = 0

theta_d = np.rad2deg(theta)[i]
theta_amp_d = np.rad2deg(theta_amp)
psi_d = np.rad2deg(psi)[i]
psi_amp_d = np.rad2deg(psi_amp)
theta_serp_d = np.rad2deg(theta_serp)


#fig, ax = plt.subplots()
#ax.axhline(0, color='gray', ls='--', lw=1.5)
#ax.plot(s, theta_d, c=bmap[0], label='theta')
#ax.plot(s, theta_amp_d, ':', c=bmap[0])
#ax.plot(s, -theta_amp_d, ':', c=bmap[0])
#ax.plot(s, psi_d, c=bmap[2], label='psi')
#ax.plot(s, psi_amp_d, ':', c=bmap[2])
#ax.plot(s, -psi_amp_d, ':', c=bmap[2])
#ax.plot(s, theta_serp_d, c=bmap[3], label='theta serp')
#ax.legend(loc='best', frameon=True)
#ax.set_xlabel('distance along body, s (m)')
#ax.set_ylabel('body angles')
#sns.despine()
#fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=1)
ax.axhline(0, color='k', lw=1)
for i in np.arange(ntime):
    ax.plot(p[i, :, 0], p[i, :, 1], lw=1, c='gray')
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=1)
ax.axhline(0, color='k', lw=1)
for i in np.arange(ntime):
    ax.plot(p_serp[i, :, 0], p_serp[i, :, 1], lw=1, c='gray')
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
fig.set_tight_layout(True)


fig = plt.figure(figsize=(8, 9.8375))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313)
axs = [ax1, ax2, ax3]

ax1.axhline(0, color='gray', ls='--', lw=1.5)
ax2.axhline(0, color='gray', ls='--', lw=1.5)
ax3.axvline(0, color='gray', lw=1)
ax3.axhline(0, color='gray', lw=1)
ax1.scatter(s, theta_d, c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax2.scatter(s, kap[i], c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax3.scatter(p[i, :, 0], p[i, :, 1], c=s, s=60, cmap=plt.cm.viridis, linewidths=0)
ax3.set_aspect('equal', adjustable='box')
ax3.axis('off')
sns.despine(ax=ax1)
sns.despine(ax=ax2)
fig.set_tight_layout(True)


# %%

#fig, (ax1, ax2, ax3) = plt.subplots()
#ax.axhline(0, color='gray', ls='--', lw=1.5)
#ax.scatter(s, theta_d, c=s, s=60, cmap=plt.cm.viridis)
#ax.scatter(s, kap, c=s, s=60, cmap=plt.cm.viridis)
#sns.despine()
#fig.set_tight_layout(True)
#
#
#fig, ax = plt.subplots()
#ax.axvline(0, color='gray', lw=1)
#ax.axhline(0, color='gray', lw=1)
#ax.scatter(p[i, :, 0], p[i, :, 1], c=s, s=60, cmap=plt.cm.viridis)
#ax.set_aspect('equal', adjustable='box')
#ax.axis('off')
#fig.set_tight_layout(True)


# %%

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

body = mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
                   tube_radius=.004, color=bmap[0])

serp = mlab.plot3d(p_serp[i, :, 0], p_serp[i, :, 1], p_serp[i, :, 2],
                   tube_radius=.004, color=bmap[2])

#mlab.plot3d(p_serp[:, 0], p_serp[:, 1], p_serp[:, 2],
#            tube_radius=.004, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2])
            serp.mlab_source.set(x=p_serp[i, :, 0], y=p_serp[i, :, 1],
                                 z=p_serp[i, :, 2])
            yield
manim = anim()
mlab.show()



# %%

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

body = mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2],
                   tube_radius=.004, color=bmap[1])

#mlab.plot3d(p_serp[:, 0], p_serp[:, 1], p_serp[:, 2],
#            tube_radius=.004, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=p[i, :, 0], y=p[i, :, 1], z=p[i, :, 2])
            yield
manim = anim()
mlab.show()


# %%

i = 0

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

body = mlab.plot3d(p_serp[i, :, 0], p_serp[i, :, 1], p_serp[i, :, 2],
                   tube_radius=.004, color=bmap[1])

#mlab.plot3d(p_serp[:, 0], p_serp[:, 1], p_serp[:, 2],
#            tube_radius=.004, color=bmap[2])

fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            body.mlab_source.set(x=p_serp[i, :, 0], y=p_serp[i, :, 1],
                                 z=p_serp[i, :, 2])
            yield
manim = anim()
mlab.show()



# %%




#    c0_1plane = c0 - np.dot(t1, c0) * t1
#    c0_1plane = c0_1plane / np.linalg.norm(c0_1plane)
#    torsion_angle[j] = np.arccos(np.dot(c0_1plane, c1))

#    c01 = c0 - np.dot(c0, t1) * t1
#    c01 = c01 / np.linalg.norm(c01)
#    b01 = b0 - np.dot(b0, t1) * t1
#    b01 = b01 / np.linalg.norm(b01)

    t0_tb = t0 - np.dot(t0, c1) * c1
    t0_tc = t0 - np.dot(t0, b1) * b1
    t0_tb = t0_tb / np.linalg.norm(t0_tb)
    t0_tc = t0_tc / np.linalg.norm(t0_tc)

    a_tb = np.arccos(np.dot(t0_tb, t1))
    a_tc = np.arccos(np.dot(t0_tc, t1))

    back_angle[j] = np.rad2deg(a_tb)
    bend_angle[j] = np.rad2deg(a_tc)



#    t_tang = t1 - np.dot(t0, t1) * t1
#    t_tang = t_tang / np.linalg.norm(t_tang)
#    t_tors = t0 - np.dot(t0, t1) * t0
#    t_tors = t_tors / np.linalg.norm(t_tors)
#    alpha_tang = np.arccos(np.dot(t_tang, t1))
#    alpha_tors = np.arccos(np.dot(t_tors, t1))

#
## %%
#
#assert(np.allclose(0, np.sum((p.T * mi).T, axis=0)))
#
#theta_d = np.rad2deg(theta)
#theta_amp_d = np.rad2deg(theta_amp)
#psi_d = np.rad2deg(psi)
#psi_amp_d = np.rad2deg(psi_amp)
#
#
#fig, ax = plt.subplots()
#ax.axhline(0, color='gray', ls='--', lw=1.5)
#ax.plot(s, theta_d, c=bmap[0], label='theta')
#ax.plot(s, theta_amp_d, ':', c=bmap[0])
#ax.plot(s, -theta_amp_d, ':', c=bmap[0])
#ax.plot(s, psi_d, c=bmap[2], label='psi')
#ax.plot(s, psi_amp_d, ':', c=bmap[2])
#ax.plot(s, -psi_amp_d, ':', c=bmap[2])
#ax.legend(loc='upper right')
#ax.set_xlabel('distance along body, s (m)')
#ax.set_ylabel('body angles')
#sns.despine()
#fig.set_tight_layout(True)
#
#
##fig, ax = plt.subplots()
##ax.plot(p[:, 0], p[:, 1])
#
#
#
## %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=.05,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

mlab.plot3d(p[:, 0], p[:, 1], p[:, 2],
            tube_radius=.002, color=bmap[1])

# airfoil orientation (c, b, t) directions
#mlab.quiver3d(p[:, 0], p[:, 1], p[:, 2],
#              cdir[:, 0], cdir[:, 1], cdir[:, 2],
#              color=bmap[0], scale_factor=.05,
#              resolution=64, mode='arrow')
#mlab.quiver3d(p[:, 0], p[:, 1], p[:, 2],
#              bdir[:, 0], bdir[:, 1], bdir[:, 2],
#              color=bmap[3], scale_factor=.05,
#              resolution=64, mode='arrow')
#mlab.quiver3d(p[:, 0], p[:, 1], p[:, 2],
#              tdir[:, 0], tdir[:, 1], tdir[:, 2],
#              color=bmap[2], scale_factor=.05,
#              resolution=64, mode='arrow')

# plane waves
#mlab.plot3d(p[:, 0], p[:, 1], 0 * p[:, 2],
#            tube_radius=.001, color=bmap[0])
#
#mlab.plot3d(p[:, 0], 0 * p[:, 1], p[:, 2],
#            tube_radius=.001, color=bmap[2])


fig.scene.isometric_view()
fig.scene.parallel_projection = True
mlab.orientation_axes()


# %%

kap = -A * k * np.sin(k * s - w * t + phi)
tang = A * np.cos(k * s - w * t + phi)

# if have neck region, make snake look in -x direction
if n_neck > 0:
    Phi = tang[n_neck]  # tangent angle at end of 'neck'
    cN = Phi / s[n_neck]
    kap_start = cN * np.ones(n_neck)
    kap[:n_neck] = kap_start
    tang[:n_neck] = kap_start * s[:n_neck]
    # tang[:n_neck] = cumtrapz(kap_start, s[:n_neck], initial=0)

x = cumtrapz(np.cos(tang), s, initial=0)
y = cumtrapz(np.sin(tang), s, initial=0)
x, y = x - x.mean(), y - y.mean()

# %%

aa = .2
n = 1 / (2 * np.pi)
c = 0
s = np.linspace(0, .7, 100)
theta = (aa * s * np.pi / 2) * np.cos(n * s) + c * s
#theta = (np.pi/2 / s[-1] * s) * np.cos(n * s) + c * s
x = cumtrapz(np.cos(theta), s, initial=0)
y = cumtrapz(np.sin(theta), s, initial=0)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(s, theta)
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(x, y, '-')
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)



# %%

# %%

i = 10
j = 60

zr = np.zeros(2)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

ctv_quiv = mlab.quiver3d(zr, zr, zr,
                         tv[i, j:j+2, 0], tv[i, j:j+2, 1], tv[i, j:j+2, 2],
                         color=bmap[2], scale_factor=.05,
                         resolution=64, mode='arrow')

cv_quiv = mlab.quiver3d(zr, zr, zr,
                        cv[i, j:j+2, 0], cv[i, j:j+2, 1], cv[i, j:j+2, 2],
                        color=bmap[0], scale_factor=.05,
                        resolution=64, mode='arrow')


bv_quiv = mlab.quiver3d(zr, zr, zr,
                        bv[i, j:j+2, 0], bv[i, j:j+2, 1], bv[i, j:j+2, 2],
                        color=bmap[1], scale_factor=.05,
                        resolution=64, mode='arrow')


t0 = tv[i, j]
t1 = tv[i, j + 1]
b1 = bv[i, j + 1]

t0_cb = t0 - np.dot(t0, t1) * t1
t0_cb = t0_cb / np.linalg.norm(t0_cb)

mlab.quiver3d([0], [0], [0],
            [t0_cb[0]], [t0_cb[1]], [t0_cb[2]],
            color=bmap[3], scale_factor=.025,
            resolution=64, mode='arrow')


t0_tc = t0 - np.dot(t0, b1) * b1
t0_tc = t0_tc / np.linalg.norm(t0_tc)

mlab.quiver3d([0], [0], [0],
            [t0_tc[0]], [t0_tc[1]], [t0_tc[2]],
            color=bmap[4], scale_factor=.025,
            resolution=64, mode='arrow')


A = np.cross(t0, t1)
A = A / np.linalg.norm(A)

mlab.quiver3d([0], [0], [0],
            [A[0]], [A[1]], [A[2]],
            color=bmap[5], scale_factor=.025,
            resolution=64, mode='arrow')
