# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 09:20:20 2014

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
import time

from scipy.linalg import eig
from scipy.optimize import newton

from matplotlib.animation import FuncAnimation

import colormaps as cmaps
import seaborn as sns
from mayavi import mlab

np.set_printoptions(suppress=True)

# so we can color the squiggle
# http://stackoverflow.com/a/25941474

import plotting
import aerodynamics
import sim

#rc = {'mathtext.fontset': 'stixsans'}
#sns.set('notebook', 'ticks', font_scale=1.35, font='Gill Sans', rc=rc)
#bmap = sns.color_palette()
rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.transparent': True}
sns.set('notebook', 'ticks', font_scale=1.5, rc=rc)
#sns.set('talk', 'ticks', font_scale=1.17, rc=rc)
bmap = sns.color_palette()

# http://xkcdcp.martinblech.com/#emerald green
emerald_green = '#028f1e'


# %%

aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)
g = 9.81  # m/s^2
rho = 1.165  # 30 C

L = .7  # .686  # m
ds = .01  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec

# nbody = 51
# s, ds = np.linspace(0, L, nbody, retstep=True)

M = .0405  # kg
rho_body = M / L  # linear mass density
mi = np.ones(nbody) * rho_body * L / nbody  # mass per segment
c = .022 * np.ones(nbody)  # m, chord length
neck_length = .05  # m
n_neck = np.floor(neck_length / ds).astype(np.int)  # number of points for neck
Stot = np.sum(c * ds)  # m^2
WS = M * g / Stot  # wing loading, N / m^2

# time parameters
ntime = 201
#dt = .0025
ts = np.arange(0, 3.001, dt)
ntime = len(ts)

# undulation parameters
# A, lamk, lamw = 1.6, .63, 1
#A, wave_length_m = 1.6, .63  # 'amplitude', wave length (m)
A = 1.6
wave_length_m = 0.6
#A = 2
freq_undulation_hz = 1.4  # Hz
period_undulation_s = 1 / freq_undulation_hz  # s
w = 2 * np.pi * freq_undulation_hz  # angular frequency, rad
# w = 0  # turn undulation off
k = 2 * np.pi / wave_length_m  # wave number, rad / m

# make the serpenoid curve zero angular momentum
ho_args = (s, ts[0], A, k, w, n_neck, mi)
phi = newton(sim.func_ho_to_zero, 0, args=ho_args)  # offset serpenoid curve


# # %% Example kinematics simulation

iners = np.zeros((ntime, 3, 3))
eig_vals = np.zeros((ntime, 3))
eig_vecs = np.zeros((ntime, 3, 3))
eig_angs = np.zeros((ntime, 3))
P = np.zeros((ntime, nbody, 3))  # ntime x nbody x [x, y, z]
dP = np.zeros((ntime, nbody, 3))  # ntime x nbody x [x, y, z]
ddP = np.zeros((ntime, nbody, 3))  # ntime x nbody x [x, y, z]
T = np.zeros((ntime, nbody))  # tangent angle
K = np.zeros((ntime, nbody))  # curvature
SS, TS = np.meshgrid(s, ts)  # time, body shape
HO = np.zeros(ntime)
TV = np.zeros((ntime, nbody, 3))  # tangent vector
CV = np.zeros((ntime, nbody, 3))  # chord vector
BV = np.zeros((ntime, nbody, 3))  # backbone vector

for i in np.arange(ntime):
    # define the shape, velocity, acceleration
    out = sim.serp_pos_vel_acc_tcb(s, ts[i], A, k, w, phi, n_neck)
    p, v, a, tang, kap, tv, cv, bv = out

    # angular momentum
    ho = np.cross(p, (mi * v.T).T).sum(axis=0)

    # principle moments
    prin_i = sim.mom_iner_2D(p, mi)
    evals, evecs = eig(prin_i)
    e1, e2, e3 = evecs.T
    th1 = np.arctan2(e1[1], e1[0])
    th2 = np.arctan2(e2[1], e2[0])
    th3 = np.arctan2(e3[1], e3[0])

    # store values
    P[i] = p
    dP[i] = v
    ddP[i] = a
    T[i] = tang
    K[i] = kap
    HO[i] = ho[2]
    TV[i] = tv
    CV[i] = cv
    BV[i] = bv
    iners[i] = prin_i
    eig_vals[i] = evals.real
    eig_vecs[i] = evecs
    eig_angs[i] = th1, th2, th3

# mean principle moment direction
evm = eig_vecs.mean(axis=0)
magevm = np.linalg.norm(evm, axis=0)
evm /= magevm

#unit_vecs = serp_coords(T)

##dP, ddP = pos2v_a(P, dt)
#interp_body = kinematics_interp(ts, s, P, dP, ddP)
#interp_serp = serp_interp(ts, s, T)

COV = np.cov(np.deg2rad(T.T))
Scov1, Scov2 = np.meshgrid(s, s)


# %%

fig, ax = plt.subplots()
ax.axvline(s[n_neck], color='gray', lw=.75)
ax.plot(s, kap, drawstyle='steps-post')
ax.plot(s, kap)

fig, ax = plt.subplots()
ax.axvline(s[n_neck], color='gray', lw=.75)
ax.plot(s, tang, drawstyle='steps-post')
ax.plot(s, tang)


# %% Plot the snake with with the tangent and chord unit vectors

p, v, a, tang, kap, tcb = sim.serp_pos_vel_acc_tcb(s, .05, A, k, w, phi, n_neck)

skip = 2
tv = tcb[::skip, 0]
cv = tcb[::skip, 1]

fig, ax = plt.subplots()
ax.plot(p[:, 0], p[:, 1], 'o-', c=emerald_green)
ax.plot(p[0, 0], p[0, 1], 'o', c=emerald_green, ms=13)
# ax.quiver(0, 0, .1, 0, color='gray', units='xy', width=.002)
# ax.quiver(0, 0, 0, .1, color='gray', units='xy', width=.002)
ax.arrow(0, 0, .03, 0, head_width=.003, fc='gray', ec='gray', lw=.5)
ax.arrow(0, 0, 0, .03, head_width=.003, fc='gray', ec='gray', lw=.5)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(p[:, 0], p[:, 1], '-', c=emerald_green)
ax.plot(p[0, 0], p[0, 1], 'o', c=emerald_green, ms=13, zorder=11)
ax.quiver(p[::skip, 0], p[::skip, 1], tv[:, 0], tv[:, 1], color=bmap[2],
          units='xy', width=.0015, zorder=10)
ax.quiver(p[::skip, 0], p[::skip, 1], cv[:, 0], cv[:, 1], color=bmap[0],
          units='xy', width=.0015)
ax.arrow(0, 0, .03, 0, head_width=.003, fc='gray', ec='gray', lw=.5)
ax.arrow(0, 0, 0, .03, head_width=.003, fc='gray', ec='gray', lw=.5)
ax.set_aspect('equal', adjustable='box')
ax.margins(.1)
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %% Plot the snake with with the tangent and chord unit vectors

i = 0
skip = 20
tv = TCB[i, ::skip, 0]
cv = TCB[i, ::skip, 1]
fig, ax = plt.subplots()
ax.plot(P[i, :, 0], P[i, :, 1], '-', c=emerald_green)
ax.plot(P[i, 0, 0], P[i, 0, 1], 'o', c=emerald_green, ms=13, zorder=11)
ax.quiver(P[i, ::skip, 0], P[i, ::skip, 1], tv[:, 0], tv[:, 1], color=bmap[2],
          units='xy', width=.0015, zorder=10)
ax.quiver(P[i, ::skip, 0], P[i, ::skip, 1], cv[:, 0], cv[:, 1], color=bmap[0],
          units='xy', width=.0015)
ax.set_aspect('equal', adjustable='box')
ax.margins(.1)
# ax.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(ts, HO)


# %% Space-time curvature field

SSmm = 1000 * SS

vmin, vmax = -.025, .025

fig, ax = plt.subplots()
cax = ax.pcolormesh(TS, SSmm, .001 * K, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")
ax.contour(TS, SSmm, K, [0], colors=emerald_green)
ax.set_xlim(TS.min(), TS.max())
ax.set_ylim(SSmm.min(), SSmm.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Movie of simulation

from scalebars import add_scalebar

fig, ax = plt.subplots()

sb = add_scalebar(ax, matchx=False, matchy=False, hidex=True, hidey=True,
                  loc=3, sizex=0.1, labelx='10 cm', sizey=0, labely='10 cm')
lbody, = ax.plot([], [], 'o-', c=emerald_green)
pbody, = ax.plot([], [], 'o', c=emerald_green)
phead, = ax.plot([], [], 'o', c=emerald_green, ms=13)
Q1 = ax.quiver(0, 0, [], [], scale=10, width=.003, color=bmap[0])
Q2 = ax.quiver(0, 0, [], [], scale=10, width=.003, color=bmap[2])
time_template = 'time = {0:.1f} sec'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.arrow(0, 0, .03, 0, head_width=.003, fc='gray', ec='gray', lw=.5)
ax.arrow(0, 0, 0, .03, head_width=.003, fc='gray', ec='gray', lw=.5)

ax.set_aspect('equal')
xlim = (1.05 * P[:, :, 0].min(), 1.05 * P[:, :, 0].max())
ylim = (-.53 * P[:, :, 1].ptp(), .53 * P[:, :, 1].ptp())
ax.set_ylim(ylim)
ax.set_xlim(xlim)

ax.axis('off')
fig.set_tight_layout(True)

def init():
    lbody.set_data([], [])
    pbody.set_data([], [])
    phead.set_data([], [])
    time_text.set_text('')
    Q1.set_UVC(0, 0)
    Q2.set_UVC(0, 0)
    return lbody, pbody, phead, time_text, Q1, Q2


def animate(i):
    xs, ys = P[i, :, 0], P[i, :, 1]
    e1, e2, _ = eig_vecs[i].T
    lbody.set_data(xs, ys)
    pbody.set_data(xs[::20], ys[::20])
    phead.set_data(xs[0], ys[0])
    Q1.set_UVC(e1[0], e1[1])
    Q2.set_UVC(e2[0], e2[1])
    time_text.set_text(time_template.format(i * dt))
    return lbody, pbody, phead, time_text, Q1, Q2

ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * 5,  # draw a frame every x ms
                    repeat=2, blit=True, init_func=init)

slowed = 5
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=True, init_func=init)

#ani.save('{}X A = {}, wave_len_m = {}, freq_hz = {} realtime.mp4'.format(
#         slowed, A, wave_length_m, freq_undulation_hz),
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


# %%

fig, ax = plt.subplots()

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

#ani.save('clean {}X A = {}, wave_len_m = {}, freq_hz = {} realtime.mp4'.format(
#         slowed, A, wave_length_m, freq_undulation_hz),
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
for j in np.arange(nbody):
    ax1.plot(ts, dP[:, j, 0])
    ax2.plot(ts, dP[:, j, 1])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
for j in np.arange(nbody):
    ax1.plot(ts, ddP[:, j, 0])
    ax2.plot(ts, ddP[:, j, 1])


# %% Plot principle moments and eigenvector components

legargs = dict(loc='lower right', frameon=True, fancybox=True, framealpha=.85)

fig, ax = plt.subplots()
ax.plot(ts, eig_vals[:, 0], c=bmap[0], label=r'$I_1$')
ax.plot(ts, eig_vals[:, 1], c=bmap[1], label=r'$I_2$')
ax.plot(ts, eig_vals[:, 2], c=bmap[2], label=r'$I_3$')
ax.legend(**legargs)
ax.set_xlim(ts[0], ts[-1])
ax.set_xlabel('time, s')
ax.set_ylabel(r'principle moments of inertia, $kg \cdot m^2$')
sns.despine()
fig.set_tight_layout(True)


# eig_deg = np.abs(np.rad2deg(eig_angs))
eig_deg = np.rad2deg(eig_angs)

fig, ax = plt.subplots()
ax.plot(ts, eig_deg[:, 0], c=bmap[0], label=r'$I_1$')
ax.plot(ts, eig_deg[:, 1], c=bmap[1], label=r'$I_2$')
ax.plot(ts, eig_deg[:, 2], c=bmap[2], label=r'$I_3$')
ax.legend(**legargs)
ax.set_xlim(ts[0], ts[-1])
ax.set_xlabel('time, s')
ax.set_ylabel('princple moment angles, deg')
sns.despine()
fig.set_tight_layout(True)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(ts, eig_vecs[:, 0, 0], label=r'$I_{1,x}$')
ax1.plot(ts, eig_vecs[:, 1, 1], label=r'$I_{2,y}$')
ax2.plot(ts, eig_vecs[:, 1, 0], label=r'$I_{1,y}$')
ax2.plot(ts, eig_vecs[:, 0, 1], label=r'$I_{2,x}$')
ax1.legend(**legargs)
ax2.legend(**legargs)
ax1.set_xlim(ts[0], ts[-1])
ax2.set_xlabel('time, s')
ax1.set_title('principle moment directions')
sns.despine()
fig.set_tight_layout(True)


# %% Run the dynamics simulation

L = .7  # .686  # m
ds = .01  # m
s = ds / 2 + np.arange(0, L, ds)  # m
nbody = len(s)
dt = .010  # sec

rho = 1.165  # 30 C
g = 9.81

mass_total = .0405  # kg
rho_body = mass_total / L  # linear mass density
mi = np.ones(nbody) * rho_body * L / nbody  # mass per segment
c = .022 * np.ones(nbody)  # m, chord length
neck_length = .05  # m
n_neck = np.floor(neck_length / ds).astype(np.int)  # number of points for neck
Stot = np.sum(c * ds)  # m^2
wing_loading = mass_total * g / Stot  # wing loading, N / m^2

A = 1.6  # 'amplitude'
wave_length_m = .6  # wave length (m) of curvature field  # .63
#A = 1.
#wave_length_m = .3
freq_undulation_hz = 1.4  # Hz
#freq_undulation_hz=0
w = 2 * np.pi * freq_undulation_hz  # angular frequency, rad
k = 2 * np.pi / wave_length_m  # wave number, rad / m

# make the serpenoid curve zero angular momentum
if w == 0:
    wtmp = 2 * np.pi * 1.4
    period_undulation_s = 0  # s
    ho_args = (s, 0, A, k, wtmp, n_neck, mi)
    phi = newton(sim.func_ho_to_zero, 0, args=ho_args)
else:
    ho_args = (s, 0, A, k, w, n_neck, mi)
    phi = newton(sim.func_ho_to_zero, 0, args=ho_args)

# aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)
# aero_interp = None

tend = None

#tend=.2
#g=0.00001

# arguments
args = (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)

params = (dt, L, nbody, mass_total, rho_body, neck_length, Stot, \
            wing_loading, wave_length_m, freq_undulation_hz)

# initial conditions
Ro0 = np.r_[0, 0, 10]
dRo0 = np.r_[1.7, 0, 0]
ang0 = np.deg2rad(np.r_[180, 0, 0])  # yaw, pitch, roll
dang0 = np.deg2rad(np.r_[0, 0, 0]) # yaw rate, pitch rate, roll rate

C0 = sim.euler2C(ang0[0], ang0[1], ang0[2])
omg0_body = np.dot(sim.dang2omg(ang0[0], ang0[1], ang0[2]), dang0)
omg0 = np.dot(C0.T, omg0_body)
# omg0 = np.deg2rad(np.r_[0, 0, 0])
soln0 = np.r_[Ro0, dRo0, omg0, ang0]

# yaw0 = 180 to align axes
# pitch up is nose up
# positive roll is to the left


# %%

# perform the integration
out = sim.integrate(soln0, args, dt, tend=tend, print_time=True)

# extract values
ts, Ro, dRo, omg, ang = out
yaw, pitch, roll = ang.T
ntime = len(ts)

savename = 'test_2016-07-01.npz'
sim.save_derived_quantities(savename, ntime, out, args, params)

#save_base = 'sims/A = {}, wave_len_m = {}, freq_hz = {}.npz'
#save_name = save_base.format(A, wave_length_m, freq_undulation_hz)

# test.npz - theta = 30, thetadot = -100 (moving along glide path)
# test2.npz - theta = 30, thetadot = 100 (backflip)
# test3.npz - theta = -30, thetadot = -100 (nose dive)
# test4.npz - theta = -30, thetadot = 100
# test5.npz - theta = 15, thetadot = -60
# test6.npz - theta = 15, thetadot = -15


# %% Load in the data

#d = np.load('test6.npz')
d = np.load('./Output/A=2.0_lk=0.4_f=1.4.npz')

#fname = 'Output/A=1.2_lk=0.3_f=1.4.npz'
#d = np.load(fname)

ntime = d['ntime']
neck_length = d['neck_length']
cop_z0 = d['cop_z0']
aoa_r = d['aoa_r']
ddR = d['ddR']
C = d['C']
aoa_d = d['aoa_d']
dR_BC = d['dR_BC']
wave_length_m = d['wave_length_m']
wing_loading = d['wing_loading']
dRo = d['dRo']
pitch = d['pitch']
R = d['R']
Ro = d['Ro']
# tcb = d['tcb']
mass_total = d['mass_total']
nbody = d['nbody']
#omg0 = d['omg0']
Stot = d['Stot']
ddr = d['ddr']
yaw = d['yaw']
ddp = d['ddp']
rho_body = d['rho_body']
ts = d['ts']
#omg0_body = d['omg0_body']
Maero_z0 = d['Maero_z0']
#ang0 = d['ang0']
Ho = d['Ho']
Fg = d['Fg']
#Ro0 = d['Ro0']
Borig = d['Borig']
Faero = d['Faero']
#C0 = d['C0']
ang = d['ang']
roll = d['roll']
Fl = d['Fl']
A = d['A']
#dang0 = d['dang0']
phi = d['phi']
omg = d['omg']
#dRo0 = d['dRo0']
L = d['L']
n_neck = d['n_neck']
domg = d['domg']
Norig = d['Norig']
kap = d['kap']
Re = d['Re']
ho = d['ho']
rho = d['rho']
#soln0 = d['soln0']
dt = d['dt']
dr = d['dr']
ds = d['ds']
dp = d['dp']
cop = d['cop']
c = d['c']
g = d['g']
Fd = d['Fd']
tang = d['tang']
# TCB = d['TCB']
k = d['k']
Ftot = d['Ftot']
mi = d['mi']
Maero = d['Maero']
p = d['p']
s = d['s']
r = d['r']
ddRo = d['ddRo']
w = d['w']
dang = d['dang']
freq_undulation_hz = d['freq_undulation_hz']
dR = d['dR']
S = d['S']
T = d['T']
Sn =d['Sn']
Tn = d['Tn']
Mm = d['Mm']
Nm = d['Nm']
Nnew = d['Nnew']
Neul = d['Neul']
tv = d['tv']
cv = d['cv']
bv = d['bv']
Tv = d['Tv']
Cv = d['Cv']
Bv = d['Bv']
beta_r = d['beta_r']
beta_d = d['beta_d']
glide_angle_r = d['glide_angle_r']
glide_angle_d = d['glide_angle_d']
heading_angle_r = d['heading_angle_r']
heading_angle_d = d['heading_angle_d']
adv_ratio = d['adv_ratio']


# %% Mesh and airfoil body shape

def _rotate_foil(xyz, th):
    Rth = np.array([[np.cos(th), -np.sin(th), 0],
                    [np.sin(th),  np.cos(th), 0],
                    [0, 0, 1]])
    return np.dot(Rth, xyz.T).T

rfoil = np.genfromtxt('data/snake0.004.bdy.txt', skip_header=1)
rfoil = rfoil - rfoil.mean(axis=0)
rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
rfoil /= np.ptp(rfoil[:, 0])
rfoil = rfoil[::5]
rfoil = np.c_[np.zeros(rfoil.shape[0]), rfoil]  # .025
rfoil = np.c_[rfoil.T, rfoil[0]].T
nfoil = rfoil.shape[0]

foil_scale = np.ones(nbody)
foil_scale[:n_neck] = np.linspace(0.55, .95, n_neck)
foil_scale *= c  # scale by the chord length


# in inertial frame
L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))
u = np.zeros((ntime, nbody, 3, 2))
bc = np.zeros((ntime, nbody, 3, 2))

# in local frame
Lb = np.zeros((ntime, nbody, 3, 2))
Db = np.zeros((ntime, nbody, 3, 2))
Ab = np.zeros((ntime, nbody, 3, 2))
ub = np.zeros((ntime, nbody, 3, 2))
bcb = np.zeros((ntime, nbody, 3, 2))

scale_velocities = .02  # 1/50th
scale_forces = 10

# airfoil shape
foils = np.zeros((ntime, nbody, nfoil, 3))
scalars = np.zeros((ntime, nbody, nfoil))
scalar_value = np.sqrt((Faero**2).sum(axis=2))
scalar_value = np.abs(beta_d)

now = time.time()
for i in np.arange(ntime):
    for j in np.arange(nbody):
        # in inertial frame
        L[i, j, :, 0] = R[i, j]
        L[i, j, :, 1] = R[i, j] + scale_forces * Fl[i, j]
        D[i, j, :, 0] = R[i, j]
        D[i, j, :, 1] = R[i, j] + scale_forces * Fd[i, j]
        A[i, j, :, 0] = R[i, j]
        A[i, j, :, 1] = R[i, j] + scale_forces * Faero[i, j]

        u[i, j, :, 0] = R[i, j]
        u[i, j, :, 1] = R[i, j] + scale_velocities * dR[i, j]

        bc[i, j, :, 0] = R[i, j]
        bc[i, j, :, 1] = R[i, j] + scale_velocities * dR_BC[i, j]

        # in local frame
        Lb[i, j, :, 0] = r[i, j]
        Lb[i, j, :, 1] = r[i, j] + scale_forces * Fl[i, j]
        Db[i, j, :, 0] = r[i, j]
        Db[i, j, :, 1] = r[i, j] + scale_forces * Fd[i, j]
        Ab[i, j, :, 0] = r[i, j]
        Ab[i, j, :, 1] = r[i, j] + scale_forces * Faero[i, j]

        ub[i, j, :, 0] = r[i, j]
        ub[i, j, :, 1] = r[i, j] + scale_velocities * dR[i, j]

        bcb[i, j, :, 0] = r[i, j]
        bcb[i, j, :, 1] = r[i, j] + scale_velocities * dR_BC[i, j]

        # airfoil
        rot = sim.rotate(C[i].T, _rotate_foil(rfoil, tang[i, j]))
        foils[i, j] = r[i, j] + foil_scale[j] * rot
        scalars[i, j] = scalar_value[i, j]

print('Elapsed time for plotting arrays {0:.3f} sec'.format(time.time() - now))


# %% Mesh on foil

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = -1

#head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
#                     scale_factor=.015, resolution=16, opacity=.5)
head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=.015, resolution=16, opacity=.5)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 color=bmap[1], opacity=.5)

# plan-view of snake
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], 0 * foils[i, :, :, 2],
#                 color=bmap[5], opacity=.5)

ml = mlab.mesh(Lb[i, :, 0], Lb[i, :, 1], Lb[i, :, 2], color=bmap[0], opacity=.8)
md = mlab.mesh(Db[i, :, 0], Db[i, :, 1], Db[i, :, 2], color=bmap[4], opacity=.8)
ma = mlab.mesh(Ab[i, :, 0], Ab[i, :, 1], Ab[i, :, 2], color=bmap[2], opacity=.8)

#mode = 'arrow'  # '2darrow'
#ql = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
#                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2], color=bmap[0],
#                   mode=mode, resolution=64, scale_factor=8, opacity=.5)
#qd = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
#                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2], color=bmap[1],
#                   mode=mode, resolution=64, scale_factor=8, opacity=.5)
#qa = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
#                   Faero[i, :, 0], Faero[i, :, 1], Faero[i, :, 2], color=bmap[2],
#                   mode=mode, resolution=64, scale_factor=8)

mlab.orientation_axes()
fig.scene.isometric_view()

savename = './anim_run/anim_{0:03d}.png'


# %% Save mesh on foil

now = time.time()

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        print('Current time: {0}'.format(ts[i]))

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=scalars[i])

        ml.mlab_source.set(x=Lb[i, :, 0], y=Lb[i, :, 1], z=Lb[i, :, 2])
        md.mlab_source.set(x=Db[i, :, 0], y=Db[i, :, 1], z=Db[i, :, 2])
        ma.mlab_source.set(x=Ab[i, :, 0], y=Ab[i, :, 1], z=Ab[i, :, 2])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim()
mlab.show()

print('Save time: {0:.3f} sec'.format(time.time() - now))


# %%

fig, ax = plt.subplots()
ax.plot(ts, glide_angle_d)
ax.plot(ts, heading_angle_d)
ax.plot(ts, np.rad2deg(pitch))
#ax.set_ylim(0, 90)
sns.despine()


# %%

fig, ax = plt.subplots()
vlim = np.max([aoa_d.max(), np.abs(aoa_d.min())])
cax = ax.pcolormesh(Tn, Sn, aoa_d, vmin=-10, vmax=vlim, cmap=cmaps.viridis)
ax.axhline(Sn[0, n_neck], color='white', lw=1, zorder=1)
ax.contour(Tn[:, n_neck:], Sn[:, n_neck:], kap[:, n_neck:], [0],
           colors='gray', linewidths=1)
ax.contour(Tn, Sn, tang, [0], colors='w', linewidths=1)
cont35 = ax.contour(Tn, Sn, aoa_d, [35], colors='w', linewidths=1.25)
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.add_lines(cont35)
cbar.set_label(r'$\alpha$, angle of attack')
ax.set_ylim(0, 1)
ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylabel('location along body')
ax.set_xlabel('time')

# add degree symbol to angles
ticks = cbar.ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
cbar.ax.set_yticklabels(newticks)

sns.despine(ax=ax)
fig.set_tight_layout(True)

# http://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, np.abs(beta_d), vmin=0, vmax=90,
                    cmap=cmaps.inferno)
ax.axhline(Sn[0, n_neck], color='white', lw=1, zorder=1)
ax.contour(Tn[:, n_neck:], Sn[:, n_neck:], kap[:, n_neck:], [0],
           colors='gray', linewidths=1)
ax.contour(Tn, Sn, tang, [0], colors='w', linewidths=1)
contb = ax.contour(Tn, Sn, np.abs(beta_d), [18.435, 26.565], colors='w',
                   linewidths=1.25)
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.set_label(r'$|\beta|$, sweep angle')
cbar.add_lines(contb)
ax.set_ylim(0, 1)
ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylabel('location along body')
ax.set_xlabel('time')

#spn = ax.axvspan(2, Tn.max(), fc='white', lw=0, alpha=1, zorder=100)
#xy = spn.get_xy()
#xy[[0, 1, 4], 0] = 4  # current time
#spn.set_xy(xy)

# add degree symbol to angles
ticks = cbar.ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
cbar.ax.set_yticklabels(newticks)

sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, Re, vmin=3000, vmax=15000, cmap=cmaps.plasma)
ax.axhline(Sn[0, n_neck], color='white', lw=1, zorder=1)
ax.contour(Tn[:, n_neck:], Sn[:, n_neck:], kap[:, n_neck:], [0],
           colors='gray', linewidths=1)
ax.contour(Tn, Sn, tang, [0], colors='w', linewidths=1)
contRe = ax.contour(Tn, Sn, Re, [9000], colors='w',
                       linewidths=1.25)
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.set_label('Reynolds number')
cbar.set_ticks(np.arange(3000, 15001, 2000))
cbar.add_lines(contRe)
ax.set_ylim(0, 1)
ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylabel('location along body')
ax.set_xlabel('time')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(R[:, :, 0], R[:, :, 1], R[:, :, 2],
          representation='surface', colormap='PRGn')
mlab.orientation_axes()


# %%

#fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# mlab.plot3d(Ro[:, 0], Ro[:, 1], Ro[:, 2], color=bmap[0], tube_radius=.003)

sk = 10
mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              Fl[::sk, :, 0], Fl[::sk, :, 1], Fl[::sk, :, 2], color=bmap[0],
              mode='arrow', resolution=64, scale_factor=10)
mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              Fd[::sk, :, 0], Fd[::sk, :, 1], Fd[::sk, :, 2], color=bmap[1],
              mode='arrow', resolution=64, scale_factor=10)

mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              dR[::sk, :, 0], dR[::sk, :, 1], dR[::sk, :, 2], color=bmap[3],
              mode='arrow', resolution=64, scale_factor=.01)
mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              dR_BC[::sk, :, 0], dR_BC[::sk, :, 1], dR_BC[::sk, :, 2],
              color=bmap[4], mode='arrow', resolution=64, scale_factor=.01)

for i in np.arange(ntime)[::sk]:
    mlab.points3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                  scale_factor=0.01)
    mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
                  scale_factor=.02)

mlab.orientation_axes()


# %%

# top view
mlab.view(azimuth=90, elevation=0, distance='auto')

# side view (x-z)
mlab.view(azimuth=-90, elevation=90, distance='auto')

# looking-up
mlab.view(-28.763817635744569,
 105.56011582018024,
 1.3184319107519193,
 np.array([ 2.85185544,  0.03199681,  5.09627075]))


# %%

now = time.time()
mlab.savefig('/Users/isaac/Desktop/test6.png', size=(2**11, 2**11))
#mlab.savefig('/Users/isaac/Desktop/test5.png', magnification=2.5)
print('Elapsed time: {0:.3f}'.format(time.time() - now))


# %%

img = plt.imread('/Users/isaac/Desktop/test2.png')

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
fig.set_tight_layout(True)


 # %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro[:, 0], Ro[:, 1], Ro[:, 2], color=bmap[2], tube_radius=.003)

sk = 5
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 0, 0], Borig[::sk, 0, 1], Borig[::sk, 0, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 1, 0], Borig[::sk, 1, 1], Borig[::sk, 1, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 2, 0], Borig[::sk, 2, 1], Borig[::sk, 2, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)

for i in np.arange(ntime)[::sk]:
    mlab.plot3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                  tube_radius=.005)
    mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
                  scale_factor=.02, mode='sphere')


# %% Plot TCB coordinate system along the body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro[:, 0], Ro[:, 1], Ro[:, 2], color=bmap[2], tube_radius=.003)

sk = 10

mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              Fl[::sk, :, 0], Fl[::sk, :, 1], Fl[::sk, :, 2], color=bmap[0],
              mode='arrow', resolution=64, scale_factor=10)
mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
              Fd[::sk, :, 0], Fd[::sk, :, 1], Fd[::sk, :, 2], color=bmap[1],
              mode='arrow', resolution=64, scale_factor=10)

#mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
#              dR[::sk, :, 0], dR[::sk, :, 1], dR[::sk, :, 2], color=bmap[3],
#              mode='arrow', resolution=64, scale_factor=.01)
#
#mlab.quiver3d(R[::sk, :, 0], R[::sk, :, 1], R[::sk, :, 2],
#              dR_BC[::sk, :, 0], dR_BC[::sk, :, 1], dR_BC[::sk, :, 2],
#              color=bmap[4], mode='arrow', resolution=64, scale_factor=.01)

for i in np.arange(ntime)[::sk]:
    mlab.plot3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                  tube_radius=.005)
    mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
                  scale_factor=.02)

#    mlab.quiver3d(R[i, :, 0], R[i, :, 1], R[i, :, 2],
#                  Tv[i, :, 0], Tv[i, :, 1], Tv[i, :, 2],
#                  color=bmap[0], mode='arrow', resolution=64, scale_factor=.05)
    mlab.quiver3d(R[i, :, 0], R[i, :, 1], R[i, :, 2],
                  Cv[i, :, 0], Cv[i, :, 1], Cv[i, :, 2],
                  color=bmap[1], mode='arrow', resolution=64, scale_factor=.05)
    mlab.quiver3d(R[i, :, 0], R[i, :, 1], R[i, :, 2],
                  Bv[i, :, 0], Bv[i, :, 1], Bv[i, :, 2],
                  color=bmap[2], mode='arrow', resolution=64, scale_factor=.05)


# %%

fig, ax = plt.subplots(figsize=(4, 7))
ax.axhline(0, color='gray', lw=.75)
ax.axvline(0, color='gray', lw=.75)
ax.plot(Ro[:, 0], Ro[:, 2])
ax.axis('equal', adjustable='box')
ax.set_ylim(Ro[:, 2].min(), Ro[:, 2].max())
sns.despine()
fig.set_tight_layout(True)

#fig.savefig('/Users/isaac/Desktop/xz.pdf', transparent=True, bbox_inches='tight')


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(ts, Ro[:, 0], label=r'$X$')
ax.plot(ts, Ro[:, 1], label=r'$Y$')
ax.plot(ts, Ro[:, 2], label=r'$Z$')
ax.legend(loc='best')
ax.set_xlabel('time (s)')
ax.set_ylabel('position of center of mass (m)')
ax.set_xlim(ts.min(), ts.max())
sns.despine()
fig.set_tight_layout(True)


# %% Euler angles

yaw, pitch, roll = ang.T

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(ts, np.rad2deg(yaw - yaw[0]), label=r'yaw')
ax.plot(ts, np.rad2deg(pitch), label=r'pitch')
ax.plot(ts, np.rad2deg(roll), label=r'roll')
ax.legend(loc='best')
ax.set_xlabel('time (sec)')
ax.set_ylabel('Euler angles')
ax.set_xlim(ts.min(), ts.max())

plt.draw()
for ax in [ax]:
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    ax.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)


# %% Euler angle rates

yawrate, pitchrate, rollrate = dang.T

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(ts, np.rad2deg(yawrate), label=r'yaw')
ax.plot(ts, np.rad2deg(pitchrate), label=r'pitch')
ax.plot(ts, np.rad2deg(rollrate), label=r'roll')
ax.legend(loc='best')
ax.set_xlabel('time (sec)')
ax.set_ylabel('Euler angle rates (deg/sec)')
ax.set_xlim(ts.min(), ts.max())

plt.draw()
for ax in [ax]:
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    ax.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)


# %% Euler angle - Euler angle rate space

# 1 / freq_undulation_hz / dt  # points/undulation cycle

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)
ln2 = ax.plot(np.rad2deg(roll), np.rad2deg(rollrate), 'o-', markevery=10, label='roll')
ln3 = ax.plot(np.rad2deg(yaw - yaw[0]), np.rad2deg(yawrate), '^-', markevery=10, label='yaw')
ln1 = ax.plot(np.rad2deg(pitch), np.rad2deg(pitchrate), 'x-', markevery=10,
              mew=1, label='pitch')

for ln in [ln1, ln2, ln3]:
    plotting.add_arrow_to_line2D(ax, ln, arrow_locs=[.95], arrowsize=2,
                         arrowstyle='->')

ax.legend(loc='best')
ax.set_xlabel('angle (deg)')
ax.set_ylabel('angle rate (deg/sec)')
sns.despine()
fig.set_tight_layout(True)

plt.draw()
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
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)


# %% Pitch - pitch rate space

from phaseport import p_pr_space, r_rr_space, vx_vz_space

d = np.load('./Output/A=1.6_lk=0.6_f=1.4.npz')

pitches_d = np.arange(-45, 11, 5)
pitch_rates_d = np.arange(-100, 21, 5)

rolls_d = np.arange(-20, 11, 5)
roll_rates_d = np.arange(-100, 101, 10)

# roll phase space
RD, RRD, UR, VR = r_rr_space(d, rolls_d, roll_rates_d, print_time=True)

np.savez('roll_space_A=1.6_lk=0.6_f=1.4.npz',
         rolls_d=rolls_d, roll_rates_d=roll_rates_d,
         RD=RD, RRD=RRD, UR=UR, VR=VR)


# %%

from phaseport import p_pr_space, r_rr_space, vx_vz_space

d = np.load('./Output/A=1.6_lk=0.6_f=1.4.npz')

pitches_d = np.arange(-45, 11, 5)
pitch_rates_d = np.arange(-100, 21, 5)

PD, PRD, UP, VP = p_pr_space(d, pitches_d, pitch_rates_d, print_time=True)

np.savez('pitch_space_A=1.6_lk=0.6_f=1.4.npz',
         pitches_d=pitches_d, pitch_rates_d=pitch_rates_d,
         PD=PD, PRD=PD, UP=UP, VP=VP)


# %%

from phaseport import p_pr_space, r_rr_space, vx_vz_space

d = np.load('./Output/A=1.6_lk=0.6_f=1.4.npz')

vxs = np.arange(0, 10.1, .5)
vzs = np.arange(0, -10.1, -.5)

VX, VZ, AX, AZ = vx_vz_space(d, vxs, vzs, print_time=True)

np.savez('vzvz_space_A=1.6_lk=0.6_f=1.4.npz',
         vxs=vxs, vzs=vzs,
         VX=VX, VZ=VZ, AX=AX, AZ=AZ)


# %%

dd = np.load('roll_space_A=1.6_lk=0.6_f=1.4.npz')
rolls_d, roll_rates_d = dd['rolls_d'], dd['roll_rates_d']
RD, RRD, UR, VR = dd['RD'], dd['RRD'], dd['UR'], dd['VR']


dd = np.load('pitch_space_A=1.6_lk=0.6_f=1.4.npz')
pitches_d, pitch_rates_d = dd['pitches_d'], dd['pitch_rates_d']
PD, PRD, UP, VP = dd['PD'], dd['PRD'], dd['UP'], dd['VP']


dd = np.load('vzvz_space_A=1.6_lk=0.6_f=1.4.npz')
vxs, vzs = dd['vxs'], dd['vzs']
VX, VZ, AX, AZ = dd['VX'], dd['VZ'], dd['AX'], dd['AZ']


# %% Roll-roll rate space

r_sim = np.rad2deg(ang[:, 2])
rr_sim = np.rad2deg(dang[:, 2])

norm = 1.1 * np.sqrt(UR**2 + VR**2)
URn, VRn = UR / norm, VR / norm

seed_points_x = np.r_[RD[0], RD[-1], RD[:, 0], RD[:, -1]]
seed_points_y = np.r_[RRD[0], RRD[-1], RRD[:, 0], RRD[:, -1]]
seed_points = np.c_[seed_points_x, seed_points_y]

i = 125

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)

# ax.quiver(RD, RRD, URn[i].T, VRn[i].T, pivot='middle', color='gray')
strm = ax.streamplot(rolls_d, roll_rates_d, UR[i].T, VR[i].T,
                     density=2, linewidth=.5, color=bmap[1],
                     arrowstyle='->')  # , start_points=.5*seed_points)
ax.plot(r_sim[:i + 1], rr_sim[:i + 1], c=bmap[0])
ax.set_xlim(rolls_d.min(), rolls_d.max())
ax.set_ylim(roll_rates_d.min(), roll_rates_d.max())

ax.set_title(i)
sns.despine()
fig.set_tight_layout(True)


# %% Roll-roll rate space movie

savename = './anim_roll/anim_{0:03d}.png'

r_sim = np.rad2deg(ang[:, 2])
rr_sim = np.rad2deg(dang[:, 2])

norm = 1. * np.sqrt(UR**2 + VR**2)
URn, VRn = UR / norm, VR / norm


time_template = 'time = {0:.2f} sec'

fig, ax = plt.subplots(figsize=(6.7, 6.5))
sns.despine()
fig.set_tight_layout(True)

now = time.time()
for i in np.arange(ntime):
    ax.clear()
    ax.axvline(0, color='gray', lw=.75)
    ax.axhline(0, color='gray', lw=.75)

    ax.streamplot(rolls_d, roll_rates_d, UR[i].T, VR[i].T,
                     density=2, linewidth=.5, color=bmap[1], arrowstyle='->',
                     arrowsize=4)

    ax.plot(r_sim[:i+1], rr_sim[:i+1], '-x', c=bmap[0], lw=2, markevery=10, mew=1)
    ax.plot(r_sim[i], rr_sim[i], 'o', c=bmap[0])

    ax.set_xlim(rolls_d.min(), rolls_d.max())
    ax.set_ylim(roll_rates_d.min(), roll_rates_d.max())

    ax.set_title(time_template.format(ts[i]))

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\dot{\phi}$   ', rotation=0)
    fig.savefig(savename.format(i))

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

savename = './anim_pitch/anim_{0:03d}.png'

ts, ntime = d['ts'], d['ntime']
p_sim = np.rad2deg(d['ang'][:, 1])
pr_sim = np.rad2deg(d['dang'][:, 1])

norm = 1. * np.sqrt(UP**2 + VP**2)
UPn, VPn = UP / norm, VP / norm


time_template = 'time = {0:.2f} sec'

fig, ax = plt.subplots(figsize=(6.7, 6.5))
sns.despine()
fig.set_tight_layout(True)

now = time.time()
for i in np.arange(ntime):
    ax.clear()
    ax.axvline(0, color='gray', lw=.75)
    ax.axhline(0, color='gray', lw=.75)

    ax.streamplot(pitches_d, pitch_rates_d, UP[i].T, VP[i].T,
                     density=2, linewidth=.5, color=bmap[1], arrowstyle='->',
                     arrowsize=4)

    ax.plot(p_sim[:i+1], pr_sim[:i+1], '-x', c=bmap[0], lw=2, markevery=10, mew=1)
    ax.plot(p_sim[i], pr_sim[i], 'o', c=bmap[0])

    ax.set_xlim(pitches_d.min(), pitches_d.max())
    ax.set_ylim(pitch_rates_d.min(), pitch_rates_d.max())

    ax.set_title(time_template.format(ts[i]))

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$   ', rotation=0)
    fig.savefig(savename.format(i))

print('elapsed time: {0:.3f} sec'.format(time.time() - now))

# to make the movie
# ffmpeg -f image2 -r 15 -i anim_%03d.png -pix_fmt yuv420p out.mp4


# %%

pp = np.rad2deg(ang[:, 1])
ppr = np.rad2deg(dang[:, 1])

PD, PRD = np.meshgrid(pitches_d, pitch_rates_d)
UD, VD = np.rad2deg(U_vecs), np.rad2deg(V_vecs)

norm = 1.1 * np.sqrt(UD**2 + VD**2)
UDn, VDn = UD / norm, VD / norm

#fig, ax = plt.subplots()
#sns.despine()
#fig.set_tight_layout(True)
#ax.axvline(0, color='gray', lw=.75)
#ax.axhline(0, color='gray', lw=.75)

i = 108

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)

ax.quiver(PD, PRD, UDn[i].T, VDn[i].T, pivot='middle', color='gray')
strm = ax.streamplot(pitches_d, pitch_rates_d, UDn[i].T, VDn[i].T,
                     density=2, linewidth=.5, color=bmap[1],
                     arrowstyle='->')
ax.plot(pp[:i], ppr[:i], c=bmap[0])

ax.set_title(i)
sns.despine()
fig.set_tight_layout(True)


# %%

pp = np.rad2deg(ang[:, 1])
ppr = np.rad2deg(dang[:, 1])

norm = 1.1 * np.sqrt(UD**2 + VD**2)
UDn, VDn = UD / norm, VD / norm

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)

traj, = ax.plot(pp[:1], ppr[:1], '-', c=bmap[0])
Q = ax.quiver(PD, PRD, UDn[0], VDn[0], pivot='middle', color='gray')

time_template = 'time = {0:.1f} sec'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_xlim(pitches_d.min(), pitches_d.max())
ax.set_ylim(pitch_rates_d.min(), pitch_rates_d.max())

sns.despine()
fig.set_tight_layout(True)

def animate(i):
    traj.set_data(pp[:i], ppr[:i])
    Q.set_UVC(UDn[i], VDn[i])
    time_text.set_text(time_template.format(i * dt))
    return traj, Q, time_text

slowed = 10
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False)


# %% Vx-Vz space movie

savename = './anim_vxvz/anim_{0:03d}.png'
dRo, ntime, ts = d['dRo'], d['ntime'], d['ts']

time_template = 'time = {0:.2f} sec'

fig, ax = plt.subplots(figsize=(8.125, 7.2))
sns.despine(ax=ax, top=False, bottom=True)

now = time.time()
for i in np.arange(ntime):
    ax.clear()

    ax.streamplot(vxs, vzs, AX[i].T, AZ[i].T,
                  density=2, linewidth=.5, color=bmap[1], arrowstyle='->',
                  arrowsize=4)
    ax.plot(dRo[:i+1, 0], dRo[:i+1, 2], '-x', c=bmap[0], markevery=10, mew=1)
    ax.plot(dRo[i, 0], dRo[i, 2], 'o', c=bmap[0])

    ax.text(7, -.5, time_template.format(ts[i]), fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(r'$v_x$', fontsize=18)
    ax.set_ylabel(r'$v_z$    ', rotation=0, fontsize=18)
    # ax.set_xlabel(r'horizontal velocity, $v_x$ (m/s)')
    # ax.set_ylabel(r'vertical velocity, $v_z$ (m/s)')
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 0)

    fig.savefig(savename.format(i))

print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

i = 130

time_template = 'time = {0:.2f} sec'

fig, ax = plt.subplots(figsize=(8.125, 7.2))

#ax.quiver(vxs2d, vzs2d, VXs[i].T, VZs[i].T, pivot='middle', color='gray')
strm = ax.streamplot(vxs, vzs, VXs[i].T, VZs[i].T,
                     density=2, linewidth=.5, color=bmap[1],
                     arrowstyle='->')
ax.plot(dRo[:i+1, 0], dRo[:i+1, 2], '-x', c=bmap[0], markevery=10, mew=1)
ax.plot(dRo[i, 0], dRo[i, 2], 'o', c=bmap[0])

ax.text(7, -.5, time_template.format(ts[i]), fontsize=16)
ax.set_aspect('equal', adjustable='box')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r'horizontal velocity, $v_x$ (m/s)')
ax.set_ylabel(r'vertical velocity, $v_z$ (m/s)')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %% Vx - Vz space

fig, ax = plt.subplots()

ax.axhline(0, color='gray', lw=.75)
ax.axvline(0, color='gray', lw=.75)
ln = ax.plot(dRo[:, 0], dRo[:, 2])
plotting.add_arrow_to_line2D(ax, ln, arrow_locs=[.8], arrowsize=2,
                         arrowstyle='->')

ax.legend()
ax.set_aspect('equal', adjustable='box')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r'horizontal velocity, $v_y$ (m/s)')
ax.set_ylabel(r'vertical velocity, $v_z$ (m/s)')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
sns.despine(ax=ax, top=False, bottom=True)


# %% angular velocity

fig, ax = plt.subplots()
ax.plot(ts, np.rad2deg(omg[:, 0]), label=r'$\omega_X$')
ax.plot(ts, np.rad2deg(omg[:, 1]), label=r'$\omega_Y$')
ax.plot(ts, np.rad2deg(omg[:, 2]), label=r'$\omega_Z$')
ax.legend()
ax.set_xlabel('time (sec)')
ax.set_ylabel('angular velocity, $\omega$ (deg/sec)')
ax.set_xlim(ts.min(), ts.max())
sns.despine()
fig.set_tight_layout(True)


# %% Angular momentum

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(ts, ho[:, 0])
ax1.plot(ts, ho[:, 1])
ax1.plot(ts, ho[:, 2])
ax2.plot(ts, Ho[:, 0])
ax2.plot(ts, Ho[:, 1])
ax2.plot(ts, Ho[:, 2])
ax2.set_xlim(ts.min(), ts.max())
ax2.set_xlabel('time (sec)')
ax1.set_ylabel(r'$h_o$')
ax2.set_ylabel(r'$H_o$')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(ts, cop[:, 0], label=r'$X$')
ax.plot(ts, cop[:, 1], label=r'$Y$')
ax.plot(ts, cop[:, 2], label=r'$Z$')
ax.legend(loc='best')
ax.set_xlim(ts.min(), ts.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel('center of pressure (m)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(ts, cop_z0[:, 0], label=r'$X$')
ax.plot(ts, cop_z0[:, 1], label=r'$Y$')
ax.plot(ts, cop_z0[:, 2], label=r'$Z$')
ax.legend(loc='best')
ax.set_xlim(ts.min(), ts.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel('center of pressure (m)')
sns.despine()
fig.set_tight_layout(True)


# %%

# cop in the local frame
cop_l = np.zeros_like(cop)
for i in np.arange(ntime):
    cop_l[i] = sim.rotate(C[i], cop[i])

fig, ax = plt.subplots()
ax.plot(ts, cop_l[:, 0], label=r'$X$')
ax.plot(ts, cop_l[:, 1], label=r'$Y$')
ax.plot(ts, cop_l[:, 2], label=r'$Z$')
ax.legend(loc='best')
ax.set_xlim(ts.min(), ts.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel('center of pressure (m)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(cop[:, 0], cop[:, 1], cop[:, 2], color=bmap[0], tube_radius=.0002)

i = 20
mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2], color=bmap[2],
            tube_radius=0.0005)
mlab.axes()
mlab.outline()


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(cop_z0[:, 0], cop_z0[:, 1], cop_z0[:, 2], color=bmap[0],
            tube_radius=.0002)

mlab.points3d(0, 0, 0, scale_factor=.005)

i = 20
mlab.plot3d(p[i, :, 0], p[i, :, 1], p[i, :, 2], color=bmap[2],
            tube_radius=0.0005)
mlab.axes()
mlab.outline()


# %% Recover the aerodynamic moment from CoP to see how close we are

Maero_rec = np.zeros((ntime, 3))
for i in np.arange(ntime):
    Maero_rec[i] = np.cross(cop[i], Faero[i]).sum(axis=0)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.plot(ts, Maero_rec[:, 0], label=r'$M_x$ recon')
ax1.plot(ts, Maero_rec[:, 1], label=r'$M_y$ recon')
ax1.plot(ts, Maero_rec[:, 2], label=r'$M_z$ recon')
ax2.plot(ts, Maero[:, 0], label=r'$M_x$')
ax2.plot(ts, Maero[:, 1], label=r'$M_y$')
ax2.plot(ts, Maero[:, 2], label=r'$M_z$')
ax2.legend(loc='best', ncol=3)
ax2.set_xlim(ts.min(), ts.max())
ax2.set_xlabel('time (sec)')
ax1.set_ylabel(r'aero moments (N$\cdot$m)')
ax2.set_ylabel(r'aero moments (N$\cdot$m)')
sns.despine()
fig.set_tight_layout(True)

def rms(ts1, ts2):
    return (ts1 - ts2)

fig, ax = plt.subplots()
ax.plot(ts, rms(Maero[:, 0], Maero_rec[:, 0]), label=r'$M_x$')
ax.plot(ts, rms(Maero[:, 1], Maero_rec[:, 1]), label=r'$M_y$')
ax.plot(ts, rms(Maero[:, 2], Maero_rec[:, 2]), label=r'$M_z$')
ax.legend(loc='best', ncol=3)
ax.set_xlim(ts.min(), ts.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel(r'aero moments (N$\cdot$m)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(T, S, Fl[:, :, 2], vmin=None, vmax=None, cmap=cmaps.viridis)
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.set_label(r'$F_L$')
ax.set_ylim(S.min(), S.max())
ax.set_xlim(T.min(), T.max())
ax.set_ylabel('distance along body, s (m)')
ax.set_xlabel('time (sec)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


fig, ax = plt.subplots()
cax = ax.pcolormesh(T, S, Fd[:, :, 2], vmin=None, vmax=None, cmap=cmaps.viridis)
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.set_label(r'$F_D$')
ax.set_ylim(S.min(), S.max())
ax.set_xlim(T.min(), T.max())
ax.set_ylabel('distance along body, s (m)')
ax.set_xlabel('time (sec)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%

now = time.time()
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
#fig.scene.disable_render = True
mlab.view(azimuth=50, elevation=28, distance=4.4781963805,
          focalpoint=np.array([-0.8398322 , -0.44114165, -2.27774457]))

mlab.plot3d(Ro[:, 0], Ro[:, 1], Ro[:, 2], color=bmap[0], tube_radius=.003)

sk = 1
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 0, 0], Borig[::sk, 0, 1], Borig[::sk, 0, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 1, 0], Borig[::sk, 1, 1], Borig[::sk, 1, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)
mlab.quiver3d(Ro[::sk, 0], Ro[::sk, 1], Ro[::sk, 2],
              Borig[::sk, 2, 0], Borig[::sk, 2, 1], Borig[::sk, 2, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.1)

for i in np.arange(ntime)[::5]:

    mlab.plot3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                tube_radius=0.008)

    mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
                  scale_factor=.02)
#fig.scene.disable_render = False
print time.time() - now


# %%

i = 20
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=1)
mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[1], opacity=1)
aa = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=1)
mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[4], opacity=1)
mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[4], opacity=.75)

tb = mlab.plot3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                tube_radius=0.003)
mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
              scale_factor=.015)
# tb.filter.capping = True
# tb.filter.number_of_sides = 12


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro[:, 0], Ro[:, 1], Ro[:, 2], color=bmap[0], tube_radius=.003)

for i in np.arange(ntime)[::10]:
    mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=.5)
    mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[1], opacity=.5)
    mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.5)
    mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.5)
    mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[4], opacity=.5)

    mlab.points3d(R[i, :, 0], R[i, :, 1], R[i, :, 2], color=bmap[1],
                  scale_factor=0.01)
    mlab.points3d(R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], color=bmap[1],
                  scale_factor=.02)


# %%

i = 200

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
mlab.orientation_axes()

#mlab.plot3d(r[i, :, 0], r[i, :, 1], r[i, :, 2], tube_radius=.005)
mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
              scale_factor=.025, resolution=16)
#_s = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#               scalars=fs, color=bmap[1])
_s = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
               scalars=scalars[i], colormap='copper', representation='surface',
               vmin=0, vmax=scalars.max())
#_s.module_manager.scalar_lut_manager.reverse_lut = True

# fig.scene.anti_aliasing_frames = 0


#img = mlab.screenshot(antialiased=False)
#
#_, ax = plt.subplots()
#ax.imshow(img)
#ax.axis('off')
#fig.set_tight_layout(True)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 30

head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], scalars[i, 0, -1],
                     colormap='GnBu', scale_mode='none',
                     vmin=scalars.min(), vmax=scalars.max(),
                     scale_factor=.015, resolution=16)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=scalars[i], colormap='GnBu',
                 representation='surface',
                 vmin=scalars.min(), vmax=scalars.max(),
                 opacity=.5)
#body.module_manager.scalar_lut_manager.reverse_lut = True

#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=scalars[i], color=bmap[1], opacity=.4)

ml = mlab.mesh(Lb[i, :, 0], Lb[i, :, 1], Lb[i, :, 2], color=bmap[0], opacity=1)
md = mlab.mesh(Db[i, :, 0], Db[i, :, 1], Db[i, :, 2], color=bmap[1], opacity=1)
ma = mlab.mesh(Ab[i, :, 0], Ab[i, :, 1], Ab[i, :, 2], color=bmap[2], opacity=1)

mlab.orientation_axes()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = -1

head = mlab.points3d(foils[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
                     scale_factor=.015, resolution=16, opacity=.5)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 color=bmap[1], opacity=.5)

ml = mlab.mesh(Lb[i, :, 0], Lb[i, :, 1], Lb[i, :, 2], color=bmap[0], opacity=.8)
md = mlab.mesh(Db[i, :, 0], Db[i, :, 1], Db[i, :, 2], color=bmap[4], opacity=.8)
ma = mlab.mesh(Ab[i, :, 0], Ab[i, :, 1], Ab[i, :, 2], color=bmap[2], opacity=.8)

mlab.orientation_axes()


# %%

import time

now = time.time()
#mlab.savefig('/Users/isaac/Desktop/meshes3.png', size=(2**11, 2**11))
mlab.savefig('/Users/isaac/Desktop/meshes4.png', size=(2**10, 2**10))
#mlab.savefig('/Users/isaac/Desktop/test5.png', magnification=2.5)
print('Elapsed time: {0:.3f}'.format(time.time() - now))


# %%

@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            print('Current time: {0}'.format(ts[i]))

            head.mlab_source.set(x=r[i, 0, 0], y=r[i, 0, 1], z=r[i, 0, 2])
            body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                                 z=foils[i, :, :, 2], scalars=scalars[i])

            ml.mlab_source.set(x=Lb[i, :, 0], y=Lb[i, :, 1], z=Lb[i, :, 2])
            md.mlab_source.set(x=Db[i, :, 0], y=Db[i, :, 1], z=Db[i, :, 2])
            ma.mlab_source.set(x=Ab[i, :, 0], y=Ab[i, :, 1], z=Ab[i, :, 2])

            yield
manim = anim()
mlab.show()


# %%

#fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(420, 420))

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(420, 420))

i = 0

head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
              scale_factor=.02, resolution=16)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=scalars[i], colormap='Greens',
                 representation='surface')  # , vmin=scalars.min(), vmax=scalars.max())
# body.module_manager.scalar_lut_manager.reverse_lut = True

#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=scalars[i], color=bmap[1], opacity=.4)

ml = mlab.mesh(Lb[i, :, 0], Lb[i, :, 1], Lb[i, :, 2], color=bmap[0], opacity=.5)
md = mlab.mesh(Db[i, :, 0], Db[i, :, 1], Db[i, :, 2], color=bmap[1], opacity=.5)
ma = mlab.mesh(Ab[i, :, 0], Ab[i, :, 1], Ab[i, :, 2], color=bmap[2], opacity=.5)


# %%

import moviepy.editor as mpy

def make_frame(t):
    i = np.abs(t - duration * ts / ts[-1]).argmin()
    print
    print('frame {0}, time {1}'.format(i, t))

    if i > 0:
        head.mlab_source.set(x=r[i, 0, 0], y=r[i, 0, 1], z=r[i, 0, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=scalars[i])

        ml.mlab_source.set(x=Lb[i, :, 0], y=Lb[i, :, 1], z=Lb[i, :, 2])
        md.mlab_source.set(x=Db[i, :, 0], y=Db[i, :, 1], z=Db[i, :, 2])
        ma.mlab_source.set(x=Ab[i, :, 0], y=Ab[i, :, 1], z=Ab[i, :, 2])

    return mlab.screenshot(antialiased=True)

duration = 10 * ts[-1]
fps = .1 * 1 / dt
times = np.arange(0, duration, 1 / fps)

#fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(320, 420))

animation = mpy.VideoClip(make_frame, duration=duration)
animation.write_gif('test_05_snake.gif', fps=fps)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 0

body = mlab.plot3d(r[i, :, 0], r[i, :, 1], r[i, :, 2], color=bmap[1],
                   tube_radius=0.005)
#body = mlab.points3d(r[i, :, 0], r[i, :, 1], r[i, :, 2], color=bmap[1],
#                     scale_factor=0.01)
head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
                     scale_factor=.02)

ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=.3)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[1], opacity=.3)
ma = mlab.mesh(Ab[i, :, 0], Ab[i, :, 1], Ab[i, :, 2], color=bmap[2], opacity=.5)
#vu = mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.5)
#bc = mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[4], opacity=.5)


# %%

@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            print('Current time: {0}'.format(ts[i]))

            body.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2])
            head.mlab_source.set(x=r[i, 0, 0], y=r[i, 0, 1], z=r[i, 0, 2])

            ml.mlab_source.set(x=Lb[i, :, 0], y=Lb[i, :, 1], z=Lb[i, :, 2])
            md.mlab_source.set(x=Db[i, :, 0], y=Db[i, :, 1], z=Db[i, :, 2])
            ma.mlab_source.set(x=Ab[i, :, 0], y=Ab[i, :, 1], z=Ab[i, :, 2])
            #ma.mlab_source.set(x=A[i, :, 0])
#            vu.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
#                               u=dR[i, :, 0], v=dR[i, :, 1], w=dR[i, :, 2])
#            bc.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
#                               u=dR_BC[i, :, 0], v=dR_BC[i, :, 1],
#                               w=dR_BC[i, :, 2])
            yield
manim = anim()
mlab.show()


# %% Good

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

FaeroT = Faero.sum(axis=1)

i = -1

body = mlab.plot3d(r[i, :, 0], r[i, :, 1], r[i, :, 2], color=bmap[1],
                   tube_radius=0.002)
#body = mlab.points3d(r[i, :, 0], r[i, :, 1], r[i, :, 2], color=bmap[1],
#                     scale_factor=0.01)
head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
                     scale_factor=.01)

mode = 'arrow'  # '2darrow'
ql = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2], color=bmap[0],
                   mode=mode, resolution=64, scale_factor=8, opacity=.5)
qd = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2], color=bmap[1],
                   mode=mode, resolution=64, scale_factor=8, opacity=.5)
qa = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
                   Faero[i, :, 0], Faero[i, :, 1], Faero[i, :, 2], color=bmap[2],
                   mode=mode, resolution=64, scale_factor=8)
#vu = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
#                   dR[i, :, 0], dR[i, :, 1], dR[i, :, 2], color=bmap[3],
#                   mode=mode, resolution=64, scale_factor=.02)
#bc = mlab.quiver3d(r[i, :, 0], r[i, :, 1], r[i, :, 2],
#                   dR_BC[i, :, 0], dR_BC[i, :, 1], dR_BC[i, :, 2],
#                   color=bmap[4], mode=mode, resolution=64,
#                   scale_factor=.02)

#qt = mlab.quiver3d(np.r_[0], np.r_[0], np.r_[0],
#                   np.r_[FaeroT[i, 0]], np.r_[FaeroT[i, 1]], np.r_[FaeroT[i, 2]],
#                   color=bmap[2], mode=mode, resolution=64, scale_factor=.1)

mlab.orientation_axes()


# %%

@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            print('Current time: {0}'.format(ts[i]))

            body.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2])
            head.mlab_source.set(x=r[i, 0, 0], y=r[i, 0, 1], z=r[i, 0, 2])

            ql.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
                               u=Fl[i, :, 0], v=Fl[i, :, 1], w=Fl[i, :, 2])
            qd.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
                               u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])
            qa.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
                               u=Faero[i, :, 0], v=Faero[i, :, 1],
                               w=Faero[i, :, 2])
#            qt.mlab_source.set(u=np.r_[FaeroT[i, 0]], v=np.r_[FaeroT[i, 1]],
#                               w=np.r_[FaeroT[i, 2]])
#            vu.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
#                               u=dR[i, :, 0], v=dR[i, :, 1], w=dR[i, :, 2])
#            bc.mlab_source.set(x=r[i, :, 0], y=r[i, :, 1], z=r[i, :, 2],
#                               u=dR_BC[i, :, 0], v=dR_BC[i, :, 1],
#                               w=dR_BC[i, :, 2])
            # fig.scene.camera.azimuth(10)
            yield
manim = anim()
mlab.show()


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

i = -1
#ev = rotate(Cs[i].T, .05 * eig_vecs[i].T)
#ep = uplot(ev)  # , coms[i])
#
#ax1 = mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[2], tube_radius=.001)
#ax2 = mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[2], tube_radius=.001)
#ax3 = mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[2], tube_radius=.001)

pos = Ris[i] - coms[i]
body = mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2], color=bmap[1],
                   tube_radius=0.003)
#body = mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], color=bmap[1],
#                     scale_factor=0.01)

mode = 'arrow'  # '2darrow'

head = mlab.points3d(pos[0, 0], pos[0, 1], pos[0, 2], color=bmap[1],
                     scale_factor=.02)

ql = mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
                   Fls[i, :, 0], Fls[i, :, 1], Fls[i, :, 2], color=bmap[0],
                   mode=mode, resolution=64, scale_factor=10)
qd = mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
                   0 * Fds[i, :, 0], 0 * Fds[i, :, 1], 0 * Fds[i, :, 2], color=bmap[1],
                   mode=mode, resolution=64, scale_factor=10)
#vu = mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
#                   dRis[i, :, 0], dRis[i, :, 1], dRis[i, :, 2], color=bmap[3],
#                   mode=mode, resolution=64, scale_factor=.02)
#bc = mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
#                   dRiBCs[i, :, 0], dRiBCs[i, :, 1], dRiBCs[i, :, 2],
#                   color=bmap[4], mode=mode, resolution=64,
#                   scale_factor=.02)


@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            # print('Current time: {0}'.format(ts[i]))
#            ev = rotate(Cs[i].T, .05 * eig_vecs[i].T)
#            ep = uplot(ev)
#
#            ax1.mlab_source.set(x=ep[0, :, 0], y=ep[0, :, 1], z=ep[0, :, 2])
#            ax2.mlab_source.set(x=ep[1, :, 0], y=ep[1, :, 1], z=ep[1, :, 2])
#            ax3.mlab_source.set(x=ep[2, :, 0], y=ep[2, :, 1], z=ep[2, :, 2])

            pos = Ris[i] - coms[i]
            body.mlab_source.set(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2])
            head.mlab_source.set(x=pos[0, 0], y=pos[0, 1], z=pos[0, 2])

            ql.mlab_source.set(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                               u=Fls[i, :, 0], v=Fls[i, :, 1], w=Fls[i, :, 2])
            qd.mlab_source.set(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                               u=Fds[i, :, 0], v=Fds[i, :, 1], w=Fds[i, :, 2])
#            vu.mlab_source.set(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#                               u=dRis[i, :, 0], v=dRis[i, :, 1], w=dRis[i, :, 2])
#            bc.mlab_source.set(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#                               u=dRiBCs[i, :, 0], v=dRiBCs[i, :, 1], w=dRiBCs[i, :, 2])
            yield
manim = anim()
mlab.show()


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

#mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[4],
#            tube_radius=.005, opacity=.3)

for i in np.arange(nuse)[::10]:
    xx, yy, zz = Ris[i, :].T
    fl = Fls[i]
    fd = Fds[i]
    vu = dRis[i]
    vbc = dRiBCs[i]

    # mlab.flow(xx, yy, zz, fl[:, 0], fl[:, 1], fl[:, 2], color=bmap[0])

# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

i = 30
mlab.mesh(us[:, i, 0], us[:, i, 1], us[:, i, 2], color=bmap[0])



# %%

arr1 = mlab.screenshot(antialiased=False)
fig.scene.anti_aliasing_frames = 0
arr2 = mlab.screenshot(antialiased=True)

ffig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.imshow(arr1)
ax2.imshow(arr2)
sns.despine()
ffig.set_tight_layout(True)


# %%

# ffmpeg -r 30 -i %04d.png -q 0 -pix_fmt yuv420p -vf scale=700:700 out.mp4
# ffmpeg -r 30 -pattern_type glob -i img0000/*.tiff' -q 0 -pix_fmt yuv420p out.mp4
savename = 'movies/tmp2/{0:04d}.png'

fig = mlab.figure(bgcolor=(1, 1, 1), size=(750, 750))

i = 0
ev = rotate(Cs[i].T, .1 * eig_vecs[i].T)
ep = uplot(ev, coms[i])

ax1 = mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[1], tube_radius=.01)
ax2 = mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[1], tube_radius=.01)
ax3 = mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[1], tube_radius=.01)

body = mlab.plot3d(Ris[i, :, 0], Ris[i, :, 1], Ris[i, :, 2], color=bmap[2],
                   tube_radius=0.01)

head = mlab.points3d(Ris[i, 0, 0], Ris[i, 0, 1], Ris[i, 0, 2], color=bmap[2],
                     scale_factor=.05)

cm = mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[4], tube_radius=.01)

for i in np.arange(ntime):
    ev = rotate(Cs[i].T, .1 * eig_vecs[i].T)
    ep = uplot(ev, coms[i])

    # ax1.mlab_source.scalars = np.array([ep[0, :, 0], ep[0, :, 1], ep[0, :, 2]])
    ax1.mlab_source.set(x=ep[0, :, 0], y=ep[0, :, 1], z=ep[0, :, 2])
    ax2.mlab_source.set(x=ep[1, :, 0], y=ep[1, :, 1], z=ep[1, :, 2])
    ax3.mlab_source.set(x=ep[2, :, 0], y=ep[2, :, 1], z=ep[2, :, 2])

    body.mlab_source.set(x=Ris[i, :, 0], y=Ris[i, :, 1], z=Ris[i, :, 2])
    head.mlab_source.set(x=Ris[i, 0, 0], y=Ris[i, 0, 1], z=Ris[i, 0, 2])
    # mlab.savefig(savename.format(j), magnification=1)


## %%
#
#import numpy as np
#import mayavi.mlab as mlab
#import  moviepy.editor as mpy
#
#duration= 2 # duration of the animation in seconds (it will loop)
#
## MAKE A FIGURE WITH MAYAVI
#
#fig_myv = mlab.figure(size=(720,720), bgcolor=(1,1,1))
#X, Y = np.linspace(-2,2,200), np.linspace(-2,2,200)
#XX, YY = np.meshgrid(X,Y)
#ZZ = lambda d: np.sinc(XX**2+YY**2)+np.sin(XX+d)
#
## ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF
#
#def make_frame(t):
#    print t
#    mlab.clf() # clear the figure (to reset the colors)
#    mlab.mesh(YY,XX,ZZ(2*np.pi*t/duration), figure=fig_myv)
#    return mlab.screenshot(antialiased=True)
#
##animation = mpy.VideoClip(make_frame, duration=duration)
##animation.write_gif("sinc.gif", fps=20)
##animation.write_videofile("my_animation.mp4", fps=24, codec='mpeg4',
##                          ffmpeg_params=['-pix_fmt', 'yuv420p']) # export as video
##animation.write_videofile("my_animation.ogv", fps=24, codec='libvorbis',
##                          ffmpeg_params=['-pix_fmt', 'yuv420p'])
##animation.write_videofile('my_animation.avi', fps=24, codec='rawvideo')
#
#import time
#now = time.time()
#frames = []
#for t in np.linspace(0, 2, 51):
#    frames.append(make_frame(t))
#print time.time() - now
#
#
## %%
#
#imageio.mimwrite('test.png', frames)

# %%

mlab.close(all=True)
fig = mlab.figure(bgcolor=(1, 1, 1), size=(715, 750))

i = 0
ev = rotate(Cs[i].T, .1 * eig_vecs[i].T)
ep = uplot(ev, coms[i])

ax1 = mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[1], tube_radius=.01)
ax2 = mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[1], tube_radius=.01)
ax3 = mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[1], tube_radius=.01)

body = mlab.plot3d(Ris[i, :, 0], Ris[i, :, 1], Ris[i, :, 2], color=bmap[2],
                   tube_radius=0.01)

head = mlab.points3d(Ris[i, 0, 0], Ris[i, 0, 1], Ris[i, 0, 2], color=bmap[2],
                     scale_factor=.05)

cm = mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[4], tube_radius=.01)

@mlab.animate(delay=100)
def anim():
    while True:
        for i in np.arange(ntime):
            print('Current time: {0}'.format(ts[i]))
            ev = rotate(Cs[i].T, .1 * eig_vecs[i].T)
            ep = uplot(ev, coms[i])

            ax1.mlab_source.set(x=ep[0, :, 0], y=ep[0, :, 1], z=ep[0, :, 2])
            ax2.mlab_source.set(x=ep[1, :, 0], y=ep[1, :, 1], z=ep[1, :, 2])
            ax3.mlab_source.set(x=ep[2, :, 0], y=ep[2, :, 1], z=ep[2, :, 2])

            body.mlab_source.set(x=Ris[i, :, 0], y=Ris[i, :, 1], z=Ris[i, :, 2])
            head.mlab_source.set(x=Ris[i, 0, 0], y=Ris[i, 0, 1], z=Ris[i, 0, 2])
            yield
manim = anim()
mlab.show()


# %%

fig = mlab.figure(size=(750, 750))

for i in np.arange(0, nuse, 10):
    pass
    # mlab.plot3d(ris[i, :, 0], ris[i, :, 1], ris[i, :, 2], color=bmap[2], tube_radius=0.01)
    # mlab.points3d([ris[i, 0, 0]], [ris[i, 0, 1]], [ris[i, 0, 2]], color=bmap[2],
    #              scale_factor=.1)

mlab.points3d(ris[0, 0, 0], ris[0, 0, 1], ris[0, 0, 2], scale_factor=.025)
for j in np.r_[0]:  # , nbody//4, nbody//2, 3 * nbody//4, -1]:
     mlab.plot3d(Ris[:, j, 0], Ris[:, j, 1], Ris[:, j, 2], color=bmap[3],
                 tube_radius=.005)


# %%


# %% PCA of shape

Td = np.rad2deg(T)

cpall = sns.diverging_palette(145, 280, s=85, l=25, as_cmap=True)
fig, ax = plt.subplots()
vlim = np.max([Td.max(), np.abs(Td.min())])
cax = ax.pcolormesh(SS, TS, Td, vmin=-vlim, vmax=vlim, cmap=plt.cm.RdBu_r)
cbar = fig.colorbar(cax, orientation='horizontal', shrink=.875)
cbar.set_label('tangent angle (deg)')
ax.set_xlim(SS.min(), SS.max())
ax.set_ylim(TS.min(), TS.max())
ax.set_xlabel('distance along body, s (m)')
ax.set_ylabel('time (sec)')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
vlim = np.max([COV.max(), np.abs(COV.min())])
cax = ax.pcolormesh(Scov1, Scov2, COV, vmin=-vlim, vmax=vlim,
                    cmap=plt.cm.RdBu_r)
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(cax, orientation='vertical', shrink=.875)
cbar.set_label('covariance')
ax.set_xlim(Scov1.min(), Scov1.max())
ax.set_ylim(Scov2.min(), Scov2.max())
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%
covvals, covvecs = eig(COV)
cvals = covvals.real
var = cvals.sum()
expl = cvals / var

fig, ax = plt.subplots()
ax.plot(np.arange(len(expl) + 1), np.r_[0, expl.cumsum()], 'o-')
ax.set_xlim(xmax=10)
ax.set_ylim(0, 1.025)
sns.despine()
fig.set_tight_layout(True)


# %%

a1 = (covvecs[:, 0].real * np.deg2rad(T)).sum(axis=0)
a2 = (covvecs[:, 1].real * np.deg2rad(T)).sum(axis=0)

a1 = a1 - a1.mean()
a2 = a2 - a2.mean()
a1 = a1 / a1.std()
a2 = a2 / a2.std()

fig, ax = plt.subplots()
ax.plot(a1, a2, '-')
# sns.jointplot(a1, a2, kind="hex")
ax.set_aspect('equal', adjustable='box')
#ax.set_xlim(-2, 2)
sns.despine()
fig.set_tight_layout(True)


# %%

th1 = a1 * covvecs[:, 0].real
th2 = a2 * covvecs[:, 1].real

dg1 = np.rad2deg(th1)
dg2 = np.rad2deg(th2)

th_act = T[0] - T[0].mean()
th_act = th_act / th_act.std()

fig, ax = plt.subplots()
ax.plot(s, dg1)
ax.plot(s, dg2)
ax.plot(s, dg1 + dg2)
ax.plot(s, th_act)
sns.despine()
fig.set_tight_layout(True)


# %% test out angles for aerodynamic forces

# current time
#units = unit_vecs[0, [0, 30]]
units = np.array([[[ 0.10127632,  0.99485834,  0.        ],
                   [ 0.99485834, -0.10127632,  0.        ],
                   [ 0.        ,  0.        ,  1.        ]],
                  [[ 0.76600743,  0.64283172,  0.        ],
                   [ 0.64283172, -0.76600743,  0.        ],
                   [ 0.        ,  0.        ,  1.        ]]])
C = sim.euler2C(*np.deg2rad(np.array([10, 20, -15])))
#C = euler2C(*np.deg2rad(np.array([0, 0, 0])))
dRis = np.array([[1, 1, 1],
                 [-1, 1, 1],
                 [1, -1, 1],
                 [1, 1, -1],
                 [-1, 1, -1]])

i = 0  # current mass
j = 4  # current velocity
dRi = dRis[j]

# boddy coordinates in local and global coordinates
# note that serp-local is the same as local
uv = units[i]
UV = sim.rotate(C.T, uv.T).T
ti, ci, bi = uv.T  # ti, ci, bi = uv[:, 0], uv[:, 1], uv[:, 2]
Ti, Ci, Bi = UV.T
#print Ti
#print Ci
#print Bi
#Ti = rotate(C.T, ti)
#Ci = rotate(C.T, ci)
#Bi = rotate(C.T, bi)

dRiT = np.dot(dRi, Ti) * Ti
dRiBC = dRi - dRiT
Ui = np.linalg.norm(dRiBC)
Dh = -dRiBC / Ui
Lh = np.cross(Ti, Dh)

Ui = np.linalg.norm(dRiBC)

# get angle of attack
radCi = np.arccos(np.dot(dRiBC, Ci) / Ui)
radBi = np.arccos(np.dot(dRiBC, Bi) / Ui)
degCi = np.rad2deg(radCi)
degBi = np.rad2deg(radBi)

if degCi < 90 and degBi >= 90:
    aoa = radCi
elif degCi < 90 and degBi < 90:
    aoa = -radCi
elif degCi >= 90 and degBi < 90:
    aoa = radCi - np.pi
    Lh = -Lh
elif degCi >= 90 and degBi >= 90:
    aoa = np.pi - radCi
    Lh = -Lh

# print degCi, degBi, np.rad2deg(aoa)
print('angle of attack: {0:.2f} deg'.format(np.rad2deg(aoa)))


fig = mlab.figure(bgcolor=(1, 1, 1), size=(750, 750))

ep = uplot(UV.T)
# b, r, g for Ti, Ci, Bi
#mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[0], tube_radius=.01)
#mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[2], tube_radius=.01)
#mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[1], tube_radius=.01)

mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[0], tube_radius=.01)
mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[2], tube_radius=.01)
mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[1], tube_radius=.01)


mlab.plot3d([0, dRi[0]], [0, dRi[1]], [0, dRi[2]], tube_radius=.01)
mlab.plot3d([0, dRiBC[0]], [0, dRiBC[1]], [0, dRiBC[2]], color=bmap[3], tube_radius=.01)

mlab.plot3d([0, Dh[0]], [0, Dh[1]], [0, Dh[2]], color=bmap[4], tube_radius=.01)
mlab.plot3d([0, Lh[0]], [0, Lh[1]], [0, Lh[2]], color=bmap[4], tube_radius=.01)

#mlab.outline(color=(.5, .5, .5))



# %%

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), size=(715, 750))
mlab.view(azimuth=50, elevation=28, distance=4.4781963805,
          focalpoint=np.array([-0.8398322 , -0.44114165, -2.27774457]))

mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[4], tube_radius=.005)

for i in np.arange(nuse)[::20]:
#    ev = rotate(Cs[i].T, .1 * eig_vecs[i].T)
#    ep = uplot(ev, coms[i])
#
#    mlab.plot3d(ep[0, :, 0], ep[0, :, 1], ep[0, :, 2], color=bmap[0], tube_radius=.005)
#    mlab.plot3d(ep[1, :, 0], ep[1, :, 1], ep[1, :, 2], color=bmap[0], tube_radius=.005)
#    mlab.plot3d(ep[2, :, 0], ep[2, :, 1], ep[2, :, 2], color=bmap[0], tube_radius=.005)


    mlab.plot3d(Ris[i, :, 0], Ris[i, :, 1], Ris[i, :, 2], color=bmap[1],
                tube_radius=0.008)

    mlab.points3d(Ris[i, 0, 0], Ris[i, 0, 1], Ris[i, 0, 2], color=bmap[1],
                  scale_factor=.03)

    # forces
    sl = 25
    sd = 25
    su = .025
    for j in np.arange(nbody)[::20]:
        xx, yy, zz = Ris[i, j, :]
        xg = [xx, xx + sl * Fgs[i, j, 0]]
        yg = [yy, yy + sl * Fgs[i, j, 1]]
        zg = [zz, zz + sl * Fgs[i, j, 2]]

        xl = [xx, xx + sl * Fls[i, j, 0]]
        yl = [yy, yy + sl * Fls[i, j, 1]]
        zl = [zz, zz + sl * Fls[i, j, 2]]

        xd = [xx, xx + sd * Fds[i, j, 0]]
        yd = [yy, yy + sd * Fds[i, j, 1]]
        zd = [zz, zz + sd * Fds[i, j, 2]]

        xa = [xx, xx + sd * Fas[i, j, 0]]
        ya = [yy, yy + sd * Fas[i, j, 1]]
        za = [zz, zz + sd * Fas[i, j, 2]]

        xt = [xx, xx + sd * Fts[i, j, 0]]
        yt = [yy, yy + sd * Fts[i, j, 1]]
        zt = [zz, zz + sd * Fts[i, j, 2]]

        xu = [xx, xx + su * dRis[i, j, 0]]
        yu = [yy, yy + su * dRis[i, j, 1]]
        zu = [zz, zz + su * dRis[i, j, 2]]

        xbc = [xx, xx + su * dRiBCs[i, j, 0]]
        ybc = [yy, yy + su * dRiBCs[i, j, 1]]
        zbc = [zz, zz + su * dRiBCs[i, j, 2]]

        mlab.plot3d(xl, yl, zl, color=bmap[0], tube_radius=.005)
        mlab.plot3d(xd, yd, zd, color=bmap[1], tube_radius=.005)
        mlab.plot3d(xa, ya, za, color=bmap[2], tube_radius=.005)
        mlab.plot3d(xt, yt, zt, color=(0, 0, 0), tube_radius=.005)
        mlab.plot3d(xg, yg, zg, color=bmap[5], tube_radius=.005)

        mlab.plot3d(xu, yu, zu, color=bmap[3], tube_radius=.005)
        mlab.plot3d(xbc, ybc, zbc, color=bmap[4], tube_radius=.005)

#        sc = mlab.pipeline.vector_scatter(Ris[i, j, 0], Ris[i, j, 1],
#                                          Ris[i, j, 2], su * dRis[i, j, 0],
#                                          su * dRis[i, j, 1],
#                                          su * dRis[i, j, 2])
#        v = mlab.pipeline.vectors(sc)
#        v.glyph.glyph.clamping=False

#        ar = Arrow_From_A_to_B(xu[0], yu[0], zu[0], xu[1], yu[1], zu[1])

#        arrow([xu[0], yu[0], zu[0]], [xu[1], yu[1], zu[1]], a_col=bmap[3])

#        mlab.quiver3d(xu[0], yu[0], zu[0], su * dRis[i, j, 0],
#                      su * dRis[i, j, 1], su * dRis[i, j, 2], mode='cone',
#                      color=bmap[3], resolution=64, scale_factor=.5)

print('Visualization time {0:.3f}'.format(time.time() - now))



# %%

Fx = fx.sum()
Fy = fy.sum()
Fz = fz.sum()
FX = fX.sum()
FZ = fZ.sum()
FL = fl.sum()
FD = fd.sum()
F = np.hypot(Fx, Fz)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(s, fx / ds)
ax2.plot(s, fz / ds)
sns.despine()
fig.set_tight_layout(True)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(s, fl / ds)
ax2.plot(s, fd / ds)
sns.despine()
fig.set_tight_layout(True)




# %% Center of pressure

Fx, Fy, Fz = fx.sum(), fy.sum(), fz.sum()
Fx_bar = ((x - cx) * fx).sum() / Fx
Fy_bar = ((y - cy) * fy).sum() / Fy
#Fz_bar =


# %% Contour plots of forces along the body

cmap = sns.light_palette("green", as_cmap=True)
ii = 0

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=.75)
ax.axvline(0, color='gray', linewidth=.75)
cax = ax.scatter(X[ii], Y[ii], s=30, c=1e3 * fxs[ii], linewidths=0, cmap=cmap)
ax.plot(coms[ii, 0], coms[ii, 1], 'o', c='gray', mec='gray')
cbar = fig.colorbar(cax, ax=ax, pad=.085)
ax.axis('equal')
ax.set_title('fx')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=.75)
ax.axvline(0, color='gray', linewidth=.75)
cax = ax.scatter(X[ii], Y[ii], s=30, c=1e3 * fzs[ii], linewidths=0, cmap=cmap)
ax.plot(coms[ii, 0], coms[ii, 1], 'o', c='gray', mec='gray')
cbar = fig.colorbar(cax, ax=ax, pad=.085)
ax.axis('equal')
ax.set_title('fz')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=.75)
ax.axvline(0, color='gray', linewidth=.75)
cax = ax.scatter(X[ii], Y[ii], s=30, c=1e3 * mxs[ii], linewidths=0, cmap=cmap)
ax.plot(coms[ii, 0], coms[ii, 1], 'o', c='gray', mec='gray')
cbar = fig.colorbar(cax, ax=ax, pad=.085)
ax.axis('equal')
ax.set_title('mx: {0:.3f} mN m'.format(1e3 * mxs[ii].sum()))
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=.75)
ax.axvline(0, color='gray', linewidth=.75)
cax = ax.scatter(X[ii], Y[ii], s=30, c=1e3 * mys[ii], linewidths=0, cmap=cmap)
ax.plot(coms[ii, 0], coms[ii, 1], 'o', c='gray', mec='gray')
cbar = fig.colorbar(cax, ax=ax, pad=.085)
ax.axis('equal')
ax.set_title('my: {0:.3f} mN m'.format(1e3 * mys[ii].sum()))
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=.75)
ax.axvline(0, color='gray', linewidth=.75)
cax = ax.scatter(X[ii], Y[ii], s=30, c=1e3 * mzs[ii], linewidths=0, cmap=cmap)
ax.plot(coms[ii, 0], coms[ii, 1], 'o', c='gray', mec='gray')
cbar = fig.colorbar(cax, ax=ax, pad=.085)
ax.axis('equal')
ax.set_title('mz: {0:.3f} mN m'.format(1e3 * mzs[ii].sum()))
sns.despine()
fig.set_tight_layout(True)

