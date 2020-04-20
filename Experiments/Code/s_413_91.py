# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:54:46 2015

%reset -f
%clear
%pylab
%load_ext autoreload
%autoreload 2

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev
from scipy.integrate import cumtrapz

import time

import seaborn as sns
from mayavi import mlab

import m_data_utils as data_utils
import m_smoothing as smoothing
import m_ukf_filter as ukf_filter
import m_asd as asd

# %%

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42}
#sns.set('notebook', 'ticks', font_scale=1.5, rc=rc)
#sns.set('notebook', 'ticks', font_scale=1.75, rc=rc)
rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# http://xkcdcp.martinblech.com/#emerald green
emerald_green = '#028f1e'


# %% Load in raw data

# this was original data
fname = '../Data/Complete trials/413_91_orig.tsv'
df = data_utils.load_qtm_tsv(fname)

## this is one Grant did
#fname = '../Data/Complete trials/413_91.tsv'
#df1 = data_utils.load_qtm_tsv(fname)
#
## there are some NaNs here, get them out
## df1 = df1[df1['Frame'] >= df['Frame'].ix[0]]
#df1 = df[df['Time'] == df1['Time']]


# %% Reconfigure data into a 3D array

pr, out = data_utils.reconfig_raw_data(df)
times, frames = out['times'], out['frames']

# since these are the same, use the 'original' one
ntime, nmark, ncoord = pr.shape

fs = 179
dt = 1 / fs


# %% Plot x, y, and z coordinates

def plot_xyz(pr, times, markers=False, title=''):
    from itertools import cycle

    ntime, nmark, ncoord = pr.shape
    # colors = sns.dark_palette(color, n_colors=nmark)
    colors = sns.husl_palette(n_colors=nmark)
    # colors = sns.light_palette(color, n_colors=nmark)

    if markers:
        marks = cycle(['o', '^', '<', '>', 's', 'h', 'd'])
    else:
        marks = cycle([''])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    for j in np.arange(nmark):
        ax1.plot(times, pr[:, j, 0], next(marks) + '-', c=colors[j], ms=3)
        ax2.plot(times, pr[:, j, 1], next(marks) + '-', c=colors[j], ms=3)
        ax3.plot(times, pr[:, j, 2], next(marks) + '-', c=colors[j], ms=3)

    ax1.margins(.03)
    ax1.set_title(title)
    ax3.set_xlabel('time (s)')
    ax1.set_ylabel('X')
    ax2.set_ylabel('Y')
    ax3.set_ylabel('Z')
    sns.despine()
    fig.set_tight_layout(True)

    return fig, (ax1, ax2, ax3)


def plot_indiv(pr, times, k, markers=False, title=''):
    from itertools import cycle

    ntime, nmark, ncoord = pr.shape
    # colors = sns.dark_palette(color, n_colors=nmark)
    colors = sns.husl_palette(n_colors=nmark)
    # colors = sns.light_palette(color, n_colors=nmark)

    if markers:
        marks = cycle(['o', '^', '<', '>', 's', 'h', 'd'])
    else:
        marks = cycle([''])

    fig, ax = plt.subplots()
    for j in np.arange(nmark):
        ax.plot(times, pr[:, j, k], next(marks) + '-', c=colors[j], ms=3)

    ax.margins(.03)
    ax.set_title(title)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('{0}'.format(['X', 'Y', 'Z'][k]))
    sns.despine()
    fig.set_tight_layout(True)

    return fig, ax


# plot the raw data
fig, axs = plot_xyz(pr, times, markers=False, title='Raw positions')
fig, ax = plot_indiv(pr, times, 0, markers=False, title='Raw position, X')

# center the data (subtract out mean coordinate)
pc, com = data_utils.shift_to_com(pr)

fig, axs = plot_xyz(pc, times, markers=False, title='Raw positions')
fig, ax = plot_indiv(pc, times, 0, markers=False, title='Raw position, X')
# fig, ax = plot_indiv(pc, times, 0, markers=False, title='Raw position, 2nd')


# %% Residual analysis

R, fcs = smoothing.residual_butter(pr, fs, df=.5, fmin=1, fmax=35)
inter, fit_slope, fcopt, rsq, flinreg = smoothing.opt_cutoff(R, fcs, rsq_cutoff=.95)


# %% Plot the residuals and optimal cutoff frequencies

def plot_residuals(pr, R, fcs, inter=None, fcopt=None, markers=True):
    from itertools import cycle

    ntime, nmark, ncoord = pr.shape

    colors = sns.husl_palette(n_colors=nmark)

    if markers:
        marks = cycle(['o', '^', '<', '>', 's', 'h', 'd'])
    else:
        marks = cycle([''])

    fig, ax = plt.subplots()
    for j in np.arange(nmark):
        mark = next(marks) + '-'
        c = colors[j]
        al = .1
        ax.plot(fcs, R[:, j, 0], mark, c=c, ms=4, alpha=al)
        ax.plot(fcs, R[:, j, 1], mark, c=c, ms=4, alpha=al)
        ax.plot(fcs, R[:, j, 2], mark, c=c, ms=4, alpha=al)

        if inter is not None and fcopt is not None:
            nn = np.zeros(3)
            ax.plot(nn, inter[j], '>', ms=5, color=c)
            ax.plot(fcopt[j], nn, '^', ms=5, color=c)
            ax.plot(fcopt[j], inter[j], mark[:-1], c=c, ms=6)

    if inter is not None:
        ax.margins(.03, .003)

    ax.set_ylim(ymax=10)
    ax.grid(True)
    ax.set(xlabel='Cutoff frequency (Hz)', ylabel='Residual (mm)')
    sns.despine()
    fig.set_tight_layout(True)

    return fig, ax


fig, ax = plot_residuals(pr, R, fcs, inter, fcopt)


# %% Try filtering each x, y, z and marker with a different fc

but_unique = smoothing.but_fcs(pr, fs, fcopt)

bup, buv, bua = but_unique['p'], but_unique['v'], but_unique['a']

# plot the filtered data
fig, axs = plot_xyz(bup, times, markers=False, title='Raw positions')
fig, ax = plot_indiv(bup, times, 0, markers=False, title='Raw position, head')


# %% Save the data

if False:
    savename = 'Filtered_' + fname.split('/')[-1].split('.')[0] + '.csv'
    savename = '../Data/Filtered data/' + savename

    pf, vf, af = bup, buv, bua

    data_utils.save_filtered_data(savename, pf, vf, af, times)

    pt, vt, at, tt = data_utils.load_filtered_data(savename)

    # check that we can save and load the data correctly
    if True:
        print(np.allclose(pf, pt))  # True
        print(np.allclose(vf, vt))  # True
        print(np.allclose(af, at))  # True
        print(np.allclose(times, tt))  # True


# %%

import pandas as pd

marker_info = pd.read_csv('../Data/Snake markers/Processed/Marker-summary_400_91.csv')
dist_btn_markers = marker_info['Dist to next, mm'].dropna().values
dist_total = marker_info['total length (mm)'].values[0]
vent_idx = np.where(marker_info['Marker type'] == 'vent')[0][0]

dist_start_markers = marker_info['Marker loc, mm'].values[0]
dist_total_markers = marker_info['Marker loc, mm'].values[-1]
dist_svl = marker_info['svl (mm)'].values[0]

dist_arclen_markers = marker_info['Marker loc, mm'].values - dist_start_markers

loc_svl = marker_info['Marker loc, in svl'].values
loc_svl_norm = (loc_svl - loc_svl.min()) / loc_svl.ptp()

ntail_mark = nmark - 1 - vent_idx
nbody_mark = vent_idx + 1

nbody_mark = np.arange(nmark)[:vent_idx + 1].size
ntail_mark = np.arange(nmark)[vent_idx + 1:].size
colors_trunk = sns.light_palette('green', n_colors=nbody_mark)
colors_trunk = np.array(colors_trunk)[:, :3]
colors_tail = sns.light_palette('purple', n_colors=ntail_mark)
colors_tail = np.array(colors_tail)[:, :3]
colors_mark = np.r_[colors_trunk, colors_tail]

# need one less for differences
nbody_mark = np.arange(nmark)[:vent_idx].size
ntail_mark = np.arange(nmark)[vent_idx + 1:].size
colors_trunk = sns.light_palette('green', n_colors=nbody_mark)
colors_trunk = np.array(colors_trunk)[:, :3]
colors_tail = sns.light_palette('purple', n_colors=ntail_mark)
colors_tail = np.array(colors_tail)[:, :3]
colors_mark = np.r_[colors_trunk, colors_tail]


# %%




# %%

# extract/rename smooth position, velocity, acceleration valueds
pf, vf, af = bup.copy(), buv.copy(), bua.copy()

# "extended" smooth points. 2nd entry for 'virtual' marker
neck_len = 30  # 30 mm or 3 cm
pfe = np.zeros((ntime, nmark + 1, 3))
pfe[:, 0] = pf[:, 0]
pfe[:, 2:] = pf[:, 1:]

# new distance between markers (for spline calculations)
# aka segment parameters
te = np.zeros(nmark)
te[0] = neck_len
te[1] = dist_btn_markers[0] - neck_len
te[2:] = dist_btn_markers[1:]

# add the virtual marker by finding rotation of pf[i, 0] into Yhat direction
for i in np.arange(ntime):
    vx, vy = vf[i, 0, 0], vf[i, 0, 1]  # of head marker
    th_pfe = np.arctan2(vx, vy)

    # rotation matrix about Zhat
    Rth = np.array([[np.cos(th_pfe), -np.sin(th_pfe), 0],
                    [np.sin(th_pfe),  np.cos(th_pfe), 0],
                    [0, 0, 1]])

    # determine the y and z offset for the virtual marker
    p1 = pf[i, 0]  # head marker
    p2 = pf[i, 1]  # 2nd marker

    # average the z-coordinate b/n head and 2nd marker
    zoff = (p2[2] - p1[2]) / 2

    # conserve neck length to get the y offset
    yoff = np.sqrt(neck_len**2 - zoff**2)

    # add the virtual marker, calculate virtual point
    p1_rot = np.dot(Rth, p1)
    p1a_rot = p1_rot + np.array([0, -yoff, zoff])

    # rotate virtual marker back to other points
    pfe[i, 1] = np.dot(Rth.T, p1a_rot)


# %% Plot the points to make sure virtual is correct

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime)[::10]:
    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)
    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
                  color=bmap[2], scale_factor=20, resolution=64)
    mlab.plot3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                color=bmap[3], tube_radius=3)
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
                color=bmap[2], tube_radius=3)
mlab.orientation_axes()
fig.scene.isometric_view()


# %% Make a movie of the glide using points

mark_pal = sns.husl_palette(n_colors=nmark)

i = 5

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(170, 851))

lines = []
pts = []
for j in np.arange(nmark):
    _c = tuple(mark_pal[j])
    if j in [0, vent_idx, nmark - 1]:
        tube_radius = 8
        scale_factor = 30
    else:
        tube_radius = 6
        scale_factor = 20

    _ln = mlab.plot3d(pf[:i + 1, j, 0], pf[:i + 1, j, 1], pf[:i + 1, j, 2],
                      color=_c, tube_radius=tube_radius)
    _pt = mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                        color=_c, scale_factor=scale_factor)
    lines.append(_ln)
    pts.append(_pt)


view = (0.0, 0.0,  8786.4793907368839, np.array([  371.32172638, -3595.79905772,  4172.91651535]))
mlab.view(*view)


# %%

if False:
#if True:

    savename = '../Movies/s_413_91/top_view/413_91_top_{0:03d}.jpg'

    now = time.time()

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(170, 851))

    for i in np.arange(ntime):

        mlab.clf()

    #    _c = (0., 0., 0.)
    #    mlab.plot3d(pf[:i + 1, :, 0], pf[:i + 1, :, 1], pf[:i + 1, :, 2],
    #                color=_c, tube_radius=6)
    #    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
    #                  color=_c, scale_factor=20)

        for j in np.arange(nmark):
            _c = tuple(mark_pal[j])
            if j in [0, vent_idx, nmark - 1]:
                tube_radius = 8
                scale_factor = 30
            else:
                tube_radius = 6
                scale_factor = 20

            _ln = mlab.plot3d(pf[:i + 1, j, 0], pf[:i + 1, j, 1], pf[:i + 1, j, 2],
                              color=_c, tube_radius=tube_radius)
            _pt = mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                                color=_c, scale_factor=scale_factor)

        fig.scene.isometric_view()
        mlab.view(*view)
        mlab.draw()

        mlab.savefig(savename.format(i), size=(8*170, 8*808))  #2 hr
#        mlab.savefig(savename.format(i), size=(3*170, 3*850))  # 14.7 min
        # mlab.savefig(savename.format(i), size=(4*170, 4*850))  # 28 min


    print('Image save time: {0:.3f} sec'.format(time.time() - now))


# %%

savename = '../Movies/s_413_91/top_view/413_91_top_{0:03d}.jpg'

now = time.time()

mlab.savefig(savename.format(i), size=(8*170, 8*808))


print('Image save time: {0:.3f} sec'.format(time.time() - now))


# %%

# %% Make a movie of the glide using points

mark_pal = sns.husl_palette(n_colors=nmark)

i = -2

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(475, 851))

lines = []
pts = []
for j in np.arange(nmark):
    _c = tuple(mark_pal[j])
    if j in [0, vent_idx, nmark - 1]:
        tube_radius = 8
        scale_factor = 30
    else:
        tube_radius = 6
        scale_factor = 20

    _ln = mlab.plot3d(pf[:i + 1, j, 0], pf[:i + 1, j, 1], pf[:i + 1, j, 2],
                      color=_c, tube_radius=tube_radius)
    _pt = mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                        color=_c, scale_factor=scale_factor)
    lines.append(_ln)
    pts.append(_pt)


view = (0.0, 90.0, 15637.004688258743, np.array([  309.86672831, -2527.4053421 ,  4172.91651535]))
mlab.view(*view)


# %%

#if False:
if True:

    savename = '../Movies/s_413_91/side_view/413_91_side_{0:03d}.jpg'

    now = time.time()

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(475, 851))

    for i in np.arange(ntime):
#    for i in [0, 50, 100]:

        mlab.clf()

    #    _c = (0., 0., 0.)
    #    mlab.plot3d(pf[:i + 1, :, 0], pf[:i + 1, :, 1], pf[:i + 1, :, 2],
    #                color=_c, tube_radius=6)
    #    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
    #                  color=_c, scale_factor=20)

        for j in np.arange(nmark):
            _c = tuple(mark_pal[j])
            if j in [0, vent_idx, nmark - 1]:
                tube_radius = 8
                scale_factor = 30
            else:
                tube_radius = 6
                scale_factor = 20

            _ln = mlab.plot3d(pf[:i + 1, j, 0], pf[:i + 1, j, 1], pf[:i + 1, j, 2],
                              color=_c, tube_radius=tube_radius)
            _pt = mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                                color=_c, scale_factor=scale_factor)

        fig.scene.isometric_view()
        mlab.view(*view)
        mlab.draw()

        mlab.savefig(savename.format(i), size=(8*475, 8*808))  # 11.1 min
#        mlab.savefig(savename.format(i), size=(2*475, 2*850))  # 11.1 min


    print('Image save time: {0:.3f} sec'.format(time.time() - now))


# %% Mass and chord length distribution information

# mass distribution information
df_mass = pd.read_csv('../Data/snake_density.csv',
                      index_col=0)
df_mass.columns = ['s', 'rho']
s_rho, body_rho = df_mass['s'].values, df_mass['rho'].values
#df_mass.columns = ['s', 'body', 'tail']
#dbody_s, dbody_rho = df_mass['s'].values, df_mass['body'].values
#dtail_s, dtail_rho = df_mass['s'].values, df_mass['tail'].values

# chord length distribution
df_chord = pd.read_csv('../Data/snake_width.csv',
                       index_col=0)
df_chord.columns = ['s', 'chord']
s_chord, body_chord = df_chord['s'].values, df_chord['chord'].values
#df_chord.columns = ['s', 'body', 'tail']
#s_chord  = df_chord['s'].values
#body_chord, tail_chord = df_chord['body'].values, df_chord['tail'].values

mass_total_meas = 71  # g

SVL = marker_info['svl (mm)'].values[0]
VTL = marker_info['tail (mm)'].values[0]

SVL_cm = SVL / 10
VTL_cm = VTL / 10
SVL_m = SVL / 1000
VTL_m = VTL / 1000


# %%

nspl = 200
#mm_per_spl_bit = np.round(te.sum() / nspl).astype(np.int)
mm_per_spl_bit = te.sum() / nspl
bits_per_seg_float = te / mm_per_spl_bit
bits_per_seg = np.round(bits_per_seg_float).astype(np.int)

nbits = bits_per_seg.sum()
if nbits > nspl:
    bits_per_seg[-1] -= nbits - nspl
elif nbits < nspl:
    bits_per_seg[-1] += nspl - nbits


# %% Fit a spline and mass distribution to body and tail using extended points

nspl = 200

Ro_I = np.zeros((ntime, 3))
times2D = np.zeros((ntime, nspl))
t_coord = np.zeros((ntime, nspl))
s_coord = np.zeros((ntime, nspl))
vent_idx_spls = np.zeros(ntime, dtype=np.int)
R_I = np.zeros((ntime, nspl, 3))  # spl
dRds_I = np.zeros((ntime, nspl, 3))  # dspl
ddRds_I = np.zeros((ntime, nspl, 3))  # ddspl
dddRds_I = np.zeros((ntime, nspl, 3))
spl_ds = np.zeros((ntime, nspl))  # length of each segment in mm
mass_spl = np.zeros((ntime, nspl))  # in g
chord_spl = np.zeros((ntime, nspl))
#spl_len_totals = np.zeros(ntime)
spl_len_errors = np.zeros((ntime, nmark - 1))  # -1 because difference b/n

for i in np.arange(ntime):

    # fit spline (fpe is the arc length coordinate of the markers)
    out = asd.global_natural_spline(pfe[i], te, nspl)
    r, dr, ddr, dddr, ts, ss, seg_lens, lengths_total_e, idx_pts = out
    # r, dr, ddr, dddr, ts, ss, _, _, _ = out

    # exclude the virtual marker for error calculations
    lengths_total = np.zeros(nmark - 1)
    lengths_total[0] = lengths_total_e[0] + lengths_total_e[1]
    lengths_total[1:] = lengths_total_e[2:]

    # arc length coordinate differences (% along spline) of markers (no virtual marker)
     # %SVL of arc length coordinate
    spl_len_error = (dist_btn_markers - lengths_total) / SVL * 100

    # index into arc length coord where vent measurement is closest
    # based on segment parameters (maybe arc length would be better,
    # but it is making the tail too short)
    vent_idx_spl = idx_pts[vent_idx]

    # mass distribution
    mass_spl_i = np.interp(ts / SVL, s_rho, body_rho)
    mass_spl_i = mass_total_meas * mass_spl_i / mass_spl_i.sum()

    # chord length distribution
    chord_spl[i] = SVL * np.interp(ts / SVL, s_chord, body_chord)

    # center of mass
    Ro_I[i] = np.sum((r.T * mass_spl_i).T, axis=0) / mass_total_meas

    # store the spline and its derivative (for tangent angle calculations)
    R_I[i] = r
    dRds_I[i] = dr
    ddRds_I[i] = ddr
    dddRds_I[i] = dddr
    spl_ds[i] = seg_lens
    mass_spl[i] = mass_spl_i
    vent_idx_spls[i] = vent_idx_spl
    times2D[i] = times[i]
    t_coord[i] = ts
    s_coord[i] = ss
    spl_len_errors[i] = spl_len_error


# %% Velocity of the CoM, using optimal Butterworth filter smoothing

Ro_I_raw = Ro_I.copy()
dRo_I_raw, ddRo_I_raw = smoothing.findiff(Ro_I_raw, dt)

Ro_resid_array = np.zeros((ntime, 1, 3))
Ro_resid_array[:, 0] = Ro_I_raw

resid_Ro, fcs_Ro = smoothing.residual_butter(Ro_resid_array, fs, df=.5, fmin=1, fmax=35)
inter, fcopt, rsq, flinreg = smoothing.opt_cutoff(resid_Ro, fcs_Ro, rsq_cutoff=.95)


# %%

fig, ax = plot_residuals(Ro_resid_array, resid_Ro, fcs, inter, fcopt)
ax.set_ylim(0, .5)


# %% Try filtering each x, y, z and marker with a different fc

Ro_unique = smoothing.but_fcs(Ro_resid_array, fs, fcopt)

Ro_I = Ro_unique['p'].reshape(-1, 3)
dRo_I = Ro_unique['v'].reshape(-1, 3)
ddRo_I = Ro_unique['a'].reshape(-1, 3)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(times, Ro_I)
ax1.plot(times, Ro_I_raw, '--')
ax2.plot(times, dRo_I)
ax2.plot(times, dRo_I_raw, '--')
ax3.plot(times, ddRo_I)
ax3.plot(times, ddRo_I_raw, '--')
sns.despine()
fig.set_tight_layout(True)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(times, Ro_I - Ro_I_raw)
ax2.plot(times, dRo_I - dRo_I_raw)
ax3.plot(times, ddRo_I - ddRo_I_raw)
sns.despine()
fig.set_tight_layout(True)


# %% Verify that we select out idx_points correctly

tcoord = t_coord[0]

idx_markers = np.zeros(nmark - 1, dtype=np.int)
for jj in np.arange(nmark - 1):
    if jj > 0:
        idx_markers[jj] = np.argmin(np.abs(tcoord - dist_btn_markers[:jj + 1].sum()))
    else:
        idx_markers[jj] = np.argmin(np.abs(tcoord - dist_btn_markers[jj]))


# %% Spline length errors

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
for j in np.arange(nmark - 1):
    jj = j + 1
    l1 = r'm$_\mathrm{' + str(jj + 1) + '}$'
    l2 = r'm$_\mathrm{' + str(jj) + '}$'
    #label = l1 + u' \u2013 ' + l2
    label = l2 + ' to ' + l1
    ax.plot(times, spl_len_errors[:, j], c=colors_mark[j], label=label)
ax.legend(loc='upper right', ncol=3, frameon=True, framealpha=.5,
          fontsize='small')
ax.set_xlabel('time (s)')
ax.set_ylabel('spline fit errors, %SVL')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig('../Figures/spline_errors.pdf', transparent=True,
#            bbox_inches='tight')


# %% Cumulative spline length errors (maybe this is the metric we report)

err_mean = spl_len_errors[:, :vent_idx].sum(axis=1).mean()
err_std = spl_len_errors[:, :vent_idx].sum(axis=1).std()

figure(); plot(times, spl_len_errors[:, :vent_idx].sum(axis=1))
axhline(err_mean, color='k')
axhline(err_mean - err_std, color='k')
axhline(err_mean + err_std, color='k')

figure(); plot(times, spl_len_errors[:, vent_idx:].sum(axis=1))
axhline(spl_len_errors[:, vent_idx:].sum(axis=1).mean(), color='k')


# %% Plot the com and spline body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(com[:, 0], com[:, 1], com[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()


# %% Store the values to reset

#R_I = spl.copy()
#dR_I = dspl.copy()
#ddR_I = ddspl.copy()
#dddR_I = dddspl.copy()

pf_I = pf.copy()
vf_I = vf.copy()
af_I = af.copy()
pfe_I = pfe.copy()

R_Ic = R_I.copy()
pf_Ic = pf_I.copy()
pfe_Ic = pfe_I.copy()

for i in np.arange(ntime):
    R_Ic[i] = R_I[i] - Ro_I[i]
    pf_Ic[i] = pf_I[i] - Ro_I[i]
    pfe_Ic[i] = pfe_Ic[i] - Ro_I[i]


# %% Calculate velocity of spline segments

# marker velocies and accelerations relative to CoM
vf_Ic = np.zeros_like(pf_I)
af_Ic = np.zeros_like(pf_I)
for j in np.arange(nmark):
    vv, aa = smoothing.findiff(pf_Ic[:, j], dt)
    vf_Ic[:, j] = vv
    af_Ic[:, j] = aa

# spline velocity and accelerations relative to CoM
dR_I = np.zeros((ntime, nspl, 3))
ddR_I = np.zeros((ntime, nspl, 3))
dR_Ic = np.zeros((ntime, nspl, 3))
ddR_Ic = np.zeros((ntime, nspl, 3))
for j in np.arange(nspl):
    vv, aa = smoothing.findiff(R_I[:, j], dt)
    dR_I[:, j] = vv
    ddR_I[:, j] = aa

    vv, aa = smoothing.findiff(R_Ic[:, j], dt)
    dR_Ic[:, j] = vv
    ddR_Ic[:, j] = aa


# %% Plot the CoM centered body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_Ic[i, :, 0], R_Ic[i, :, 1], R_Ic[i, :, 2],
                color=bmap[2], tube_radius=3)
#    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
#                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()


# %%

dxdy = np.diff(Ro_I, axis=0)
# dxdy_t = Ro_I[1:] - Ro_I[:-1]

mt = -np.arctan2(dxdy[:, 0], dxdy[:, 1])
mtt = -np.arctan2(dRo_I[:, 0], dRo_I[:, 1])
mt = mtt

R_I2S = np.zeros((ntime, 3, 3))
for i in np.arange(ntime - 1):
    mu = mt[i]
    Rmu = np.array([[np.cos(mu), np.sin(mu), 0],
                    [-np.sin(mu),  np.cos(mu), 0],
                    [0, 0, 1]])
    R_I2S[i] = Rmu
R_I2S[-1] = R_I2S[-2]

pf_St = np.zeros_like(pf_I)
pf_Sct = np.zeros_like(pf_Ic)
#Ro_St = R_I.copy()
pf_Sold = np.zeros_like(pf_S)
R_St = np.zeros_like(R_I)
R_Sct = np.zeros_like(R_Ic)
for i in np.arange(ntime):
#    Ro_St[i] = np.dot(R_I2S[i], Ro_I[i].T).T
    pf_Sct[i] = np.dot(R_I2S[i], pf_Ic[i].T).T
    R_Sct[i] = np.dot(R_I2S[i], R_Ic[i].T).T
    pf_St[i] = pf_Sct[i] + Ro_S[i]
    R_St[i] = R_Sct[i] + Ro_S[i]
    pf_Sold[i] = pf_Sc[i] - Ro_S[0] + Ro_S[i]

fig, ax = plt.subplots()
#ax.plot(times[:-1], np.rad2deg(mt))
ax.plot(times, np.rad2deg(mt))
ax.plot(times, np.rad2deg(mus).cumsum())
sns.despine()
fig.set_tight_layout(True)

# plot the com trajectory in the YX plane
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
#ax2.plot(Ro_St[:, 0], Ro_St[:, 1], 'b')
ax2.plot(Ro_S[:, 0], Ro_S[:, 1], 'b')
for i in np.arange(ntime)[1::20]:
    ax1.plot(pf_I[i, :, 0], pf_I[i, :, 1], 'ok', ms=3)
    ax2.plot(pf_St[i, :, 0], pf_St[i, :, 1], 'ok', ms=3)
    ax1.plot(R_I[i, :, 0], R_I[i, :, 1], c='g', lw=1)
    ax2.plot(R_St[i, :, 0], R_St[i, :, 1], c='g', lw=1)
    ax2.plot(R_S[i, :, 0], R_S[i, :, 1], c='b', lw=1)
    ax2.plot(pf_Sold[i, :, 0], pf_Sold[i, :, 1], 'or', ms=3)
#    ax2.plot(pf_Sc[i, :, 0], pf_Sc[i, :, 1], 'or', ms=3)
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
#ax1.axis('off')
#ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %% Glide angle

gamma = -np.arctan2(dRo_I[:, 2], dRo_I[:, 1])

# not sure what this is...
sss = np.unwrap(np.arctan2(dRo_I[:, 0], dRo_I[:, 2]))

fig, ax = plt.subplots()
ax.plot(times, np.rad2deg(gamma))
#ax.plot(times, np.rad2deg(sss))
#ax.set_ylim(0, 90)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
#ax.plot(times, np.rad2deg(gamma))
ax.plot(times, np.rad2deg(sss))
#ax.set_ylim(0, 90)
sns.despine()
fig.set_tight_layout(True)



# %%

# plot the com trajectory in the YX plane
fig, ax = plt.subplots()
ax.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
#ax2.plot(Ro_St[:, 0], Ro_St[:, 1], 'b')
ax.plot(Ro_S[:, 0], Ro_S[:, 1], 'b')
for i in np.arange(ntime)[1::20]:
    ax.plot(pf_I[i, :, 0], pf_I[i, :, 1], 'ok', ms=3)
    ax.plot(pfe_I[i, :, 0], pfe_I[i, :, 1], 'ok', ms=2)
#    ax.plot(pf_St[i, :, 0], pf_St[i, :, 1], 'ok', ms=3)
    ax.plot(R_I[i, :, 0], R_I[i, :, 1], c='g', lw=1)
#    ax.plot(R_St[i, :, 0], R_St[i, :, 1], c='g', lw=1)
    ax.plot(R_S[i, :, 0], R_S[i, :, 1], c='b', lw=1)
#    ax.plot(pf_Sold[i, :, 0], pf_Sold[i, :, 1], 'or', ms=3)
    ax.plot(pf_S[i, :, 0], pf_S[i, :, 1], 'or', ms=3)
    ax.plot(pfe_S[i, :, 0], pfe_S[i, :, 1], 'ok', ms=2)
#plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
ax.set_aspect('equal', adjustable='box')
#ax1.axis('off')
#ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)



# %%

# plot the com trajectory in the YX plane
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(Ro_I[:, 1], Ro_I[:, 2], 'r')
#ax2.plot(Ro_St[:, 0], Ro_St[:, 1], 'b')
ax2.plot(Ro_S[:, 1], Ro_S[:, 2], 'b')
for i in np.arange(ntime)[1::20]:
    ax1.plot(pf_I[i, :, 1], pf_I[i, :, 2], 'ok', ms=3)
#    ax2.plot(pf_St[i, :, 1], pf_St[i, :, 2], 'ok', ms=3)
    ax1.plot(R_I[i, :, 1], R_I[i, :, 2], c='g', lw=1)
#    ax2.plot(R_St[i, :, 1], R_St[i, :, 2], c='g', lw=1)
    ax2.plot(pf_S[i, :, 1], pf_S[i, :, 2], 'ok', ms=3)
    ax2.plot(R_S[i, :, 1], R_S[i, :, 2], c='g', lw=1)
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
ax1.axis('off')
ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %% Check that the straightened glide length is the same

glide_path_len_I = cumtrapz(np.sqrt(np.sum(dRo_I**2, axis=1)), times, initial=0) / 1000
glide_path_len_S = cumtrapz(np.sqrt(np.sum(dRo_S**2, axis=1)), times, initial=0) / 1000

fig, ax = plt.subplots()
ax.plot(times, glide_path_len_I)
ax.plot(times, glide_path_len_S)
ax.set_xlabel('time (s)')
ax.set_ylabel('cumulative distance along glide path (m)')
sns.despine()
fig.set_tight_layout(True)


# %%

C_I2S_t = R_I2S.copy()
for i in np.arange(1, ntime):
    C_I2S_t[i] = np.dot(R_I2S[i], C_I2S_t[i - 1])


# %% Straighten the trajectory so it is along Yhat direction

nmark = pf_I.shape[1]

# shift to 'com'. Note, this is not strictly needed
Ro0 = Ro_I[0]
Ro_S = Ro_I - Ro0  # straightened CoM position
dRo_S = dRo_I.copy()
ddRo_S = ddRo_I.copy()
R_S = R_I - Ro0
pfe_S = pfe_I - Ro0

# apply to original markers
pf_S = pf_I - Ro0
vf_S = vf_I.copy()
af_S = af_I.copy()

# spline derivatives in straightened frame
dRds_S = dRds_I.copy()
ddRds_S = ddRds_I.copy()
dddRds_S = dddRds_I.copy()

# velocity and accelerations of the spline
dR_S = dR_I.copy()
ddR_S = ddR_I.copy()
dR_Sc = dR_Ic.copy()
ddR_Sc = ddR_Ic.copy()

# rotation angles to straighten trajectory
mus = np.zeros(ntime)
Rmus = np.zeros((ntime, 3, 3))
for i in np.arange(ntime):
    Rmus[i] = np.eye(3)

Ro_S_plot = np.zeros((ntime, ntime, 3))

# iterate through the points a find the successive roations
for i in np.arange(ntime):
    uu = Ro_S[i]
    # mu = np.arctan2(uu[0], uu[1])  # tan^-1(px / py)
    # Rmu = np.array([[np.cos(mu), -np.sin(mu), 0],
    #                 [np.sin(mu),  np.cos(mu), 0],
    #                 [0, 0, 1]])  #NOTE: Rmu = R3(yaw).T

    mu = np.arctan2(-uu[0], uu[1])  # tan^-1(-px / py)
    Rmu = np.array([[np.cos(mu), np.sin(mu), 0],
                    [-np.sin(mu),  np.cos(mu), 0],
                    [0, 0, 1]])

    if i == 1:
        start = 0
    else:
        start = i

    mus[start] = mu
    Rmus[start] = Rmu

    # apply the rotation to each point along the spline
    for ii in np.arange(start, ntime):
        Ro_S[ii] = np.dot(Rmu, Ro_S[ii].T).T  # com
        Ro_S_plot[i, ii] = Ro_S[ii]
        R_S[ii] = np.dot(Rmu, R_S[ii].T).T  # body
        dRo_S[ii] = np.dot(Rmu, dRo_S[ii].T).T  # com velocity
        ddRo_S[ii] = np.dot(Rmu, ddRo_S[ii].T).T  # com acceleration
        pfe_S[ii] = np.dot(Rmu, pfe_S[ii].T).T
        dRds_S[ii] = np.dot(Rmu, dRds_S[ii].T).T
        ddRds_S[ii] = np.dot(Rmu, ddRds_S[ii].T).T
        dddRds_S[ii] = np.dot(Rmu, dddRds_S[ii].T).T
        pf_S[ii] = np.dot(Rmu, pf_S[ii].T).T
        vf_S[ii] = np.dot(Rmu, vf_S[ii].T).T
        af_S[ii] = np.dot(Rmu, af_S[ii].T).T
        dR_S[ii] = np.dot(Rmu, dR_S[ii].T).T
        ddR_S[ii] = np.dot(Rmu, ddR_S[ii].T).T
        dR_Sc[ii] = np.dot(Rmu, dR_Sc[ii].T).T
        ddR_Sc[ii] = np.dot(Rmu, ddR_Sc[ii].T).T

# com centered spline and points
R_Sc = np.zeros((ntime, nspl, 3))
pf_Sc = np.zeros((ntime, nmark, 3))
pfe_Sc = np.zeros((ntime, nmark + 1, 3))
for i in np.arange(ntime):
    R_Sc[i] = R_S[i] - Ro_S[i]
    pf_Sc[i] = pf_S[i] - Ro_S[i]
    pfe_Sc[i] = pfe_S[i] - Ro_S[i]

# move the trajectory back to the inital com position, except com_X = 0
#Ro0_move = np.r_[0, Ro0[1:]]
Ro0_move = Ro0
Ro_S = Ro_S + Ro0_move
R_S = R_S + Ro0_move
pf_Sc = pf_Sc + Ro0_move
pfe_S = pfe_S + Ro0_move

for i in np.arange(ntime):
    Ro_S_plot[i] += Ro0_move


# %%

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
sns.despine()
plt.pause(1)
for i in np.arange(ntime):
    ax.cla()
    ax.plot(Ro_S_plot[1, :, 0], Ro_S_plot[1, :, 1], 'k')
    ax.plot(Ro_S_plot[i, :, 0], Ro_S_plot[i, :, 1])
    ax.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
    ax.axvline(Ro_I[0, 0], color='gray', lw=1)
    fig.canvas.draw()
    plt.pause(.01)


# %%

# plot the com trajectory in the YX plane
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
ax2.plot(Ro_S[:, 0], Ro_S[:, 1], 'b')
for i in np.arange(ntime)[1::20]:
    ax1.plot(R_I[i, :, 0], R_I[i, :, 1], c='gray', lw=1)
    ax2.plot(R_S[i, :, 0], R_S[i, :, 1], c='gray', lw=1)
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
#ax1.axis('off')
#ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %%

# plot the com trajectory in the YX plane
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(Ro_I[:, 1], Ro_I[:, 2], 'r')
ax2.plot(Ro_S[:, 1], Ro_S[:, 2], 'b')
for i in np.arange(ntime)[1::20]:
    ax1.plot(R_I[i, :, 1], R_I[i, :, 2], c='gray', lw=1)
    ax2.plot(R_S[i, :, 1], R_S[i, :, 2], c='gray', lw=1)
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
ax1.axis('off')
ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %% Fit a more robust plane to the snake

R_Sc = R_Sct.copy()

nhat = np.zeros((ntime, 3))
for i in np.arange(ntime):
#    x, y, z = R_Ic[i].T
    x, y, z = R_Sc[i].T

    # z error to calculate mid-line plane
#    Mm = np.array([[(x**2).sum(), (x * y).sum(), x.sum()],
#                   [(x * y).sum(), (y**2).sum(), y.sum()],
#                   [x.sum(), y.sum(), len(x)]])

    # x error to calculate roll?
    Mm = np.array([[(y**2).sum(), (y * z).sum(), y.sum()],
                   [(y * z).sum(), (z**2).sum(), z.sum()],
                   [y.sum(), z.sum(), len(x)]])
    bm = np.array([(y * x).sum(), (z * x).sum(), x.sum()])
#    nhat[i] = np.linalg.solve(Mm, bm)
    nhat[i] = np.linalg.lstsq(Mm, bm)[0]

nhat = (nhat.T / np.linalg.norm(nhat, axis=1)).T
idx_flip = np.where(nhat[:, 2] < 0)[0]
nhat[idx_flip] = nhat[idx_flip] * -1


# %%

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(R_Sc[:, :, 0].T, R_Sc[:, :, 1].T, c='gray', alpha=.15)
ax.plot(R_Sc[0, :, 0], R_Sc[0, :, 1])  # ti = 0
ax.plot(R_Sc[1, :, 0], R_Sc[1, :, 1])  # ti = 1
ax.plot(R_Sc[2, :, 0], R_Sc[2, :, 1])  # ti = 1
#ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(R_Ic[:, :, 0].T, R_Ic[:, :, 1].T, c='gray', alpha=.15)
ax.plot(R_Ic[0, :, 0], R_Ic[0, :, 1])  # ti = 0
ax.plot(R_Ic[1, :, 0], R_Ic[1, :, 1])  # ti = 1
ax.plot(R_Ic[2, :, 0], R_Ic[2, :, 1])  # ti = 1
#ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %%

Ro_S_test = np.zeros(Ro_S.shape)

C_I2S = Rmus.copy()
for i in np.arange(1, ntime):
    C_I2S[i] = np.dot(Rmus[i], C_I2S[i - 1])

Ro_I_test = Ro_I - Ro0
R_Sc_test = np.zeros_like(R_S)
for i in np.arange(ntime):
    Ro_S_test[i] = np.dot(C_I2S[i], Ro_I_test[i].T).T
    R_Sc_test[i] = np.dot(C_I2S[i], R_Ic[i].T).T
Ro_S_test += Ro0_move

#com_cont = com_cont + com0_move
dcom = Ro_S_test[0] - Ro_S_test[-1]
dcom_ang = np.rad2deg(np.arctan(dcom[0] / dcom[1]))

# plot the com trajectory in the YX plane
fig, ax = plt.subplots()
#ax.plot(com[:, 1], com[:, 0], '-')
ax.plot(Ro_S[:, 1], Ro_S[:, 0], '-', label='Ro_S')
ax.plot(Ro_S_test[:, 1], Ro_S_test[:, 0], '-', label='Ro_S_test')
ax.legend(loc='best')
#ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)

## plot the com trajectory in the YZ plane
#fig, ax = plt.subplots()
##ax.plot(com[:, 1], com[:, 2], '-')
#ax.plot(com_cont[:, 1], com_cont[:, 2], '-')
#ax.set_aspect('equal', adjustable='box')
#sns.despine()
#fig.set_tight_layout(True)

# make sure our concatinated rotation matrices and the iterative method
# are the same
assert np.allclose(Ro_S, Ro_S_test)
assert np.allclose(R_Sc_test, R_Sc)


# %%

C_I2S_test = Rmus.copy()
for i in np.arange(1, ntime):
    C_I2S_test[i] = np.dot(Rmus[i].T, C_I2S_test[i - 1])

# try transposing the rotation matrix
R_S_test = np.zeros((ntime, nspl, 3))
for i in np.arange(ntime):
    R_S_test[i] = np.dot(C_I2S_test[i], R_Ic[i].T).T
    R_S_test[i] += Ro_S[i]
#    R_S_test[i] += Ro_I[i]


# %%

# plot the com trajectory in the YX plane
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
ax2.plot(Ro_S[:, 0], Ro_S[:, 1], 'b')
#ax2.plot(Ro_I[:, 0], Ro_I[:, 1], 'b')
for i in np.arange(ntime)[::20]:
    ax1.plot(R_I[i, :, 0], R_I[i, :, 1], c='gray', lw=1)
#    ax2.plot(R_S[i, :, 0], R_S[i, :, 1], c='gray', lw=1)
    ax2.plot(R_S_test[i, :, 0], R_S_test[i, :, 1], c='gray', lw=1)
#ax.plot(com_cont[:, 1], com_cont[:, 0], '-')
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
ax1.axis('off')
ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
for i in np.arange(ntime)[::20]:
    ax.plot(R_I[i, :, 0], R_I[i, :, 1], c='b', lw=1)
#    ax2.plot(R_S[i, :, 0], R_S[i, :, 1], c='gray', lw=1)
    ax.plot(R_S_test[i, :, 0], R_S_test[i, :, 1], c='r', lw=1)
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %% Plot the heading angle subsequenct rotations

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
#ax.plot(times, np.rad2deg(mus), 'o-')
ax.plot(times, np.rad2deg(mus), 'o-')
#ax.plot(times[2:], np.rad2deg(mus[1:]), 'o-')
ax.margins(.03)
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
#ax.plot(times, np.rad2deg(mus), 'o-')
ax.plot(times, np.rad2deg(mus).cumsum(), 'o-')
ax.plot(times, np.rad2deg(yaw), 'r-')
#ax.plot(times[2:], np.rad2deg(mus[1:]), 'o-')
ax.margins(.03)
sns.despine()
fig.set_tight_layout(True)


# %%

# plot the com trajectory in the YZ plane
fig, ax = plt.subplots()
ax.plot(Ro_S[:, 1], Ro_S[:, 2], 'o-')
#ax.plot(com_cont[:, 1], com_cont[:, 2], '-')
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %%

# plot the com trajectory in the YX plane
fig, ax = plt.subplots()
ax.plot(Ro_I[:, 0], Ro_I[:, 1], 'r')
ax.plot(Ro_S[:, 0], Ro_S[:, 1], 'b')
for i in np.arange(ntime)[::10]:
    ax.plot(R_I[i, :, 0], R_I[i, :, 1], c='gray', lw=1)
#ax.plot(com_cont[:, 1], com_cont[:, 0], '-')
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)




# %%

# plot the com trajectory  velocity in the YX plane
fig, ax = plt.subplots()
ax.plot(dRo_I[:, 0], dRo_I[:, 1], 'r')
ax.plot(dRo_S[:, 0], dRo_S[:, 1], 'o-')
#ax.plot(com_cont[:, 1], com_cont[:, 0], '-')
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %% Verify com and points were rotated correctly

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro_S[:, 0], Ro_S[:, 1], Ro_S[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_S[i, :, 0], R_S[i, :, 1], R_S[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_S[i, :, 0], pfe_S[i, :, 1], pfe_S[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()


# %% Verify com and points were rotated correctly

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro_I[:, 0], Ro_I[:, 1], Ro_I[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_I[i, :, 0], pfe_I[i, :, 1], pfe_I[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()


# %% Verify com and points were rotated correctly

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro_I[:, 0], Ro_I[:, 1], Ro_I[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_I[i, :, 0], pfe_I[i, :, 1], pfe_I[i, :, 2],

                  color=bmap[3], scale_factor=20, resolution=64)
mlab.plot3d(Ro_S[:, 0], Ro_S[:, 1], Ro_S[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_S[i, :, 0], R_S[i, :, 1], R_S[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_S[i, :, 0], pfe_S[i, :, 1], pfe_S[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()


# %% Tangent, chord, and backbone coordinate system

Tdir_S = np.zeros((ntime, nspl, 3))
Cdir_S = np.zeros((ntime, nspl, 3))
Bdir_S = np.zeros((ntime, nspl, 3))

Cb_S = np.zeros((ntime, nspl, 3, 3))

a_angs = np.zeros((ntime, nspl))
b_angs = np.zeros((ntime, nspl))

kap_signed = np.zeros((ntime, nspl))
kap_unsigned =np.zeros((ntime, nspl))
tau = np.zeros((ntime, nspl))

for i in np.arange(ntime):

    # use rotated spline derivative from straightening procedure
    dr = dRds_S[i]

    # TNB frame
    # https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
    tdir = (dr.T / np.linalg.norm(dr, axis=1)).T
    tdir0 = tdir[0]  # this will be point back in -Yhat direction

    xhat = np.r_[1, 0, 0]  # cdir0 should nominally be in xhat direciton
    cdir0 = xhat - tdir0 * np.dot(tdir0, xhat)
    cdir0 = cdir0 / np.linalg.norm(cdir0)
    bdir0 = np.cross(tdir0, cdir0)

    j = 0
    Tdir_S[i] = tdir
    Cdir_S[i, j] = cdir0
    Bdir_S[i, j] = bdir0

    # rotation matrix for foil shape defined in (y - z plane, with x = 0)
    Cr_foil = np.zeros((3, 3))
    Cr_foil[:, 0] = tdir0
    Cr_foil[:, 1] = cdir0
    Cr_foil[:, 2] = bdir0
    Cb_S[i, j] = Cr_foil

    # now iterate along the body, finding successive rotations
    # Bloomenthal (1990)
    for j in np.arange(1, nspl):
        T0 = Tdir_S[i, j - 1]  # tangent direction at head
        T1 = Tdir_S[i, j]
        T0 = T0 / np.linalg.norm(T0)
        T1 = T1 / np.linalg.norm(T1)
        A = np.cross(T0, T1)
        A = A / np.linalg.norm(A)  # why have to do this?

        # components of rotation matrix
        Ax, Ay, Az = A  # axis of rotation
        sqx, sqy, sqz = A**2
        cos = np.dot(T0, T1)
        cos1 = 1 - cos
        xycos1 = Ax * Ay * cos1
        yzcos1 = Ay * Az * cos1  # check on Az
        zxcos1 = Ax * Az * cos1
        sin = np.sqrt(1 - cos**2)
        xsin, ysin, zsin =  A * sin

        # make the rotation matrix
        Cr = np.array([[sqx + (1 - sqx) * cos, xycos1 + zsin, zxcos1 - ysin],
                       [xycos1 - zsin, sqy + (1 - sqy) * cos, yzcos1 + xsin],
                       [zxcos1 + ysin, yzcos1 - xsin, sqz + (1 - sqz) * cos]])

        # not 100% on why need to transpose (active vs. passive rotation?)
        # https://en.wikipedia.org/wiki/Active_and_passive_transformation
        Cr = Cr.T

        # store rotation matrix for the foil
        Cb_S[i, j] = np.dot(Cr, Cb_S[i, j - 1])

        C0 = Cdir_S[i, j - 1]
        B0 = Bdir_S[i, j - 1]
        C1 = np.dot(Cr, C0)
        B1 = np.dot(Cr, B0)
        Cdir_S[i, j] = C1
        Bdir_S[i, j] = B1

        # bending and twisting angles of the snake/frame
        T0_CT1 = T0 - np.dot(T0, B1) * B1
        T0_CT1 = T0_CT1 / np.linalg.norm(T0_CT1)
        alpha = np.arccos(np.dot(T1, T0_CT1))
        alpha_sign = np.sign(np.dot(C1, T0_CT1))
        alpha = alpha_sign * alpha

        C0_BC1 = C0 - np.dot(C0, T1) * T1
        C0_BC1 = C0_BC1 / np.linalg.norm(C0_BC1)
        beta = np.arccos(np.dot(C1, C0_BC1))
        beta_sign = np.sign(np.dot(B1, C0_BC1))
        beta = beta_sign * beta

        a_angs[i, j] = alpha
        b_angs[i, j] = beta

        # a_angs is the lateral bending
        # b_angs is vertibral twist
#        a_angs[i, j] = np.arccos(np.dot(T0_BT1, T1))  # bending angle about B1
#        b_angs[i, j] = np.arccos(np.dot(C0_BC1, C1))  # twisting angle about T1
#        a_cos = np.dot(T0_BT1, T1)
#        a_sin = np.linalg.norm(np.cross(T0_BT1, T1))
#        b_cos = np.dot(C0_BC1, C1)
#        b_sin = np.linalg.norm(np.cross(C0_BC1, C1))
#
#        a_angs[i, j] = np.arctan2(a_sin, a_cos)
#        b_angs[i, j] = np.arctan2(b_sin, b_cos)
#        a_angs[i, j] = np.arctan(a_sin / a_cos)
#        b_angs[i, j] = np.arctan(b_sin / b_cos)

#        # bending and twisting angles of the snake/frame
#        T0_BT1 = T0 - np.dot(T0, C1) * C1
#        C0_BC1 = C0 - np.dot(C0, T1) * T1
#        T0_BT1 = T0_BT1 / np.linalg.norm(T0_BT1)
#        C0_BC1 = C0_BC1 / np.linalg.norm(C0_BC1)
##        a_angs[i, j] = np.arccos(np.dot(T0_BT1, T1))  # bending angle about B1
##        b_angs[i, j] = np.arccos(np.dot(C0_BC1, C1))  # twisting angle about T1
#        a_cos = np.dot(T0_BT1, T1)
#        a_sin = np.linalg.norm(np.cross(T0_BT1, T1))
#        b_cos = np.dot(C0_BC1, C1)
#        b_sin = np.linalg.norm(np.cross(C0_BC1, C1))
#
#        a_angs[i, j] = np.arctan2(a_sin, a_cos)
#        b_angs[i, j] = np.arctan2(b_sin, b_cos)
##        a_angs[i, j] = np.arctan(a_sin / a_cos)
##        b_angs[i, j] = np.arctan(b_sin / b_cos)


    # calculate signed and unsigned curvature and torsion
    dx, dy, dz = dRds_S[i].T
    ddx, ddy, ddz = ddRds_S[i].T
    dddx, dddy, dddz = dddRds_S[i].T
    for j in np.arange(nspl):
        k1 = ddz[j] * dy[j] - ddy[j] * dz[j]
        k2 = ddx[j] * dz[j] - ddz[j] * dx[j]
        k3 = ddy[j] * dx[j] - ddx[j] * dy[j]
        kn = (dx[j]**2 + dy[j]**2 + dz[j]**2)**1.5

        t1 = dddx[j] * k1
        t2 = dddy[j] * k2
        t3 = dddz[j] * k3
        tn = k1**2 + k2**2 + k3**2

        kap_signed[i, j] = (k1 + k2 + k3) / kn
        kap_unsigned[i, j] = np.sqrt(k1**2 + k2**2 + k3**2) / kn
        tau[i, j] = (t1 + t2 + t3) / tn


# %% Rotate the coordinate system to the interial frame

Tdir_I = np.zeros_like(Tdir_S)
Cdir_I = np.zeros_like(Cdir_S)
Bdir_I = np.zeros_like(Bdir_S)
Cb_I = np.zeros_like(Cb_S)

for i in np.arange(ntime):
    Tdir_I[i] = np.dot(C_I2S[i].T, Tdir_S[i].T).T
    Cdir_I[i] = np.dot(C_I2S[i].T, Cdir_S[i].T).T
    Bdir_I[i] = np.dot(C_I2S[i].T, Bdir_S[i].T).T

    # torsion minimizing frame, but in inertial frame
    for j in np.arange(nspl):
        Cb_I[i, j] = np.dot(C_I2S[i].T, Cb_S[i, j])


# %%



# %% Plot the bending and twisting angles

i = 0

fig, ax = plt.subplots()

for i in np.arange(20):
#    ax.plot(t_coord[i], np.rad2deg(a_angs[i]))
    ax.plot(t_coord[i], np.rad2deg(b_angs[i]))


# %%

i = 0

fig, ax = plt.subplots()

for i in np.arange(ntime):
    ax.plot(t_coord[i], np.rad2deg(a_angs[i]))
#    ax.plot(t_coord[i], np.rad2deg(b_angs[i]))


# %% 3D surface of the bending angle

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#skip = 5
#for i in np.arange(ntime)[::skip]:
#    mlab.plot3d(t_coord[i], np.rad2deg(a_angs[i]), times[i] * np.ones(nspl))

b_angs[np.isnan(b_angs)] = 0

mlab.surf(.001 * t_coord, times2D, .1 * np.rad2deg(a_angs))#, colormap='RdBu')
#mlab.surf(.001 * t_coord, times2D, 1 * np.rad2deg(b_angs))#, colormap='YlGn')


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord, np.rad2deg(a_angs), vmin=-5, vmax=5,
                    cmap=plt.cm.coolwarm)
fig.colorbar(cax, ax=ax)
sns.despine(ax=ax)


fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord, np.rad2deg(b_angs), vmin=-.25, vmax=.25,
                    cmap=plt.cm.coolwarm)
fig.colorbar(cax, ax=ax)
sns.despine(ax=ax)


# %% Verify the coordinate system looks correct

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(pfe_S[:, vent_idx + 1, 0], pfe_S[:, vent_idx + 1, 1],
            pfe_S[:, vent_idx + 1, 2], color=bmap[0], tube_radius=2)

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_S[i, :, 0], R_S[i, :, 1], R_S[i, :, 2],
                color=bmap[3], tube_radius=3)

sk = 1
for i in np.arange(ntime)[::20]:
    # that
#    mlab.quiver3d(spl[i, ::sk, 0], spl[i, ::sk, 1], spl[i, ::sk, 2],
#              Tdir[i, ::sk, 0], Tdir[i, ::sk, 1], Tdir[i, ::sk, 2],
#              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # chat
    mlab.quiver3d(R_S[i, ::sk, 0], R_S[i, ::sk, 1], R_S[i, ::sk, 2],
              Cdir_S[i, ::sk, 0], Cdir_S[i, ::sk, 1], Cdir_S[i, ::sk, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(R_S[i, ::sk, 0], R_S[i, ::sk, 1], R_S[i, ::sk, 2],
              Bdir_S[i, ::sk, 0], Bdir_S[i, ::sk, 1], Bdir_S[i, ::sk, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

mlab.orientation_axes()
fig.scene.isometric_view()


# %% Verify the coordinate system looks correct Interial and Straightened

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# straightened frame
for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_S[i, :, 0], R_S[i, :, 1], R_S[i, :, 2],
                color=bmap[3], tube_radius=3)

sk = 1
for i in np.arange(ntime)[::20]:
    # that
    mlab.quiver3d(R_S[i, ::sk, 0], R_S[i, ::sk, 1], R_S[i, ::sk, 2],
              Tdir_S[i, ::sk, 0], Tdir_S[i, ::sk, 1], Tdir_S[i, ::sk, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # chat
    mlab.quiver3d(R_S[i, ::sk, 0], R_S[i, ::sk, 1], R_S[i, ::sk, 2],
              Cdir_S[i, ::sk, 0], Cdir_S[i, ::sk, 1], Cdir_S[i, ::sk, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(R_S[i, ::sk, 0], R_S[i, ::sk, 1], R_S[i, ::sk, 2],
              Bdir_S[i, ::sk, 0], Bdir_S[i, ::sk, 1], Bdir_S[i, ::sk, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

# inertial frame
for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
                color=bmap[0], tube_radius=3)

sk = 1
for i in np.arange(ntime)[::20]:
    # that
    mlab.quiver3d(R_I[i, ::sk, 0], R_I[i, ::sk, 1], R_I[i, ::sk, 2],
              Tdir_I[i, ::sk, 0], Tdir_I[i, ::sk, 1], Tdir_I[i, ::sk, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # chat
    mlab.quiver3d(R_I[i, ::sk, 0], R_I[i, ::sk, 1], R_I[i, ::sk, 2],
              Cdir_I[i, ::sk, 0], Cdir_I[i, ::sk, 1], Cdir_I[i, ::sk, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(R_I[i, ::sk, 0], R_I[i, ::sk, 1], R_I[i, ::sk, 2],
              Bdir_I[i, ::sk, 0], Bdir_I[i, ::sk, 1], Bdir_I[i, ::sk, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

mlab.orientation_axes()
fig.scene.isometric_view()


# %%




# %% Orientation of the airfoil (for plotting and movies)

# in straightened frame
foils_Sc, foil_color = asd.apply_airfoil_shape(R_Sc, chord_spl, Cb_S)

# in inertial frame
foils_Ic, foil_color = asd.apply_airfoil_shape(R_Ic, chord_spl, Cb_I)

# verify inertial frame and rotated from straight frame are the same
foils_Ic_test = np.zeros_like(foils_Sc)
for i in np.arange(ntime):
    for j in np.arange(foils_Sc.shape[1]):
        foils_Ic_test[i, j] = np.dot(C_I2S[i].T, foils_Sc[i, j].T).T

assert(np.allclose(foils_Ic, foils_Ic_test))

# for plotting the whole glide
foils_S, foils_I = np.zeros_like(foils_Sc), np.zeros_like(foils_Ic)
for i in np.arange(ntime):
    foils_S[i] = foils_Sc[i] + Ro_S[i]
    foils_I[i] = foils_Ic[i] + Ro_I[i]


# %% Plot the airfoil shape (light is 0 index, dark in -1 index)

rfoil = np.genfromtxt('../Data/Foil/snake0.004.bdy.txt', skip_header=1)
rfoil = rfoil - rfoil.mean(axis=0)
rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
#rfoil[:, 1] -= rfoil[:, 1].min()  # center at top of airfoil  #TODO
rfoil /= np.ptp(rfoil[:, 0])
rfoil = rfoil[::5]
_r0 = np.zeros(rfoil.shape[0])  # 0 in Yhat direction to start
rfoil = np.c_[rfoil[:, 0], _r0, rfoil[:, 1]]  # in XZ frame to start
rfoil = np.c_[rfoil.T, rfoil[0]].T
nfoil = rfoil.shape[0]

fig, ax = plt.subplots()
#ax.plot(rfoil[:, 0], rfoil[:, 2], 'o-')
ax.scatter(rfoil[:, 0], rfoil[:, 2], c=np.arange(nfoil))
ax.set_aspect('equal', adjustable='box')
ax.margins(.03)
sns.despine()
fig.set_tight_layout(True)


# %% Plot foil at discrete time steps

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = ntime - 20
i = 70
#i = 110
i = 189
#i = 220
i = 0
i = 85

#head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
#                     scale_factor=.015, resolution=16, opacity=.5)
#head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
#                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 color=bmap[1], opacity=1)
#body = mlab.mesh(foils_Sc[i, :, :, 0], foils_Sc[i, :, :, 1], foils_Sc[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#body = mlab.mesh(foils_Ic_test[i, :, :, 0], foils_Ic_test[i, :, :, 1],
#                 foils_Ic_test[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=.5,
#                 vmin=0, vmax=1)

# body.module_manager.scalar_lut_manager.reverse_lut = True

#mlab.points3d(pfe_c[i, :, 0], pfe_c[i, :, 1],
#            pfe_c[i, :, 2], color=bmap[2], scale_factor=15)

#mlab.orientation_axes()
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


##savename = './anim_run/anim_{0:03d}.png'
#
#
# %% Save mesh on foil

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
#        print('Current time: {0}'.format(times[i]))

#        head.mlab_source.set(x=foils_Ic[i, 0, 0, 0],
#                             y=foils_Ic[i, 0, 0, 1],
#                             z=foils_Ic[i, 0, 50, 2])
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])

#        ml.mlab_source.set(x=Lb[i, :, 0], y=Lb[i, :, 1], z=Lb[i, :, 2])
#        md.mlab_source.set(x=Db[i, :, 0], y=Db[i, :, 1], z=Db[i, :, 2])
#        ma.mlab_source.set(x=Ab[i, :, 0], y=Ab[i, :, 1], z=Ab[i, :, 2])

#        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim()
mlab.show()


# %%

mlab.savefig('../Figures/i189.png', size=(3*1286, 3*426))



# %% Fit plane to snake in the INERTIAL frame
#TODO: the Yhat portion causes problesm, going to use the straightened frame

# See: http://stackoverflow.com/a/9243785

from scipy.linalg import svd

mass_body_weight = mass_spl[0, :vent_idx_spl + 1]  # head to vent
#mass_body_weight = mass_spl[0, :vent_idx_spls[0] + 1]  # head to vent
weights = mass_body_weight / mass_body_weight.sum()

Xhat = np.r_[1, 0, 0]
Yhat = np.r_[0, 1, 0]
Zhat = np.r_[0, 0, 1]
Nhat_I = np.zeros((ntime, 3))

Xp_S = np.zeros((ntime, 3))  # Yhat x Np to define the plane coordinate system
Yp_S = np.zeros((ntime, 3))  # forward velocity, projected into plane
Zp_S = np.zeros((ntime, 3))  # normals of the plane

# coordinates from SVD
V0 = np.zeros((ntime, 3))
V1 = np.zeros((ntime, 3))
V2 = np.zeros((ntime, 3))
#V_idxs = np.zeros(ntime, dtype=np.int)

# rotation matrices
C_I2B = np.zeros((ntime, 3, 3))
C1s = np.zeros((ntime, 3, 3))
C2s = np.zeros((ntime, 3, 3))
C3s = np.zeros((ntime, 3, 3))

Sfrac = np.zeros((ntime, 3))  # measure of 'planeness'
planar_fit_error = np.zeros(ntime)

Mws = np.zeros((ntime, len(weights), 3))

for i in np.arange(ntime):
#    idx = vent_idx_spls[i] + 1
    idx = vent_idx_spl + 1
    M = R_Ic[i, :idx]
    Mw = (M.T * weights).T  # weighted points
    Mws[i] = Mw

    U, S, V = svd(Mw)
    Svar = S**2

    # S is sorted singular values in descending order
    # rows of V are singluar values; we want row three
    n = V[2]

#    # check that if the snake is in a funny configuration, we get the
#    # direction closest to the z
#    V_idx = np.abs(np.dot(Zhat, V)).argmax()
#    V_idxs[i] = V_idx
#    n = V[V_idx]

    # make sure the normal points up
    if np.dot(n, Zhat) < 0:
        n = -n

    Nhat_I[i] = n

    # rotate normal vector into straightened frame
    na = np.dot(C_I2S[i], n)

    # calculate pitch angle
    pitch_i = np.arctan2(na[1], na[2])  # tan^-1(na_y, na_z)

    # C1(pitch)
    C1_i = np.array([[1, 0, 0],
                     [0, np.cos(pitch_i), np.sin(pitch_i)],
                     [0, -np.sin(pitch_i), np.cos(pitch_i)]])
    C1_i = C1_i.T  #TODO do this? yes (2016-10-19, but not sure why yet)
    # this is opposite convention Diebel, p. 5, p. 28 (2-1-3 rotation)
    # but we use this opposite rotation convention for yaw (C_I2S)

    # rotate normal about x-axis (Xhat?) into XZ plane
    nb = np.dot(C1_i, na)

    # calculate the roll angle
    roll_i = np.arctan2(nb[0], nb[2])  # tan^-1(nb_x, nb_z)

    # C2(roll)
    C2_i = np.array([[np.cos(roll_i), 0, -np.sin(roll_i)],
                     [0, 1, 0],
                     [np.sin(roll_i), 0, np.cos(roll_i)]])
#    C2_i = C2_i.T  #TODO do this? (2016-10-19, not, but not sure why yet)

    C1s[i] = C1_i
    C2s[i] = C2_i
    C3s[i] = C_I2S[i]

    # C = C2(roll) * C1(pitch) * C3(yaw)
    C_I2B[i] = np.dot(C2s[i], np.dot(C1s[i], C3s[i]))
    #NOTE: np.dot(C_I2B[i], Nhat[i]) = Xhat

    Sfrac[i] = Svar / Svar.sum()
#    planar_fit_error[i] = np.sqrt(np.sum(np.dot(M, n)**2))
    planar_fit_error[i] = np.sqrt(np.sum(np.dot(Mw, n)**2))

    V0[i] = V[0]
    V1[i] = V[1]
    V2[i] = V[2]


# %%



# %% Try fitting a plane, with z as the error

# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
# http://stackoverflow.com/a/12618938
# http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx

def plane_fit_zerr(R):
    """Fit a plane to the x, y coordinates of the snake,
    where we want to minimize error in the z-direction.

    z = ax + by + c
    """

    x, y, z = R.T
    nspl = len(x)

    A = np.c_[x, y, np.ones(nspl)]
    C, resid, rank, sing_vals = np.linalg.lstsq(A, z)    # coefficients

    # evaluate it on grid
    Zplane = C[0] * x + C[1] * y + C[2]
#    Rfit = np.c_[x, y, z - Zplane]
    Rfit = np.c_[x, y, Zplane]

    return Rfit, C, resid, rank, sing_vals


def plane_fit_zerr(R):
    """Fit a plane to the x, y coordinates of the snake,
    where we want to minimize error in the z-direction.

    z = ax + by
    """

    x, y, z = R.T
    nspl = len(x)

    A = np.c_[x, y]
    C, resid, rank, sing_vals = np.linalg.lstsq(A, z)    # coefficients

    # evaluate it on grid
    Zplane = C[0] * x + C[1] * y
#    Rfit = np.c_[x, y, z - Zplane]
    Rfit = np.c_[x, y, Zplane]

    return Rfit, C, resid, rank, sing_vals


i = 170
#R = d['R_Sc'][i]
R = d['R_Ic'][i]

Rfit, C, resid, rank, sing_vals = plane_fit_zerr(R)

# remove the z component
Rsub = Rfit.copy()
Rsub[:, 2] = R[:, 2] - Rfit[:, 2]

#n = np.r_[C[0] / C[2], C[1] / C[2], 1 / C[2]]
n = np.r_[C[0], C[1], -1]
n = n / np.linalg.norm(n)
if np.dot(n, np.r_[0, 0, 1]) < 1:
    n = -n


# %%

R_B = d['R_B']

i = 170

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

mlab.plot3d(Rsub[:, 0], Rsub[:, 1], Rsub[:, 2], color=bmap[2], tube_radius=3)
mlab.plot3d(R_B[i, :, 0], R_B[i, :, 1], R_B[i, :, 2], color=bmap[0], tube_radius=3)


# %%

XY_I = d['XY_I']

#i = 160
i = 170

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

#pts = mlab.points3d(pfe_B[i, :, 0], pfe_B[i, :, 1], pfe_B[i, :, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=64)

#body = mlab.mesh(foils_B[i, :, :, 0],
#                 foils_B[i, :, :, 1],
#                 foils_B[i, :, :, 2],
#                 scalars=foil_color[i],
#                 colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

#pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0],
                 foils_Ic[i, :, :, 1],
                 foils_Ic[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=.5,
                 vmin=0, vmax=1)

XY_mesh = mlab.mesh(XY_I[i, :, :, 0], XY_I[i, :, :, 1], XY_I[i, :, :, 2],
                    color=bmap[0], opacity=.25)

mlab.plot3d(Rfit[:, 0], Rfit[:, 1], Rfit[:, 2], color=bmap[0], tube_radius=3)
mlab.plot3d(Rsub[:, 0], Rsub[:, 1], Rsub[:, 2], color=bmap[2], tube_radius=3)


# %%

R_Sc, dR_Sc, ddR_Sc = d['R_Sc'], d['dR_Sc'], d['ddR_Sc']
#R_Sc, dR_Sc, ddR_Sc = d['R_Ic'], d['dR_Ic'], d['ddR_Ic']
#R_Sc, dR_Sc, ddR_Sc = d['R_B'], d['dR_B'], d['ddR_B']
m = d['mass_spl']

ho = np.zeros_like(R_Sc)
dho = np.zeros_like(R_Sc)
for i in np.arange(ntime):
    mv = (m[i] * dR_Sc[i].T).T
    ma = (m[i] * ddR_Sc[i].T).T
#    mv = dR_Sc[i]
#    ma = ddR_Sc[i]
    ho[i] = np.cross(R_Sc[i], mv)
    dho[i] = np.cross(R_Sc[i], ma)

Ho = ho.sum(axis=1)
dHo = dho.sum(axis=1)

figure(); plot(Ho)
figure(); plot(dHo)


# %%

R_Sc = d['R_Sc']
C_I2S = d['C_I2S']
pfe_Sc = d['pfe_Sc']
foils_Sc = d['foils_Sc']

mass_spl = d['mass_spl']
vent_idx_spl = d['vent_idx_spl']

mass_body_weight = mass_spl[0, :vent_idx_spl + 1]  # head to vent
weights = mass_body_weight / mass_body_weight.sum()

planar_fit_error = np.zeros(ntime)
Mws = np.zeros((ntime, len(weights), 3))
Nhat_I = np.zeros((ntime, 3))

C_I2B = np.zeros((ntime, 3, 3))
C1s = np.zeros((ntime, 3, 3))
C2s = np.zeros((ntime, 3, 3))
C3s = np.zeros((ntime, 3, 3))

yaw, pitch, roll = np.zeros((ntime, 3)).T

idx = vent_idx_spl + 1
R_Sc = R_Sc[:, :idx]

R_B = R_Sc.copy()
pfe_B = pfe_Sc.copy()
foils_B = foils_Sc.copy()

for i in np.arange(ntime):
    M = R_Sc[i]
    Mw = (M.T * weights).T  # weighted points
    Mws[i] = Mw

    x, y, z = Mw.T

    A = np.c_[x, y]
    C, resid, rank, sing_vals = np.linalg.lstsq(A, z)

    zfit = np.dot(M[:, :2], C)
    R_B[i, :, 2] -= zfit

    zfit = np.dot(pfe_Sc[i, :, :2], C)
    pfe_B[i, :, 2] -= zfit

    for j in np.arange(foils_B.shape[1]):
        zfit = np.dot(foils_Sc[i, j, :, :2], C)
        foils_B[i, j, :, 2] -= zfit



# %%

i = 160
i = 170

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

pts = mlab.points3d(pfe_B[i, :, 0], pfe_B[i, :, 1], pfe_B[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_B[i, :, :, 0],
                 foils_B[i, :, :, 1],
                 foils_B[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

pts = mlab.points3d(pfe_Sc[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_Sc[i, :, :, 0],
                 foils_Sc[i, :, :, 1],
                 foils_Sc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=.5,
                 vmin=0, vmax=1)

mlab.plot3d(Rfit[:, 0], Rfit[:, 1], Rfit[:, 2], color=bmap[0], tube_radius=3)
mlab.plot3d(Rsub[:, 0], Rsub[:, 1], Rsub[:, 2], color=bmap[2], tube_radius=3)



# %%

R_Ic = d['R_Ic']
C_I2S = d['C_I2S']
mass_spl = d['mass_spl']
vent_idx_spl = d['vent_idx_spl']

mass_body_weight = mass_spl[0, :vent_idx_spl + 1]  # head to vent
weights = mass_body_weight / mass_body_weight.sum()

planar_fit_error = np.zeros(ntime)
Mws = np.zeros((ntime, len(weights), 3))
Nhat_I = np.zeros((ntime, 3))

C_I2B = np.zeros((ntime, 3, 3))
C1s = np.zeros((ntime, 3, 3))
C2s = np.zeros((ntime, 3, 3))
C3s = np.zeros((ntime, 3, 3))

yaw, pitch, roll = np.zeros((ntime, 3)).T

#R_Bf = np.zeros_like(R_Ic)

for i in np.arange(ntime):
#    idx = vent_idx_spls[i] + 1
    idx = vent_idx_spl + 1
    M = R_Ic[i, :idx]
    Mw = (M.T * weights).T  # weighted points
    Mws[i] = Mw

    x, y, z = Mw.T

    A = np.c_[x, y]
    C, resid, rank, sing_vals = np.linalg.lstsq(A, z)

#    Zplane = C[0] * x + C[1] * y
#    Rfit = np.c_[x, y, z - Zplane]
#    Rfit = np.c_[x, y, Zplane]
#    Rfit

    # remove the z component
#    Rsub = Rfit.copy()
#    Rsub[:, 2] = R[:, 2] - Rfit[:, 2]

    n = np.r_[C[0], C[1], -1]
    n = n / np.linalg.norm(n)
    if np.dot(n, np.r_[0, 0, 1]) < 1:
        n = -n

    Nhat_I[i] = n

        # rotate normal vector into straightened frame
    na = np.dot(C_I2S[i], n)

    # calculate pitch angle
#    pitch_i = np.arctan2(-na[1], na[2])  # tan^-1(-na_y, na_z)
    pitch_i = np.arctan2(na[1], na[2])  # tan^-1(na_y, na_z)

    # C1(pitch)
    C1_i = np.array([[1, 0, 0],
                     [0, np.cos(pitch_i), np.sin(pitch_i)],
                     [0, -np.sin(pitch_i), np.cos(pitch_i)]])
    C1_i = C1_i.T  #TODO do this? yes (2016-10-19, but not sure why yet)
    # this is opposite convention Diebel, p. 5, p. 28 (2-1-3 rotation)
    # but we use this opposite rotation convention for yaw (C_I2S)

    # rotate normal about x-axis (Xhat?) into XZ plane
    nb = np.dot(C1_i, na)

    # calculate the roll angle
    roll_i = np.arctan2(nb[0], nb[2])  # tan^-1(nb_x, nb_z)

    # C2(roll)
    C2_i = np.array([[np.cos(roll_i), 0, -np.sin(roll_i)],
                     [0, 1, 0],
                     [np.sin(roll_i), 0, np.cos(roll_i)]])
#    C2_i = C2_i.T  #TODO do this? (2016-10-19, not, but not sure why yet)

    C1s[i] = C1_i
    C2s[i] = C2_i
    C3s[i] = C_I2S[i]

    # C = C2(roll) * C1(pitch) * C3(yaw)
    C_I2B[i] = np.dot(C2s[i], np.dot(C1s[i], C3s[i]))

    yaw_i = np.arccos(C_I2S[i][0, 0])
    yaw[i], pitch[i], roll[i] = yaw_i, pitch_i, roll_i


# %%

R_Ic = d['R_Ic']
dRds_I = d['dRds_I']

pfe_B = np.zeros_like(pfe_Ic)
foils_B = np.zeros_like(foils_Ic)
R_B = np.zeros_like(R_Ic)
dRds_B = np.zeros_like(dRds_I)

for i in np.arange(ntime):
    pfe_B[i] = np.dot(C_I2B[i], pfe_Ic[i].T).T
    R_B[i] = np.dot(C_I2B[i], R_Ic[i].T).T
    dRds_B[i] = np.dot(C_I2B[i], dRds_I[i].T).T

    for j in np.arange(foils_Ic.shape[1]):
        foils_B[i, j] = np.dot(C_I2B[i], foils_Ic[i, j].T).T


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

#pts = mlab.points3d(pfe_Sc[i, :, 0], pfe_Sc[i, :, 1], pfe_Sc[i, :, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=64)
#
#body = mlab.mesh(foils_Sc[i, :, :, 0],
#                 foils_Sc[i, :, :, 1],
#                 foils_Sc[i, :, :, 2],
#                 scalars=foil_color[i],
#                 colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0],
                 foils_Ic[i, :, :, 1],
                 foils_Ic[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

mlab.plot3d(R[:, 0], R[:, 1], R[:, 2], color=bmap[1], tube_radius=3)
mlab.plot3d(Rfit[:, 0], Rfit[:, 1], Rfit[:, 2], color=bmap[0], tube_radius=3)
mlab.plot3d(Rsub[:, 0], Rsub[:, 1], Rsub[:, 2], color=bmap[2], tube_radius=3)

#mlab.quiver3d(C[0], C[1], C[2], scale_factor=50,
#              color=bmap[1], mode='arrow', opacity=1, resolution=64)

mlab.quiver3d(n[0], n[1], n[2], scale_factor=50,
              color=bmap[5], mode='arrow', opacity=1, resolution=64)


# %%


fig, ax = plt.subplots()
ax.plot(Nhat_I)
#ax.plot(Nhat)
sns.despine()
fig.set_tight_layout(True)




# %% construct rotation matrix to get snake in Xp, Yp, Zp frame

# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix

# rotation matrix from I to B
#Rps_S2B = np.zeros((ntime, 3, 3))
#Rps_I2B = np.zeros((ntime, 3, 3))

# CoM velocity and accerlation
dRo_B = np.zeros_like(dRo_I)
ddRo_B = np.zeros_like(ddRo_I)

# spline derivatives
dRds_B = np.zeros_like(dRds_I)
ddRds_B = np.zeros_like(ddRds_I)

# spline velocity and accerlation
R_B = np.zeros_like(R_Ic)
dR_B = np.zeros_like(dR_I)
ddR_B = np.zeros_like(ddR_I)

# markers
pf_B = np.zeros_like(pf_I)
vf_B = np.zeros_like(vf_I)
af_B = np.zeros_like(af_I)
pfe_B = np.zeros_like(pfe_I)

# relative to CoM [...] velocities and accelerations
dR_Bc = np.zeros_like(dR_Ic)
ddR_Bc = np.zeros_like(ddR_Ic)
vf_Bc = np.zeros_like(vf_Ic)
af_Bc = np.zeros_like(af_Ic)

# body coordinate system and foils for plotting
foils_B = np.zeros_like(foils_Ic)
Tdir_B = np.zeros_like(Tdir_I)
Cdir_B = np.zeros_like(Cdir_I)
Bdir_B = np.zeros_like(Bdir_I)

for i in np.arange(ntime):
    dRo_B[i] = np.dot(C_I2B[i], dRo_I[i].T).T
    ddRo_B[i] = np.dot(C_I2B[i], ddRo_I[i].T).T

    dRds_B[i] = np.dot(C_I2B[i], dRds_I[i].T).T
    ddRds_B[i] = np.dot(C_I2B[i], ddRds_I[i].T).T

    R_B[i] = np.dot(C_I2B[i], R_Ic[i].T).T
    dR_B[i] = np.dot(C_I2B[i], dR_I[i].T).T
    ddR_B[i] = np.dot(C_I2B[i], ddR_I[i].T).T

    pf_B[i] = np.dot(C_I2B[i], pf_Ic[i].T).T
    vf_B[i] = np.dot(C_I2B[i], vf_I[i].T).T
    af_B[i] = np.dot(C_I2B[i], af_I[i].T).T
    pfe_B[i] = np.dot(C_I2B[i], pfe_Ic[i].T).T

    dR_Bc[i] = np.dot(C_I2B[i], dR_Ic[i].T).T
    ddR_Bc[i] = np.dot(C_I2B[i], ddR_Ic[i].T).T
    vf_Bc[i] = np.dot(C_I2B[i], vf_Ic[i].T).T
    af_Bc[i] = np.dot(C_I2B[i], af_Ic[i].T).T

    for j in np.arange(nspl):
        Tdir_B[i, j] = np.dot(C_I2B[i], Tdir_I[i, j])
        Cdir_B[i, j] = np.dot(C_I2B[i], Cdir_I[i, j])
        Bdir_B[i, j] = np.dot(C_I2B[i], Bdir_I[i, j])

    for j in np.arange(foils_Ic.shape[1]):
        foils_B[i, j] = np.dot(C_I2B[i], foils_Ic[i, j].T).T


# %%

def C2euler(C):
    """Euler angles from rotation matrix, using 3-1-2 convention.

    Parameters
    ----------
    C : proper rotation matrix which converts vectors from the inertial
        frame to the body frame

    Returns
    -------
    yaw, pitch, and roll angles in radians
    """

    # eqn 361 Diebel, but swap ROWS
    yaw = np.arctan2(-C[1, 0], C[1, 1])
    pitch = np.arcsin(C[1, 2])
    roll = np.arctan2(-C[0, 2], C[2, 2])

    return yaw, pitch, roll


yaw, pitch, roll = np.zeros(ntime), np.zeros(ntime), np.zeros(ntime)
for i in np.arange(ntime):
    yaw[i], pitch[i], roll[i] = C2euler(C_I2B[i])
#    yaw[i], pitch[i], roll[i] = C2euler(Rps_I2B[i])
#    yaw[i], pitch[i], roll[i] = C2euler(C_I2S[i])
#    yaw[i], pitch[i], roll[i] = C2euler(Rmus[i])

yaw_d, pitch_d, roll_d = np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)

dyaw, ddyaw = smoothing.findiff(yaw, dt)
dpitch, ddpitch= smoothing.findiff(pitch, dt)
droll, ddroll = smoothing.findiff(roll, dt)

dyaw_d, dpitch_d, droll_d = np.rad2deg(dyaw), np.rad2deg(dpitch), np.rad2deg(droll)


# %%

fig, ax = plt.subplots()
ax.plot(times, yaw_d, c='b', label='yaw')
ax.plot(times, pitch_d, c='r', label='pitch')
ax.plot(times, roll_d, c='g', label='roll')
ax.legend(loc='best')
ax.set_xlabel('time (s)')
ax.set_ylabel('Euler angles (deg)')
sns.despine()
fig.set_tight_layout(True)
#fig.savefig('../Figures/s_413_91/Euler angles vs time.pdf')


fig, ax = plt.subplots()
ax.plot(times, dyaw_d, c='b', label='yaw')
ax.plot(times, dpitch_d, c='r', label='pitch')
ax.plot(times, droll_d, c='g', label='roll')
ax.legend(loc='best')
ax.set_xlabel('time (s)')
ax.set_ylabel('Euler angle rates (deg/sec)')
sns.despine()
fig.set_tight_layout(True)
#fig.savefig('../Figures/s_413_91/Euler angle rates vs time.pdf')


# %%


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(yaw_d, dyaw_d, 'bx-', mew=2, markevery=20)
ax.grid(True)
ax.set_xlabel('yaw (deg)')
ax.set_ylabel('yaw rate (deg/sec)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig('../Figures/s_413_91/yaw_space.pdf')


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(pitch_d, dpitch_d, 'rx-', mew=2, markevery=20)
ax.grid(True)
ax.set_xlabel('pitch (deg)')
ax.set_ylabel('pitch rate (deg/sec)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig('../Figures/s_413_91/pitch_space.pdf')


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(roll_d, droll_d, 'gx-', mew=2, markevery=20)
ax.grid(True)
ax.set_xlabel('roll (deg)')
ax.set_ylabel('roll rate (deg/sec)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig('../Figures/s_413_91/roll_space.pdf')


#fig, ax = plt.subplots()
#ax.plot(times * 1.4, yaw_d, label='yaw')
#ax.plot(times * 1.4, pitch_d, label='pitch')
#ax.plot(times * 1.4, roll_d, label='roll')
#ax.legend(loc='best')
#ax.set_xlabel('time/Tund=.714s')
#ax.set_ylabel('Euler angles (deg)')
#sns.despine()
#fig.set_tight_layout(True)


# %% Angular velocity

def dang2omg(yaw, pitch, roll):
    """Body components of angular velocity from Euler angle rates. This
    is used to determine initial conditions of omega.

    Parameters
    ----------
    yaw : float
        yaw angle about z-axis in radians
    pitch : float
        pitch angle about y-axis in radians
    roll : float
        roll angle about x-axis in radians

    Returns
    -------
    euler_rates_matrix : array, size (3, 3)
        body-axis components of absolute angular velocity in terms of
        Euler angle rates. This needs to be dotted with Euler angle rates
        and C.T to get omega_0 in expressed in the inertial frame.
    """
    # p. 28 Diebel, eqn 365, but swap first and last COLUMNS
    c, s = np.cos, np.sin
    Kinv = np.array([[-c(pitch) * s(roll), c(roll), 0],
                     [s(pitch), 0, 1],
                     [c(pitch) * c(roll), s(roll), 0]])
    return Kinv


dang = np.c_[dyaw, dpitch, droll]
omg_B = np.zeros_like(dang)
omg_I = np.zeros_like(dang)
for i in np.arange(ntime):
    Kinv = dang2omg(yaw[i], pitch[i], roll[i])
    omg_B[i] = np.dot(Kinv, dang[i])
    omg_I[i] = np.dot(C_I2B[i].T, omg_B[i])

#TODO maybe these need to be smoothed?
domg_B, ddomg_B = smoothing.findiff(omg_B, dt)
domg_I, ddomg_I = smoothing.findiff(omg_I, dt)


# %%

fig, ax = plt.subplots()
ax.plot(times, omg_B)
ax.plot(times, omg_I, '--')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, domg_B)
ax.plot(times, domg_I, '--')
sns.despine()
fig.set_tight_layout(True)


# %% Inertial terms


# %% Reconstruct the serpenoid curve
#TODO: flesh this out more

#cos_theta = -dRds_B[:, :, 0] / np.cos(psi_body)  # x
#sin_theta = dRds_B[:, :, 1] / np.cos(psi_body)  # y
#theta_body_x = np.arccos(cos_theta)
#theta_body_y = np.arcsin(sin_theta)
#tan_theta = -dRds_B[:, :, 1] / dRds_B[:, :, 0]
#theta_body = np.arctan(tan_theta)

psi_body = np.arcsin(dRds_B[:, :, 2])
theta_body = np.arctan2(dRds_B[:, :, 0], -dRds_B[:, :, 1])

psi_body = np.unwrap(psi_body, axis=1)
theta_body = np.unwrap(theta_body, axis=1)

#i = 189
i = 50
fig, ax = plt.subplots()
#ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body_x[:, i]))
#ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body_y[:, i]))
ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body[i]))
ax.plot(t_coord[i] / SVL, np.rad2deg(psi_body[i]))
ax.set_ylim(-115, 115)
ax.axhline(-90, color='k', lw=1)
ax.axhline(90, color='k', lw=1)
ax.grid(True)
sns.despine()
fig.set_tight_layout(True)



# %%

fig, ax = plt.subplots()
sns.despine()
fig.set_tight_layout(True)


#ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body_x[:, i]))
#ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body_y[:, i]))

for i in np.arange(ntime):
    ax.cla()
    ax.plot(t_coord[i] / SVL, np.rad2deg(theta_body[i]))
    ax.plot(t_coord[i] / SVL, np.rad2deg(psi_body[i]))
    ax.set_ylim(-115, 115)
    ax.axhline(-90, color='k', lw=1)
    ax.axhline(90, color='k', lw=1)
    ax.grid(True)
    fig.canvas.draw()
    plt.pause(.01)


# %% Non-linear curve fit to the vertical wave

from scipy.optimize import curve_fit

def func(snon, psi_max, nu_psi, phi_psi, frac_psi):
    psi_off = psi_max * frac_psi
    psi_amp = np.linspace(psi_max - psi_off, psi_max + psi_off, len(snon))
    return psi_amp * np.cos(2 * np.pi * nu_psi * snon + phi_psi)
#    return psi_max * np.cos(2 * np.pi * nu_psi * snon + phi_psi)

snon = t_coord[0, :vent_idx_spl + 1] / SVL

i = 189

ydata = psi_body[i, :vent_idx_spl + 1]

pguess = np.r_[np.rad2deg(15), 2 * 1.1, 0, .1]

#pguess = popt

popt, pcov = curve_fit(func, snon, ydata, p0=pguess)

yfit = func(snon, *popt)


# %%

fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(ydata), label='data')
ax.plot(snon, np.rad2deg(yfit), label='fit')
ax.legend(loc='best')
ax.grid(True)
ax.set_xlabel('length (SVL)')
ax.set_ylabel('vertical wave angle (deg)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig('../Figures/s_413_91/vertical wave (i=189).pdf')


# %% Non-linear curve fit to the lateral wave

from scipy.optimize import curve_fit

def func(snon, theta_max, nu_theta, phi_theta, frac_theta):
    theta_off = theta_max * frac_theta
    start, stop = theta_max - theta_off, theta_max + theta_off
    theta_amp = np.linspace(start, stop, len(snon))
    intwave = np.cos(2 * np.pi * nu_theta * snon + phi_theta)
    return theta_amp * np.sin(np.pi / 2 * intwave)


def rms_error(data, fit):
    return np.sqrt(np.mean((data - fit)**2))


snon = t_coord[0, :vent_idx_spl + 1] / SVL

#i = 115
i = 189
#i = 70

ydata = theta_body[i, :vent_idx_spl + 1]

pguess = np.r_[np.rad2deg(90), 1.1, 0, .1]

#pguess = popt

popt, pcov = curve_fit(func, snon, ydata, p0=pguess)

yfit = func(snon, *popt)



# %%

fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(ydata), label='data')
ax.plot(snon, np.rad2deg(yfit), label='fit')
ax.legend(loc='best')
ax.grid(True)
ax.set_xlabel('length (SVL)')
ax.set_ylabel('lateral wave angle (deg)')
sns.despine()
fig.set_tight_layout(True)
fig.savefig('../Figures/s_413_91/lateral wave (i=189).pdf')


# %% COD analysis of bending angles

def cod(Xorig):
    """"Complex orthogonal decomposition analysis.

    Parameters
    ----------
    Xorig : array, size (ntime, nbody)
        Angles of the body

    Returns
    -------
    cod_dict : dict
        Dictionary with the COD variables.

    """

    from scipy.signal import hilbert

    X = Xorig.copy().T  # transpose so that (nbody, ntime)
    m_nbody, n_ntime = X.shape

    Z = hilbert(X, axis=1)  # check the axis (across time)
    R = np.dot(Z, Z.conj().T) / n_ntime
    lamb, W = np.linalg.eig(R)
    lamb = lamb.real
    idx = np.argsort(lamb)[::-1]
    W = W[:, idx]  # columns are the eigenvectors

    lamb_norm = lamb / lamb.sum()
    A = np.sqrt(lamb)

    Q = np.dot(W.conj().T, Z)  # modal coordinates (length of ntime)
    #np.allclose(np.dot(W[:, 0].conj().T, Z), Q[0])
    #np.allclose(np.dot(W[:, 2].conj().T, Z), Q[2])

    out = dict(lamb=lamb, lamb_norm=lamb_norm, A=A, Z=Z, W=W, Q=Q)

    return out


def cod_mode_decomp(W, nmodes=5):
    """Deconstruct the modal coordinate into traveling and standing waves.

    Parameters
    ----------
    cod_dict : dict
        Dictionary with the COD varaibles (output from cod)

    Returns
    -------

    """

    Ws = np.zeros((W.shape[0], nmodes)).astype(np.complex)  # standing wave modes
    Wt = np.zeros((W.shape[0], nmodes)).astype(np.complex)  # traveling wave modes
    traveling_index = np.zeros(nmodes)

    for i in np.arange(nmodes):
        c, d = W[:, i].real, W[:, i].imag
        # 1 = traveling, 0 = standing
        traveling_index[i] = 1 / np.linalg.cond(np.c_[c, d])

        e_c = c / np.linalg.norm(c)
        d_s = np.dot(d, e_c) * e_c
        d_t = d - d_s
        c_t = np.linalg.norm(d_t) * e_c
        c_s = c - c_t

        w_s = c_s + 1j * d_s
        w_t = c_t + 1j * d_t

        # w_s = w_s.reshape(-1, 1)
        # w_t = w_t.reshape(-1, 1)
        Ws[:, i] = w_s
        Wt[:, i] = w_t

    return Ws, Wt, traveling_index


def cod_reanimate(cod_dict, mode_number, times):

    a = cod_dict['A'][mode_number]
    w = cod_dict['W'][:, mode_number]
    f = cod_dict['f']

    ntime = len(times)
    nbody = len(w)
    Y = np.zeros((ntime, nbody))

    for i in np.arange(ntime):
        t = times[i]
        amp = a * np.exp(1j * 2 * np.pi * f * t)
        # zz = amp * (w.real + 1j * w.imag)  # if -1j, wave goes wrong way
        zz = amp * w
        yy = zz.real
        Y[i] = yy

    return Y


def cod_reconstruct_body(theta, psi, mass, sdim, times):
    """
    """

    from scipy.integrate import cumtrapz

    # derivatives wrt body
    dy = -np.cos(psi) * np.cos(theta)
    dx =  np.cos(psi) * np.sin(theta)
    dz = np.sin(psi)

    # integrate for backbone
    ntime, nbody = len(times), len(sdim)
    p_cod = np.zeros((ntime, nbody, 3))
    for i in np.arange(ntime):
        p_cod[i, :, 0] = cumtrapz(dx[i], sdim, initial=dx[i, 0])
        p_cod[i, :, 1] = cumtrapz(dy[i], sdim, initial=dy[i, 0])
        p_cod[i, :, 2] = cumtrapz(dz[i], sdim, initial=dz[i, 0])
        # p_cod[i, :, 0] = cumtrapz(dx[i], sdim, initial=0)
        # p_cod[i, :, 1] = cumtrapz(dy[i], sdim, initial=0)
        # p_cod[i, :, 2] = cumtrapz(dz[i], sdim, initial=0)

        com_shift = np.sum((p_cod[i].T * mass[i]).T, axis=0) / mass[i].sum()
        p_cod[i] = p_cod[i] - com_shift

    return p_cod


# %%

## reconstruct the motion using on a few modes
#r = 1  # number of modes ot use
#Wr = W[:, :r]
##Wr = W[:, r-1:r]  # just use second mode
##Wr = W[:, 0].reshape(-1, 1)
#Zr = np.dot(Wr, np.dot(Wr.conj().T, Z))
#Xr = np.real(Zr)


# wave decomposition
#Zr_s = np.dot(w_s, np.dot(w_s.conj().T, Z))
#Zr_t = np.dot(w_t, np.dot(w_t.conj().T, Z))
#Xr_s = np.real(Zr_s)
#Xr_t = np.real(Zr_t)

#Zr_s = np.dot(w_s, np.dot(w_s.conj().T, Z))
#Zr_t = np.dot(w_t, np.dot(w_t.conj().T, Z))
#Xr_s = np.real(Zr_s)
#Xr_t = np.real(Zr_t)

sdim = t_coord[0, :vent_idx_spl + 1]
snon = sdim / SVL
theta_reduc = theta_body[:, :vent_idx_spl + 1]
psi_reduc =psi_body[:, :vent_idx_spl + 1]

# perform cod analysis
theta_cod = cod(theta_reduc)
psi_cod = cod(psi_reduc)

# decompose modes into standing and traveling waves
_theta_out = cod_mode_decomp(theta_cod['W'])
_psi_out = cod_mode_decomp(psi_cod['W'])
theta_cod['Ws'], theta_cod['Wt'], theta_cod['traveling_index'] = _theta_out
psi_cod['Ws'], psi_cod['Wt'], psi_cod['traveling_index'] = _psi_out


# %% Importance of modes

fig, ax = plt.subplots()
ax.plot(100 * theta_cod['lamb_norm'].cumsum(), 'o-', label='lateral')
ax.plot(100 * psi_cod['lamb_norm'].cumsum(), 'o-', label='vertical')
ax.legend(loc='lower right')
ax.set_xlim(-.25, 5.25)
ax.set_ylim(49, 102)
ax.grid(True)
ax.set_xlabel('mode number')
ax.set_ylabel('cumulative variance')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig('../Figures/s_413_91/cod_var.pdf')


# %% Temporal frequency for dominant mode

theta_Qang = np.unwrap(np.angle(theta_cod['Q'][0]))
psi_Qang = np.unwrap(np.angle(psi_cod['Q'][0]))

theta_Qfit = np.polyfit(times, theta_Qang, 1)  # rad / sec
psi_Qfit = np.polyfit(times, psi_Qang, 1)

# temporal frequency
theta_f = theta_Qfit[0] / (2 * np.pi)  # slope of the best fit
psi_f = psi_Qfit[0] / (2 * np.pi)
theta_cod['f'] = theta_f
psi_cod['f'] = psi_f

linear_fit_Qtheta = np.polyval(theta_Qfit, times)
linear_fit_Qpsi = np.polyval(psi_Qfit, times)

fig, ax = plt.subplots()
ax.plot(times, theta_Qang)
ax.plot(times, psi_Qang)
ax.plot(times, linear_fit_Qtheta, 'b--')
ax.plot(times, linear_fit_Qpsi, 'g--')
ax.set_xlabel('time (sec)')
ax.set_ylabel('unwound angle')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
#ax.plot(theta_cod['Q'][0].real, theta_cod['Q'][0].imag, '-o',
#        markevery=1000)
#ax.plot(psi_cod['Q'][0].real, psi_cod['Q'][0].imag, '-o',
#        markevery=1000)
ax.plot(psi_cod['Q'][2].real, psi_cod['Q'][2].imag, '-o',
        markevery=1000)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Re')
ax.set_ylabel('Im')
sns.despine()
fig.set_tight_layout(True)


# %% Plot the mode shapes

fig, ax = plt.subplots()
ax.plot(times, theta_cod['Q'][0].real, 'b')
ax.plot(times, theta_cod['Q'][0].imag, 'b--')
ax.plot(times, psi_cod['Q'][0].real, 'g')
ax.plot(times, psi_cod['Q'][0].imag, 'g--')
ax.set_xlabel('time (sec)')
ax.set_ylabel('modal activity')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(theta_cod['Q'][0].real, psi_cod['Q'][0].real, 'b')
#ax.plot(times, theta_cod['Q'][1], 'b--')
#ax.plot(times, psi_cod['Q'][0], 'g')
#ax.plot(times, psi_cod['Q'][1], 'g--')
ax.set_aspect('equal', adjustable='box')
#ax.set_xlabel('time (sec)')
#ax.set_ylabel('modal activity')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(psi_cod['Q'][0].real, psi_cod['Q'][1].real, 'b')
#ax.plot(times, theta_cod['Q'][1], 'b--')
#ax.plot(times, psi_cod['Q'][0], 'g')
#ax.plot(times, psi_cod['Q'][1], 'g--')
ax.set_aspect('equal', adjustable='box')
#ax.set_xlabel('time (sec)')
#ax.set_ylabel('modal activity')
sns.despine()
fig.set_tight_layout(True)


# %% Spatial frequency for dominant mode

theta_Wang = np.unwrap(np.angle(theta_cod['W'][:, 0]))
psi_Wang = np.unwrap(np.angle(psi_cod['W'][:, 0]))

theta_Wfit = np.polyfit(snon, theta_Wang, 1)  # rad
psi_Wfit = np.polyfit(snon, psi_Wang, 1)

# spatial frequency
theta_nu = theta_Wfit[0] / (2 * np.pi)  # slope of the best fit
psi_nu = psi_Wfit[0] / (2 * np.pi)
theta_cod['nu'] = theta_nu
psi_cod['nu'] = psi_nu

linear_fit_Wtheta = np.polyval(theta_Wfit, snon)
linear_fit_Wpsi = np.polyval(psi_Wfit, snon)

fig, ax = plt.subplots()
ax.plot(snon, theta_Wang)
ax.plot(snon, psi_Wang)
ax.plot(snon, linear_fit_Wtheta, 'b--')
ax.plot(snon, linear_fit_Wpsi, 'g--')
ax.set_xlabel('length (SVL)')
ax.set_ylabel('unwound angle')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(theta_cod['W'][:, 0].real, theta_cod['W'][:, 0].imag, '-o',
        markevery=1000)
ax.plot(psi_cod['W'][:, 0].real, psi_cod['W'][:, 0].imag, '-o',
        markevery=1000)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Re')
ax.set_ylabel('Im')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['W'][:, 0].real), 'b')
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['W'][:, 0].imag), 'b--')
#ax.plot(snon, theta_cod['W'][:, 1], 'b--')
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['W'][:, 0].real), 'g')
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['W'][:, 0].imag), 'g--')
#ax.plot(snon, psi_cod['W'][:, 1], 'g--')
ax.set_xlabel('time (sec)')
ax.set_ylabel('modal activity')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['Wt'][:, 0].real))
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['Wt'][:, 0].imag))
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['Ws'][:, 0].real))
ax.plot(snon, np.rad2deg(theta_cod['A'][0] * theta_cod['Ws'][:, 0].imag))
#ax.plot(snon, np.rad2deg())
#ax.set_xlabel('length (SVL)')
#ax.set_ylabel('unwound angle')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['Wt'][:, 0].real))
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['Wt'][:, 0].imag))
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['Ws'][:, 0].real))
ax.plot(snon, np.rad2deg(psi_cod['A'][0] * psi_cod['Ws'][:, 0].imag))
#ax.plot(snon, np.rad2deg())
#ax.set_xlabel('length (SVL)')
#ax.set_ylabel('unwound angle')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(snon, np.rad2deg(psi_cod['A'][1] * psi_cod['Wt'][:, 1].real))
ax.plot(snon, np.rad2deg(psi_cod['A'][1] * psi_cod['Wt'][:, 1].imag))
ax.plot(snon, np.rad2deg(psi_cod['A'][1] * psi_cod['Ws'][:, 1].real))
ax.plot(snon, np.rad2deg(psi_cod['A'][1] * psi_cod['Ws'][:, 1].imag))
#ax.plot(snon, np.rad2deg())
#ax.set_xlabel('length (SVL)')
#ax.set_ylabel('unwound angle')
sns.despine()
fig.set_tight_layout(True)


# %% Reanimate the modes

theta_0 = cod_reanimate(theta_cod, 0, times)
psi_0 = cod_reanimate(psi_cod, 0, times)
psi_1 = cod_reanimate(psi_cod, 1, times)
psi_01 = psi_0 + psi_1


# %%

fig, ax = plt.subplots()
sns.despine()
fig.set_tight_layout(True)

for i in np.arange(ntime):
    ax.cla()
    ax.axhline(90, color='gray', lw=1)
    ax.axhline(-90, color='gray', lw=1)
    ax.plot(snon, np.rad2deg(theta_0[i]))
    ax.plot(snon, np.rad2deg(psi_0[i]))
    ax.plot(snon, np.rad2deg(psi_1[i]))
    ax.plot(snon, np.rad2deg(psi_01[i]))
#    ax.plot(snon, np.rad2deg(psi_1[i]))
#    ax.plot(snon, np.rad2deg(Xr.T[i]), 'r')
#    ax.plot(snon, np.rad2deg(A * w.real), 'k')
#    ax.plot(snon, np.rad2deg(A * w.imag), 'gray')
    fig.canvas.draw()
    ax.set_ylim(-110, 100)
    plt.pause(.01)


# %% Reconstruct the body

psi_recon = psi_01

mass_spl_red = mass_spl[:, :vent_idx_spl + 1]

p_cod = cod_reconstruct_body(theta_0, psi_recon, mass_spl_red, sdim, times)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 0

body = mlab.plot3d(p_cod[i, :, 0], p_cod[i, :, 1], p_cod[i, :, 2],
            color=bmap[2], tube_radius=3)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=p_cod[i, :, 0],
                             y=p_cod[i, :, 1],
                             z=p_cod[i, :, 2])
        yield
manim = anim()
mlab.show()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro_I[:, 0], Ro_I[:, 1], Ro_I[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_I[i, :, 0], pfe_I[i, :, 1], pfe_I[i, :, 2],

                  color=bmap[3], scale_factor=20, resolution=64)
mlab.plot3d(Ro_S[:, 0], Ro_S[:, 1], Ro_S[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(R_S[i, :, 0], R_S[i, :, 1], R_S[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe_S[i, :, 0], pfe_S[i, :, 1], pfe_S[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)

mlab.orientation_axes()
fig.scene.isometric_view()

# %%

# %%

def corr(x, y):
    c = np.correlate(x, y, mode='full')  # theta, psi
    c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    return c

c = corr(theta_cod['Q'][0].real, psi_cod['Q'][0].real)
tt = np.r_[-times[::-1], times[1:]]

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)
ax.plot(tt, c)
sns.despine()
fig.set_tight_layout(True)


# %%

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



# %%

fig, ax = plt.subplots()
ax.plot(t_coord.T / SVL, np.rad2deg(theta_body.T))
ax.plot(t_coord.T / SVL, np.rad2deg(psi_body.T))
sns.despine()
fig.set_tight_layout(True)


# %% Plot the body in the body frame

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2], scale_factor=.01,
              color=bmap[0], mode='arrow', resolution=64)

mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2], scale_factor=.01,
              color=bmap[1], mode='arrow', resolution=64)

mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

#mlab.quiver3d(V0_I[i, 0], V0_I[i, 1], V0_I[i, 2], scale_factor=75,
#              color=bmap[1], mode='2darrow', resolution=64)
#mlab.quiver3d(V1_I[i, 0], V1_I[i, 1], V1_I[i, 2], scale_factor=75,
#              color=bmap[2], mode='2darrow', resolution=64)
#mlab.quiver3d(V2_I[i, 0], V2_I[i, 1], V2_I[i, 2], scale_factor=75,
#              color=bmap[0], mode='2darrow', resolution=64)

body = mlab.mesh(foils_B[i, :, :, 0], foils_B[i, :, :, 1], foils_B[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='PuBu', opacity=1,
                 vmin=0, vmax=1)

#mlab.orientation_axes()
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %% Moment of inertia tensor and principle directions

def Iten(r, m):
    """Full moment of inertia tensor.

    Parameters
    ----------
    r : array, size (nbody, 3)
        [x, y, z] coordinates of the point masses
    m : array, size (nbody)
        mass of each point

    Returns
    -------
    Ifull : array, size (3, 3)
        moment of inerta tensor
    """
    x, y, z = r.T
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y)
    Ixz = -np.sum(m * x * z)
    Iyz = -np.sum(m * y * z)
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])


E1_I = np.zeros((ntime, 3))
E2_I = np.zeros((ntime, 3))
E3_I = np.zeros((ntime, 3))
EV = np.zeros((ntime, 3))
for i in np.arange(ntime):
    #idx = vent_idx_spls[i] + 1
    idx = vent_idx_spl
    m_i = mass_spl[i, :idx]
    M = R_Ic[i, :idx]
    Iten_i = Iten(M, m_i)

    evals, evecs = np.linalg.eig(Iten_i)
    evals_idx = np.argsort(evals)
    evals = evals[evals_idx]
    e1, e2, e3 = evecs[evals_idx].T

    E1_I[i] = e1
    E2_I[i] = e2
    E3_I[i] = e3
    EV[i] = evals


# %% Plot the principle axes

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#i = 0
i = 30
i = 189
i = 70


# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

Yp_quiv = mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

# TODO animate this!
E1_quiv = mlab.quiver3d(E1_I[i, 0], E1_I[i, 1], E1_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=3)
E2_quiv = mlab.quiver3d(E2_I[i, 0], E2_I[i, 1], E2_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=3)
E3_quiv = mlab.quiver3d(E3_I[i, 0], E3_I[i, 1], E3_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=3)



#YZ_mesh = mlab.mesh(YZ_I[i, :, :, 0], YZ_I[i, :, :, 1], YZ_I[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_I[i, :, :, 0], XZ_I[i, :, :, 1], XZ_I[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
#XY_mesh = mlab.mesh(XY_I[i, :, :, 0], XY_I[i, :, :, 1], XY_I[i, :, :, 2],
#                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])
        dRo_quiv.mlab_source.set(u=[dRo_I[i, 0]],
                                 v=[dRo_I[i, 1]],
                                 w=[dRo_I[i, 2]])

        Yp_quiv.mlab_source.set(u=[Yp_I[i, 0]],
                                v=[Yp_I[i, 1]],
                                w=[Yp_I[i, 2]])
        Xp_quiv.mlab_source.set(u=[Xp_I[i, 0]],
                                v=[Xp_I[i, 1]],
                                w=[Xp_I[i, 2]])
        Zp_quiv.mlab_source.set(u=[Zp_I[i, 0]],
                                v=[Zp_I[i, 1]],
                                w=[Zp_I[i, 2]])

        E1_quiv.mlab_source.set(u=[E1_I[i, 0]],
                                v=[E1_I[i, 1]],
                                w=[E1_I[i, 2]])
        E2_quiv.mlab_source.set(u=[E2_I[i, 0]],
                                v=[E2_I[i, 1]],
                                w=[E2_I[i, 2]])
        E3_quiv.mlab_source.set(u=[E3_I[i, 0]],
                                v=[E3_I[i, 1]],
                                w=[E3_I[i, 2]])


        yield
manim = anim()
mlab.show()


# %% Plot the principle axes

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#i = 0
i = 30
i = 189
i = 70
i=0

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

Yp_quiv = mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_I[i, :, :, 0], YZ_I[i, :, :, 1], YZ_I[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_I[i, :, :, 0], XZ_I[i, :, :, 1], XZ_I[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
XY_mesh = mlab.mesh(XY_I[i, :, :, 0], XY_I[i, :, :, 1], XY_I[i, :, :, 2],
                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        pts.mlab_source.set(x=pfe_Ic[i, :, 0],
                            y=pfe_Ic[i, :, 1],
                            z=pfe_Ic[i, :, 2])
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])
        dRo_quiv.mlab_source.set(u=[dRo_I[i, 0]],
                                 v=[dRo_I[i, 1]],
                                 w=[dRo_I[i, 2]])

        Yp_quiv.mlab_source.set(u=[Yp_I[i, 0]],
                                v=[Yp_I[i, 1]],
                                w=[Yp_I[i, 2]])
        Xp_quiv.mlab_source.set(u=[Xp_I[i, 0]],
                                v=[Xp_I[i, 1]],
                                w=[Xp_I[i, 2]])
        Zp_quiv.mlab_source.set(u=[Zp_I[i, 0]],
                                v=[Zp_I[i, 1]],
                                w=[Zp_I[i, 2]])
        XY_mesh.mlab_source.set(x=XY_I[i, :, :, 0],
                                y=XY_I[i, :, :, 1],
                                z=XY_I[i, :, :, 2])

        yield
manim = anim()
mlab.show()




# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#i = 0
i = 30
i = 189
i = 70


# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

mlab.points3d(pfe_B[i, :, 0], pfe_B[i, :, 1], pfe_B[i, :, 2],
              color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_B[i, :, :, 0], foils_B[i, :, :, 1], foils_B[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)



# %% Body coordinate system planes for plotting

_nmesh = 21


#Xp_S = np.zeros((ntime, 3))  # Yhat x Np to define the plane coordinate system
#Yp_S = np.zeros((ntime, 3))  # forward velocity, projected into plane
#Zp_S = np.zeros((ntime, 3))  # normals of the plane

# axes for plotting
Xp_B = np.zeros((ntime, 3))
Yp_B = np.zeros((ntime, 3))
Zp_B = np.zeros((ntime, 3))
Xp_I = np.zeros((ntime, 3))
Yp_I = np.zeros((ntime, 3))
Zp_I = np.zeros((ntime, 3))
Xp_S = np.zeros((ntime, 3))
Yp_S = np.zeros((ntime, 3))
Zp_S = np.zeros((ntime, 3))

# planes for plotting
YZ_B = np.zeros((ntime, _nmesh, _nmesh, 3))
XZ_B = np.zeros((ntime, _nmesh, _nmesh, 3))
XY_B = np.zeros((ntime, _nmesh, _nmesh, 3))

YZ_I = np.zeros((ntime, _nmesh, _nmesh, 3))
XZ_I = np.zeros((ntime, _nmesh, _nmesh, 3))
XY_I = np.zeros((ntime, _nmesh, _nmesh, 3))

YZ_S = np.zeros((ntime, _nmesh, _nmesh, 3))
XZ_S = np.zeros((ntime, _nmesh, _nmesh, 3))
XY_S = np.zeros((ntime, _nmesh, _nmesh, 3))

# extents of the mesh
xx = np.linspace(-200, 200, _nmesh)
yy = np.linspace(-200, 200, _nmesh)
zz = np.linspace(-75, 75, _nmesh)
YZ_y, YZ_z = np.meshgrid(yy, zz)
XZ_x, XZ_z = np.meshgrid(xx, zz)
XY_x, XY_y = np.meshgrid(xx, yy)

for i in np.arange(ntime):
    Xp_B[i] = np.array([1, 0, 0])
    Yp_B[i] = np.array([0, 1, 0])
    Zp_B[i] = np.array([0, 0, 1])

    Xp_I[i] = np.dot(C_I2B[i].T, Xp_B[i])
    Yp_I[i] = np.dot(C_I2B[i].T, Yp_B[i])
    Zp_I[i] = np.dot(C_I2B[i].T, Zp_B[i])

    Xp_S[i] = np.dot(C_I2S[i], Xp_I[i])
    Yp_S[i] = np.dot(C_I2S[i], Yp_I[i])
    Zp_S[i] = np.dot(C_I2S[i], Zp_I[i])

    YZ_B[i, :, :, 0] = 0 * YZ_y
    YZ_B[i, :, :, 1] = YZ_y
    YZ_B[i, :, :, 2] = YZ_z

    XZ_B[i, :, :, 0] = XZ_x
    XZ_B[i, :, :, 1] = 0 * XZ_x
    XZ_B[i, :, :, 2] = XZ_z

    XY_B[i, :, :, 0] = XY_x
    XY_B[i, :, :, 1] = XY_y
    XY_B[i, :, :, 2] = 0 * XY_x

    for j in np.arange(_nmesh):
        # .T on rotation matrix b/c _B is in the body frame, so convert to _I
        YZ_I[i, j] = np.dot(C_I2B[i].T, YZ_B[i, j].T).T
        XZ_I[i, j] = np.dot(C_I2B[i].T, XZ_B[i, j].T).T
        XY_I[i, j] = np.dot(C_I2B[i].T, XY_B[i, j].T).T

        YZ_S[i, j] = np.dot(C_I2S[i], YZ_I[i, j].T).T
        XZ_S[i, j] = np.dot(C_I2S[i], XZ_I[i, j].T).T
        XY_S[i, j] = np.dot(C_I2S[i], XY_I[i, j].T).T


# %% Plot the body in the body frame

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

i = 0
#i = 30
#i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

#mlab.points3d(pfe_I[i, :, 0] - Ro_I[i, 0],
#              pfe_I[i, :, 1] - Ro_I[i, 1],
#              pfe_I[i, :, 2] - Ro_I[i, 2],
#                  color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)


Yp_quiv = mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_I[i, :, :, 0], YZ_I[i, :, :, 1], YZ_I[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_I[i, :, :, 0], XZ_I[i, :, :, 1], XZ_I[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
XY_mesh = mlab.mesh(XY_I[i, :, :, 0], XY_I[i, :, :, 1], XY_I[i, :, :, 2],
                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])
        dRo_quiv.mlab_source.set(u=[dRo_I[i, 0]],
                                 v=[dRo_I[i, 1]],
                                 w=[dRo_I[i, 2]])

        Yp_quiv.mlab_source.set(u=[Yp_I[i, 0]],
                                v=[Yp_I[i, 1]],
                                w=[Yp_I[i, 2]])
        Xp_quiv.mlab_source.set(u=[Xp_I[i, 0]],
                                v=[Xp_I[i, 1]],
                                w=[Xp_I[i, 2]])
        Zp_quiv.mlab_source.set(u=[Zp_I[i, 0]],
                                v=[Zp_I[i, 1]],
                                w=[Zp_I[i, 2]])
#        YZ_mesh.mlab_source.set(x=YZ_I[i, :, :, 0],
#                                y=YZ_I[i, :, :, 1],
#                                z=YZ_I[i, :, :, 2])
#        XZ_mesh.mlab_source.set(x=XZ_I[i, :, :, 0],
#                                y=XZ_I[i, :, :, 1],
#                                z=XZ_I[i, :, :, 2])
        XY_mesh.mlab_source.set(x=XY_I[i, :, :, 0],
                                y=XY_I[i, :, :, 1],
                                z=XY_I[i, :, :, 2])

#        for ii in np.arange(3):
#            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
#                                       v=nframe[i, ii, 1],
#                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()


# %% Plot the body in the BODY FRAME

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

i = 0
#i = 30
#i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

#mlab.points3d(pfe_I[i, :, 0] - Ro_I[i, 0],
#              pfe_I[i, :, 1] - Ro_I[i, 1],
#              pfe_I[i, :, 2] - Ro_I[i, 2],
#              color=(.85, .85, .85), scale_factor=10, resolution=64)

pts = mlab.points3d(pfe_B[i, :, 0], pfe_B[i, :, 1], pfe_B[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_B[i, :, :, 0], foils_B[i, :, :, 1], foils_B[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)


Yp_quiv = mlab.quiver3d(Yp_B[i, 0], Yp_B[i, 1], Yp_B[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_B[i, 0], Xp_B[i, 1], Xp_B[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_B[i, 0], Zp_B[i, 1], Zp_B[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_I[i, :, :, 0], YZ_I[i, :, :, 1], YZ_I[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_I[i, :, :, 0], XZ_I[i, :, :, 1], XZ_I[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
XY_mesh = mlab.mesh(XY_B[i, :, :, 0], XY_B[i, :, :, 1], XY_B[i, :, :, 2],
                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_B[i, :, :, 0],
                             y=foils_B[i, :, :, 1],
                             z=foils_B[i, :, :, 2],
                             scalars=foil_color[i])
        pts.mlab_source.set(x=pfe_B[i, :, 0],
                            y=pfe_B[i, :, 1],
                            z=pfe_B[i, :, 2])
        dRo_quiv.mlab_source.set(u=[dRo_B[i, 0]],
                                 v=[dRo_B[i, 1]],
                                 w=[dRo_B[i, 2]])

        Yp_quiv.mlab_source.set(u=[Yp_B[i, 0]],
                                v=[Yp_B[i, 1]],
                                w=[Yp_B[i, 2]])
        Xp_quiv.mlab_source.set(u=[Xp_B[i, 0]],
                                v=[Xp_B[i, 1]],
                                w=[Xp_B[i, 2]])
        Zp_quiv.mlab_source.set(u=[Zp_B[i, 0]],
                                v=[Zp_B[i, 1]],
                                w=[Zp_B[i, 2]])
#        YZ_mesh.mlab_source.set(x=YZ_B[i, :, :, 0],
#                                y=YZ_B[i, :, :, 1],
#                                z=YZ_B[i, :, :, 2])
#        XZ_mesh.mlab_source.set(x=XZ_B[i, :, :, 0],
#                                y=XZ_B[i, :, :, 1],
#                                z=XZ_B[i, :, :, 2])
        XY_mesh.mlab_source.set(x=XY_B[i, :, :, 0],
                                y=XY_B[i, :, :, 1],
                                z=XY_B[i, :, :, 2])

#        for ii in np.arange(3):
#            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
#                                       v=nframe[i, ii, 1],
#                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()




# %% Plot body, frame, planes in STRAIGHTENED frame

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

i = 0
#i = 30
i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

#mlab.points3d(pfe_I[i, :, 0] - Ro_I[i, 0],
#              pfe_I[i, :, 1] - Ro_I[i, 1],
#              pfe_I[i, :, 2] - Ro_I[i, 2],
#                  color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_S[i, 0], dRo_S[i, 1], dRo_S[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_Sc[i, :, :, 0], foils_Sc[i, :, :, 1], foils_Sc[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)


Yp_quiv = mlab.quiver3d(Yp_S[i, 0], Yp_S[i, 1], Yp_S[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_S[i, 0], Xp_S[i, 1], Xp_S[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_S[i, 0], Zp_S[i, 1], Zp_S[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

YZ_mesh = mlab.mesh(YZ_S[i, :, :, 0], YZ_S[i, :, :, 1], YZ_S[i, :, :, 2],
                    color=bmap[2], opacity=.25)
XZ_mesh = mlab.mesh(XZ_S[i, :, :, 0], XZ_S[i, :, :, 1], XZ_S[i, :, :, 2],
                    color=bmap[1], opacity=.25)
XY_mesh = mlab.mesh(XY_S[i, :, :, 0], XY_S[i, :, :, 1], XY_S[i, :, :, 2],
                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(Ro_I[:, 0], Ro_I[:, 1], Ro_I[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
#    mlab.plot3d(R_I[i, :, 0], R_I[i, :, 1], R_I[i, :, 2],
#                color=bmap[2], tube_radius=3)
    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                  color=(.9, .9, .9), scale_factor=5, resolution=32)

    mlab.mesh(foils_Ic[i, :, :, 0] + Ro_I[i, 0],
              foils_Ic[i, :, :, 1] + Ro_I[i, 1],
              foils_Ic[i, :, :, 2] + Ro_I[i, 2],
              scalars=foil_color[i], colormap='YlGn', opacity=1,
              vmin=0, vmax=1)

mlab.orientation_axes()
fig.scene.isometric_view()



# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#
dRo_quiv = mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

#mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2], scale_factor=.01,
#              color=bmap[1], mode='arrow', resolution=64)


body = mlab.mesh(foils_B[i, :, :, 0], foils_B[i, :, :, 1], foils_B[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)


#mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
#              color=bmap[1], mode='arrow', resolution=64)
#mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
#              color=bmap[2], mode='arrow', resolution=64)
#mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
#              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_B[i, :, :, 0], YZ_B[i, :, :, 1], YZ_B[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_B[i, :, :, 0], XZ_B[i, :, :, 1], XZ_B[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
#XY_mesh = mlab.mesh(XY_B[i, :, :, 0], XY_B[i, :, :, 1], XY_B[i, :, :, 2],
#                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_B[i, :, :, 0],
                             y=foils_B[i, :, :, 1],
                             z=foils_B[i, :, :, 2],
                             scalars=foil_color[i])
        dRo_quiv.mlab_source.set(u=[dRo_B[i, 0]],
                                 v=[dRo_B[i, 1]],
                                 w=[dRo_B[i, 2]])

#        Yp_quiv.mlab_source.set(u=[Yp_I[i, 0]],
#                                v=[Yp_I[i, 1]],
#                                w=[Yp_I[i, 2]])
#        Xp_quiv.mlab_source.set(u=[Xp_I[i, 0]],
#                                v=[Xp_I[i, 1]],
#                                w=[Xp_I[i, 2]])
#        Zp_quiv.mlab_source.set(u=[Zp_I[i, 0]],
#                                v=[Zp_I[i, 1]],
#                                w=[Zp_I[i, 2]])
#        YZ_mesh.mlab_source.set(x=YZ_I[i, :, :, 0],
#                                y=YZ_I[i, :, :, 1],
#                                z=YZ_I[i, :, :, 2])
#        XZ_mesh.mlab_source.set(x=XZ_I[i, :, :, 0],
#                                y=XZ_I[i, :, :, 1],
#                                z=XZ_I[i, :, :, 2])
#        XY_mesh.mlab_source.set(x=XY_I[i, :, :, 0],
#                                y=XY_I[i, :, :, 1],
#                                z=XY_I[i, :, :, 2])

#        for ii in np.arange(3):
#            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
#                                       v=nframe[i, ii, 1],
#                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

#i = 189
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230
i = 0

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#
#dRo_quiv = mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2],
#                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
#                         resolution=64)
#
##mlab.quiver3d(dRo_B[i, 0], dRo_B[i, 1], dRo_B[i, 2], scale_factor=.01,
##              color=bmap[1], mode='arrow', resolution=64)
#
#
#body = mlab.mesh(foils_B[i, :, :, 0], foils_B[i, :, :, 1], foils_B[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)


#mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
#              color=bmap[1], mode='arrow', resolution=64)
#mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
#              color=bmap[2], mode='arrow', resolution=64)
#mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
#              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_B[i, :, :, 0], YZ_B[i, :, :, 1], YZ_B[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_B[i, :, :, 0], XZ_B[i, :, :, 1], XZ_B[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
#XY_mesh = mlab.mesh(XY_B[i, :, :, 0], XY_B[i, :, :, 1], XY_B[i, :, :, 2],
#                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])
#        dRo_quiv.mlab_source.set(u=[dRo_Ic[i, 0]],
#                                 v=[dRo_Ic[i, 1]],
#                                 w=[dRo_Ic[i, 2]])
        yield
manim = anim()
mlab.show()


# %%

mlab.figure()

X = np.array([0, 1, 0, 1, .5])
Y = np.array([0, 0, 1, 1, .5])
Z = np.array([1, 1, 1, 1, 1])

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")
mlab.show()





# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 189
#i = 60
#i = 10
#i = 220
i = 230

mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2], scale_factor=.0075,
              color=(0, 0, 0), mode='arrow', resolution=64)

mlab.quiver3d(Yp[i, 0], Yp[i, 1], Yp[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
mlab.quiver3d(Xp[i, 0], Xp[i, 1], Xp[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
mlab.quiver3d(Zp[i, 0], Zp[i, 1], Zp[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

mlab.quiver3d(V0[i, 0], V0[i, 1], V0[i, 2], scale_factor=75,
              color=bmap[1], mode='2darrow', resolution=64)
mlab.quiver3d(V1[i, 0], V1[i, 1], V1[i, 2], scale_factor=75,
              color=bmap[2], mode='2darrow', resolution=64)
mlab.quiver3d(V2[i, 0], V2[i, 1], V2[i, 2], scale_factor=75,
              color=bmap[0], mode='2darrow', resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#mlab.orientation_axes()
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

fig, ax = plt.subplots()
ax.plot(times, 100 * Sfrac[:, 2])
#ax.plot(times, 100 * Sfrac)
ax.set_ylim(ymin=0)
#ax.set_ylim((0, 100))
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(times, planar_fit_error)
ax.set(xlabel='time (s)', ylabel='planar error (mm)')
sns.despine()
fig.set_tight_layout(True)



# %%

# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 189
#i = 60
#i = 10
#i = 220
#i = 230

mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2], scale_factor=.0075,
              color=(0, 0, 0), mode='arrow', resolution=64)

mlab.quiver3d(Yp[i, 0], Yp[i, 1], Yp[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
mlab.quiver3d(Xp[i, 0], Xp[i, 1], Xp[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
mlab.quiver3d(Zp[i, 0], Zp[i, 1], Zp[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#mlab.orientation_axes()
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %% Project snake into the normal plane

spl_c_Np = np.zeros((ntime, nspl, 3))
spl_c_Yp = np.zeros((ntime, nspl, 3))
spl_c_Xp = np.zeros((ntime, nspl, 3))

for i in np.arange(ntime):
    # proj = (spl_c[i].T - np.dot(spl_c[i], Np[i])).T
    for j in np.arange(nspl):
        proj = spl_c[i, j] - np.dot(spl_c[i, j], Np[i]) * Np[i]
        spl_c_Np[i, j] = proj

        proj = spl_c[i, j] - np.dot(spl_c[i, j], Yp[i]) * Yp[i]
        spl_c_Yp[i, j] = proj

        proj = spl_c[i, j] - np.dot(spl_c[i, j], Xp[i]) * Xp[i]
        spl_c_Xp[i, j] = proj


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=25,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 189
i = 60
#i = 230

mlab.quiver3d(Yp[i, 0], Yp[i, 1], Yp[i, 2], scale_factor=50,
              color=bmap[1], mode='arrow', resolution=64)
mlab.quiver3d(Xp[i, 0], Xp[i, 1], Xp[i, 2], scale_factor=50,
              color=bmap[2], mode='arrow', resolution=64)
mlab.quiver3d(Np[i, 0], Np[i, 1], Np[i, 2], scale_factor=50,
              color=bmap[0], mode='arrow', resolution=64)

body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#mlab.plot3d(spl_c_Np[i, :, 0], spl_c_Np[i, :, 1],
#            spl_c_Np[i, :, 2], color=bmap[0], tube_radius=3)
#mlab.plot3d(spl_c_Yp[i, :, 0], spl_c_Yp[i, :, 1],
#            spl_c_Yp[i, :, 2], color=bmap[1], tube_radius=3)
#mlab.plot3d(spl_c_Xp[i, :, 0], spl_c_Xp[i, :, 1],
#            spl_c_Xp[i, :, 2], color=bmap[2], tube_radius=3)

mlab.orientation_axes()
fig.scene.isometric_view()
mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)


# %% construct rotation matrix to get snake in Yhat, Np, Xp frame

# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix

Rps = np.zeros((ntime, 3, 3))
spl_c_rot = np.zeros((ntime, nspl, 3))
foils_rot = np.zeros_like(foils)
Cdir_rot = np.zeros_like(Cdir)
Bdir_rot = np.zeros_like(Bdir)

for i in np.arange(ntime):
    Rp = np.zeros((3, 3))
    Rp[:, 0] = Xp[i]
    Rp[:, 1] = Yp[i]
    Rp[:, 2] = Np[i]

    Rp = Rp.T

    Rps[i] = Rp

    # rotate the spline and foil for plotting
#    spl_c_rot[i] = np.dot(spl_c[i], Rp)
#    foils_rot[i] = np.dot(foils[i], Rp)
    spl_c_rot[i] = np.dot(Rp, spl_c[i].T).T
#    foils_rot[i] = np.dot(Rp, foils[i].T).T
    for j in np.arange(nspl):
        foils_rot[i, j] = np.dot(Rp, foils[i, j].T).T
        Cdir_rot[i, j] = np.dot(Rp, Cdir[i, j])
        Bdir_rot[i, j] = np.dot(Rp, Bdir[i, j])


back_bend = np.zeros((ntime, nspl))
lateral_bend = np.zeros((ntime, nspl))
for i in np.arange(ntime):
    for j in np.arange(1, nspl):
        # back bend and lateral bend angles
        # http://stackoverflow.com/a/10145056
        c0 = Cdir[i, j - 1]
        c1 = Cdir_rot[i, j]
        b0 = Bdir[i, j - 1]
        b1 = Bdir[i, j]

        sin_b = np.cross(b0, b1)
        cos_b = np.dot(b0, b1)
        sin_l = np.cross(c0, c1)
        cos_l = np.dot(c0, c1)

        # so that the sign is correct
        sign_b = np.sign(np.dot(sin_b, c1))
        sign_l = np.sign(np.dot(sin_l, b1))

        sin_b_mag = np.linalg.norm(sin_b)
        sin_l_mag = np.linalg.norm(sin_l)

        back_bend[i, j] = sign_b * np.arctan2(sin_b_mag, cos_b)
        lateral_bend[i, j] = sign_l * np.arctan2(sin_l_mag, cos_l)


# %%

i = 0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)

for i in np.arange(0, ntime, 40):
    ax1.plot(ts, np.rad2deg(lateral_bend[i]), c=bmap[0])
    ax2.plot(ts, np.rad2deg(back_bend[i]), c=bmap[1])
sns.despine()
#    plt.cla()
#    plt.draw()
fig.set_tight_layout(True)


SS, TT = np.meshgrid(ts, times)

fig, ax = plt.subplots()
vmax = .75 * np.rad2deg(np.max(np.abs(lateral_bend)))
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
vmax = .75 * np.rad2deg(np.max(np.abs(back_bend)))
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


# %% Calculate velocity of spline segments

v_spl = np.zeros((ntime, nspl, 3))
a_spl = np.zeros((ntime, nspl, 3))

v_spl_c = np.zeros((ntime, nspl, 3))  # in CoM frame
a_spl_c = np.zeros((ntime, nspl, 3))  # in CoM frame

v_spl_c_rot = np.zeros((ntime, nspl, 3))  # rotated CoM frame
a_spl_c_rot = np.zeros((ntime, nspl, 3))  # rotated CoM frame

for j in np.arange(nspl):
    vv, aa = smoothing.findiff(spl[:, j], dt)
    v_spl[:, j] = vv
    a_spl[:, j] = aa

    vv, aa = smoothing.findiff(spl_c[:, j], dt)
    v_spl_c[:, j] = vv
    a_spl_c[:, j] = aa

    vv, aa = smoothing.findiff(spl_c_rot[:, j], dt)
    v_spl_c_rot[:, j] = vv
    a_spl_c_rot[:, j] = aa


# %% Velocity of the center of mass
#TODO we should probably filter this first

v_com, a_com = smoothing.findiff(com, dt)


# %% Verify the smoothness of the velocities

j = 0

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(times, spl[:, j, 0])
ax2.plot(times, spl[:, j, 1])
ax3.plot(times, spl[:, j, 2])
sns.despine()
fig.set_tight_layout(True)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(times, v_spl[:, j, 0])
ax2.plot(times, v_spl[:, j, 1])
ax3.plot(times, v_spl[:, j, 2])
sns.despine()
fig.set_tight_layout(True)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
for j in [0, -1]:
    ax1.plot(times, v_spl_c[:, j, 0])
    ax2.plot(times, v_spl_c[:, j, 1])
    ax3.plot(times, v_spl_c[:, j, 2])
sns.despine()
fig.set_tight_layout(True)


# %% Perform SVD to get singular values for different directions

Sfrac_rot = np.zeros((ntime, 3))

for i in np.arange(ntime):
    idx = vent_idx_spls[i] + 1
    M = spl_c_rot[i, :idx]
    #Mw = (M.T * weights[i]).T  # weighted points
    Mw = (M.T * weights).T  # weighted points
    U, S, V = svd(Mw)

    Sfrac_rot[i] = S / S.sum()


# %% Plot the singular values

fig, ax = plt.subplots()
#ax.plot(times, 100 * Sfrac)
ax.plot(times, 100 * Sfrac_rot)
ax.set_ylim(0, 100)
ax.set_xlabel('time (s)')
ax.set_ylabel('singluar values (% of variance)')
sns.despine()
fig.set_tight_layout(True)

fig.savefig('../Figures/singular_values_comoving_413_91.pdf', transparent=True,
            bbox_inches='tight')


# %% Angle between the comoving plane normal and gravity

Xhat = np.r_[1, 0, 0]
Zhat = np.r_[0, 0, 1]

ang_Np = np.zeros(ntime)
sign_Np = np.zeros(ntime)

for i in np.arange(ntime):
    ang_Np[i] = np.rad2deg(np.arccos(np.dot(Zhat, Np[i])))

    # this doesn't quite work yet
    # correct the sign of the angle (+ if has component in +X direction)
    sign_tmp = np.dot(Xhat, Np[i])
    if sign_tmp > 0:
        sign_Np[i] = 1
    else:
        sign_Np[i] = -1

fig, ax = plt.subplots()
ax.plot(times, ang_Np)
#ax.plot(times, sign_Np * ang_Np)
ax.set_xlabel('time (s)')
ax.set_ylabel('angle offset of co-moving plane')
sns.despine()

fig.canvas.draw()
# add degree symbol to angles
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

fig.set_tight_layout(True)


fig.savefig('../Figures/angle_offset_comoving_413_91.pdf', transparent=True,
            bbox_inches='tight')



# %% Animation of rotated body and body coordinate system

spl_c_plane = spl_c_rot.copy()


fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 0

body_rot = mlab.mesh(foils_rot[i, :, :, 0], foils_rot[i, :, :, 1],
                 foils_rot[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

# chat
cv_quiv = mlab.quiver3d(spl_c_plane[i, :, 0], spl_c_plane[i, :, 1],
                        spl_c_plane[i, :, 2],
                        Cdir_rot[i, :, 0], Cdir_rot[i, :, 1],
                        Cdir_rot[i, :, 2],
                        color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
# bhat
bv_quiv = mlab.quiver3d(spl_c_plane[i, :, 0], spl_c_plane[i, :, 1],
                        spl_c_plane[i, :, 2],
                        Bdir_rot[i, :, 0], Bdir_rot[i, :, 1],
                        Bdir_rot[i, :, 2],
                        color=bmap[2], mode='arrow', resolution=64, scale_factor=25)


# %%

@mlab.animate(delay=200)
def anim_body():
    for i in np.arange(ntime):
        body_rot.mlab_source.set(x=foils_rot[i, :, :, 0],
                                 y=foils_rot[i, :, :, 1],
                                 z=foils_rot[i, :, :, 2],
                                 scalars=foil_color[i])
        cv_quiv.mlab_source.set(x=spl_c_plane[i, :, 0], y=spl_c_plane[i, :, 1],
                                z=spl_c_plane[i, :, 2],
                                u=Cdir_rot[i, :, 0], v=Cdir_rot[i, :, 1],
                                w=Cdir_rot[i, :, 2])
        bv_quiv.mlab_source.set(x=spl_c_plane[i, :, 0], y=spl_c_plane[i, :, 1],
                                z=spl_c_plane[i, :, 2],
                                u=Bdir_rot[i, :, 0], v=Bdir_rot[i, :, 1],
                                w=Bdir_rot[i, :, 2])
        yield

manim = anim_body()
mlab.show()


# %% Plot in the rotated frame

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 189
#i = 60

#mlab.quiver3d(Yhat[0], Yhat[1], Yhat[2], scale_factor=50,
#              color=bmap[1], mode='arrow', resolution=64)
#mlab.quiver3d(Xp[i, 0], Xp[i, 1], Xp[i, 2], scale_factor=50,
#              color=bmap[2], mode='arrow', resolution=64)
#mlab.quiver3d(Np[i, 0], Np[i, 1], Np[i, 2], scale_factor=50,
#              color=bmap[0], mode='arrow', resolution=64)

body_iner = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=.5,
                 vmin=0, vmax=1)

body_rot = mlab.mesh(foils_rot[i, :, :, 0], foils_rot[i, :, :, 1],
                 foils_rot[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#mlab.plot3d(spl_c_rot[i, :, 0], spl_c_rot[i, :, 1],
#            0 * spl_c_rot[i, :, 2], color=bmap[0], tube_radius=3)
#
#mlab.plot3d(0 * spl_c_rot[i, :, 0], spl_c_rot[i, :, 1],
#            spl_c_rot[i, :, 2], color=bmap[2], tube_radius=3)

#mlab.plot3d(spl_c_rot[i, :, 0], spl_c_rot[i, :, 1],
#            spl_c_rot[i, :, 2], color=bmap[0], tube_radius=3)

#mlab.plot3d(spl_c_Np[i, :, 0], spl_c_Np[i, :, 1],
#            spl_c_Np[i, :, 2], color=bmap[0], tube_radius=3)
#mlab.plot3d(spl_c_Yp[i, :, 0], spl_c_Yp[i, :, 1],
#            spl_c_Yp[i, :, 2], color=bmap[1], tube_radius=3)
#mlab.plot3d(spl_c_Xp[i, :, 0], spl_c_Xp[i, :, 1],
#            spl_c_Xp[i, :, 2], color=bmap[2], tube_radius=3)


# %%

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        body_iner.mlab_source.set(x=foils[i, :, :, 0],
                                  y=foils[i, :, :, 1],
                                  z=foils[i, :, :, 2],
                                  scalars=foil_color[i])

        body_rot.mlab_source.set(x=foils_rot[i, :, :, 0],
                                 y=foils_rot[i, :, :, 1],
                                 z=foils_rot[i, :, :, 2],
                                 scalars=foil_color[i])

        yield

manim = anim_body()
mlab.show()


# %%
sk = 1
for i in np.arange(ntime)[::20]:
    # that
#    mlab.quiver3d(spl[i, ::sk, 0], spl[i, ::sk, 1], spl[i, ::sk, 2],
#              Tdir[i, ::sk, 0], Tdir[i, ::sk, 1], Tdir[i, ::sk, 2],
#              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # chat
    mlab.quiver3d(spl[i, ::sk, 0], spl[i, ::sk, 1], spl[i, ::sk, 2],
              Cdir[i, ::sk, 0], Cdir[i, ::sk, 1], Cdir[i, ::sk, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(spl[i, ::sk, 0], spl[i, ::sk, 1], spl[i, ::sk, 2],
              Bdir[i, ::sk, 0], Bdir[i, ::sk, 1], Bdir[i, ::sk, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

mlab.orientation_axes()
fig.scene.isometric_view()


# %%

i = 0

# transverse component
fig, ax = plt.subplots()
ax.plot(s_coord[i], spl_c_rot[i, :, 0], c=bmap[0])
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# fore-aft component
fig, ax = plt.subplots()
ax.plot(s_coord[i], spl_c_rot[i, :, 1], c=bmap[0])
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# vertical component
fig, ax = plt.subplots()
ax.plot(s_coord[i], spl_c_rot[i, :, 2], c=bmap[0])
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(spl_c_rot[i, :, 0], spl_c_rot[i, :, 1], c=bmap[0])
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# tangent angle
tang_ang = np.arctan2(spl_c_rot[i, :, 1], spl_c_rot[i, :, 0])
tang_ang[tang_ang < 0] += 2 * np.pi
tang_ang = np.rad2deg(tang_ang)

fig, ax = plt.subplots()
ax.plot(s_coord[i] / SVL, tang_ang)
sns.despine()
fig.set_tight_layout(True)


#fig, ax = plt.subplots()
#ax.plot(spl_c_rot[:, 1], spl_c_rot[:, 2], c=bmap[2])
#ax.set_aspect('equal', adjustable='box')
#sns.despine()
#fig.set_tight_layout(True)


# %% Plot Cdir and Bdir on foil body to verify directions

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
i = 110

#head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
#                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 scalars=foil_color[i], colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 color=bmap[1], opacity=.5)

#mlab.points3d(pfe_c[i, :, 0], pfe_c[i, :, 1],
#            pfe_c[i, :, 2], color=bmap[2], scale_factor=15)

sk = 10
# chat
mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
          Cdir[i, ::sk, 0], Cdir[i, ::sk, 1], Cdir[i, ::sk, 2],
          color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
# bhat
mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
          Bdir[i, ::sk, 0], Bdir[i, ::sk, 1], Bdir[i, ::sk, 2],
          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

#mlab.orientation_axes()
fig.scene.isometric_view()


# %% Plot velocity vectors on the body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
i = 110

#head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
#                     scale_factor=.015, resolution=16, opacity=.5)
head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
#                 color=bmap[1], opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

sk = 1
mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
              v_spl[i, ::sk, 0], v_spl[i, ::sk, 1], v_spl[i, ::sk, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=.01)

# plot the original velocity vectors
mlab.quiver3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1], pf_rot_c[i, :, 2],
              vf_rot[i, :, 0], vf_rot[i, :, 1], vf_rot[i, :, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=.01)

#mlab.points3d(pfe_c[i, :, 0], pfe_c[i, :, 1],
#            pfe_c[i, :, 2], color=bmap[2], scale_factor=15)

#mlab.orientation_axes()
fig.scene.isometric_view()


# %% Functions for aerodynamic forces

from numba import jit

@jit(nopython=True)
def nbcross(a, b):
    """A faster cross product. 'a' and 'b' are both length 3 arrays.
    """
    i = a[1] * b[2] - a[2] * b[1]
    j = -a[0] * b[2] + a[2] * b[0]
    k = a[0] * b[1] - a[1] * b[0]
    return i, j, k


def aero_forces_old(Tdir, Cdir, Bdir, dRis, ds, c, rho, aero_interp):
    """Aerodynamic forces or each segment. This is a modified version
    from the simulation code.

    Parameters
    ----------
    Tdir : array, size (nbody, 3)
        tangent vector in inertial coordinates
    Cdir : array, size (nbody, 3)
        chord vector in inertial coordinates
    Bdir : array, size (nbody, 3)
        backbone vector in inertial coordinates
    dRis : array, size (nbody, 3)
        velocities of each mass in mm/s, expressed in the inertial frame
    ds : array, size (nbody)
        length of each piece in mm of spline body
    c : array, size (nbody)
        chord length in mm of each piece of mass
    aero_interp : function
        function that returns force coeffients when passed
        aoa, U, c, and nu (optional)

    Returns
    -------
    Fl : array, size (nbody)
        lift force in N
    Fd : array, size (nbody)
        drag force in N
    dRiBCs : array, size (nbody)
        velocity in the BC-plane of the snake airfoil, expressed in
        the inertial frame
    aoas : array, size (nbody)
        angles of attack in radians
    Res : array, size (nbody)
        Reynolds number
    betas : array, size (nbody)
        sweep angle in radians
    dRiTCs : array, size (nbody)
        velocity in the TC-plane of the snake airfoil, expressed in
        the inertial frame
    """

    nbody = dRis.shape[0]
    Fl = np.zeros((nbody, 3))
    Fd = np.zeros((nbody, 3))
    dRiBCs = np.zeros((nbody, 3))
    aoas = np.zeros(nbody)
    Res = np.zeros(nbody)
    betas = np.zeros(nbody)  # sweep angle
    dRiTCs = np.zeros((nbody, 3))  # velocity along body (TODO: check this)

    # we need consistent units -- meters
    mm2m = .001  # conversion from mm to m (length unit of c, ds, dRi)
    c = mm2m * c.copy()
    ds = mm2m * ds.copy()
    dRis = mm2m * dRis.copy()

    for j in np.arange(nbody):
        dRi = dRis[j]

        # coordinate along the body
        Ti = Tdir[j]
        Ci = Cdir[j]
        Bi = Bdir[j]

        dRiT = np.dot(dRi, Ti) * Ti  # the * Ti makes it a vector
        dRiBC = dRi - dRiT
        Ui = np.linalg.norm(dRiBC)  # reduced velocity in plane
        Uti = np.linalg.norm(dRi)  # total velocity hitting mass (for Re calc)

        # TODO: should fix this, so that the other values are filled in
        # well, if Ui is zero, then everything else should be zero
        if Ui == 0:
            Fl[j] = 0
            Fd[j] = 0
            continue

        uhat = dRiBC / Ui
        Dh = -uhat
        Lh = np.array(nbcross(Ti, Dh))
        # Lh = np.cross(Ti, Dh)

        # angle of velocity in BC coordinate system
        cos_c = np.dot(uhat, Ci)
        cos_b = np.dot(uhat, Bi)
        if cos_c >= 0 and cos_b < 0:
            aoa = np.arccos(np.dot(uhat, Ci))
            # Lh = -np.array(nbcross(Ti, uhat))
        elif cos_c >= 0 and cos_b >= 0:
            aoa = -np.arccos(np.dot(uhat, Ci))
            Lh = -Lh
            # Lh = np.array(nbcross(Ti, uhat))
        elif cos_c < 0 and cos_b >= 0:
            aoa = -np.arccos(np.dot(uhat, -Ci))
            # Lh = -np.array(nbcross(Ti, uhat))
        elif cos_c < 0 and cos_b < 0:
            aoa = np.arccos(np.dot(uhat, -Ci))
            Lh = -Lh
            # Lh = np.array(nbcross(Ti, uhat))

        # now get the forces
        cl, cd, clcd, Re = aero_interp(aoa, Uti, c[j])
        dynP = .5 * rho * Ui**2 * ds[j] * c[j]
        Flj = dynP * cl * Lh
        Fdj = dynP * cd * Dh

        # sweep angle beta
        dRiB = np.dot(dRi, Bi) * Bi
        dRiTC = dRi - dRiB  # velocity in T-C plane
        Ui = np.linalg.norm(dRiTC)  # reduced velocity in plane
        bTi = np.dot(dRiTC, Ti) / Ui
        # beta_di = np.rad2deg(np.arccos(bTi)) - 90
        beta = np.arccos(bTi) - np.pi / 2

        # store the values (back into mm)
        dRiBCs[j] = dRiBC / mm2m
        aoas[j] = aoa
        Res[j] = Re
        betas[j] = beta
        dRiTCs[j] = dRiTC / mm2m
        Fl[j] = Flj  # N
        Fd[j] = Fdj  # N

    return Fl, Fd, dRiBCs, aoas, Res, betas, dRiTCs


# %%

def dot(a, b):
    """Dot product for two (n x 3) arrays. This is a dot product
    for two vectors.

    Example from the aerodynamics function:
    a1 = np.array([np.dot(dR[i], Tv[i]) for i in np.arange(nbody)])
    a2 = np.diag(np.inner(dR, Tv))
    a3 = np.sum(dR * Tv, axis=1)
    np.allclose(a1, a2)
    np.allclose(a1, a3)
    """
    return np.sum(a * b, axis=1)


#@jit(nopython=True)
#def nbcross(a, b):
#    """A faster cross product. 'a' and 'b' are both length 3 arrays.
#    """
#    i = a[1] * b[2] - a[2] * b[1]
#    j = -a[0] * b[2] + a[2] * b[0]
#    k = a[0] * b[1] - a[1] * b[0]
#    return i, j, k
#
#
#def cross(a, b):
#    """A faster cross product for two n x 3 arrays.
#
#    a = np.random.rand(3 * 300).reshape(-1, 3)
#    b = np.random.rand(3 * 300).reshape(-1, 3)
#
#    %timeit np.cross(a, b)  # 10000 loops, best of 3: 30.9 µs per loop
#    %timeit np.array(sim.nbcross(a.T, b.T)).T  # 100000 loops, best of 3: 6.04 µs per loop
#    """
#
#    return np.array(nbcross(a.T, b.T)).T


def aero_forces(Tv, Cv, Bv, dR, ds, c, rho, aero_interp, full_out=False):
    """Aerodynamic forces or each segment.

    Parameters
    ----------
    Tv : array, size (nbody, 3)
        tangent vector in interial frame
    Tv : array, size (nbody, 3)
        chord vector in inertial frame
    Tv : array, size (nbody, 3)
        backbone vector in inertial frame
#    C : array, size (3, 3)
#        rotation matrix at the current time step
    dR : array, size (nbody, 3)
        velocities of each mass, expressed in the inertial frame
    ds : float
        length of each piece of mass
    c : array, size (nbody)
        chord length in m of each piece of mass
    rho : float
        density of air in kg/m^3. A good value is rho = 1.165  # 30 C
    aero_interp : function
        function that returns force coeffients when passed
        aoa, U, c, and nu (optional)

    TODO UPDATE THE RETURN DICTIONARY
    Returns
    -------
    Fl : array, size (nbody)
        lift force in N
    Fd : array, size (nbody)
        drag force in N
    dRiBCs : array, size (nbody)
        velocity in the BC-plane of the snake airfoil, expressed in
        the inertial frame
    aoas : array, size (nbody)
        angles of attack in radians
    Res : array, size (nbody)
        Reynolds number
    """

    nbody = dR.shape[0]

#    # body coordinate system in intertial frame
#    Tv = rotate(C.T, tv)
#    Cv = rotate(C.T, cv)
#    Bv = rotate(C.T, bv)

    # we need consistent units -- meters
    mm2m = .001  # conversion from mm to m (length unit of c, ds, dRi)
    c = mm2m * c.copy()
    ds = mm2m * ds.copy()
    dR = mm2m * dR.copy()

    # velocity components parallel and perpendicular to arifoil
    dR_T = (dot(dR, Tv) * Tv.T).T  # dR_T = dot(dR, Tv) * Tv
    dR_BC = dR - dR_T  # velocity in B-C plan

    U_BC = np.linalg.norm(dR_BC, axis=1)  # reduced velocity in BC plane
    U_tot = np.linalg.norm(dR, axis=1)  # total velocity hitting mass (for Re calc)

    # angle of velocity in BC coordinate system
    cos_c = dot(dR_BC, Cv) / U_BC
    cos_b = dot(dR_BC, Bv) / U_BC

    # arccos is constrainted to [-1, 1] (due to numerical error)
    rad_c = np.arccos(np.clip(cos_c, -1, 1))
    rad_b = np.arccos(np.clip(cos_b, -1, 1))
    deg_c = np.rad2deg(rad_c)
    deg_b = np.rad2deg(rad_b)

    # unit vectors for drag and lift directions
    Dh = (-dR_BC.T / U_BC).T  # -dR_BC / U_BC
    Lh = np.cross(Tv, Dh)  # np.cross(Ti, Dh)
    aoa = np.zeros(nbody)

    # chat in -xhat, bhat = chat x that, bhat in +zhat
    Q1 = (deg_c < 90) & (deg_b >= 90)  # lower right
    Q2 = (deg_c < 90) & (deg_b < 90)  # upper right
    Q3 = (deg_c >= 90) & (deg_b < 90)  # upper left
    Q4 = (deg_c >= 90) & (deg_b >= 90)  # lower left

    # get sign and value of aoa and sign of Lh vector correct
    aoa = np.zeros(nbody)
    aoa[Q1] = rad_c[Q1]
    aoa[Q2] = -rad_c[Q2]
    aoa[Q3] = rad_c[Q3] - np.pi
    aoa[Q4] = np.pi - rad_c[Q4]
    Lh[Q1] = -Lh[Q1]
    Lh[Q2] = -Lh[Q2]

    # dynamic pressure
    dynP = .5 * rho * U_BC**2
    dA = ds * c  # area of each segment

    # now calculate the forces
    cl, cd, clcd, Re = aero_interp(aoa, U_tot, c)
    Fl = (dynP * dA * cl * Lh.T).T  # Fl = dynP * cl * Lh
    Fd = (dynP * dA * cd * Dh.T).T  # Fd = dynP * cd * Dh
    Fa = Fl + Fd  # total aerodynamic force

    if full_out:
        # sweep angle beta
        dR_B = (dot(dR, Bv) * Bv.T).T  # dR_B = np.dot(dR, Bv) * Bv
        dR_TC = dR - dR_B  # velocity in T-C plane
        U_TC = np.linalg.norm(dR_TC, axis=1)  # reduced velocity in TC plane
        cos_beta = dot(dR_TC, Tv) / U_TC
        beta = np.arccos(np.clip(cos_beta, -1, 1)) - np.pi / 2

        # fraction of dynP because of simple sweep theory assumption
        dynP_frac = U_BC**2 / U_tot**2
#        dypP_dA_frac = (

        # save aerodynamic variables in a dictionary
        out = dict(Fl=Fl, Fd=Fd, Fa=Fa, dR_T=dR_T, dR_BC=dR_BC, U_BC=U_BC,
                   U_tot=U_tot, Dh=Dh, Lh=Lh, aoa=aoa, dynP=dynP,
                   cl=cl, cd=cd, clcd=clcd, Re=Re,
                   dR_B=dR_B, dR_TC=dR_TC, U_TC=U_TC, beta=beta,
                   dynP_frac=dynP_frac)

        return out

    return Fa


# %% Quantify AoA, sweep angle, aerodynamic forces

import m_aerodynamics as aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)

rho = 1.17  # kg/m^3
mm2m = .001

Fl_I = np.zeros((ntime, nspl, 3))
Fd_I = np.zeros((ntime, nspl, 3))
Fa_I = np.zeros((ntime, nspl, 3))
Ml_I = np.zeros((ntime, nspl, 3))
Md_I = np.zeros((ntime, nspl, 3))
Ma_I = np.zeros((ntime, nspl, 3))
Fl_B = np.zeros((ntime, nspl, 3))
Fd_B = np.zeros((ntime, nspl, 3))
Fa_B = np.zeros((ntime, nspl, 3))
Ml_B = np.zeros((ntime, nspl, 3))
Md_B = np.zeros((ntime, nspl, 3))
Ma_B = np.zeros((ntime, nspl, 3))
Re = np.zeros((ntime, nspl))
aoa = np.zeros((ntime, nspl))
beta = np.zeros((ntime, nspl))
dynP = np.zeros((ntime, nspl))
dynP_frac = np.zeros((ntime, nspl))
dR_BC_I = np.zeros((ntime, nspl, 3))
dR_TC_I = np.zeros((ntime, nspl, 3))
U_BC_I = np.zeros((ntime, nspl))
U_TC_I = np.zeros((ntime, nspl))
cl = np.zeros((ntime, nspl))
cd = np.zeros((ntime, nspl))
clcd = np.zeros((ntime, nspl))

for i in np.arange(ntime):
    # aerodynamic forces, angles
    out = aero_forces(Tdir_I[i], Cdir_I[i], Bdir_I[i], dR_I[i],
                      spl_ds[i], chord_spl[i], rho, aero_interp,
                      full_out=True)

    # store the values
    Fl_I[i] = out['Fl']
    Fd_I[i] = out['Fd']
    Fa_I[i] = out['Fa']
    Ml_I[i] = np.cross(R_Ic[i], Fl_I[i])  # Nmm
    Md_I[i] = np.cross(R_Ic[i], Fd_I[i])  # Nmm
    Ma_I[i] = np.cross(R_Ic[i], Fa_I[i])  # Nmm
    Re[i] = out['Re']
    aoa[i] = out['aoa']
    beta[i] = out['beta']
    dynP[i] = out['dynP']
    dynP_frac[i] = out['dynP_frac']
    dR_BC_I[i] = out['dR_BC']
    dR_TC_I[i] = out['dR_TC']
    U_BC_I[i] = out['U_BC']
    U_TC_I[i] = out['U_TC']
    cl[i] = out['cl']
    cd[i] = out['cd']
    clcd[i] = out['clcd']

    # in the body frame
    Fl_B[i] = np.dot(C_I2B[i], Fl_I[i].T).T
    Fd_B[i] = np.dot(C_I2B[i], Fd_I[i].T).T
    Fa_B[i] = np.dot(C_I2B[i], Fa_I[i].T).T
    Ml_B[i] = np.dot(C_I2B[i], Ml_I[i].T).T
    Md_B[i] = np.dot(C_I2B[i], Md_I[i].T).T
    Ma_B[i] = np.dot(C_I2B[i], Ma_I[i].T).T


# %% Check Newton's laws in INERTIAL FRAME

# check Newton's laws
weight = (mass_total_meas / 1000 * 9.81)
F_tot_I = Fa_I.sum(axis=1) + np.r_[0, 0, -weight]  # N
m_times_ddRo_I = (mass_total_meas / 1000) * (ddRo_I / 1000)

F_tot_norm_I = F_tot_I / weight
m_times_ddRo_norm_I = m_times_ddRo_I / weight

rmsd_newton_I = np.sqrt(np.mean((m_times_ddRo_I - F_tot_I)**2, axis=0))
rmsd_newton_norm_I = rmsd_newton_I / weight

fig, ax = plt.subplots()
ax.plot(times, F_tot_I)
ax.plot(times, m_times_ddRo_I)
sns.despine()
fig.set_tight_layout(True)

#fig, ax = plt.subplots(figsize=(7.5, 6.5))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))
for ax in (ax1, ax2, ax3):
    ax.axhline(0, color='gray', lw=1)
ax1.plot(times, m_times_ddRo_norm_I[:, 0], label=r'$m\ddot{R}_o$')
ax1.plot(times, F_tot_norm_I[:, 0], label=r'$\sum F$')
ax2.plot(times, m_times_ddRo_norm_I[:, 1])
ax2.plot(times, F_tot_norm_I[:, 1])
ax3.plot(times, m_times_ddRo_norm_I[:, 2])
ax3.plot(times, F_tot_norm_I[:, 2])
ax1.legend(loc='lower left')
ax3.set_xlabel('time (sec)')
ax1.set_ylabel('inertial X, RMSE = {0:.2f}'.format(rmsd_newton_norm_I[0]))
ax2.set_ylabel('inertial Y, RMSE = {0:.2f}'.format(rmsd_newton_norm_I[1]))
ax3.set_ylabel('inertial Z, RMSE = {0:.2f}'.format(rmsd_newton_norm_I[2]))
#ax1.set_ylim(-1, .4)
ax1.set_yticks([-1, -.75, -.5, -.25, 0, .25, .5])
sns.despine()
fig.set_tight_layout(True)


# %%

fig.savefig('../Figures/s_413_91/newton_forces_comparison.pdf',
            bbox_inches='tight', transparent=True)



# %% Check Newton's laws in BODY FRAME

# check Newton's laws
weight = (mass_total_meas / 1000 * 9.81)
weight_B = np.zeros((ntime, 3))
ddRo_B_temp = np.zeros_like(ddRo_I)
for i in np.arange(ntime):
    weight_B[i] = np.dot(C_I2B[i], r_[0, 0, -weight])
    ddRo_B_temp[i] = np.dot(C_I2B[i], ddRo_I[i])
F_tot_B = Fa_B.sum(axis=1) + weight_B # N
m_times_ddRo_B = (mass_total_meas / 1000) * (ddRo_B / 1000)

F_tot_norm_B = F_tot_B / weight
m_times_ddRo_norm_B = m_times_ddRo_B / weight

rmsd_newton_B = np.sqrt(np.mean((m_times_ddRo_B - F_tot_B)**2, axis=0))
rmsd_newton_norm_B = rmsd_newton_I / weight

fig, ax = plt.subplots()
ax.plot(times, F_tot_B)
ax.plot(times, m_times_ddRo_B)
sns.despine()
fig.set_tight_layout(True)

#fig, ax = plt.subplots(figsize=(7.5, 6.5))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))
for ax in (ax1, ax2, ax3):
    ax.axhline(0, color='gray', lw=1)
ax1.plot(times, m_times_ddRo_norm_B[:, 0], label=r'$m\ddot{R}_o$')
ax1.plot(times, F_tot_norm_B[:, 0], label=r'$\sum F$')
ax2.plot(times, m_times_ddRo_norm_B[:, 1])
ax2.plot(times, F_tot_norm_B[:, 1])
ax3.plot(times, m_times_ddRo_norm_B[:, 2])
ax3.plot(times, F_tot_norm_B[:, 2])
ax1.legend(loc='lower left')
ax3.set_xlabel('time (sec)')
ax1.set_ylabel('inertial X, RMSE = {0:.2f}'.format(rmsd_newton_norm_B[0]))
ax2.set_ylabel('inertial Y, RMSE = {0:.2f}'.format(rmsd_newton_norm_B[1]))
ax3.set_ylabel('inertial Z, RMSE = {0:.2f}'.format(rmsd_newton_norm_B[2]))
#ax1.set_ylim(-1, .4)
ax1.set_yticks([-1, -.75, -.5, -.25, 0, .25, .5])
sns.despine()
fig.set_tight_layout(True)


# %%

pfill_c = np.zeros_like(pfill)
for j in np.arange(nmark):
    pfill_c[:, j] = pfill[:, j] - Ro_I


# %%

fig, ax = plt.subplots()
for j in np.arange(nmark):
    ax.plot(times, pfill_c[:, j, 0])


# %%

fig, ax = plt.subplots()
for j in np.arange(nmark):
    ax.plot(times, pf_Ic[:, j, 0])


# %%

fig, ax = plt.subplots()
for j in np.arange(nmark):
    ax.plot(times, pf_Sc[:, j, 0])


# %%

k = 2

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.plot(times, pf_Sc[:, 0, k])
ax.plot(times, pf_Sc[:, vent_idx, k])


# %%

def Iten(r, m):
    """Full moment of inertia tensor.

    Parameters
    ----------
    r : array, size (nbody, 3)
        [x, y, z] coordinates of the point masses
    m : array, size (nbody)
        mass of each point

    Returns
    -------
    Ifull : array, size (3, 3)
        moment of inerta tensor
    """
    x, y, z = r.T
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y)
    Ixz = -np.sum(m * x * z)
    Iyz = -np.sum(m * y * z)
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])


def dIten_dt(r, dr, m):
    """Time-deriative of the full moment of inertia tensor.

    Parameters
    ----------
    r : array, size (nbody, 3)
        [x, y, z] coordinates of the point masses
    dr : array, size (nbody, 3)
        [dx, dy, dz] derivative of the coordinates of the point masses
    m : array, size (nbody)
        mass of each point

    Returns
    -------
    dIfull : array, size (3, 3)
        time derivative of the moment of inerta tensor
    """
    x, y, z = r.T
    dx, dy, dz = dr.T
    dIxx = np.sum(m * (2 * y * dy + 2 * z * dz))
    dIyy = np.sum(m * (2 * x * dx + 2 * z * dz))
    dIzz = np.sum(m * (2 * x * dx + 2 * y * dy))
    dIxy = -np.sum(m * (dx * y + x * dy))
    dIxz = -np.sum(m * (dx * z + x * dz))
    dIyz = -np.sum(m * (dy * z + y * dz))
    return np.array([[dIxx, dIxy, dIxz],
                     [dIxy, dIyy, dIyz],
                     [dIxz, dIyz, dIzz]])


rb1 = np.zeros((ntime, 3))
rb2 = np.zeros((ntime, 3))
vg1 = np.zeros((ntime, 3))
vg2 = np.zeros((ntime, 3))
vg3 = np.zeros((ntime, 3))
dho = np.zeros((ntime, nspl, 3))
RHS_mom = np.zeros((ntime, 3))
for i in np.arange(ntime):
    m_i = mass_spl[i]

    I_i = Iten(R_Ic[i], m_i)
    dI_i = dIten_dt(R_Ic[i], dR_Ic[i], m_i)
    omg_i = omg_I[i]
    domg_i = domg_I[i]
#    omg_i = omg_B[i]
#    domg_i = domg_B[i]
    ho_i = np.cross(R_Ic[i], (m_i * dR_Ic[i].T).T).sum(axis=0)
    dho_i = np.cross(R_Ic[i], (m_i * ddR_Ic[i].T).T).sum(axis=0)
    dho[i] = np.cross(R_B[i], (m_i * ddR_Bc[i].T).T)  #NOTE _B frame

    rb1_i = np.dot(I_i, domg_i)
    rb2_i = np.cross(omg_i, np.dot(I_i, omg_i))
    vg1_i = np.dot(dI_i, omg_i)
    vg2_i = np.cross(omg_i, ho_i)
    vg3_i = dho_i

    #TODO : rb1 seems to be corrupted with noise; "better" fit without it...
    RHS_mom[i] = rb1_i + rb2_i + vg1_i + vg2_i + vg3_i #- rb1_i  #TODO
    rb1[i] = rb1_i
    rb2[i] = rb2_i
    vg1[i] = vg1_i
    vg2[i] = vg2_i
    vg3[i] = vg3_i

# convert from mm and g to m and kg (Nm)
#RHS_mom = RHS_mom / 1000**3
#dho = dho / 1000**3
RHS_mom = RHS_mom / 1000**3 * 100  # Nmm
dho = dho / 1000**3 * 100  # Nmm

# Integrated moments
M_tot_I = Ma_I.sum(axis=1)  #  # Nmm

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))
for ax in (ax1, ax2, ax3):
    ax.axhline(0, color='gray', lw=1)
ax1.plot(times, RHS_mom[:, 0], label='RHS moment')
ax1.plot(times, M_tot_I[:, 0], label=r'$\sum M$')
ax2.plot(times, RHS_mom[:, 1])
ax2.plot(times, M_tot_I[:, 1])
ax3.plot(times, RHS_mom[:, 2])
ax3.plot(times, M_tot_I[:, 2])
ax1.legend(loc='lower left')
ax3.set_xlabel('time (sec)')
ax1.set_ylabel('inertial X (Nmm)')
ax2.set_ylabel('inertial Y')
ax3.set_ylabel('inertial Z')
#ax.set_ylim(-5, 5)
sns.despine()
fig.set_tight_layout(True)


## %%
#
#fig.savefig('../Figures/s_413_91/euler_without_Idot_domg.pdf',
#            bbox_inches='tight', transparent=True)
#
#
## %%
#
#fig.savefig('../Figures/s_413_91/euler_with_Idot_domg.pdf',
#            bbox_inches='tight', transparent=True)



# %%

domg_test_I = np.gradient(omg_I, dt, axis=0, edge_order=2)

fig, ax = plt.subplots()
ax.plot(times, domg_I)
ax.plot(times, domg_test_I)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
#ax1.plot(times, domg_I)
#ax2.plot(times, domg_B)
ax1.plot(times, omg_I)
ax2.plot(times, omg_B)


fig, ax = plt.subplots()
#ax.plot(times, domg_I - domg_B)
ax.plot(times, omg_I - omg_B)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

for var in [rb1, rb2, vg1, vg2, vg3]:
#for var in [vg3]:
    ax1.plot(times, var[:, 0])
    ax2.plot(times, var[:, 1])
    ax3.plot(times, var[:, 2])

sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord / SVL, dho[:, :, 2])
cbar = fig.colorbar(cax, ax=ax)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(6, 9))

ax1.plot(times, dho.sum(axis=1)[:, 0])
ax2.plot(times, dho.sum(axis=1)[:, 1])
ax3.plot(times, dho.sum(axis=1)[:, 2])

sns.despine()
fig.set_tight_layout(True)


# %% Dynamic pressure fraction

dynP_frac_mean = dynP_frac.mean(axis=1)
dynP_frac_std = dynP_frac.std(axis=1)

fig, ax = plt.subplots()
ax.plot(times, 100 * dynP_frac_mean)
ax.plot(times, 100 * (dynP_frac_mean - dynP_frac_std), 'b--')
ax.plot(times, 100 * (dynP_frac_mean + dynP_frac_std), 'b--')
ax.set_ylim(0, 100)
sns.despine()
fig.set_tight_layout(True)


# %% Scale by the area

dA = spl_ds * chord_spl
dA_weights = (dA.T / dA.sum(axis=1)).T

dynP_frac_scaled = dynP_frac * dA_weightsde


# %%

fig, ax = plt.subplots()
ax.plot(times, 100 * dynP_frac_mean, 'b', lw=3)
ax.fill_between(times, 100 * (dynP_frac_mean - dynP_frac_std),
                100 * (dynP_frac_mean + dynP_frac_std), color='b', alpha=.5)
ax.set_ylim(0, 100)
ax.set_xlabel('time (s)')
ax.set_ylabel('dynamic pressure fraction (mean +/- std')
sns.despine()
fig.set_tight_layout(True)

fig.savefig('../Figures/s_413_91/dynamic_pressure_fraction.pdf',
            bbox_inches='tight', transparent=True)


# %%

fig, ax = plt.subplots()
ax.plot(times, 100 * dynP_frac[:, 10], 'o')
#ax.plot(times, 100 * (dynP_frac_mean - dynP_frac_std))
#ax.plot(times, 100 * (dynP_frac_mean + dynP_frac_std))
ax.set_ylim(0, 100)
sns.despine()
fig.set_tight_layout(True)


# %% Turn velocities and forces into meshes for plotting

# in inertial frame
Lmesh_I = np.zeros((ntime, nspl, 3, 2))
Dmesh_I = np.zeros((ntime, nspl, 3, 2))
Amesh_I = np.zeros((ntime, nspl, 3, 2))
#u = np.zeros((ntime, nspl, 3, 2))
#bc = np.zeros((ntime, nspl, 3, 2))
#tc = np.zeros((ntime, nspl, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 10 * 1000

for i in np.arange(ntime):
    for j in np.arange(nspl):
        # in inertial frame
        Lmesh_I[i, j, :, 0] = R_Ic[i, j]
        Lmesh_I[i, j, :, 1] = R_Ic[i, j] + scale_forces * Fl_I[i, j]
        Dmesh_I[i, j, :, 0] = R_Ic[i, j]
        Dmesh_I[i, j, :, 1] = R_Ic[i, j] + scale_forces * Fd_I[i, j]
        Amesh_I[i, j, :, 0] = R_Ic[i, j]
        Amesh_I[i, j, :, 1] = R_Ic[i, j] + scale_forces * Fa_I[i, j]

#        u[i, j, :, 0] = spl_c[i, j]
#        u[i, j, :, 1] = spl_c[i, j] + scale_velocities * v_spl[i, j]
#
#        bc[i, j, :, 0] = spl_c[i, j]
#        bc[i, j, :, 1] = spl_c[i, j] + scale_velocities * dR_BC[i, j]
#
#        tc[i, j, :, 0] = spl_c[i, j]
#        tc[i, j, :, 1] = spl_c[i, j] + scale_velocities * dR_TC[i, j]


# %%

# %% Plot the body in the body frame

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=75,
#                      color=frame_c[ii], mode='arrow', opacity=.5, resolution=3)

i = 0
#i = 30
#i = 189
#i = 190
#i = 60
#i = 70
#i = 10
#i = 220
#i = 230

# inertial axies
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)
#

#mlab.points3d(pfe_I[i, :, 0] - Ro_I[i, 0],
#              pfe_I[i, :, 1] - Ro_I[i, 1],
#              pfe_I[i, :, 2] - Ro_I[i, 2],
#              color=(.85, .85, .85), scale_factor=10, resolution=64)

mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
              color=(.85, .85, .85), scale_factor=10, resolution=64)

dRo_quiv = mlab.quiver3d(dRo_I[i, 0], dRo_I[i, 1], dRo_I[i, 2],
                         scale_factor=.01, color=(0, 0, 0), mode='arrow',
                         resolution=64)

body = mlab.mesh(foils_Ic[i, :, :, 0], foils_Ic[i, :, :, 1], foils_Ic[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

Yp_quiv = mlab.quiver3d(Yp_I[i, 0], Yp_I[i, 1], Yp_I[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
Xp_quiv = mlab.quiver3d(Xp_I[i, 0], Xp_I[i, 1], Xp_I[i, 2], scale_factor=75,
              color=bmap[2], mode='arrow', resolution=64)
Zp_quiv = mlab.quiver3d(Zp_I[i, 0], Zp_I[i, 1], Zp_I[i, 2], scale_factor=75,
              color=bmap[0], mode='arrow', resolution=64)

#YZ_mesh = mlab.mesh(YZ_I[i, :, :, 0], YZ_I[i, :, :, 1], YZ_I[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_I[i, :, :, 0], XZ_I[i, :, :, 1], XZ_I[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
#XY_mesh = mlab.mesh(XY_I[i, :, :, 0], XY_I[i, :, :, 1], XY_I[i, :, :, 2],
#                    color=bmap[0], opacity=.25)

op = .6
ml = mlab.mesh(Lmesh_I[i, :, 0], Lmesh_I[i, :, 1], Lmesh_I[i, :, 2],
               color=bmap[0], opacity=op)
md = mlab.mesh(Dmesh_I[i, :, 0], Dmesh_I[i, :, 1], Dmesh_I[i, :, 2],
               color=bmap[4], opacity=op)
#ma = mlab.mesh(Amesh_I[i, :, 0], Amesh_I[i, :, 1], Amesh_I[i, :, 2],
#               color=bmap[2], opacity=op / 2)
#mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(tc[i, :, 0], tc[i, :, 1], tc[i, :, 2], color=bmap[3], opacity=.8)

#ql = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
#                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2],
#                   scale_factor=scale_forces, color=bmap[0],
#                   opacity=op)
#
#qd = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
#                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2],
#                   scale_factor=scale_forces, color=bmap[4],
#                   opacity=op)

#mlab.orientation_axes()
fig.scene.parallel_projection = True
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
                             y=foils_Ic[i, :, :, 1],
                             z=foils_Ic[i, :, :, 2],
                             scalars=foil_color[i])
        dRo_quiv.mlab_source.set(u=[dRo_I[i, 0]],
                                 v=[dRo_I[i, 1]],
                                 w=[dRo_I[i, 2]])

        Yp_quiv.mlab_source.set(u=[Yp_I[i, 0]],
                                v=[Yp_I[i, 1]],
                                w=[Yp_I[i, 2]])
        Xp_quiv.mlab_source.set(u=[Xp_I[i, 0]],
                                v=[Xp_I[i, 1]],
                                w=[Xp_I[i, 2]])
        Zp_quiv.mlab_source.set(u=[Zp_I[i, 0]],
                                v=[Zp_I[i, 1]],
                                w=[Zp_I[i, 2]])
#        YZ_mesh.mlab_source.set(x=YZ_I[i, :, :, 0],
#                                y=YZ_I[i, :, :, 1],
#                                z=YZ_I[i, :, :, 2])
#        XZ_mesh.mlab_source.set(x=XZ_I[i, :, :, 0],
#                                y=XZ_I[i, :, :, 1],
#                                z=XZ_I[i, :, :, 2])
        XY_mesh.mlab_source.set(x=XY_I[i, :, :, 0],
                                y=XY_I[i, :, :, 1],
                                z=XY_I[i, :, :, 2])

#        for ii in np.arange(3):
#            bframe[ii].mlab_source.set(u=nframe[i, ii, 0],
#                                       v=nframe[i, ii, 1],
#                                       w=nframe[i, ii, 2])
#        mlab.savefig('../Movies/s_serp3d/sample_glide/iso_forces_{0:03d}.png'.format(i),
#                     size=(2 * 750, 2 * 708), figure=fig)
        yield
manim = anim()
mlab.show()


# %% Plot forces on the body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
#                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
i = 110
#i = 140
#i = 0
#i = 175
#i = 220
#i = 46

head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

op = .6
ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(tc[i, :, 0], tc[i, :, 1], tc[i, :, 2], color=bmap[3], opacity=.8)

ql = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2],
                   scale_factor=scale_forces, color=bmap[0],
                   opacity=op)

qd = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2],
                   scale_factor=scale_forces, color=bmap[4],
                   opacity=op)

#qcom = mlab.quiver3d([0], [0], [0],
#                   [0], [v_com[i, 1]], [v_com[i, 2]],
#                   scale_factor=scale_velocities, color=bmap[0],
#                   mode='arrow', resolution=64)

## color of the markers
#gc = 200
#gray = tuple(np.r_[gc, gc, gc] / 255)
#
## recorded markers
#markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
#                        pf_rot_c[i, :, 2], color=gray, scale_factor=10)
#
## virtual marker
#markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
#                             pfe_c[i, 1, 2], color=gray, scale_factor=7,
#                             mode='sphere', opacity=1)

#sk = 1
## chat
#mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
#          Cdir[i, ::sk, 0], Cdir[i, ::sk, 1], Cdir[i, ::sk, 2],
#          color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
## bhat
#mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
#          Bdir[i, ::sk, 0], Bdir[i, ::sk, 1], Bdir[i, ::sk, 2],
#          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

#mlab.orientation_axes()
fig.scene.isometric_view()
fig.scene.parallel_projection = True


if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')




# %% Dynamic pressure fraction

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord / SVL, 100 * dynP_frac, cmap=plt.cm.viridis,
                    vmin=0, vmax=100)
cbar = fig.colorbar(cax, ax=ax)


fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord / SVL, np.rad2deg(aoa), cmap=plt.cm.viridis,
                    vmin=-10, vmax=90)
cbar = fig.colorbar(cax, ax=ax)


fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord / SVL, np.rad2deg(beta), cmap=plt.cm.viridis,
                    vmin=-90, vmax=90)
cbar = fig.colorbar(cax, ax=ax)


# %% Inertial velocities as heatmaps

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, t_coord / SVL, dR_I[:, :, 2], cmap=plt.cm.viridis)
cbar = fig.colorbar(cax, ax=ax)


# %% Check the wing loading makes sense

Faero_sum = np.linalg.norm(Faero, axis=2).sum(axis=1)
Faero_sum_novent = np.linalg.norm(Faero[:, :vent_idx_spl + 1], axis=2).sum(axis=1)

Faero_sum = Faero[:, :, 2].sum(axis=1)
Faero_sum_novent = Faero[:, :vent_idx_spl + 1, 2].sum(axis=1)

# calcualte the area
#chord_len_m = chord_spl[0, :vent_idx_spl + 1] / 1000
#dt_coord_m = np.gradient(t_coord[0, :vent_idx_spl + 1], edge_order=2) / 1000
chord_len_m = chord_spl[0] / 1000
dt_coord_m = np.gradient(t_coord[0], edge_order=2) / 1000
snake_area = (chord_len_m * dt_coord_m).sum()  # m^2

mass_total_kg = mass_total / 1000
SVL_m = SVL / 1000
#chord_len_m = chord_len / 1000
#snake_area = SVL_m * chord_len_m
grav = 9.81  # m/s^2
Ws = mass_total_kg * grav / snake_area
Faero_Ws = Faero_sum / snake_area
Faero_Ws_novent = Faero_sum_novent / snake_area

fig, ax = plt.subplots()
ax.axhline(Ws, color='gray', linestyle='--')
ax.plot(times, Faero_Ws, label='including forces on tail')
ax.plot(times, Faero_Ws_novent, label='excluding forces on tail')
ax.legend(loc='upper left')
ax.set_xlabel('time (s)')
ax.set_ylabel(r'wing loading (N/m$^2$)')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig('../Figures/wing-loading_413_91_zdir.pdf', transparent=True,
#             bbox_inches='tight')


# %% Vertical force in units of weight

mg = mass_total_kg * grav

Faero_bw = Faero_sum / mg
Faero_bw_novent = Faero_sum_novent / mg

Faero_X = Faero[:, :, 0].sum(axis=1) / mg
Faero_Y = Faero[:, :, 1].sum(axis=1) / mg

fig, ax = plt.subplots()
ax.axhline(1, color='gray', linestyle='--')
ax.plot(times, Faero_X, label='X')
ax.plot(times, Faero_Y, label='Y')
ax.plot(times, Faero_bw, label='including forces on tail')
ax.plot(times, Faero_bw_novent, label='including forces on tail')
ax.legend(loc='upper left')
ax.set_xlabel('time (s)')
ax.set_ylabel(r'vertical aerodynamic force (body weight)')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig('../Figures/vertical_force_413_91_zdir.pdf', transparent=True,
#             bbox_inches='tight')


# %% Contour plots of quantities of interest

f_undulation = 1.4  # Hz
T_undulation = 1 / f_undulation
# normalized body position
#Sn = s_coord / SVL
Sn = t_coord / SVL
Tn = times2D / T_undulation

vent_plot = Sn[:, vent_idx_spls + 2][:, 0]


fig1, ax1 = plt.subplots()
cax1 = ax1.pcolormesh(Tn, Sn, aoa, vmin=-10, vmax=90,
               cmap=plt.cm.viridis)
cont1 = ax1.contour(times2D, Sn, aoa, [35], colors='w', linewidths=1.25)
#ax1.plot(Tn[:, 0], vent_plot, c='k', lw=1)
cbar1 = fig1.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.875)
cbar1.add_lines(cont1)
cbar1.set_ticks([0, 30, 60, 90])
#cbar1.set_label(r'$\alpha$, angle of attack')
cbar1.set_label(r'    $\alpha$', rotation=0)
ax1.set_ylabel(r'$s$    ', rotation=0)
ax1.set_xlabel(r'$t$')
ax1.axhline(1, color='gray', lw=1)
ax1.set_xlim(Tn.min(), Tn.max())
ax1.set_ylim(0, Sn.max())
sns.despine(ax=ax1)
fig1.set_tight_layout(True)


# %% AoA contour plot

fig1, ax1 = plt.subplots()
#cax1 = ax1.pcolormesh(Tn, Sn, aoa, vmin=-10, vmax=90,
#               cmap=plt.cm.viridis)
cax1 = ax1.contourf(Tn, Sn, aoa, np.arange(40, 91, 10), vmin=40, vmax=90,
               cmap=plt.cm.viridis)
#cont1 = ax1.contour(times2D, Sn, aoa, [35], colors='w', linewidths=1.25)
#ax1.plot(Tn[:, 0], vent_plot, c='k', lw=1)
cbar1 = fig1.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.875)
#cbar1.add_lines(cont1)
cbar1.set_ticks(np.arange(40, 91, 10))
#cbar1.set_ticks([45, 60, 75, 90])
#cbar1.set_ticks([0, 30, 60, 90])
#cbar1.set_label(r'$\alpha$, angle of attack')
cbar1.set_label(r'    $\alpha$', rotation=0)
ax1.set_ylabel(r'$s$    ', rotation=0)
ax1.set_xlabel(r'$t$')
ax1.axhline(1, color='gray', lw=1)
ax1.set_xlim(Tn.min(), Tn.max())
ax1.set_ylim(0, Sn.max())
sns.despine(ax=ax1)
fig1.set_tight_layout(True)


# %% AoA  and sweep angle with better limits

# aoa
fig1, ax1 = plt.subplots()
cax1 = ax1.pcolormesh(Tn, Sn, aoa, vmin=40, vmax=90,
               cmap=plt.cm.viridis)
#ax1.plot(Tn[:, 0], vent_plot, c='k', lw=1)
cont1 = ax1.contour(Tn, Sn, aoa, [60], colors='w', linewidths=1.25)
cbar1 = fig1.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.875)
cbar1.add_lines(cont1)
cbar1.set_ticks(np.arange(40, 91, 10))
#cbar1.set_label(r'$\alpha$, angle of attack')
cbar1.set_label(r'angle of attack, $\alpha$')
ax1.set_ylabel('length (SVL)')
ax1.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
ax1.axhline(1, color='gray', lw=1)
ax1.set_xlim(Tn.min(), Tn.max())
ax1.set_ylim(0, Sn.max())
sns.despine(ax=ax1)
fig1.set_tight_layout(True)


# sweep angle
fig2, ax2 = plt.subplots()
cax2 = ax2.pcolormesh(Tn, Sn, np.abs(beta), vmin=0, vmax=90,
               cmap=plt.cm.inferno)
#cont2 = ax2.contour(Tn, Sn, np.abs(beta), [18.435, 26.565], colors='w',
#                    linewidths=1.25)
cbar2 = fig2.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.875)
#cbar2.set_label(r'$|\beta|$, sweep angle')
#cbar2.set_label(r'    $|\beta|$', rotation=0)
#ax2.set_ylabel(r'$s$    ', rotation=0)
#ax2.set_xlabel(r'$t$')
cbar2.set_label(r'sweep angle, $|\beta|$')
ax2.set_ylabel('length (SVL)')
ax2.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar2.set_ticks([0, 30, 60, 90])
#cbar2.add_lines(cont2)
ax2.axhline(1, color='gray', lw=1)
ax2.set_xlim(Tn.min(), Tn.max())
ax2.set_ylim(0, Sn.max())
sns.despine(ax=ax2)
fig2.set_tight_layout(True)


# add degree symbol to angles
for cbar in [cbar1, cbar2]:
    ticks = cbar.ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    cbar.ax.set_yticklabels(newticks)


fig1.savefig('../Figures/aoa_vent.png', transparent=True,
             bbox_inches='tight')

fig2.savefig('../Figures/beta_vent.png', transparent=True,
             bbox_inches='tight')


# %%

#beta_red = 1 - np.cos(np.abs(np.deg2rad(beta)))**2

sweep_vel_ratio = np.linalg.norm(dR_BC, axis=2) / np.linalg.norm(v_spl, axis=2)
beta_red = sweep_vel_ratio**2

# sweep angle
fig2, ax2 = plt.subplots()
cax2 = ax2.pcolormesh(Tn, Sn, beta_red, vmin=0, vmax=1,
               cmap=plt.cm.plasma)
#ax2.contour(Tn, Sn, beta_red, np.arange(0, 101, 25), vmin=0, vmax=100)
#cont2 = ax2.contour(Tn, Sn, np.abs(beta), [18.435, 26.565], colors='w',
#                    linewidths=1.25)
cbar2 = fig2.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.875)
#cbar2.set_label(r'$|\beta|$, sweep angle')
#cbar2.set_label(r'    $|\beta|$', rotation=0)
cbar2.set_label('velocity fraction')
ax2.set_ylabel('length (SVL)')
ax2.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
#ax2.set_ylabel(r'$s$    ', rotation=0)
#ax2.set_xlabel(r'$t$')
cbar2.set_ticks([0, .25, .50, .75, 1])
#cbar2.set_ticks([0, 25, 50, 75, 100])
#cbar2.add_lines(cont2)
ax2.axhline(1, color='gray', lw=1)
#ax2.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray')
ax2.set_xlim(Tn.min(), Tn.max())
ax2.set_ylim(0, Sn.max())
sns.despine(ax=ax2)
fig2.set_tight_layout(True)

fig2.savefig('../Figures/velocity_fraction.png', transparent=True,
             bbox_inches='tight')


# %%# %% Plot angle of attack, sweep angle, and Reynolds number

#Tn = times2D
#Sn = s_coord / SVL
#Sn = t_coord / SVL

figsize = (6.8, 9)
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)
ax1, ax2, ax3 = axs.flatten()

# angle of attack
#vlim_aoa = np.max([aoa_d.max(), np.abs(aoa_d.min())])
cax1 = ax1.pcolormesh(Tn, Sn, aoa, vmin=-10, vmax=90,
                      cmap=plt.cm.viridis)
cont1 = ax1.contour(Tn, Sn, aoa, [35], colors='w', linewidths=1.25)
cbar1 = fig.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.875)
cbar1.add_lines(cont1)
cbar1.set_ticks([0, 30, 60, 90])
#cbar1.set_label(r'$\alpha$, angle of attack')
cbar1.set_label(r'    $\alpha$', rotation=0)

# sweep angle
cax2 = ax2.pcolormesh(Tn, Sn, np.abs(beta), vmin=0, vmax=90,
                      cmap=plt.cm.inferno)
cont2 = ax2.contour(Tn, Sn, np.abs(beta), [18.435, 26.565], colors='w',
                    linewidths=1.25)
cbar2 = fig.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.875)
#cbar2.set_label(r'$|\beta|$, sweep angle')
cbar2.set_label(r'    $|\beta|$', rotation=0)
cbar2.set_ticks([0, 30, 60, 90])
cbar2.add_lines(cont2)

# Reynolds number
cax3 = ax3.pcolormesh(Tn, Sn, Re, vmin=1000, vmax=15000, cmap=plt.cm.plasma)
#cont3 = ax3.contour(Tn, Sn, Re, [9000], colors='w', linewidths=1.25)
cbar3 = fig.colorbar(cax3, ax=ax3, orientation='vertical', shrink=.875)
#cbar3.set_label('Reynolds number')
cbar3.set_label(r'    $Re$', rotation=0)
cbar3.set_ticks(np.arange(0, 15001, 2000))
#cbar3.add_lines(cont3)

for ax in [ax1, ax2, ax3]:
#    ax.contour(Tn[:, n_neck:], Sn[:, n_neck:], kap[:, n_neck:], [0],
#               colors='gray', linewidths=1, zorder=1)
#    ax.plot(Tn[:, 0], vent_plot, c='gray', lw=1)
    ax.axhline(1, color='gray', lw=1)
    ax.set_xlim(Tn.min(), Tn.max())
    ax.set_ylim(0, Sn.max())
    sns.despine(ax=ax)

ax1.set_ylabel(r'$s$    ', rotation=0)
ax2.set_ylabel(r'$s$    ', rotation=0)
ax3.set_ylabel(r'$s$    ', rotation=0)
ax3.set_xlabel(r'$t$')

# add degree symbol to angles
for cbar in [cbar1, cbar2]:
    ticks = cbar.ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    cbar.ax.set_yticklabels(newticks)

#ax1.set_title(label_base.format(Ap, lkp, fp) + r' $\mathrm{Hz}$')

#fig.suptitle(label_base.format(Ap, lkp, fp),
#             fontsize=plt.rcParams[u'axes.titlesize'])

fig.set_tight_layout(True)

fig.savefig('../Figures/aero_summary_vent.png', transparent=True,
            bbox_inches='tight')

#fig.savefig('../Figures/aero_summary.png', transparent=True,
#            bbox_inches='tight')


# %% Plot the bending and twisting angles

bending_ang = np.rad2deg(a_angs)
twisting_ang = np.rad2deg(b_angs)

figsize = (6.8, 7)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
ax1, ax2 = axs.flatten()

# angle of attack
#vlim_aoa = np.max([aoa_d.max(), np.abs(aoa_d.min())])
cax1 = ax1.pcolormesh(Tn, Sn, bending_ang, vmin=0, vmax=15,
                      cmap=plt.cm.viridis)
#cont1 = ax1.contour(Tn, Sn, aoa, [35], colors='w', linewidths=1.25)
cbar1 = fig.colorbar(cax1, ax=ax1, orientation='vertical', shrink=.875)
#cbar1.add_lines(cont1)
#cbar1.set_ticks([0, 30, 60, 90])
#cbar1.set_label(r'$\alpha$, angle of attack')
cbar1.set_label(r'    $\alpha$', rotation=0)

# sweep angle
cax2 = ax2.pcolormesh(Tn, Sn, twisting_ang, #, vmin=0, vmax=90,
                      cmap=plt.cm.inferno)
#cont2 = ax2.contour(Tn, Sn, np.abs(beta), [18.435, 26.565], colors='w',
#                    linewidths=1.25)
cbar2 = fig.colorbar(cax2, ax=ax2, orientation='vertical', shrink=.875)
#cbar2.set_label(r'$|\beta|$, sweep angle')
cbar2.set_label(r'    $|\beta|$', rotation=0)
#cbar2.set_ticks([0, 30, 60, 90])
#cbar2.add_lines(cont2)


# %% Plot the x extent

b_coord = t_coord

vmax = np.abs(np.r_[spl_c[:, :, 0].min(), spl_c[:, :, 0].max()]).max()
vmax = 200
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, b_coord, spl_c[:, :, 0], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
#cax = ax.pcolormesh(Tn, Sn, spl_c[:, :, 0], cmap=plt.cm.coolwarm,
#                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)

#cax.cmap.set_under('k')

cbar.set_label('Lateral excursion (mm)')

ax.set_xlim(times2D.min(), times2D.max())
ax.set_ylim(0, b_coord.max())
#ax.set_xlim(Tn.min(), Tn.max())
#ax.set_ylim(0, Sn.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel('body length (mm)')

ax.contour(times2D, s_coord, spl_c[:, :, 0],
           [0], colors='gray')


ax.axhline(SVL, color='gray', lw=1)
#ax.contour(Tn, Sn, spl_c[:, :, 0],
#           [0], colors='gray')

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig('../Figures/lateral_mm.png', transparent=True,
            bbox_inches='tight')


# %% Plot the x extent (rotated co-moving snake frame)

b_coord = t_coord

vmax = np.abs(np.r_[spl_c[:, :, 0].min(), spl_c[:, :, 0].max()]).max()
vmax = 200
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(times2D, b_coord, spl_c_rot[:, :, 0], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
#cax = ax.pcolormesh(Tn, Sn, spl_c[:, :, 0], cmap=plt.cm.coolwarm,
#                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)

#cax.cmap.set_under('k')

cbar.set_label('Lateral excursion (mm)')

ax.set_xlim(times2D.min(), times2D.max())
ax.set_ylim(0, b_coord.max())
#ax.set_xlim(Tn.min(), Tn.max())
#ax.set_ylim(0, Sn.max())
ax.set_xlabel('time (sec)')
ax.set_ylabel('body length (mm)')

ax.contour(times2D, s_coord, spl_c[:, :, 0],
           [0], colors='gray')


ax.axhline(SVL, color='gray', lw=1)
#ax.contour(Tn, Sn, spl_c[:, :, 0],
#           [0], colors='gray')

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Normalized lateral excursion

vmax = np.abs(np.r_[spl_c[:, :, 1].min(), spl_c[:, :, 1].max()]).max()
vmax = 300 / SVL
vmin = -vmax
vmax = .2
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c[:, :, 0] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

cbar.set_label('lateral excursion (SVL)')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
cbar.set_ticks([-.2, -.1, 0, .1, .2])

ax.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray')

ax.axhline(1, color='gray', lw=1)

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/lateral_norm_no_zero.png', transparent=True,
#            bbox_inches='tight')

#fig.savefig('../Figures/lateral_norm_yes_zero.png', transparent=True,
#            bbox_inches='tight')


# %% Normalized lateral excursion (co-moving rotated frame)

vmax = np.abs(np.r_[spl_c[:, :, 1].min(), spl_c[:, :, 1].max()]).max()
vmax = 300 / SVL
vmin = -vmax
vmax = .2
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c_rot[:, :, 0] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

cbar.set_label('lateral excursion (SVL)')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
cbar.set_ticks([-.2, -.1, 0, .1, .2])

ax.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray')

ax.axhline(1, color='gray', lw=1)

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/lateral_norm_no_zero_comoving_frame.png', transparent=True,
#            bbox_inches='tight')

#fig.savefig('../Figures/lateral_norm_yes_zero_comoving_frame.png', transparent=True,
#            bbox_inches='tight')


# %% Normalized fore-aft excursion

vmax = np.abs(np.r_[spl_c[:, :, 1].min(), spl_c[:, :, 1].max()]).max()
vmax = 300 / SVL
vmin = -vmax
vmax = .4
vmin = -.4

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c[:, :, 1] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

cbar.set_label('fore-aft excursion (SVL)')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')

cbar.set_ticks([-.4, -.2, 0, .2, .4])

#ax.contour(Tn, Sn, spl_c[:, :, 0],
#           [0], colors='gray')

ax.axhline(1, color='gray', lw=1)

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/fore_aft_norm.png', transparent=True,
#            bbox_inches='tight')


# %% Normalized fore-aft excursion (rotated co-moving snake frame)

vmax = np.abs(np.r_[spl_c[:, :, 1].min(), spl_c[:, :, 1].max()]).max()
vmax = 300 / SVL
vmin = -vmax
vmax = .4
vmin = -.4

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c_rot[:, :, 1] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

cbar.set_label('fore-aft excursion (SVL)')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')

cbar.set_ticks([-.4, -.2, 0, .2, .4])

#ax.contour(Tn, Sn, spl_c[:, :, 0],
#           [0], colors='gray')

ax.axhline(1, color='gray', lw=1)

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/fore_aft_norm_comoving_frame.png', transparent=True,
#            bbox_inches='tight')


# %% Normalized vertical excursion

vmax = np.abs(np.r_[spl_c[:, :, 2].min(), spl_c[:, :, 2].max()]).max()
vmax = 100
vmin = -vmax
vmax = .1
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c[:, :, 2] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

#ax.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray', alpha=.5)

cbar.set_label('vertical excursion (SVL)')

ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')

cbar.set_ticks([-.1, -.05, 0, .05, .1])

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/vertical_norm.png', transparent=True,
#            bbox_inches='tight')


# %% Normalized vertical excursion (rotated co-moving frame)

vmax = np.abs(np.r_[spl_c[:, :, 2].min(), spl_c[:, :, 2].max()]).max()
vmax = 100
vmin = -vmax
vmax = .075
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, spl_c_rot[:, :, 2] / SVL, cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
#cbar.set_ticks(np.linspace(vmin, vmax, 6))

#cax.cmap.set_under('k')

#ax.contour(Tn, Sn, spl_c_rot[:, :, 2] / SVL, [0], colors='gray')

cbar.set_label('vertical excursion (SVL)')

ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$s$')
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')

cbar.set_ticks([-.1, -.05, 0, .05, .1])

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig('../Figures/vertical_norm_comoving_frame.png', transparent=True,
            bbox_inches='tight')


# %% Vx spline velocity field

v_spl_c_plot = v_spl_c / 1000  # m/s
vmax = 2
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 0], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray')
#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('lateral velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/lateral_velocity_no_zero.png', transparent=True,
#            bbox_inches='tight')


# %% Vx spline velocity field (rotated frame)

v_spl_c_plot = v_spl_c_rot / 1000  # m/s
vmax = 2
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 0], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, spl_c[:, :, 0], [0], colors='gray')
#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('lateral velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

# %% Vy spline velocity field

v_spl_c_plot = v_spl_c / 1000  # m/s
vmax = 1
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('fore-aft velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/foreaft_velocity_yes_zero.png', transparent=True,
#            bbox_inches='tight')


# %% Vy spline velocity field (rotated snake frame)

v_spl_c_plot = v_spl_c_rot / 1000  # m/s
vmax = 1
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('fore-aft velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Vz spline velocity field

v_spl_c_plot = v_spl_c / 1000  # m/s
vmax = 1
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 2], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/vertical_velocity_yes_zero.png', transparent=True,
#            bbox_inches='tight')


# %% Vz spline velocity field (rotated snake frame)

v_spl_c_plot = v_spl_c_rot / 1000  # m/s
vmax = 1
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, v_spl_c_plot[:, :, 2], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

#ax.contour(Tn, Sn, v_spl_c_plot[:, :, 0], [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical velocity (m/s)')

sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Ax spline acceleration field

a_spl_c_plot = a_spl_c / 9810  # gs
vmax = 3
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, a_spl_c_plot[:, :, 0], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('lateral acceleration (g)')


# %% Ay spline acceleration field

a_spl_c_plot = a_spl_c / 9810  # gs
vmax = 3
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, a_spl_c_plot[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('fore-aft acceleration (g)')


# %% Az spline acceleration field

a_spl_c_plot = a_spl_c / 9810  # gs
vmax = 3
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, a_spl_c_plot[:, :, 2], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical acceleration (g)')


# %% Faero field in Y

vmax = 4
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Faero[:, :, 1], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical acceleration (g)')


# %% Faero field in Z

vmax = 10
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Faero[:, :, 2], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical acceleration (g)')


# %% Fl field in Y

vmax = 8
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Fl[:, :, 1], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

# for the colorbar in the slide
#cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', shrink=.875)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('forward lift force (mN)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig('../Figures/Fl_yhat.png', transparent=True,
            bbox_inches='tight')

#ax.set_xlabel('')

#fig.savefig('../Figures/Fl_yhat_horiz.png', transparent=True,
#            bbox_inches='tight')


# %% Fd field in Z

vmax = 8
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Fd[:, :, 2], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical drag force (mN)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig('../Figures/Fd_zhat.png', transparent=True,
            bbox_inches='tight')


# %%

# %% Faero field in Y

vmax = 8
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Faero[:, :, 1], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('forward aerodynamic force (mN)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/Fl_yhat.png', transparent=True,
#            bbox_inches='tight')

#ax.set_xlabel('')

#fig.savefig('../Figures/Fl_yhat_horiz.png', transparent=True,
#            bbox_inches='tight')


# %% Faero field in Z

vmax = 8
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, 1000 * Faero[:, :, 2], cmap=plt.cm.plasma,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical aerodynamic force (mN)')

sns.despine(ax=ax)
fig.set_tight_layout(True)

#fig.savefig('../Figures/Fd_zhat.png', transparent=True,
#            bbox_inches='tight')


# %% Fl / Fd field in Y

vmax = 3
vmin = -vmax

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, Fl[:, :, 1] / Fd[:, :, 1], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical acceleration (g)')


# %% Fl / Fd field in Z

vmax = 1
vmin = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, Fl[:, :, 2] / Fd[:, :, 2], cmap=plt.cm.coolwarm,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('vertical acceleration (g)')



# %% Inertial terms

mass_spl_kg = mass_spl / 1000  # kg

spl_c_m = spl_c / 1000  # m
#a_spl_m = a_spl / 1000  # m/s^2
a_spl_m = a_spl_c / 1000  # m/s^2  # local acceleration

# in co-moving frame
spl_c_m = spl_c_rot / 1000  # in body frame
v_spl_m = v_spl_c_rot / 1000  # in body frame
a_spl_m = a_spl_c_rot / 1000  # in body frame

inertial_term = np.zeros((ntime, nspl, 3))
Faero_term = np.zeros((ntime, nspl, 3))
ho = np.zeros((ntime, nspl, 3))

for i in np.arange(ntime):
    for j in np.arange(nspl):

        # angular momenum
        tmp_iner = np.cross(spl_c_m[i, j], mass_spl_kg[i, j] * a_spl_m[i, j])
        inertial_term[i, j] = tmp_iner

        tmp_aero = np.cross(spl_c_m[i, j], Faero[i, j])
        Faero_term[i, j] = tmp_aero

        ho[i, j] = np.cross(spl_c_m[i, j], mass_spl_kg[i, j] * v_spl_m[i, j])

        # newton's linear momentum
#        tmp_iner = mass_spl_kg[i, j] * a_spl_m[i, j]
#        inertial_term[i, j] = tmp_iner
#
#        tmp_aero = Faero[i, j]
#        Faero_term[i, j] = tmp_aero


# %%

dho = inertial_term.sum(axis=1)
Ma = Faero_term.sum(axis=1)
ho_sum = ho.sum(axis=1)

fig, ax = plt.subplots()
ax.plot(times, dho[:, 0])
ax.plot(times, dho[:, 1])
ax.plot(times, dho[:, 2])

fig, ax = plt.subplots()
ax.plot(times, Ma[:, 0])
ax.plot(times, Ma[:, 1])
ax.plot(times, Ma[:, 2])

fig, ax = plt.subplots()
ax.plot(times, ho_sum[:, 0])
ax.plot(times, ho_sum[:, 1])
ax.plot(times, ho_sum[:, 2])


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, ho, cmap=cmap,
                    vmin=vmin, vmax=vmax)
#cax = ax.contourf(Tn, Sn, aero_to_iner, np.arange(0, 6), cmap=cmap,
#                  vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('inertial to aero')


# %%

aero_to_iner = np.linalg.norm(Faero_term, axis=2) / np.linalg.norm(inertial_term, axis=2)

vmax = 10
vmin = 1

cmap = plt.cm.plasma
cmap.set_under = 'gray'

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, aero_to_iner, cmap=cmap,
                    vmin=vmin, vmax=vmax)
#cax = ax.contourf(Tn, Sn, aero_to_iner, np.arange(0, 6), cmap=cmap,
#                  vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
ax.axhline(1, color='gray', lw=1)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_ylabel('length (SVL)')
ax.set_xlabel(r'time / T$_\mathrm{\mathsf{undulation}}$')
cbar.set_label('inertial to aero')



# %% Curvature field

startx = np.r_[.0239, .0069, .1552, .5942, .9511]
starty = np.r_[382.65, 152.936, 2.719, 4.8654, 7.01196]
stopx = np.r_[.5900, .7173, 1.3286, 1.36141, 1.36551]
stopy = np.r_[854.89, 738.98, 801., 590.71, 344.015]

slope = (stopy - starty) / (stopx - startx)  # mm / s
slope_svl = slope / dist_svl

startx = startx / T_undulation
starty = starty / SVL
stopx = stopx / T_undulation
stopy = stopy / SVL

vmin, vmax = -.05, .05

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, kap_signed, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
cbar.set_label(r'curvature, $\kappa$ (1/mm)')#, fontsize='medium')
cbar.solids.set_edgecolor("face")
cbar.set_ticks([vmin, 0, vmax])

sk = 2
ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
           [0], colors='gray')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_xlabel('$t$')
ax.set_ylabel('$s$')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)

fig.savefig('../Figures/curvature.png', transparent=True,
            bbox_inches='tight')

# overlay the hand selected
for ii in np.arange(len(startx)):
    xx = [startx[ii], stopx[ii]]
    yy = [starty[ii], stopy[ii]]
    ax.plot(xx, yy, '--', color='white')

fig.savefig('../Figures/curvature_overlay.png', transparent=True,
            bbox_inches='tight')


# %% Torsion field

vmin, vmax = -.05, .05

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, tau, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
cbar.set_label(r'torsion, $\tau$ (1/mm)')
#cbar.set_label(r'torsion, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")
cbar.set_ticks([vmin, 0, vmax])

#sk = 2
#ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
#           [0], colors=emerald_green)

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$s$')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)

fig.savefig('../Figures/torsion.png', transparent=True,
            bbox_inches='tight')


# %% Project Faero along gravity (Zhat) and Yhat to get force components

Fa_yhat = np.zeros((ntime, nspl))
Fa_zhat = np.zeros((ntime, nspl))
#yhat = np.array([0, 1, 0])
#zhat = np.array([0, 1, 1])
for i in np.arange(ntime):
    den = np.linalg.norm(Faero[i], axis=1)
#    Fa_zhat[i] = np.dot(Faero[i], zhat) / den
#    Fa_yhat[i] = np.dot(Faero[i], yhat) / den
    Fa_zhat[i] = Faero[i, :, 2] / den
    Fa_yhat[i] = Faero[i, :, 1] / den

Fa_ratio = Fa_yhat / Fa_zhat

#Fa_ratio[Fa_ratio < 0] = -.25
#Fa_ratio[Fa_ratio >= 0] = .25

vmin = 0
vmax = 0.5

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, Fa_ratio, cmap=plt.cm.viridis,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)
cbar.set_ticks(np.linspace(vmin, vmax, 6))

cax.cmap.set_under('k')

cbar.set_label('aerodynamic effectiveness')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$s$')

sk = 2
ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig('../Figures/aero_effectivenss.png', transparent=True,
            bbox_inches='tight')

#fig.savefig('../Figures/aero_effectivenss_0_to_1.png', transparent=True,
#            bbox_inches='tight')


# %%

Fa_ratio_abs = np.abs(Fa_ratio)

vmin = 0
vmax = .5

fig, ax = plt.subplots()
cax = ax.pcolormesh(Tn, Sn, Fa_ratio_abs, cmap=plt.cm.viridis,
                    vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.875)

cbar.set_label('aerodynamic effectiveness')

ax.set_xlim(Tn.min(), Tn.max())
ax.set_ylim(0, Sn.max())
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$s$')

sk = 2
ax.contour(Tn[sk:-sk, sk:-sk], Sn[sk:-sk, sk:-sk], kap_signed[sk:-sk, sk:-sk],
           [0], colors='gray')

sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Turn velocities and forces into meshes for plotting

# in inertial frame
L = np.zeros((ntime, nspl, 3, 2))
D = np.zeros((ntime, nspl, 3, 2))
A = np.zeros((ntime, nspl, 3, 2))
u = np.zeros((ntime, nspl, 3, 2))
bc = np.zeros((ntime, nspl, 3, 2))
tc = np.zeros((ntime, nspl, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 10 * 1000

for i in np.arange(ntime):
    for j in np.arange(nspl):
        # in inertial frame
        L[i, j, :, 0] = spl_c[i, j]
        L[i, j, :, 1] = spl_c[i, j] + scale_forces * Fl[i, j]
        D[i, j, :, 0] = spl_c[i, j]
        D[i, j, :, 1] = spl_c[i, j] + scale_forces * Fd[i, j]
        A[i, j, :, 0] = spl_c[i, j]
        A[i, j, :, 1] = spl_c[i, j] + scale_forces * Faero[i, j]

        u[i, j, :, 0] = spl_c[i, j]
        u[i, j, :, 1] = spl_c[i, j] + scale_velocities * v_spl[i, j]

        bc[i, j, :, 0] = spl_c[i, j]
        bc[i, j, :, 1] = spl_c[i, j] + scale_velocities * dR_BC[i, j]

        tc[i, j, :, 0] = spl_c[i, j]
        tc[i, j, :, 1] = spl_c[i, j] + scale_velocities * dR_TC[i, j]


# %% Plot forces on the body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
#                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)
i = 110
#i = 140
#i = 0
#i = 175
#i = 220
#i = 46

head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

op = .6
ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(tc[i, :, 0], tc[i, :, 1], tc[i, :, 2], color=bmap[3], opacity=.8)

ql = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2],
                   scale_factor=scale_forces, color=bmap[0],
                   opacity=op)

qd = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2],
                   scale_factor=scale_forces, color=bmap[4],
                   opacity=op)

#qcom = mlab.quiver3d([0], [0], [0],
#                   [0], [v_com[i, 1]], [v_com[i, 2]],
#                   scale_factor=scale_velocities, color=bmap[0],
#                   mode='arrow', resolution=64)

## color of the markers
#gc = 200
#gray = tuple(np.r_[gc, gc, gc] / 255)
#
## recorded markers
#markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
#                        pf_rot_c[i, :, 2], color=gray, scale_factor=10)
#
## virtual marker
#markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
#                             pfe_c[i, 1, 2], color=gray, scale_factor=7,
#                             mode='sphere', opacity=1)

#sk = 1
## chat
#mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
#          Cdir[i, ::sk, 0], Cdir[i, ::sk, 1], Cdir[i, ::sk, 2],
#          color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
## bhat
#mlab.quiver3d(spl_c[i, ::sk, 0], spl_c[i, ::sk, 1], spl_c[i, ::sk, 2],
#          Bdir[i, ::sk, 0], Bdir[i, ::sk, 1], Bdir[i, ::sk, 2],
#          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

#mlab.orientation_axes()
fig.scene.isometric_view()
fig.scene.parallel_projection = True


if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')


# %% ANIMATIONS OF THE GLIDE

# color of the markers
gc = 200
gray = tuple(np.r_[gc, gc, gc] / 255)

i = 0


# %% 1) ANIMATE JUST THE BODY

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#qcom = mlab.quiver3d([0], [0], [0],
#                   [0], [v_com[i, 1]], [v_com[i, 2]],
#                   scale_factor=scale_velocities, color=(0, 0, 0),
#                   mode='arrow', resolution=64)

# recorded markers
markers = mlab.points3d(pfe_c[i, :, 0], pfe_c[i, :, 1],
                        pfe_c[i, :, 2], color=gray, scale_factor=10,
                        resolution=64)

fig.scene.isometric_view()
fig.scene.parallel_projection = True


# %%

savename = '../anim_glide/0_body_iso_markers/anim_{0:03d}.png'

@mlab.animate(delay=100)
def ():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        markers.mlab_source.set(x=pfe_c[i, :, 0], y=pfe_c[i, :, 1],
                                z=pfe_c[i, :, 2])

#        markers.mlab_source.set(x=pf_rot_c[i, :, 0], y=pf_rot_c[i, :, 1],
#                                z=pf_rot_c[i, :, 2])

#        markers_virt.mlab_source.set(x=pfe_c[i, 1, 0], y=pfe_c[i, 1, 1],
#                                     z=pfe_c[i, 1, 2])

#        qcom.mlab_source.set(v=[v_com[i, 1]], w=[v_com[i, 2]])

        mlab.savefig(savename.format(i), size=(2*750, 2*750))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/2_body_side/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        markers.mlab_source.set(x=pf_rot_c[i, :, 0], y=pf_rot_c[i, :, 1],
                                z=pf_rot_c[i, :, 2])

        markers_virt.mlab_source.set(x=pfe_c[i, 1, 0], y=pfe_c[i, 1, 1],
                                     z=pfe_c[i, 1, 2])

#        qcom.mlab_source.set(v=[v_com[i, 1]], w=[v_com[i, 2]])

        mlab.view(azimuth=0, elevation=90, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/3_body_top/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        markers.mlab_source.set(x=pf_rot_c[i, :, 0], y=pf_rot_c[i, :, 1],
                                z=pf_rot_c[i, :, 2])

        markers_virt.mlab_source.set(x=pfe_c[i, 1, 0], y=pfe_c[i, 1, 1],
                                     z=pfe_c[i, 1, 2])

        qcom.mlab_source.set(v=[v_com[i, 1]], w=[v_com[i, 2]])

        mlab.view(azimuth=0, elevation=0, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/3b_body_back/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        markers.mlab_source.set(x=pf_rot_c[i, :, 0], y=pf_rot_c[i, :, 1],
                                z=pf_rot_c[i, :, 2])

        markers_virt.mlab_source.set(x=pfe_c[i, 1, 0], y=pfe_c[i, 1, 1],
                                     z=pfe_c[i, 1, 2])

        qcom.mlab_source.set(v=[v_com[i, 1]], w=[v_com[i, 2]])

        mlab.view(azimuth=-90, elevation=90, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %% ANIMATE LIFT AND DRAG FORCES ON THE BODY

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

op = .6
ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(u[i, :, 0], u[i, :, 1], u[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(bc[i, :, 0], bc[i, :, 1], bc[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(tc[i, :, 0], tc[i, :, 1], tc[i, :, 2], color=bmap[3], opacity=.8)

ql = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fl[i, :, 0], Fl[i, :, 1], Fl[i, :, 2],
                   scale_factor=scale_forces, color=bmap[0],
                   opacity=op)

qd = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Fd[i, :, 0], Fd[i, :, 1], Fd[i, :, 2],
                   scale_factor=scale_forces, color=bmap[4],
                   opacity=op)

fig.scene.isometric_view()
fig.scene.parallel_projection = True


# %%

savename = '../anim_glide/4_fl_fd_iso/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        ml.mlab_source.set(x=L[i, :, 0], y=L[i, :, 1],
                           z=L[i, :, 2])
        md.mlab_source.set(x=D[i, :, 0], y=D[i, :, 1],
                           z=D[i, :, 2])

        ql.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fl[i, :, 0], v=Fl[i, :, 1], w=Fl[i, :, 2])
        qd.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])

        mlab.view(azimuth=45, elevation=54.736, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/5_fl_fd_side/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        ml.mlab_source.set(x=L[i, :, 0], y=L[i, :, 1],
                           z=L[i, :, 2])
        md.mlab_source.set(x=D[i, :, 0], y=D[i, :, 1],
                           z=D[i, :, 2])

        ql.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fl[i, :, 0], v=Fl[i, :, 1], w=Fl[i, :, 2])
        qd.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])

        mlab.view(azimuth=0, elevation=90, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/6_fl_fd_top/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        ml.mlab_source.set(x=L[i, :, 0], y=L[i, :, 1],
                           z=L[i, :, 2])
        md.mlab_source.set(x=D[i, :, 0], y=D[i, :, 1],
                           z=D[i, :, 2])

        ql.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fl[i, :, 0], v=Fl[i, :, 1], w=Fl[i, :, 2])
        qd.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])

        mlab.view(azimuth=0, elevation=0, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%

savename = '../anim_glide/7_fl_fd_back/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        ml.mlab_source.set(x=L[i, :, 0], y=L[i, :, 1],
                           z=L[i, :, 2])
        md.mlab_source.set(x=D[i, :, 0], y=D[i, :, 1],
                           z=D[i, :, 2])

        ql.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fl[i, :, 0], v=Fl[i, :, 1], w=Fl[i, :, 2])
        qd.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])

        mlab.view(azimuth=-90, elevation=90, distance=1410,
                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %% ANIMATE TOTAL AERODYNAMIC FORCE ON THE BODY

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
                     color=bmap[1], scale_factor=20, resolution=16, opacity=1)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

op = .6
ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=.8)

qa = mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                   Faero[i, :, 0], Faero[i, :, 1], Faero[i, :, 2],
                   scale_factor=scale_forces, color=bmap[2],
                   opacity=op)

fig.scene.isometric_view()
fig.scene.parallel_projection = True


# %%

savename = '../anim_glide/8_faero_iso/anim_{0:03d}.png'

@mlab.animate(delay=100)
def anim_body():
    for i in np.arange(ntime):

        head.mlab_source.set(x=foils[i, 0, 0, 0], y=foils[i, 0, 0, 1],
                             z=foils[i, 0, 50, 2])
        body.mlab_source.set(x=foils[i, :, :, 0], y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2], scalars=foil_color[i])

        ma.mlab_source.set(x=A[i, :, 0], y=A[i, :, 1],
                           z=A[i, :, 2])

        qa.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Faero[i, :, 0], v=Faero[i, :, 1], w=Faero[i, :, 2])
        qd.mlab_source.set(x=spl_c[i, :, 0], y=spl_c[i, :, 1],
                           z=spl_c[i, :, 2],
                           u=Fd[i, :, 0], v=Fd[i, :, 1], w=Fd[i, :, 2])

#        mlab.view(azimuth=-90, elevation=90, distance=1410,
#                  focalpoint=np.r_[-2.17619705, -69.13372803, 61.65791321])

        mlab.savefig(savename.format(i), size=(2**10, 2**10))

        yield

manim = anim_body()
mlab.show()


# %%


# %% ANIMATE SPLINE FITTING

import time

# a visually plesaing time point
i = 110

# a good camera angle and distance
view_save = (55.826409254334664, 58.922421240924997, 670.61545304290667,
             np.array([-50.66807202, -48.27570399,  -2.28887983]))
azimuth = view_save[0]
elevation = view_save[1]
distance = view_save[2]
focalpoint = view_save[3]

# color of the markers (look like IR tape color)
gc = 200
gray = tuple(np.r_[gc, gc, gc] / 255)

#spline_frames = np.r_[1, np.arange(nspl)[::5][1:], nspl - 1]
spline_frames = np.arange(1, nspl)


# %% 1) Spline growing from points

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

# recorded markers
markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
                        pf_rot_c[i, :, 2], color=gray, scale_factor=15,
                        resolution=64)

# virtual marker
markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                             pfe_c[i, 1, 2], color=gray, scale_factor=10,
                             mode='sphere', opacity=1, resolution=64)

# spline fit
body_spl = mlab.plot3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                       color=bmap[1], tube_radius=3)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)


# %% 1) Spline growing from points

savename = '../anim_i110/1_spline/anim_{0:03d}.png'

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(1, 6):

    mlab.clf()

    # recorded markers
    markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
                            pf_rot_c[i, :, 2], color=gray, scale_factor=15,
                            resolution=64)

    # virtual marker
    markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                                 pfe_c[i, 1, 2], color=gray, scale_factor=10,
                                 mode='sphere', opacity=1, resolution=64)

    # spline fit
    body_spl = mlab.plot3d(spl_c[i, :j, 0], spl_c[i, :j, 1], spl_c[i, :j, 2],
                           color=bmap[1], tube_radius=3)

    fig.scene.isometric_view()
    mlab.view(azimuth, elevation, distance, focalpoint)
    mlab.draw()
    mlab.savefig(savename.format(j), size=(2**10, 2**10))

print('Image save time: {0:.3f} sec'.format(time.time() - now))


# %% 2) Cdir and Bdir along spline

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# recorded markers
markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
                        pf_rot_c[i, :, 2], color=gray, scale_factor=15,
                        resolution=64)

# virtual marker
markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                             pfe_c[i, 1, 2], color=gray, scale_factor=10,
                             mode='sphere', opacity=1, resolution=64)

# spline fit
body_spl = mlab.plot3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                       color=bmap[1], tube_radius=3)

# Cdir
mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
          Cdir[i, :, 0], Cdir[i, :, 1], Cdir[i, :, 2],
          color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
# bhat
mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
          Bdir[i, :, 0], Bdir[i, :, 1], Bdir[i, :, 2],
          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

savename = '../anim_i110/2_cb_coord/anim_{0:03d}.png'


# %% 2) Cdir and Bdir along spline

savename = '../anim_i110/2_cb_coord/anim_{0:03d}.png'

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for j in spline_frames:

    mlab.clf()

    # recorded markers
    markers = mlab.points3d(pf_rot_c[i, :, 0], pf_rot_c[i, :, 1],
                            pf_rot_c[i, :, 2], color=gray, scale_factor=15,
                            resolution=64)

    # virtual marker
    markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                                 pfe_c[i, 1, 2], color=gray, scale_factor=10,
                                 mode='sphere', opacity=1, resolution=64)

    # spline
    body_spl = mlab.plot3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                           color=bmap[1], tube_radius=3)

    # Cdir
    mlab.quiver3d(spl_c[i, :j, 0], spl_c[i, :j, 1], spl_c[i, :j, 2],
              Cdir[i, :j, 0], Cdir[i, :j, 1], Cdir[i, :j, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(spl_c[i, :j, 0], spl_c[i, :j, 1], spl_c[i, :j, 2],
              Bdir[i, :j, 0], Bdir[i, :j, 1], Bdir[i, :j, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

    fig.scene.isometric_view()
    mlab.view(azimuth, elevation, distance, focalpoint)
    mlab.draw()
    mlab.savefig(savename.format(j), size=(2**10, 2**10))

print('Image save time: {0:.3f} sec'.format(time.time() - now))


# %% 3) Airfoil shape

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

# recorded markers
markers = mlab.points3d(pf_rot_c[i, 1:, 0], pf_rot_c[i, 1:, 1],
                        pf_rot_c[i, 1:, 2], color=gray, scale_factor=15,
                        resolution=64)

# virtual marker
markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                             pfe_c[i, 1, 2], color=gray, scale_factor=10,
                             mode='sphere', opacity=1, resolution=64)

# spline fit
body_spl = mlab.plot3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
                       color=bmap[1], tube_radius=3)

# Cdir
mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
          Cdir[i, :, 0], Cdir[i, :, 1], Cdir[i, :, 2],
          color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
# bhat
mlab.quiver3d(spl_c[i, :, 0], spl_c[i, :, 1], spl_c[i, :, 2],
          Bdir[i, :, 0], Bdir[i, :, 1], Bdir[i, :, 2],
          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

body = mlab.mesh(foils[i, :j+1, :, 0], foils[i, :j+1, :, 1],
                     foils[i, :j+1, :, 2],
                     scalars=foil_color[i, :j+1], colormap='YlGn', opacity=1,
                     vmin=0, vmax=1)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)


# %% 3) Airfoil shape

savename = '../anim_i110/3_airfoil_APS/anim_{0:03d}.png'

now = time.time()

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for j in spline_frames:

    mlab.clf()

    # recorded markers
    markers = mlab.points3d(pf_rot_c[i, 1:, 0], pf_rot_c[i, 1:, 1],
                            pf_rot_c[i, 1:, 2], color=gray, scale_factor=15,
                            resolution=64)

    # virtual marker
    markers_virt = mlab.points3d(pfe_c[i, 1, 0], pfe_c[i, 1, 1],
                                 pfe_c[i, 1, 2], color=gray, scale_factor=10,
                                 mode='sphere', opacity=1, resolution=64)

    # spline
    body_spl = mlab.plot3d(spl_c[i, j:, 0], spl_c[i, j:, 1], spl_c[i, j:, 2],
                           color=bmap[1], tube_radius=3)

    # j = nspl
    # mlab.clf()
    ## now run everything below

    # Cdir
    mlab.quiver3d(spl_c[i, j:, 0], spl_c[i, j:, 1], spl_c[i, j:, 2],
              Cdir[i, j:, 0], Cdir[i, j:, 1], Cdir[i, j:, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(spl_c[i, j:, 0], spl_c[i, j:, 1], spl_c[i, j:, 2],
              Bdir[i, j:, 0], Bdir[i, j:, 1], Bdir[i, j:, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

    # head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
    #                  color=bmap[1], scale_factor=20, resolution=16, opacity=1)
    body = mlab.mesh(foils[i, :j+1, :, 0], foils[i, :j+1, :, 1],
                     foils[i, :j+1, :, 2],
                     scalars=foil_color[i, :j+1], colormap='YlGn', opacity=1,
                     vmin=0, vmax=1)

    fig.scene.isometric_view()
    mlab.view(azimuth, elevation, distance, focalpoint)
    mlab.draw()
    mlab.savefig(savename.format(j), size=(2*750, 2*750))

print('Image save time: {0:.3f} sec'.format(time.time() - now))


# Note, then set j = nspl, and run just body = ... code

# ffmpeg -f image2 -r 30 -i anim_%03d.png -pix_fmt yuv420p out.mp4
# convert -delay 1 -loop 0 anim_*.png out.gif


# %% MIDLINE ANALYSIS


# %% Calculate the snake 'wings'

ixs_wing, ixs_pos, ixs_neg = [], [], []
nwings = np.zeros(ntime, dtype=np.int)
limx = np.zeros((ntime, 2))
limy = np.zeros((ntime, 2))
for i in np.arange(ntime):
    # just select out body points (most aerodynamic effect)
    xx, yy, zz = spl_c[i, :vent_idx_spls[i]].T

    # indices where we have zero crossings
    # (ix_l is the left index, ix_r is the right)
    ix_l = np.where(xx[:-1] * xx[1:] < 0)[0]
    ix_r = ix_l + 1

    # determine which index is closest to zero
    nwing = len(ix_l)
    ix = np.zeros(nwing, dtype=np.int)
    for jj in np.arange(nwing):
        if np.abs(xx[ix_l[jj]]) < np.abs(xx[ix_r[jj]]):
            ix[jj] = ix_l[jj]
        else:
            ix[jj] = ix_r[jj]

    # determine sign change at the intersections
    ix_p = ix[np.where(xx[ix_r] - xx[ix_l] > 0)[0]]
    ix_n = ix[np.where(xx[ix_r] - xx[ix_l] < 0)[0]]

    # store the wing indices
    nwings[i] = nwing
    ixs_wing.append(ix)
    ixs_pos.append(ix_p)
    ixs_neg.append(ix_n)

    # determine axis limits based on the body extremes in y and z
    body_y = spl_c[i, :vent_idx_spls[i] + 1, 1]
    body_z = spl_c[i, :vent_idx_spls[i] + 1, 2]
    limx[i] = body_y.min(), body_y.max()
    limy[i] = body_z.min(), body_z.max()


# %% Plot the wingedness

fig, ax = plt.subplots()
ax.step(times, nwings, where='pre')
ax.set_ylim(0, 4)
ax.set_yticks([0, 1, 2, 3, 4])
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.step(np.arange(ntime), nwings, where='pre')
ax.set_ylim(0, 4)
ax.set_yticks([0, 1, 2, 3, 4])
sns.despine()
fig.set_tight_layout(True)

print (nwings == 2).sum()
print (nwings == 3).sum()


# %% Extract y an z values for the wings

ix2 = np.where(nwings == 3)[0]

wing2_y, wing2_z = [], []
wing3_y, wing3_z = [], []
aoa_wings, beta_wings = np.r_[0], np.r_[0]
for i in np.arange(ntime):

    # select out spline values at wings
    wing_spl = spl_c[i,  ixs_wing[i]]

    aoa_wings = np.r_[aoa_wings, aoa[i,  ixs_wing[i]]]
    beta_wings = np.r_[beta_wings, beta[i,  ixs_wing[i]]]

    if nwings[i] == 2:
        wing2_y.append(wing_spl[:, 1])
        wing2_z.append(wing_spl[:, 2])
    elif nwings[i] == 3:
       wing3_y.append(wing_spl[:, 1])
       wing3_z.append(wing_spl[:, 2])

aoa_wings = aoa_wings[1:]
beta_wings = beta_wings[1:]

wing2_y = np.vstack(wing2_y)
wing2_z = np.vstack(wing2_z)

wing3_y = np.vstack(wing3_y)
wing3_z = np.vstack(wing3_z)

dwing2_y = np.diff(wing2_y, axis=1).flatten()
dwing2_z = np.diff(wing2_z, axis=1).flatten()

dwing3_y = np.diff(wing3_y, axis=1).flatten()
dwing3_z = np.diff(wing3_z, axis=1).flatten()

dwing_y = np.r_[dwing2_y, dwing3_y]
dwing_z = np.r_[dwing2_z, dwing3_z]


# %% Histogram of angles of attack and sweep angles

fig, ax = plt.subplots()
nn, bn, _ = ax.hist(aoa_wings, 90, range=(0, 90), normed=True)
ax.set_xlabel('angle of attack')
ax.set_ylabel('probability density')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
nn, bn, _ = ax.hist(beta_wings, 180 / 2, range=(-90, 90), normed=True)
ax.set_xlabel('sweep angle')
ax.set_ylabel('probability density')
ax.set_xlim(-90, 90)
ax.set_xticks(np.arange(-90, 91, 15))
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
nn, bn, _ = ax.hist(np.abs(beta_wings), 90, range=(0, 90), normed=True)
ax.set_xlabel('sweep angle')
ax.set_ylabel('probability density')
sns.despine()
fig.set_tight_layout(True)


# %%

g = sns.jointplot(dwing_y / chord_len, dwing_z / chord_len,
                  kind="kde", size=7, space=0, stat_func=None)
g.ax_joint.plot(dwing_y / chord_len, dwing_z / chord_len, 'o',
                alpha=.1, c=bmap[2])

g.set_axis_labels(xlabel=r'$\Delta y$', ylabel=r'$\Delta z$')

g.ax_joint.set_aspect('equal')


# %%

fig = plt.gcf()

fig.savefig('../Figures/wing_space_joint.png', transparent=True,
            bbox_inches='tight')


# %%

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)
ax.plot(wing2_y / chord_len, wing2_z / chord_len, 'o', c=bmap[0], alpha=.75)
ax.plot(wing3_y / chord_len, wing3_z / chord_len, 'o', c=bmap[1], alpha=.75)
ax.set_aspect('equal')
ax.set_xlabel(r'$y$ (chords)')
ax.set_ylabel(r'$z$ (chords)')
sns.despine()
fig.set_tight_layout(True)

fig.savefig('../Figures/wing_trajectories.pdf', transparent=True,
            bbox_inches='tight')


# %%

histargs = {'histtype': 'stepfilled', 'align': 'mid'}

fig, ax = plt.subplots()
_, bins, _ = ax.hist(dwing_y / chord_len, 22, range=(-12, -1), **histargs)
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.hist(dwing_z / chord_len, 20, alpha=1)
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.hist(dwing2_y / chord_len, 20, alpha=.5)
ax.hist(dwing3_y / chord_len, 20, alpha=.5)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.hist(dwing2_z / chord_len, 20, alpha=.5)
ax.hist(dwing3_z / chord_len, 20, alpha=.5)
sns.despine()
fig.set_tight_layout(True)


# %% Velocity and acceleration of CoM

fig,ax = plt.subplots()
ax.plot(times, v_com[:, 0])
ax.plot(times, v_com[:, 1])
ax.plot(times, v_com[:, 2])
sns.despine()
fig.set_tight_layout(True)

fig,ax = plt.subplots()
ax.plot(times, a_com[:, 0])
ax.plot(times, a_com[:, 1])
ax.plot(times, a_com[:, 2])
sns.despine()
fig.set_tight_layout(True)

# need to scale the velocity vector
vel_scale = np.linalg.norm(v_com, axis=1).max()
v_com_norm = v_com / vel_scale


# %%

figure(); plot(v_com[:, 1], v_com[:, 2]); axis('equal'); xlim(0, 10000);


# %% Movie of wingedness with velocity overlay

from matplotlib.animation import FuncAnimation

#from scalebars import add_scalebar

fig, ax = plt.subplots(figsize=(12.7, 4.17))
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

#sb = add_scalebar(ax, matchx=False, matchy=False, hidex=True, hidey=True,
#                  loc=3, sizex=100, labelx='100 mm', sizey=0, labely='100 mm')

time_template = 'time = {0:.3f} sec'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

alpha = 1
Qc = ax.quiver(0, 0, [], [], scale=10, width=.003, color=bmap[0])
head, = ax.plot([], [], 'o', c=bmap[1], ms=9, alpha=alpha)
body, = ax.plot([], [], '-', c=bmap[1], alpha=alpha)
tail, = ax.plot([], [], '-', c=bmap[3], alpha=alpha)
pos, = ax.plot([], [], 'o', c=bmap[2], ms=12)
neg, = ax.plot([], [], 'o', ms=12, mew=2, mfc='none', mec=bmap[2])

ax.set_aspect('equal')#, adjustable='box')
ax.set_xlim(np.r_[-np.abs(limx).max(), np.abs(limx).max()] / chord_len)
ax.set_ylim(np.r_[-np.abs(limy).max(), np.abs(limy).max()] / chord_len)

ax.set_xlabel('y position (chords)')
ax.set_ylabel('z position (chords)')

sns.despine()
fig.set_tight_layout(True)


def init():
    head.set_data([], [])
    body.set_data([], [])
    tail.set_data([], [])
    pos.set_data([], [])
    neg.set_data([], [])
    time_text.set_text('')
    Qc.set_UVC(0, 0)
    Qc.set_offsets(np.array([[0, 0]]))
    return head, body, tail, pos, neg, time_text, Qc


def animate(i):
    xx, yy, zz = spl_c[i].T / chord_len
    ix_pos, ix_neg = ixs_pos[i], ixs_neg[i]

    body.set_data(yy[:vent_idx_spls[i] + 1], zz[:vent_idx_spls[i] + 1])
    tail.set_data(yy[vent_idx_spls[i]:], zz[vent_idx_spls[i]:])
    head.set_data(yy[0], zz[0])
    pos.set_data(yy[ix_pos], zz[ix_pos])
    neg.set_data(yy[ix_neg], zz[ix_neg])
    time_text.set_text(time_template.format(i * dt))

#    # velocities for quiver plot
#    vxx, vyy, vzz = v_spl[i, ixs_wing[i]].T / vel_scale
#    vy = np.r_[v_com_norm[i, 1], vyy]
#    vz = np.r_[v_com_norm[i, 2], vzz]
#    py = np.r_[0, yy[ixs_wing[i]]]
#    pz = np.r_[0, zz[ixs_wing[i]]]
#    pquiver = np.c_[py, pz]
#    Qc.set_UVC(vy, vz)
#    Qc.set_offsets(pquiver)

    Qc.set_UVC(v_com_norm[i, 1], v_com_norm[i, 2])

    return head, body, tail, pos, neg, time_text, Qc


slowed = 10
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False, init_func=init)


#ani.save('../anim_i110/wings_com_quiver.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


# %% Movie of the wingedness

from matplotlib.animation import FuncAnimation

#from scalebars import add_scalebar

fig, ax = plt.subplots(figsize=(12.7, 4.17))
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

#sb = add_scalebar(ax, matchx=False, matchy=False, hidex=True, hidey=True,
#                  loc=3, sizex=100, labelx='100 mm', sizey=0, labely='100 mm')

time_template = 'time = {0:.3f} sec'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

alpha = 1
head, = ax.plot([], [], 'o', c=bmap[1], ms=9, alpha=alpha)
body, = ax.plot([], [], '-', c=bmap[1], alpha=alpha)
tail, = ax.plot([], [], '-', c=bmap[3], alpha=alpha)
pos, = ax.plot([], [], 'o', c=bmap[2], ms=12)
neg, = ax.plot([], [], 'o', ms=12, mew=2, mfc='none', mec=bmap[2])

ax.set_aspect('equal')#, adjustable='box')
ax.set_xlim(np.r_[-np.abs(limx).max(), np.abs(limx).max()] / chord_len)
ax.set_ylim(np.r_[-np.abs(limy).max(), np.abs(limy).max()] / chord_len)
sns.despine()
fig.set_tight_layout(True)


def init():
    head.set_data([], [])
    body.set_data([], [])
    tail.set_data([], [])
    pos.set_data([], [])
    neg.set_data([], [])
    time_text.set_text('')
    # Q1.set_UVC(0, 0)
    # Q2.set_UVC(0, 0)
    return head, body, tail, pos, neg, time_text


def animate(i):
    xx, yy, zz = spl_c[i].T / chord_len
    ix_pos, ix_neg = ixs_pos[i], ixs_neg[i]

#    nburn = 10
#    if i < nburn:
#        ix_pos_old = np.hstack(ixs_pos[:i])
#        ix_neg_old = np.hstack(ixs_neg[:i])
#    else:
#        ix_pos_old = np.hstack(ixs_pos[i - nburn:i])
#        ix_neg_old = np.hstack(ixs_neg[i - nburn:i])
#
#    npos_old = len(ix_pos_old)
#    yz_pos = np.zeros((npos_old, 2))
#    for ii in np.arange(npos_old):
#        yz_pos[
#
#    nneg_old = len(ix_neg_old)
#    yz_neg = np.zeros((nneg_old, 2))

    body.set_data(yy[:vent_idx_spls[i] + 1], zz[:vent_idx_spls[i] + 1])
    tail.set_data(yy[vent_idx_spls[i]:], zz[vent_idx_spls[i]:])
    head.set_data(yy[0], zz[0])
    pos.set_data(yy[ix_pos], zz[ix_pos])
    neg.set_data(yy[ix_neg], zz[ix_neg])
#    pos.set_data(yy[ix_pos_old], zz[ix_pos_old])
#    neg.set_data(yy[ix_neg_old], zz[ix_neg_old])

    # Q1.set_UVC(e1[0], e1[1])
    # Q2.set_UVC(e2[0], e2[1])
    time_text.set_text(time_template.format(i * dt))
    return head, body, tail, pos, neg, time_text


slowed = 10
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False, init_func=init)


#ani.save('{}X A = {}, wave_len_m = {}, freq_hz = {} realtime.mp4'.format(
#         slowed, A, wave_length_m, freq_undulation_hz),
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])



# %%

i = 110


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
#ax.plot(ix, xx[ix], 'o', c=bmap[0])
#ax.plot(spl_c[100:110, :, 0].T)
ax.plot(xx, c=bmap[0])
ax.plot(ix_pos, xx[ix_pos], 'o', c=bmap[2], ms=12)
ax.plot(ix_neg, xx[ix_neg], 'o', ms=12, mew=2, mfc='none', mec=bmap[2])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots(figsize=(12.7, 4.17))
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)
ax.plot(yy[:vent_idx_spls[i] + 1], zz[:vent_idx_spls[i] + 1], '-', c=bmap[1])
ax.plot(yy[vent_idx_spls[i]:], zz[vent_idx_spls[i]:], '-', c=bmap[3])
ax.plot(yy[0], zz[0], 'o', c=bmap[1], ms=9)
ax.plot(yy[ix_pos], zz[ix_pos], 'o', c=bmap[2], ms=12)
ax.plot(yy[ix_neg], zz[ix_neg], 'o', ms=12, mew=2, mfc='none', mec=bmap[2])
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(np.r_[-np.abs(limx).max(), np.abs(limx).max()])
ax.set_ylim(np.r_[-np.abs(limy).max(), np.abs(limy).max()])
sns.despine()
fig.set_tight_layout(True)


    # %% Plot forces on the body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 110

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


# %% Intersections with the mid plane

# locate where the body pierces the plane (x direction changes sign)
#ix = np.where(us[:-1] * us[1:] < 0)[0]


# %% Movie of intersections with the mid plane





# %%

#sk = 1
# chat
#mlab.quiver3d(sple_s[i, ::sk, 0], sple_s[i, ::sk, 1], sple_s[i, ::sk, 2],
#          tcbe[i, ::sk, 1, 0], tcbe[i, ::sk, 1, 1], tcbe[i, ::sk, 1, 2],
#          color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
## bhat
#mlab.quiver3d(sple_s[i, ::sk, 0], sple_s[i, ::sk, 1], sple_s[i, ::sk, 2],
#          tcbe[i, ::sk, 2, 0], tcbe[i, ::sk, 2, 1], tcbe[i, ::sk, 2, 2],
#          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

# plan-view of snake
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], 0 * foils[i, :, :, 2],
#                 color=bmap[5], opacity=.5)

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

# %% Try rotating the body given the COM

#mu = np.deg2rad(35)
#Rmu = np.array([[np.cos(mu), -np.sin(mu), 0],
#                [np.sin(mu), np.cos(mu), 0],
#                [0, 0, 1]])
#

from scipy import linalg

spl = SPL.copy()
#for i in np.arange(ntime):
#    spl[i] = np.dot(Rmu, spl[i].T).T

# shift to 'com'. Note, this is not strictly needed
com = spl.mean(axis=1)
com0 = com[0]
spl = spl - com0
com = com - com0
pfs = pf - com0

com_rot = com.copy()
spl_rot = spl.copy()
pf_rot = pfs.copy()

mu_rot = np.zeros(ntime - 1)
mu_rot_old = np.zeros(ntime - 1)
rot_mat = np.zeros((ntime - 1, 3, 3))

colors = sns.dark_palette("purple", n_colors=ntime)
fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)

# iterate through the points a find the successive roations
mu_comps = []
imax = ntime - 1
#for i in np.arange(imax): #(ntime - 1):
for i in np.arange(1, ntime):
#    uu = com_rot[i + 1] - com_rot[i]
    uu = com_rot[i]
    uhat = uu / np.linalg.norm(uu)
    mu = np.arctan2(uhat[0], uhat[1])  # tan^-1(px / py)
    Rmu = np.array([[np.cos(mu), -np.sin(mu), 0],
                    [np.sin(mu),  np.cos(mu), 0],
                    [0, 0, 1]])
    mu_rot[i - 1] = mu
    rot_mat[i - 1] = Rmu

    # apply the rotation to each point
    # com_rot[i + 1:] = np.dot(Rmu, com_rot[i + 1:].T).T
    for ii in np.arange(i, ntime):
        com_rot[ii] = np.dot(Rmu, com_rot[ii].T).T
        spl_rot[ii] = np.dot(Rmu, spl_rot[ii].T).T
        pf_rot[ii] = np.dot(Rmu, pf_rot[ii].T).T

    if i % 20 == 0:
        # ax.plot(com_rot[:, 0], com_rot[:, 1], c=colors[i], alpha=.5)
        ax.plot(com_rot[i:, 0], com_rot[i:, 1], c=colors[i], alpha=.5)
        # ax.plot(com_rot[:i, 0], com_rot[:i, 1], c=colors[i], alpha=.5)

#ax.plot(com[:, 0], com[:, 1], c=bmap[2])
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(times[1:], np.rad2deg(mu_rot), 'o-')
ax.margins(.03)
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.plot(com[:, 0], com[:, 1], 'o-', c=bmap[2])
#ax.plot(com_rot[:, 0], com_rot[:, 1], 'o-', c=bmap[0])
ax.plot(com_rot[:, 0], com_rot[:, 1], '-', c=bmap[0])
ax.plot(com_rot[:, 0], com_rot[:, 1], 'o', c=bmap[0])
ax.set_aspect('equal', adjustable='box')
#ax.set_xticks([])
#ax.set_yticks([])
sns.despine()
fig.set_tight_layout(True)


# %% Now iterate back through, after adding a neck

neck_len = 30  # 30 mm or 3 cm
pfe = np.zeros((ntime, nmark + 1, 3))  # "extended" smooth points
pfe[:, 0] = pfs[:, 0]
pfe[:, 1] = pfs[:, 0] + np.array([0, -neck_len, 0])
pfe[:, 2:] = pfs[:, 1:]

sple = np.zeros((ntime, nspl, 3))  # inertial frame
sple_s = np.zeros((ntime, nspl, 3))  # com centered
tcbe = np.zeros((ntime, nspl, 3, 3))  # tangent, chord, backbone directions
Crs = np.zeros((ntime, nspl, 3, 3))  # rotation matrices along body
Rgams = np.zeros((ntime, nspl, 3, 3))  # rotation matrix for tangent angles

lengths_norm = np.zeros((ntime, nmark - 1))
lengths_norm_per = np.zeros((ntime, nmark - 1))

for i in np.arange(ntime):

    # spline fit
    tck, fpe = splprep(pfe[i, :, :].T.tolist(), k=3, s=0)
    ts = np.linspace(0, 1, nspl)

    xs, ys, zs = splev(ts, tck)
    dx, dy, dz = splev(ts, tck, der=1)
    ddx, ddy, ddz = splev(ts, tck, der=2)
    dddx, dddy, dddz = splev(ts, tck, der=3)

    r = np.c_[xs, ys, zs]
    dr = np.c_[dx, dy, dz]
    ddr = np.c_[ddx, ddy, ddz]
    dddr = np.c_[dddx, dddy, dddz]

    # account for the 'virtual' marker
    fp = np.r_[fpe[0], fpe[2:]]

    ds = np.sqrt(dx**2 + dy**2 + dz**2)  # arc length to integrate
    total_lens[i] = cumtrapz(ds, ts, initial=0)[-1]

    spl_diff = np.diff((fp + loc_svl[0]) * loc_svl.ptp())
    meas_diff = np.diff(loc_svl)
    lengths_norm[i] = (meas_diff - spl_diff) * 100
    lengths_norm_per[i] = (meas_diff - spl_diff) / meas_diff * 100

    sple[i] = np.c_[xs, ys, zs]
    sple_s[i] = sple[i] - com_rot[i]  # TODO: do this afterwards when recenter

    # TNB frame
    # https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
    Tdir = (dr.T / np.linalg.norm(dr, axis=1)).T
    Tdir0 = Tdir[0]  # this will be point back in -Yhat direction

    # find angle between Tdir0 and Yhat (rotate about Zhat)
    gam1 = np.arctan2(Tdir0[0], Tdir0[1])
    Rgam = np.array([[np.cos(gam1), -np.sin(gam1), 0],
                    [np.sin(gam1),  np.cos(gam1), 0],
                    [0, 0, 1]])
    Rgam = Rgam.T

    # Cdir0 in rotated is along -Xhat
    Cdir0 = np.dot(Rgam, np.array([1, 0, 0]))
    Bdir0 = np.cross(Cdir0, Tdir0)

    j = 0
    tcbe[i, :, 0] = Tdir
    tcbe[i, j, 1] = Cdir0
    tcbe[i, j, 2] = Bdir0
    Rgams[i, j] = Rgam
    Crs[i, j] = np.eye(3)  # no rotation at the start

    # now iterate along the body, finding successive rotations
    # Bloomenthal (1990)
    # successive rotations of the previous coordinate system
    for j in np.arange(1, nspl):
        T0 = tcbe[i, j - 1, 0]  # tangent direction at head
        T1 = tcbe[i, j, 0]
        T0 = T0 / np.linalg.norm(T0)
        T1 = T1 / np.linalg.norm(T1)
        A = np.cross(T0, T1)
        A = A / np.linalg.norm(A)  # why have to do this?

        # components of rotation matrix
        a_ang = np.arccos(np.dot(T0, T1))  # 'bending' angle (rotate about A)
        Ax, Ay, Az = A
        sqx, sqy, sqz = A**2
        cos = np.dot(T0, T1)
        cos1 = 1 - cos
        xycos1 = Ax * Ay * cos1
        yzcos1 = Ay * Az * cos1  # check on Az
        zxcos1 = Ax * Az * cos1
        sin = np.sqrt(1 - cos**2)
        xsin, ysin, zsin =  A * sin

        # make the rotation matrix
        Cr = np.array([[sqx + (1 - sqx) * cos, xycos1 + zsin, zxcos1 - ysin],
                       [xycos1 - zsin, sqy + (1 - sqy) * cos, yzcos1 + xsin],
                       [zxcos1 + ysin, yzcos1 - xsin, sqz + (1 - sqz) * cos]])

        # not 100% on why need to transpose (active vs. passive rotation?)
        # https://en.wikipedia.org/wiki/Active_and_passive_transformation
        Cr = Cr.T

        # store rotation matrix for the foil
        Crs[i, j] = Cr

        # store tangent angle rotation matrix
        gam1 = np.arctan2(T1[0], T1[1])
        Rgam = np.array([[np.cos(gam1), -np.sin(gam1), 0],
                        [np.sin(gam1),  np.cos(gam1), 0],
                        [0, 0, 1]])
        Rgam = Rgam.T
        Rgams[i, j] = Rgam

        # rotate the old chord and backbone coordinates
        C0 = tcbe[i, j - 1, 1]  # chord direction at head
        B0 = tcbe[i, j - 1, 2]  # backbone direction at head
        C1 = np.dot(Cr, C0)
        B1 = np.dot(Cr, B0)
        tcbe[i, j, 1] = C1
        tcbe[i, j, 2] = B1
        b_ang = np.arccos(np.dot(C0, C1))  # 'twisting' angle


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
for j in np.arange(nmark - 1):
    l1 = r'm$_\mathrm{' + str(j + 1) + '}$'
    l2 = r'm$_\mathrm{' + str(j) + '}$'
    label = l1 + u' \u2013 ' + l2
    ax.plot(times, lengths_norm[:, j], c=colors_mark[j], label=label)
ax.legend(loc='best', ncol=3, frameon=True, framealpha=.5)
ax.set_xlabel('time (s)')
ax.set_ylabel('spline fit, % svl')
sns.despine()
fig.set_tight_layout(True)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
#mlab.plot3d(com_rot[:, 0], com_rot[:, 1], com_rot[:, 2],
#            tube_radius=10, color=bmap[0])
#mlab.plot3d(com[:, 0], com[:, 1], com[:, 2],
#            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(spl_rot[i, :, 0], spl_rot[i, :, 1], spl_rot[i, :, 2],
                color=bmap[0], tube_radius=3)
    mlab.points3d(pf_rot[i, :, 0], pf_rot[i, :, 1], pf_rot[i, :, 2],
                  color=bmap[0], scale_factor=20, resolution=64)
    mlab.plot3d(spl[i, :, 0], spl[i, :, 1], spl[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfs[i, :, 0], pfs[i, :, 1], pfs[i, :, 2],
                  color=bmap[2], scale_factor=20, resolution=64)
    mlab.plot3d(sple[i, :, 0], sple[i, :, 1], sple[i, :, 2],
                color=bmap[3], tube_radius=3)
    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)
mlab.orientation_axes()
fig.scene.isometric_view()


# %% Orientation of the airfoil

rfoil = np.genfromtxt('../Data/Foil/snake0.004.bdy.txt', skip_header=1)
rfoil = rfoil - rfoil.mean(axis=0)
rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
rfoil /= np.ptp(rfoil[:, 0])
rfoil = rfoil[::5]
# rfoil = np.c_[np.zeros(rfoil.shape[0]), rfoil]
_r0 = np.zeros(rfoil.shape[0])  # 0 in Yhat direction to start
rfoil = np.c_[rfoil[:, 0], _r0, rfoil[:, 1]]  # in XZ frame to start
rfoil = np.c_[rfoil.T, rfoil[0]].T
nfoil = rfoil.shape[0]

chord_len = 22  # mm
foil_scale = np.ones(nspl)
# foil_scale[:n_neck] = np.linspace(0.55, .95, n_neck)
foil_scale *= chord_len  # scale by the chord length

# airfoil shape
foils = np.zeros((ntime, nspl, nfoil, 3))

for i in np.arange(ntime):
    for j in np.arange(nspl):
        # rotate into XZ plane
        foil0 = np.dot(Rgams[i, j], rfoil.T).T

        # rotate into CB plane
        rotated_foil = np.dot(Crs[i, j], foil0.T).T

        # scale and move to position along the body
        foils[i, j] = sple_s[i, j] + foil_scale[j] * rotated_foil


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 125

#head = mlab.points3d(r[i, 0, 0], r[i, 0, 1], r[i, 0, 2], color=bmap[1],
#                     scale_factor=.015, resolution=16, opacity=.5)
#head = mlab.points3d(foils[i, 0, 0, 0], foils[i, 0, 0, 1], foils[i, 0, 50, 2],
#                     color=bmap[1], scale_factor=15, resolution=16, opacity=.5)
body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 color=bmap[1], opacity=1)

sk = 1
# chat
#mlab.quiver3d(sple_s[i, ::sk, 0], sple_s[i, ::sk, 1], sple_s[i, ::sk, 2],
#          tcbe[i, ::sk, 1, 0], tcbe[i, ::sk, 1, 1], tcbe[i, ::sk, 1, 2],
#          color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
## bhat
#mlab.quiver3d(sple_s[i, ::sk, 0], sple_s[i, ::sk, 1], sple_s[i, ::sk, 2],
#          tcbe[i, ::sk, 2, 0], tcbe[i, ::sk, 2, 1], tcbe[i, ::sk, 2, 2],
#          color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

# plan-view of snake
#body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], 0 * foils[i, :, :, 2],
#                 color=bmap[5], opacity=.5)

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

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#mlab.mesh(spl_rot[:, :, 0], spl_rot[:, :, 1], spl_rot[:, :, 2],
#          opacity=.25, representation='surface')

#mlab.mesh(sple[:, :, 0], sple[:, :, 1], sple[:, :, 2],
#          opacity=.25, representation='surface')

mlab.plot3d(pfe[:, vent_idx + 1, 0], pfe[:, vent_idx + 1, 1],
            pfe[:, vent_idx + 1, 2], color=bmap[0], tube_radius=2)

for i in np.arange(ntime)[::10]:
    mlab.plot3d(sple[i, :, 0], sple[i, :, 1], sple[i, :, 2],
                color=bmap[3], tube_radius=3)

sk = 1
for i in np.arange(ntime)[::10]:
    # that
#    mlab.quiver3d(sple[i, ::sk, 0], sple[i, ::sk, 1], sple[i, ::sk, 2],
#              tcbe[i, ::sk, 0, 0], tcbe[i, ::sk, 0, 1], tcbe[i, ::sk, 0, 2],
#              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
    # chat
    mlab.quiver3d(sple[i, ::sk, 0], sple[i, ::sk, 1], sple[i, ::sk, 2],
              tcbe[i, ::sk, 1, 0], tcbe[i, ::sk, 1, 1], tcbe[i, ::sk, 1, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    # bhat
    mlab.quiver3d(sple[i, ::sk, 0], sple[i, ::sk, 1], sple[i, ::sk, 2],
              tcbe[i, ::sk, 2, 0], tcbe[i, ::sk, 2, 1], tcbe[i, ::sk, 2, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

mlab.orientation_axes()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime)[::20]:
    mlab.points3d(pf_rot[i, :, 0], pf_rot[i, :, 1], pf_rot[i, :, 2],
                  color=bmap[0], scale_factor=20, resolution=64)
    mlab.points3d(pfs[i, :, 0], pfs[i, :, 1], pfs[i, :, 2],
                  color=bmap[2], scale_factor=20, resolution=64)
    mlab.points3d(pfe[i, :, 0], pfe[i, :, 1], pfe[i, :, 2],
                  color=bmap[3], scale_factor=20, resolution=64)
mlab.orientation_axes()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
mlab.plot3d(com_rot[:, 0], com_rot[:, 1], com_rot[:, 2],
            tube_radius=10, color=bmap[0])
mlab.plot3d(com[:, 0], com[:, 1], com[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(spl_rot[i, :, 0], spl_rot[i, :, 1], spl_rot[i, :, 2],
                color=bmap[0], tube_radius=3)
    mlab.points3d(pf_rot[i, :, 0], pf_rot[i, :, 1], pf_rot[i, :, 2],
                  color=bmap[0], scale_factor=20, resolution=64)
    mlab.plot3d(spl[i, :, 0], spl[i, :, 1], spl[i, :, 2],
                color=bmap[2], tube_radius=3)
    mlab.points3d(pfs[i, :, 0], pfs[i, :, 1], pfs[i, :, 2],
                  color=bmap[2], scale_factor=20, resolution=64)
mlab.orientation_axes()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
mlab.plot3d(com_rot[:, 0], com_rot[:, 1], com_rot[:, 2],
            tube_radius=10, color=bmap[0])
#mlab.plot3d(com[:, 0], com[:, 1], com[:, 2],
#            tube_radius=10, color=bmap[2])

mlab.mesh(spl_rot[:, :, 0], spl_rot[:, :, 1], spl_rot[:, :, 2],
          opacity=.25, representation='surface')

for i in np.arange(ntime)[::10]:
    mlab.plot3d(spl_rot[i, :, 0], spl_rot[i, :, 1], spl_rot[i, :, 2],
                   color=bmap[0], tube_radius=3)
#    mlab.plot3d(spl[i, :, 0], spl[i, :, 1], spl[i, :, 2],
#                   color=bmap[2], tube_radius=3)
mlab.orientation_axes()


# %%

fig, ax = plt.subplots()
for i in np.arange(50):
    ax.axhline(0, color='gray', lw=1)
    ax.plot(spl_rot[i, :, 0])
sns.despine()
fig.set_tight_layout(True)


# %%

a = np.r_[1, 0, 0]
c = np.r_[0, 1, 0]
a = np.c_[a, c].T

mu = np.deg2rad(.0005)
Rmu = np.array([[np.cos(mu), -np.sin(mu), 0],
                [np.sin(mu),  np.cos(mu), 0],
                [0, 0, 1]])

b = np.dot(Rmu, a.T).T

print np.rad2deg(np.arctan2(b[:, 1], b[:, 0]))

fig, ax = plt.subplots()
ax.plot([a[0], b[0]], [a[1], b[1]], '-o')
ax.axvline(1, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %%

dcom_rot = np.diff(com_rot, axis=0)
dcom_rot = (dcom_rot.T / np.linalg.norm(dcom_rot)).T

dcom_rot = np.diff(com_rot, axis=0) / dt
np.cumsum(np.sqrt(np.sum(dcom_rot**2, axis=1)))

#dcom_rot = np.zeros((ntime, 3))
#dcom_rot[:, 1] = 1

np.dot(dcom_rot, np.r_[1, 0, 0])


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
mlab.plot3d(com_rot[:, 0], com_rot[:, 1], com_rot[:, 2],
            tube_radius=10, color=bmap[0])
mlab.plot3d(com[:, 0], com[:, 1], com[:, 2],
            tube_radius=10, color=bmap[2])

for i in np.arange(ntime)[::10]:
    mlab.plot3d(spl_rot[i, :, 0], spl_rot[i, :, 1], spl_rot[i, :, 2],
                   color=bmap[0], tube_radius=3)
    mlab.plot3d(spl[i, :, 0], spl[i, :, 1], spl[i, :, 2],
                   color=bmap[2], tube_radius=3)
mlab.orientation_axes()


# %% Prepare data for 3D plots

pf, vf, af = bup, buv, bua

# shift the data to the centroid (rough com)
com = pf.mean(axis=1)
pf_s = (pf.swapaxes(0, 1) - com).swapaxes(0, 1)
SPL_s = (SPL.swapaxes(0, 1) - com).swapaxes(0, 1)

com_r = pr.mean(axis=1)
pr_s = (pr.swapaxes(0, 1) - com_r).swapaxes(0, 1)

# 2D marker and time arraws
M, T = np.meshgrid(np.arange(nmark) + 1, times)

colors_mark = sns.husl_palette(n_colors=nmark)
# time_colors = sns.color_palette('Greens', n_colors=ntime)


# %% Try PCA to the body (without regard to velocity) to describe shape

def cpm(v):
    """Skew-symmetric cross-product matrix.

    Parameters
    -----------
    v : array, size (3)
        vector to make cross-product matrix from

    Returns
    -------
    cpm_v : array, size (3, 3)
        cross-product matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot_prin_iner_old(a_prin, b_iner):
    """Find the rotation matrix to take the vector in the principle
    frame and align it with the vector in the inertial frame.
    """

    a = a_prin / np.linalg.norm(a_prin)
    b = b_iner / np.linalg.norm(b_iner)

    unum = np.dot(a, b) * a
    vnum = b - unum
    if np.allclose(unum, 0):
        u = unum
    else:
        u = unum / np.linalg.norm(unum)
    v = vnum / np.linalg.norm(vnum)
    w = np.cross(b, a)
    F = np.c_[u, v, w].T

    v = np.cross(a, b)
    s = np.linalg.norm(v)  # sine of angle
    c = np.dot(a, b)  # cosine of angle
    G = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])

    R = np.linalg.inv(F).dot(G.dot(F))
    return R


def rot_prin_iner(a_prin, b_iner):
    """Find the rotation matrix to take the vector in the principle
    frame and align it with the vector in the inertial frame.

    See: http://math.stackexchange.com/a/476311
    """

    a = a_prin / np.linalg.norm(a_prin)
    b = b_iner / np.linalg.norm(b_iner)

    v = np.cross(a, b)
    s = np.linalg.norm(v)  # sine of angle
    c = np.dot(a, b)  # cosine of angle
    vt = cpm(v)
    if s == 0:
        R = np.eye(3)
    else:
        R = np.eye(3) + vt + vt**2 * (1 - c) / s**2

    return R


from scipy import linalg

i = 200

# select just the trunk
pf_b = pf[i, :vent_idx + 1, :]
com_b = pf_b.mean(axis=0)
pf_bc = pf_b - com_b
pf_tot = pf[i] - com_b

# spline fit trunk
nspl = 200
tck, fp = splprep(pf_bc.T.tolist(), k=3, s=0)
ts = np.linspace(0, 1, nspl)
xs, ys, zs = splev(ts, tck)
spl_bc = np.c_[xs, ys, zs]

# spline fit the tail
tckt, fpt = splprep(pf_tot.T.tolist(), k=3, s=0)
ts_tail = np.linspace(fpt[vent_idx], 1, 25)
xt, yt, zt = splev(ts_tail, tckt)
spl_tail = np.c_[xt, yt, zt]

# X = SPL_s[i].T
X = spl_bc.T

R = np.dot(X, X.conj().T)
q = np.linalg.matrix_rank(R)

pov, pom = linalg.eig(R)

index_pca = np.argsort(pov.real)[::-1]  # pov.imag should be 0
pov = pov[index_pca].real
pov = pov / pov.sum()

pom = pom[:, index_pca]  # cols are the different eigenvectors
e1_orig, e2_orig, e3_orig = pom.T
e_orig = [e1_orig, e2_orig, e3_orig]

xhat, yhat, zhat = np.eye(3).T
ider = [yhat, xhat, zhat]  # glide nominally along yhat

e_new = []
for i in np.arange(3):
    if np.dot(e_orig[i], ider[i]) < 0:
        e_new.append(-e_orig[i])
    else:
        e_new.append(e_orig[i])

e1, e2, e3 = e_new
pom_old = pom
pom = np.array(e_new).T
pom_R = np.array([e_new[1], e_new[0], e_new[2]])  # swap order so y first

R1 = rot_prin_iner(e_new[0], ider[0])
R2 = rot_prin_iner(e_new[1], ider[1])
R3 = rot_prin_iner(e_new[2], ider[2])

# now plot the body and PCA
ntail_mark = nmark - 1 - vent_idx
nbody_mark = vent_idx + 2
colors_trunk = sns.light_palette('green', as_cmap=True)(fp)[:, :3]
colors_tail = sns.light_palette('purple', n_colors=ntail_mark)
colors_tail = np.array(colors_tail)[:, :3]
colors_mark = np.r_[colors_trunk, colors_tail]

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

pmarks = []
for jj in np.arange(nmark):
    cc = tuple(colors_mark[jj])
    _plot = mlab.points3d(pf_tot[jj, 0], pf_tot[jj, 1], pf_tot[jj, 2],
                          color=cc, scale_factor=10, scale_mode='none',
                          resolution=64)
    pmarks.append(_plot)

# plot the spline fit
pspl = mlab.plot3d(spl_bc[:, 0], spl_bc[:, 1], spl_bc[:, 2],
                   color=bmap[1], tube_radius=3)

tail_color = tuple(colors_tail.mean(axis=0))
tspl = mlab.plot3d(spl_tail[:, 0], spl_tail[:, 1], spl_tail[:, 2],
                   color=tail_color, tube_radius=3)

# rotation into principle frame (probably don't want transpose)
#spl_rot = np.dot(pom_R.T, spl_bc.T).T
#pspl = mlab.plot3d(spl_rot[:, 0], spl_rot[:, 1], spl_rot[:, 2],
#                   color=bmap[0], tube_radius=3)

## rotation into principle frame
#spl_rot = np.dot(pom_R, spl_bc.T).T
#spl_rot_tail = np.dot(pom_R, spl_tail.T).T
#pspl = mlab.plot3d(spl_rot[:, 0], spl_rot[:, 1], spl_rot[:, 2],
#                   color=bmap[1], tube_radius=3, opacity=.5)
#pspl = mlab.plot3d(spl_rot_tail[:, 0], spl_rot_tail[:, 1], spl_rot_tail[:, 2],
#                   color=tail_color, tube_radius=3, opacity=.5)

Nframe = np.eye(3)
for ii in np.arange(3):
    mlab.quiver3d(pom[0, ii], pom[1, ii], pom[2, ii], scale_factor=40,
                  color=bmap[ii], mode='arrow', resolution=64, line_width=3)
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=bmap[ii], mode='arrow', opacity=.5)

# plot the planes
mins = 1.1 * pf_tot.min(axis=0)
maxs = 1.1 * pf_tot.max(axis=0)
lims_xy = np.c_[mins, maxs].T[:, :2]
xx, yy = np.meshgrid(lims_xy[:, 0], lims_xy[:, 1])
zz_sinus = -1 / e3[2] * (e3[0] * xx + e3[1] * yy)


zz_saggital = -1 / e2[2] * (e2[0] * xx + e2[1] * yy)

mlab.mesh(xx, yy, zz_sinus, opacity=.2, color=bmap[2])
#mlab.mesh(xx, yy, zz_saggital, opacity=.4, color=bmap[1])


# %% Distance from spline to sinus plane

d_body = np.zeros(nspl)
for jj in np.arange(nspl):
    d_body[jj] = np.dot(e3, spl_bc[jj]) / np.linalg.norm(e3)

d_tail = np.zeros(spl_tail.shape[0])
for jj in np.arange(spl_tail.shape[0]):
    d_tail[jj] = np.dot(e3, spl_tail[jj]) / np.linalg.norm(e3)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(np.r_[d_body, d_tail], c=bmap[2])
sns.despine()


# %% Distance from spline to sagittal plane

d_body = np.zeros(nspl)
for jj in np.arange(nspl):
    d_body[jj] = np.dot(e2, spl_bc[jj]) / np.linalg.norm(e2)

d_tail = np.zeros(spl_tail.shape[0])
for jj in np.arange(spl_tail.shape[0]):
    d_tail[jj] = np.dot(e2, spl_tail[jj]) / np.linalg.norm(e2)

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(np.r_[d_body, d_tail], c=bmap[1])
sns.despine()


# %%

nmark, ntime = X.shape

R = np.dot(X, X.conj().T)
q = np.linalg.matrix_rank(R)

pov, pom = linalg.eig(R)

index_pca = np.argsort(pov.real)[::-1]  # pov.imag should be 0
pov = pov[index_pca].real
pom = pom[:, index_pca]

pov_norm = pov / pov.sum()



# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 100

# plot the colored markers
pmarks = []
for j in np.arange(nmark):
    cc = tuple(colors_mark[j])
    _plot = mlab.points3d(pf_s[i, j, 0], pf_s[i, j, 1], pf_s[i, j, 2],
                          color=cc, scale_factor=20, scale_mode='none')
    pmarks.append(_plot)
    mlab.points3d(pr_s[i, j, 0], pr_s[i, j, 1], pr_s[i, j, 2],
                  color=cc, scale_factor=15, scale_mode='none',
                  mode='cube')

# plot the spline fit
pspl = mlab.plot3d(SPL_s[i, :, 0], SPL_s[i, :, 1], SPL_s[i, :, 2],
                   color=bmap[1], tube_radius=5)


# %%

# get the time delay
dtms = dt * 1000
delay = np.around(dtms * 10).astype(np.int)

@mlab.animate(delay=delay)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            print('Current time: {0}'.format(times[i]))

            for j in np.arange(nmark):
                pmarks[j].mlab_source.set(
                    x=pf_s[i, j, 0], y=pf_s[i, j, 1], z=pf_s[i, j, 2])

            pspl.mlab_source.set(
                x=SPL_s[i, :, 0], y=SPL_s[i, :, 1], z=SPL_s[i, ::, 2])

            yield

manim = anim()
mlab.show()


# %%

import mayavi.mlab as mlab
import  moviepy.editor as mpy

#fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(764, 771))

i = 0
# plot the colored markers
pmarks = []
for j in np.arange(nmark):
    cc = tuple(colors_mark[j])
    _plot = mlab.points3d(pf_s[i, j, 0], pf_s[i, j, 1], pf_s[i, j, 2],
                          color=cc, scale_factor=20, scale_mode='none')
    pmarks.append(_plot)

# plot the spline fit
pspl = mlab.plot3d(SPL_s[i, :, 0], SPL_s[i, :, 1], SPL_s[i, :, 2],
                   color=bmap[1], tube_radius=5)


# %%

def make_frame(t):
    i = np.int(np.around(t / dt))
    print('frame {0} for frame ntime'.format(i))

    for j in np.arange(nmark):
        pmarks[j].mlab_source.set(
            x=pf_s[i, j, 0], y=pf_s[i, j, 1], z=pf_s[i, j, 2])

    pspl.mlab_source.set(
        x=SPL_s[i, :, 0], y=SPL_s[i, :, 1], z=SPL_s[i, ::, 2])

    return mlab.screenshot(antialiased=True)

animation = mpy.VideoClip(make_frame, duration=times[-1])
#
## animation.write_gif("sinc.gif", fps=20)
animation.write_videofile('413_91_60fps.mp4', fps=179, codec='libx264',
                          audio=False, ffmpeg_params=['-pix_fmt', 'yuv420p'])


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(nmark):
    cc = tuple(colors_mark[j])
    mlab.points3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                  color=cc, scale_factor=15, scale_mode='none')
    mlab.plot3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                color=cc, tube_radius=4)


# %% 3D grid of points and spline

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(pcom[:, 0], pcom[:, 1], pcom[:, 2],
            color=bmap[0], tube_radius=10)

for j in np.arange(nmark):
    cc = tuple(colors[j])
    mlab.points3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                  color=cc, scale_factor=15, scale_mode='none')
    mlab.plot3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                color=cc, tube_radius=4)

for i in np.arange(ntime)[::5]:
    mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], opacity=1,
                color=time_colors[i], tube_radius=5)
    mlab.points3d(SPL[i, 0, 0], SPL[i, 0, 1], SPL[i, 0, 2], opacity=1,
                  color=time_colors[i], scale_factor=20, scale_mode='none')


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for i in [50]:
    # http://stackoverflow.com/a/19667996
    _s = mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], KAPs[i],
                colormap='RdBu', tube_radius=4, vmin=vmin, vmax=vmax)
    _s.module_manager.scalar_lut_manager.reverse_lut = True

    for j in np.arange(nmark):
        mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                      color=tuple(colors_mark[j]), scale_factor=15)

    sk = 1
#    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
#                  TNB[i, ::sk, 0, 0], TNB[i, ::sk, 0, 1], TNB[i, ::sk, 0, 2],
#                  color=bmap[0], mode='arrow', resolution=64, scale_factor=50)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 1, 0], TNB[i, ::sk, 1, 1], TNB[i, ::sk, 1, 2],
                  color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 2, 0], TNB[i, ::sk, 2, 1], TNB[i, ::sk, 2, 2],
                  color=bmap[2], mode='arrow', resolution=64, scale_factor=25)


# %%

colors_mark = sns.husl_palette(n_colors=nmark)

vmin, vmax = -.05, .05
fig, ax = plt.subplots()
cax = ax.pcolormesh(TT, SS, KAPs, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

ax.contour(TT, SS, KAPs, [0], colors=emerald_green)

ax.plot(times, MARK[:, vent_idx], color=colors_mark[vent_idx])

for j in np.arange(nmark):
    ax.plot(times, MARK[:, j], color=colors_mark[j])
    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')

ax.contour(TT, SS, TAU, [-.05, .05], colors=('w', 'k'))

ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %% Curvature map, but swap axes (x=length, y=time)

colors_mark = sns.husl_palette(n_colors=nmark)

vmin, vmax = -.05, .05
fig, ax = plt.subplots()
cax = ax.pcolormesh(SS, TT, KAPs, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

ax.contour(SS, TT, KAPs, [0], colors=emerald_green)

ax.plot(MARK[:, vent_idx], times, color=colors_mark[vent_idx])

for j in np.arange(nmark):
    ax.plot(MARK[:, j], times, color=colors_mark[j])
    ax.axvline(dist_arclen_markers[j], color=colors_mark[j], ls='--')

ax.contour(SS, TT, TAU, [-.05, .05], colors=('w', 'k'))

ax.set_ylim(TT.min(), TT.max())
ax.set_xlim(SS.min(), SS.max())
ax.set_ylabel('time (s)')
ax.set_xlabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %%

#d = np.load('/Users/isaac/Desktop/413_91_markers_spline_Gary.npz')
#
#KAPs = d['curvature_signed']
#TT = d['spline_times']
#SS = d['spline_coord']

colors_mark = sns.husl_palette(n_colors=nmark)

vmin, vmax = -.05, .05
clines = np.linspace(vmin, vmax, 51)
fig, ax = plt.subplots()
cax = ax.contourf(TT, SS, KAPs, clines, cmap=plt.cm.coolwarm)# vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

ax.contour(TT, SS, KAPs, [0], colors=emerald_green)

ax.plot(times, MARK[:, vent_idx] , color=colors_mark[vent_idx])
#
#for j in np.arange(nmark):
#    ax.plot(times, MARK[:, j] , color=colors_mark[j])
#    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')
#
#ax.contour(TT, SS, TAU, [-.05, .05], colors=('w', 'k'))

ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %% Slope of the 0 curvature lines

startx = np.r_[.0239, .0069, .1552, .5942, .9511]
starty = np.r_[382.65, 152.936, 2.719, 4.8654, 7.01196]
stopx = np.r_[.5900, .7173, 1.3286, 1.36141, 1.36551]
stopy = np.r_[854.89, 738.98, 801., 590.71, 344.015]

slope = (stopy - starty) / (stopx - startx)  # mm / s
slope_svl = slope / dist_svl


vmin, vmax = -.05, .05
clines = np.linspace(vmin, vmax, 51)
fig, ax = plt.subplots()
cax = ax.contourf(TT, SS, KAPs, clines, cmap=plt.cm.coolwarm,
                  vmin=vmin, vmax=vmax, extend='both')
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

for ii in np.arange(len(startx)):
    xx = [startx[ii], stopx[ii]]
    yy = [starty[ii], stopy[ii]]
    ax.plot(xx, yy, '--', color='white')

ax.contour(TT, SS, KAPs, [0], colors=[bmap[1]])

ax.plot(times, MARK[:, vent_idx] , color=colors_mark[vent_idx])
#
#for j in np.arange(nmark):
#    ax.plot(times, MARK[:, j] , color=colors_mark[j])
#    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')
#
#ax.contour(TT, SS, TAU, [-.05, .05], colors=('w', 'k'))

ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %% Time between peaks in curvature

dists = np.r_[200, 300, 400, 500]
k200 = np.r_[.3281, .6594, 1.0658, 1.3489]
k300 = np.r_[.4688, .8116, 1.1930]
k400 = np.r_[.1355, .5900, .906]
k500 = np.r_[.0478, .3626, .6833, 1.067]
k_dists = [k200, k300, k400, k500]

dk200 = np.diff(k200)
dk300 = np.diff(k300)
dk400 = np.diff(k400)
dk500 = np.diff(k500)

T200 = 2 * dk200.mean()
T300 = 2 * dk300.mean()
T400 = 2 * dk400.mean()
T500 = 2 * dk500.mean()

periods = np.r_[T200, T300, T400, T500]
freqs = 1 / periods
freq_mean = freqs.mean()
freq_std = freqs.std()

# plot the curvature profiles and the frequencies
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True,
                        figsize=(4.85, 9))

for ii in np.arange(len(dists)):
    dist = dists[ii]
    idx = np.argmin(np.abs(SS - dist), axis=1)

    kap_s = np.zeros(ntime)
    for nn in np.arange(ntime):
        kap_s[nn] = KAPs[nn, idx[nn]]

    axs[ii].axhline(0, color='gray', lw=.5)
    k_peaks = k_dists[ii]
    for jj in np.arange(len(k_peaks)):
        axs[ii].axvline(k_peaks[jj], color='gray', lw=.75)
    axs[ii].plot(times, kap_s)

    text = 's = {0} mm, f = {1:.2f} Hz'.format(dists[ii], freqs[ii])
    axs[ii].text(0.05, 0.85, text, transform=axs[ii].transAxes,
        color='k', fontsize='medium')
    axs[ii].set_ylim(-.15, .15)
    axs[ii].set_yticks([-.1, 0, .1])

axs[-1].set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %% Length between peaks in curvature

time_test = .4
idx = np.argmin(np.abs(TT - time_test), axis=0)

kap_t = np.zeros(nspl)
for ns in np.arange(nspl):
    ii = idx[ns]
    kap_t[ns] = KAPs[ii, ns]


fig, ax = plt.subplots()
ax.axvline(dist_svl, color='gray', lw=.75)
ax.plot(SS[ii, :], kap_t)
sns.despine()
fig.set_tight_layout(True)



# %%

ii = 50
dii = 25

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(SS[ii], KAPs[ii])
ax.plot(SS[ii + dii], KAPs[ii + dii])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(SS[ii], TAU[ii])
ax.plot(SS[ii + dii], TAU[ii + dii])
sns.despine()
fig.set_tight_layout(True)


# %%

colors_mark = sns.husl_palette(n_colors=nmark)

vmin, vmax = -.05, .05
fig, ax = plt.subplots()
cax = ax.pcolormesh(TT, SS, KAPs, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")
#for j in np.arange(nmark):
#    ax.plot(times, MARK[:, j] , color=colors_mark[j])
#    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')
ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %%

vmin, vmax = -.025, .025
fig, ax = plt.subplots()
cax = ax.pcolormesh(TT, SS, TAU, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5)
cbar.set_label(r'torsion, $\tau$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

ax.contour(TT, SS, KAPs, [0], colors=emerald_green)

ax.plot(times, MARK[:, vent_idx] , color=colors_mark[vent_idx])

#for j in np.arange(nmark):
#    ax.plot(times, MARK[:, j] , color=colors_mark[j])
#    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')
ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %%

colors_mark = sns.husl_palette(n_colors=nmark)

vmin, vmax = 0, .05
clines = np.linspace(0, .05, 41)
fig, ax = plt.subplots()
#cax = ax.pcolormesh(TT, SS, KAPu, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
cax = ax.contourf(TT, SS, KAPu, clines, cmap=plt.cm.coolwarm)#, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=.5,
                    ticks=[0, .025, .05])
cbar.set_label(r'curvature, $\kappa$ (1/mm)', fontsize='medium')
cbar.solids.set_edgecolor("face")

ax.contour(TT, SS, KAPu, [.005], colors='black')

#ax.plot(times, MARK[:, vent_idx] , color=colors_mark[vent_idx])
#
#for j in np.arange(nmark):
#    ax.plot(times, MARK[:, j] , color=colors_mark[j])
#    ax.axhline(dist_arclen_markers[j], color=colors_mark[j], ls='--')
#
#ax.contour(TT, SS, TAU, [-.05, .05], colors=('w', 'k'))

ax.set_xlim(TT.min(), TT.max())
ax.set_ylim(SS.min(), SS.max())
ax.set_xlabel('time (s)')
ax.set_ylabel('distance along body (mm)')
sns.despine(ax=ax)
#sns.despine(fig=fig, ax=cax, left=True, right=False)
fig.set_tight_layout(True)


# %% Prepare data for 3D plots

pf, vf, af = bup, buv, bua

# rought com assuming uniform linear mass density
_, coms = data_utils.shift_to_com(pf)

# 2D marker and time arraws
M, T = np.meshgrid(np.arange(nmark) + 1, times)

colors = sns.husl_palette(n_colors=nmark)
time_colors = sns.color_palette('Greens', n_colors=ntime)


# %% Centroid velocities and accelerations

# using the trunk region
pcom = pf[:, :vent_idx + 1].mean(axis=1)
vcom, acom = smoothing.findiff(pcom, dt)
acom /= 9810

nhat = np.array([0, 0, -1])  # angle from horizontal down is positive
glide_angle_mag = np.dot(nhat, vcom.T) / np.linalg.norm(vcom, axis=1)
glide_angle = np.rad2deg(np.arcsin(glide_angle_mag))

vmag_com = np.sqrt(np.sum(vcom**2, axis=1))
KE_norm = .5 * vmag_com**2
PE_norm = 9810 * pcom[:, 2]
ET_norm = KE_norm + PE_norm

KE_norm_rate, KE_norm_rate2 = smoothing.findiff(KE_norm, dt)
PE_norm_rate, PE_norm_rate2 = smoothing.findiff(PE_norm, dt)
ET_norm_rate, ET_norm_rate2 = smoothing.findiff(ET_norm, dt)


# %% 3D grid of points and spline

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(pcom[:, 0], pcom[:, 1], pcom[:, 2],
            color=bmap[0], tube_radius=10)

for j in np.arange(nmark):
    cc = tuple(colors[j])
    mlab.points3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                  color=cc, scale_factor=15, scale_mode='none')
    mlab.plot3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                color=cc, tube_radius=4)

for i in np.arange(ntime)[::5]:
    mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], opacity=1,
                color=time_colors[i], tube_radius=5)
    mlab.points3d(SPL[i, 0, 0], SPL[i, 0, 1], SPL[i, 0, 2], opacity=1,
                  color=time_colors[i], scale_factor=20, scale_mode='none')


# %%

Ws = 29 * 1000**2
c = 22.0
g = 9810
rho = 1.2
eps = (rho * g / 2) * (c / Ws)

# non-dimensional and rescaled velocity
vcom_res = vcom / np.sqrt(c * g / eps)

import plotting

fig, ax = plt.subplots()
ln = ax.plot(.001 * vcom[:, 1], .001 * vcom[:, 2])
plotting.add_arrow_to_line2D(ax, ln, arrow_locs=[.05, .25, .75, .9], arrowsize=2,
                             arrowstyle='->')
ax.set_aspect('equal')#, adjustable='box')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r'horizontal velocity, $v_x$ (m/s)')
ax.set_ylabel(r'vertical velocity, $v_z$ (m/s)')
ax.set_xlim(0, 8)
ax.set_ylim(-8, 0)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ln = ax.plot(vcom_res[:, 1], vcom_res[:, 2])
plotting.add_arrow_to_line2D(ax, ln, arrow_locs=[.05, .25, .75, .9], arrowsize=2,
                             arrowstyle='->')
ax.set_aspect('equal', adjustable='box')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r'horizontal velocity, $\hat{v}_x$')
ax.set_ylabel(r'vertical velocity, $\hat{v}_z$')
ax.set_xlim(0, 1.5)
ax.set_ylim(-1.5, 0)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %% VPD with all of the markers

fig, ax = plt.subplots()

for j in np.arange(nmark):
    ax.plot(.001 * vf[:, j, 1], .001 * vf[:, j, 2], color=colors[j], lw=.5)

ln = ax.plot(.001 * vcom[:, 1], .001 * vcom[:, 2])
plotting.add_arrow_to_line2D(ax, ln, arrow_locs=[.05, .25, .75, .9], arrowsize=2,
                             arrowstyle='->')

ax.set_aspect('equal', adjustable='box')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel(r'horizontal velocity, $v_y$ (m/s)')
ax.set_ylabel(r'vertical velocity, $v_z$ (m/s)')
ax.set_xlim(0, 8)
ax.set_ylim(-8, 0)
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %% 3D VPD

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(vcom[:, 0], vcom[:, 1], vcom[:, 2],
            color=bmap[0], tube_radius=15)

skip = 5

#for j in np.arange(nmark):
for j in np.arange(vent_idx + 1):
    mlab.plot3d(vf[:, j, 0], vf[:, j, 1], vf[:, j, 2],
                color=tuple(colors[j]), tube_radius=10)
    mlab.points3d(vf[::skip, j, 0], vf[::skip, j, 1], vf[::skip, j, 2],
                  color=tuple(colors[j]), scale_factor=30, scale_mode='none')
    mlab.points3d(vf[0, j, 0], vf[0, j, 1], vf[0, j, 2],
                  color=tuple(colors[j]), scale_factor=35, scale_mode='none')


# %% 3D acceleration plot

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(acom[:, 0], acom[:, 1], acom[:, 2],
            color=bmap[0], tube_radius=15)

skip = 5

#for j in np.arange(nmark):
#for j in np.arange(vent_idx + 1):
for j in [0]:
    mlab.plot3d(af[:, j, 0], af[:, j, 1], af[:, j, 2],
                color=tuple(colors[j]), tube_radius=100)
    mlab.points3d(af[::skip, j, 0], af[::skip, j, 1], af[::skip, j, 2],
                  color=tuple(colors[j]), scale_factor=30, scale_mode='none')
    mlab.points3d(af[0, j, 0], af[0, j, 1], af[0, j, 2],
                  color=tuple(colors[j]), scale_factor=35, scale_mode='none')



# %% Center the velocity

vf_cent = (vf.swapaxes(0, 1) - vcom).swapaxes(0, 1)

fig, ax = plt.subplots()
for j in np.arange(vent_idx + 1):
    print j
    ax.plot(vf_cent[:, j, 1], vf_cent[:, j, 2], color=colors[j], lw=.5)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(vent_idx + 1):
    mlab.plot3d(vf_cent[:, j, 0], vf_cent[:, j, 1], vf_cent[:, j, 2],
                color=tuple(colors[j]), tube_radius=5)
    mlab.points3d(vf_cent[0, j, 0], vf_cent[0, j, 1], vf_cent[0, j, 2],
                  color=tuple(colors[j]), scale_factor=25, scale_mode='none')


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in [-1]:
    mlab.plot3d(vf_cent[:, j, 0], vf_cent[:, j, 1], vf_cent[:, j, 2],
                color=tuple(colors[j]), tube_radius=5)
    mlab.points3d(vf_cent[0, j, 0], vf_cent[0, j, 1], vf_cent[0, j, 2],
                  color=tuple(colors[j]), scale_factor=25, scale_mode='none')


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.15, 7))
ax1.axhline(0, color='gray', lw=.75)
ax2.axhline(0, color='gray', lw=.75)
ax1.plot(times, .001 * vcom[:, 0], label='X')
ax1.plot(times, .001 * vcom[:, 1], label='Y')
ax1.plot(times, .001 * vcom[:, 2], label='Z')
ax2.plot(times, acom[:, 0])
ax2.plot(times, acom[:, 1])
ax2.plot(times, acom[:, 2])
ax1.legend(loc='best', ncol=3)
ax2.set_xlabel('time (s)')
ax1.set_ylabel('velocities (m/s)')
ax2.set_ylabel('acceleration (g)')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ln = ax.plot(.001 * pcom[:, 1], .001 * pcom[:, 2])
plotting.add_arrow_to_line2D(ax, ln, arrowstyle='->', arrowsize=2)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([-5, -4, -3, -2, -1, 0])
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, 1e-6 * KE_norm, label='KE')
ax.plot(times, 1e-6 * PE_norm, label='PE')
ax.plot(times, 1e-6 * ET_norm, label=r'$E_T$')
ax.legend(loc='best')
ax.set_xlabel('time (s)')
ax.set_ylabel('specific energy (J/kg)')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=.75)
ax.plot(times, 1e-6 * KE_norm_rate, label='KE')
ax.plot(times, 1e-6 * PE_norm_rate, label='PE')
ax.plot(times, 1e-6 * ET_norm_rate, label=r'$E_T$')
ax.set_xlabel('time (s)')
ax.set_ylabel(r'energy rate (J/kg$\cdot$ s)')
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, glide_angle)
ax.set_ylim(0, 90)
ax.set_xlabel('time (s)')
ax.set_ylabel(r'glide angle, $\gamma$')
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


# %%

def straighten_trajectory(data):
    """Align the trajectory so we are in the 'glide-polar'
    2D projection.
    """

    # non-destructive updates
    ds = data.copy()
    npts = data.shape[0]
    nmark = data.shape[1] // 3

    cs = calc_com(data)
    ds, cs = center_data(ds, cs)

    for i in range(npts):
        th = np.arctan2(cs[i, 0], cs[i, 1])
        cs[i:] = rotate(cs[i:], th)

        for j in range(nmark):
            start, stop = j * nmark, (j + 1) * nmark
            ds[i:, start:stop] = rotate(ds[i:, start:stop], th)

    return ds, cs


# %%

vmin, vmax = -.05, .05

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for i in [50]:
    # http://stackoverflow.com/a/19667996
    _s = mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], KAPs[i],
                colormap='RdBu', tube_radius=4, vmin=vmin, vmax=vmax)
    _s.module_manager.scalar_lut_manager.reverse_lut = True

    for j in np.arange(nmark):
        mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
                      color=tuple(colors_mark[j]), scale_factor=15)

    sk = 1
#    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
#                  TNB[i, ::sk, 0, 0], TNB[i, ::sk, 0, 1], TNB[i, ::sk, 0, 2],
#                  color=bmap[0], mode='arrow', resolution=64, scale_factor=50)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 1, 0], TNB[i, ::sk, 1, 1], TNB[i, ::sk, 1, 2],
                  color=bmap[1], mode='arrow', resolution=64, scale_factor=25)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 2, 0], TNB[i, ::sk, 2, 1], TNB[i, ::sk, 2, 2],
                  color=bmap[2], mode='arrow', resolution=64, scale_factor=25)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for i in [50]:
#    mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], KAP[i],
#                colormap='RdBu', tube_radius=4, vmin=vmin, vmax=vmax)
#
#    for j in np.arange(nmark):
#        mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2],
#                      color=tuple(colors_mark[j]), scale_factor=15)

    sk = 1
#    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
#                  TNB[i, ::sk, 0, 0], TNB[i, ::sk, 0, 1], TNB[i, ::sk, 0, 2],
#                  color=bmap[0], mode='arrow', resolution=64, scale_factor=50)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 1, 0], TNB[i, ::sk, 1, 1], TNB[i, ::sk, 1, 2],
                  color=bmap[1], mode='arrow', resolution=64, scale_factor=50)
    mlab.quiver3d(SPL[i, ::sk, 0], SPL[i, ::sk, 1], SPL[i, ::sk, 2],
                  TNB[i, ::sk, 2, 0], TNB[i, ::sk, 2, 1], TNB[i, ::sk, 2, 2],
                  color=bmap[2], mode='arrow', resolution=64, scale_factor=50)


# %%

SPL_s, _ = data_utils.shift_to_com(SPL)

sk = 1

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))
#tquiv = mlab.quiver3d(SPL_s[i, ::sk, 0], SPL_s[i, ::sk, 1], SPL_s[i, ::sk, 2],
#                      TNB[i, ::sk, 0, 0], TNB[i, ::sk, 0, 1], TNB[i, ::sk, 0, 2],
#                      color=bmap[0], mode='arrow', resolution=64, scale_factor=50)
nquiv = mlab.quiver3d(SPL_s[i, ::sk, 0], SPL_s[i, ::sk, 1], SPL_s[i, ::sk, 2],
                      TNB[i, ::sk, 1, 0], TNB[i, ::sk, 1, 1], TNB[i, ::sk, 1, 2],
                  color=bmap[1], mode='arrow', resolution=64, scale_factor=40)
bquiv = mlab.quiver3d(SPL_s[i, ::sk, 0], SPL_s[i, ::sk, 1], SPL_s[i, ::sk, 2],
                      TNB[i, ::sk, 2, 0], TNB[i, ::sk, 2, 1], TNB[i, ::sk, 2, 2],
                      color=bmap[2], mode='arrow', resolution=64, scale_factor=40)

@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for i in np.arange(ntime):
            print('Current time: {0}'.format(times[i]))

#            tquiv.mlab_source.set(
#                x=SPL_s[i, ::sk, 0], y=SPL_s[i, ::sk, 1], z=SPL_s[i, ::sk, 2],
#                u=TNB[i, ::sk, 0, 0], v=TNB[i, ::sk, 0, 1], w=TNB[i, ::sk, 0, 2])
            nquiv.mlab_source.set(
                x=SPL_s[i, ::sk, 0], y=SPL_s[i, ::sk, 1], z=SPL_s[i, ::sk, 2],
                u=TNB[i, ::sk, 1, 0], v=TNB[i, ::sk, 1, 1], w=TNB[i, ::sk, 1, 2])
            bquiv.mlab_source.set(
                x=SPL_s[i, ::sk, 0], y=SPL_s[i, ::sk, 1], z=SPL_s[i, ::sk, 2],
                u=TNB[i, ::sk, 2, 0], v=TNB[i, ::sk, 2, 1], w=TNB[i, ::sk, 2, 2])

            yield

manim = anim()
mlab.show()



# %%

i = 50
nspl = 2000

# spline fit
tck, fp = splprep(pf[i, :, :].T.tolist(), k=3, s=0)
ts = np.linspace(0, 1, nspl)

xs, ys, zs = splev(ts, tck)
dx, dy, dz = splev(ts, tck, der=1)
ddx, ddy, ddz = splev(ts, tck, der=2)
dddx, dddy, dddz = splev(ts, tck, der=3)

r = np.c_[xs, ys, zs]
dr = np.c_[dx, dy, dz]
ddr = np.c_[ddx, ddy, ddz]
dddr = np.c_[dddx, dddy, dddz]

# kap = (np.cross(dr, ddr).T / np.linalg.norm(dr, axis=1)**3).T
kap, tau = np.zeros(nspl), np.zeros(nspl)
for k in np.arange(nspl):
    k1 = ddz[k] * dy[k] - ddy[k] * dz[k]
    k2 = ddx[k] * dz[k] - ddz[k] * dx[k]
    k3 = ddy[k] * dx[k] - ddx[k] * dy[k]
    kn = (dx[k]**2 + dy[k]**2 + dz[k]**2)**1.5

    # t1 = dddx[k] * (dy[k] * ddz[k] - ddy[k] * dz[k])
    # t2 = dddy[k] * (ddx[k] * dz[k] - dx[k] * ddz[k])
    # t3 = dddz[k] * (dx[k] * ddy[k] - ddx[k] * dy[k])
    t1 = dddx[k] * k1
    t2 = dddy[k] * k2
    t3 = dddz[k] * k3
    tn = k1**2 + k2**2 + k3**2

    kap[k] = (k1 + k2 + k3) / kn
    tau[k] = (t1 + t2 + t3) / tn

ds = np.sqrt(dx**2 + dy**2 + dz**2)  # arc length to integrate
total_len = cumtrapz(ds, ts, initial=0)[-1]
ss = ts * total_len  # scale the arc length coordinate

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(ss, xs, ss, ys, ss, zs)
ax[1].plot(ss, dx, ss, dy, ss, dz)
ax[2].plot(ss, ddx, ss, ddy, ss, ddz)
ax[3].plot(ss, dddx, ss, dddy, ss, dddz)
ax[0].set_xlim(ss[0], ss[-1])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(ss, kap)
ax.set_xlim(ss[0], ss[-1])
ax.set_xlabel(r'arc length, $s$ (mm)')
ax.set_ylabel(r'curvature, $\kappa$ (1/mm)')
sns.despine()

fig, ax = plt.subplots()
ax.plot(ss, tau)
ax.set_xlim(ss[0], ss[-1])
ax.set_xlabel(r'arc length, $s$ (mm)')
ax.set_ylabel(r'torsion, $\tau$ (1/mm)')
sns.despine()


# %% Fit a physical spline

import time

import c2ph
reload(c2ph)

# TODO gamma should be less than 1, but it is not alway!
#l[gamma < 1] *= 1.1

i = 50
dpf = np.diff(pf[i], axis=0)

N = nmark - 1
l = dist_btn_markers.copy()
dx, dy, dz = dpf.T
dpmag = np.sqrt(dx**2 + dy**2 + dz**2)
gamma = l / dpmag

args = (N, l, dpf)
a, b = c2ph.make_ic(pf[i], N, l, dpf, ends='cubic')
x0 = np.r_[a.real, a.imag, b.real, b.imag] # 4N + 8 scalars to find

ss = np.linspace(0, 1, 101)

# calculate optimal spline coefficients
now = time.time()
xn, out = c2ph.calc_spine_coeffs(x0, args, iter=100, iprint=2)
print('Elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

colors_mark = sns.husl_palette(n_colors=nmark)
colors_seg = sns.husl_palette(n_colors=N, h=.05)

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(nmark):
    mlab.points3d(pf[i, j, 0], pf[i, j, 1], pf[i, j, 2], scale_factor=15,
                  color=tuple(colors_mark[j]))

for j in np.arange(N):
    rspl = c2ph.eval_spline_bezier(x0, pf[i], j, ss)
    mlab.plot3d(rspl[:, 0], rspl[:, 1], rspl[:, 2], tube_radius=2,
                color=tuple(colors_seg[j]), opacity=.4)

for j in np.arange(N):
    rspl = c2ph.eval_spline_bezier(xn, pf[i], j, ss)
    mlab.plot3d(rspl[:, 0], rspl[:, 1], rspl[:, 2], tube_radius=2,
                color=tuple(colors_seg[j]))

#tck, fp = splprep(pf[i, :, :].T.tolist(), k=3, s=0)
#ts = np.linspace(0, 1, len(ss) * N)
#xs, ys, zs = splev(ts, tck)   # fit the spline
#mlab.plot3d(xs, ys, zs, tube_radius=3, color=bmap[1], opacity=.4)

mlab.plot3d(SPL[i, :, 0], SPL[i, :, 1], SPL[i, :, 2], tube_radius=3,
            color=bmap[1], opacity=.4)


# %% Try with nlopt

def f(x, grad):
    """NLopt objective function.
    """
    if grad.size > 0:
        grad[:] = c2ph.fprime(x, N, l, dpf)

    obj = c2ph.f_objective(x, N, l, dpf)
    print obj
    return obj


def c(res, x, grad):
    """NLopt constraint function.
    """
    if grad.size > 0:
        'print in grad_cons'
        grad[:] = c2ph.fprime_eqcons(x, N, l, dpf)

    res[:] = c2ph.f_eqcons(x, N, l, dpf)


import nlopt

# formulate splines ICs
args = (N, l, dpf)
a, b = c2ph.make_ic(pf, N, l, dpf, ends='cubic')
x0 = np.r_[a.real, a.imag, b.real, b.imag]  # 4N + 8 scalars to find

algorithm = nlopt.GN_ISRES
algorithm = nlopt.LD_SLSQP
nopt = 4 * N + 8
tols = 1e-6 * np.ones(4 * N)

opt = nlopt.opt(algorithm, nopt)
opt.set_min_objective(f)
opt.add_equality_mconstraint(c, tols)
opt.set_ftol_abs(1e-8)
opt.set_xtol_rel(1e-8)
opt.set_xtol_abs(1e-8)

#opt.set_min_objective(lambda x, grad: f(x, grad, N, l, dp))
#opt.add_equality_mconstraint(lambda res, x, grad: c(res, x, grad, N, l, dp), tols)


# calculate optimal spline coefficients
now = time.time()
xopt = opt.optimize(x0)
opt_val = opt.last_optimum_value()
result = opt.last_optimize_result()
# xn, out = c2ph.calc_spine_coeffs(x0, args, iprint=1)
print('Elapsed time: {0:.3f} sec'.format(time.time() - now))


# %% Prepare data for 3D plots

pf, vf, af = bup, buv, bua

# rought com assuming uniform linear mass density
_, coms = data_utils.shift_to_com(pf)

# 2D marker and time arraws
M, T = np.meshgrid(np.arange(nmark) + 1, times)

colors = sns.husl_palette(n_colors=nmark)
time_colors = sns.color_palette('Greens', n_colors=ntime)


# %% 3D plot of the raw data

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(pr[:, :, 0], pr[:, :, 1], pr[:, :, 2], opacity=.5, scalars=T,
          colormap='copper', representation='surface')
mlab.points3d(pr[:, :, 0], pr[:, :, 1], pr[:, :, 2], opacity=.5,
              color=bmap[2], scale_factor=15)


# %% Overlsay spline and raw fit

from scipy.interpolate import splprep, splev

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[0],
            tube_radius=6)
mlab.mesh(pf[:, :, 0], pf[:, :, 1], pf[:, :, 2], opacity=.25, scalars=T,
          colormap='copper', representation='surface')
# mlab.points3d(ps[:, :, 0], ps[:, :, 1], ps[:, :, 2], opacity=.5,
#               color=bmap[0], scale_factor=15)

for i in np.arange(ntime)[::5]:
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                color=bmap[1], tube_radius=5)
    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                  color=bmap[1], scale_factor=15)

    # spline fit
    tck, fp = splprep(pf[i + 1, :, :].T.tolist(), k=3, s=.1)
    ts = np.linspace(0, 1, 101)
    xs, ys, zs = splev(ts, tck)

    mlab.plot3d(xs, ys, zs, opacity=.5,
                color=bmap[2], tube_radius=5)
    mlab.points3d(pf[i + 1, :, 0], pf[i + 1, :, 1], pf[i + 1, :, 2],
                  opacity=.5, color=bmap[2], scale_factor=15)
    # mlab.points3d(xs, ys, zs, opacity=.5,
    #               color=bmap[2], scale_factor=15)


# %% Overlay velocity vectors

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime)[-15:-13]:
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                color=bmap[1], tube_radius=5)
    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                  color=bmap[1], scale_factor=15)

    mlab.quiver3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
                  vf[i, :, 0], vf[i, :, 1], vf[i, :, 2], color=bmap[3],
                  mode='arrow', resolution=64, scale_factor=.015)


# %% Overlay velocity on mesh

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(pr[:, :, 0], pr[:, :, 1], pr[:, :, 2], opacity=.5, scalars=T,
          colormap='copper', representation='surface')

for i in np.arange(ntime):
#    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
#                color=bmap[1], tube_radius=5)
#    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
#                  color=bmap[1], scale_factor=15)

    mlab.quiver3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
                  vf[i, :, 0], vf[i, :, 1], vf[i, :, 2], color=bmap[3],
                  mode='arrow', resolution=64, scale_factor=.015)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(nmark):
    cc = tuple(colors[j])
    mlab.points3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                  color=cc, scale_factor=15, scale_mode='none')
    mlab.plot3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.4,
                color=cc, tube_radius=4)

for i in np.arange(ntime)[::5]:
    tck, fp = splprep(pf[i, :, :].T.tolist(), k=3, s=.1)
    ts = np.linspace(0, 1, 101)
    xs, ys, zs = splev(ts, tck)
    mlab.plot3d(xs, ys, zs, opacity=1, color=time_colors[i], tube_radius=5)
    mlab.points3d(xs[0], ys[0], zs[0], opacity=1,
                  color=time_colors[i], scale_factor=20, scale_mode='none')


# %%

pff = .001 * pf
shift = pff.mean(axis=0).mean(axis=0)
sx, sy, sz = shift

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(nmark):
    cc = tuple(colors[j])
#    mlab.points3d(pff[:, j, 0] - sx, pff[:, j, 1] - sy, pff[:, j, 2] - sz,
#                  opacity=.4, color=cc, scale_factor=.015, scale_mode='none')
    mlab.plot3d(pff[:, j, 0] - sx, pff[:, j, 1] - sy, pff[:, j, 2] - sz,
                opacity=.4, color=cc, tube_radius=.004)

#for i in np.arange(ntime)[::5]:
#    tck, fp = splprep((pff[i, :, :] - shift).T.tolist(), k=2, s=.001)
#    ts = np.linspace(0, 1, 101)
#    xs, ys, zs = splev(ts, tck)
#    xs = xs + .5
#    # xs, ys, zs = pff[i, :, :].T
#    mlab.plot3d(xs, ys, zs, opacity=1, color=time_colors[i], tube_radius=.005)
#    mlab.points3d(xs[0], ys[0], zs[0], opacity=1,
#                  color=time_colors[i], scale_factor=.0225, scale_mode='none')

#import time
#now = time.time()
#mlab.savefig('/Users/isaac/Desktop/413_91_rainbow.obj')
#print('elapsed time: {0:.3f} sec'.format(time.time() - now))


# %%

vff = .001 * vf

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime)[::5]:
    tck, fp = splprep((pff[i, :, :] - shift).T.tolist(), k=3, s=0)
    ts = np.linspace(0, 1, 101)
    xs, ys, zs = splev(ts, tck)

    mlab.plot3d(xs, ys, zs, opacity=1, color=time_colors[i], tube_radius=.005)
    mlab.points3d(xs[0], ys[0], zs[0], opacity=1,
                  color=time_colors[i], scale_factor=.0225, scale_mode='none')

    mlab.quiver3d(pff[i, :, 0] - sx, pff[i, :, 1] - sy, pff[i, :, 2] - sz,
                  vff[i, :, 0], vff[i, :, 1], vff[i, :, 2], color=bmap[3],
                  mode='arrow', resolution=64, scale_factor=.01)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

for j in np.arange(nmark):
    cc = tuple(colors[j])
#    mlab.points3d(pf[:, j, 0], pf[:, j, 1], pf[:, j, 2], opacity=.5,
#                  color=cc, scale_factor=15, scale_mode='none')
    mlab.plot3d(pf[:, j, 0].T, pf[:, j, 1], pf[:, j, 2], opacity=.5,
                color=cc, tube_radius=4)

for i in np.arange(ntime)[::5]:
#    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
#                color=bmap[1], tube_radius=5)
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                color=time_colors[i], tube_radius=5)

#    mlab.quiver3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2],
#                  vf[i, :, 0], vf[i, :, 1], vf[i, :, 2], color=bmap[3],
#                  mode='arrow', resolution=64, scale_factor=.015)


# %%

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))
#fig = mlab.figure(bgcolor=(0, 0, 0), fgcolor=(0, 0, 0), size=(750, 750))

mlab.points3d(pf[:, :, 0], pf[:, :, 1], pf[:, :, 2], M, opacity=.5,
              colormap='gist_ncar', scale_factor=15, scale_mode='none')
#mlab.plot3d(pf[:, :, 0].T, pf[:, :, 1].T, pf[:, :, 2].T)#, M)#, opacity=.5,
#              colormap='gist_ncar')#, tube_radius=5)

for i in np.arange(ntime)[::5]:
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                color=bmap[1], tube_radius=5)


# %% Plot the body and com

fig = mlab.figure(bgcolor=(1, 1, 0.92157), fgcolor=(0, 0, 0), size=(750, 750))

mlab.plot3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[2],
            tube_radius=4)

for i in np.arange(ntime)[::5]:
    mlab.plot3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                color=bmap[1], tube_radius=5)

    mlab.points3d(pf[i, :, 0], pf[i, :, 1], pf[i, :, 2], opacity=.5,
                  color=bmap[1], scale_factor=15)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

ix = 0
xx, yy, zz = (pf[ix, :, :] - pf[ix, :, :].mean(axis=0)).T
tck, fp = splprep([xx, yy, zz], k=3, s=.1)
ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)

pts = mlab.points3d(xx, yy, zz, color=bmap[1], scale_factor=16)
bdy = mlab.plot3d(xs, ys, zs, color=bmap[1], tube_radius=3)


@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for ix in np.arange(ntime):
            print('frame {0} of {1}'.format(ix, ntime))
            xx, yy, zz = (pf[ix, :, :] - pf[ix, :, :].mean(axis=0)).T
            tck, fp = splprep([xx, yy, zz], k=3, s=.1)
            ts = np.linspace(0, 1, 101)
            xs, ys, zs = splev(ts, tck)

            pts.mlab_source.set(x=xx, y=yy, z=zz)
            bdy.mlab_source.set(x=xs, y=ys, z=zs)
            # bdy.mlab_source.set(x=xx, y=yy, z=zz)

            yield

manim = anim()
mlab.show()


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

ix = 0
xx, yy, zz = (pf[ix, :, :] - pf[ix, :, :].mean(axis=0)).T
tck, fp = splprep([xx, yy, zz], k=3, s=.1)
ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)

pts = mlab.points3d(xx, yy, zz, color=bmap[1], scale_factor=16)
bdy = mlab.plot3d(xs, ys, zs, color=bmap[1], tube_radius=3)
# bdy = mlab.plot3d(xx, yy, zz, color=bmap[1], tube_radius=3)

vel = mlab.quiver3d(0 * pf[ix, :, 0], 0 * pf[ix, :, 1], 0 * pf[ix, :, 2],
                    0 * vf[ix, :, 0], 0 * vf[ix, :, 1], 0 * vf[ix, :, 2],
                    color=bmap[3],
                    mode='arrow', resolution=64, scale_factor=.015)

#acc = mlab.quiver3d(0 * pf[ix, :, 0], 0 * pf[ix, :, 1], 0 * pf[ix, :, 2],
#                    0 * af[ix, :, 0], 0 * af[ix, :, 1], 0 * af[ix, :, 2],
#                    color=bmap[2],
#                    mode='arrow', resolution=64, scale_factor=.005)

@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for ix in np.arange(ntime):
            xx, yy, zz = (pf[ix, :, :] - pf[ix, :, :].mean(axis=0)).T
            tck, fp = splprep([xx, yy, zz], k=3, s=.1)
            ts = np.linspace(0, 1, 101)
            xs, ys, zs = splev(ts, tck)

            pts.mlab_source.set(x=xx, y=yy, z=zz)
            bdy.mlab_source.set(x=xs, y=ys, z=zs)
            vel.mlab_source.set(x=xx, y=yy, z=zz,
                                u=vf[ix, :, 0], v=vf[ix, :, 1], w=vf[ix, :, 2])
#            acc.mlab_source.set(x=xx, y=yy, z=zz,
#                                u=af[ix, :, 0], v=af[ix, :, 1], w=af[ix, :, 2])
            # bdy.mlab_source.set(x=xx, y=yy, z=zz)

            yield

manim = anim()
mlab.show()


# %% Play with exporting the body waveform

i = 0
bc, bcom = data_utils.shift_to_com(bup)

fig, ax = plt.subplots()
ax.plot(bc[i, :, 0], bc[i, :, 1])
ax.set_aspect('equal')
sns.despine()
fig.set_tight_layout(True)

nrepeat = 4
br = np.repeat(bc[i], nrepeat).reshape((nmark, 3, nrepeat))
fig, ax = plt.subplots()
ax.plot(br[:, 0], br[:, 1])
ax.set_aspect('equal')
sns.despine()
fig.set_tight_layout(True)


# %% Play with exporting the data as sound

j, k = -1, 0
pos = bup[:, j, k]
vel = buv[:, j, k]
acc = bua[:, j, k]

arr = vel

c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
d_minor = ['D', 'E', 'F', 'G', 'A', 'Bb', 'C']

from miditime.MIDITime import MIDITime

mymidi = MIDITime(tempo=120 * 100, outfile='413_91_tail.mid',
                  seconds_per_year=dt, octave_range=4)

note_list = []
for i in np.arange(ntime):

    # Where does this data point sit in the domain of your data?
    scale_pct = mymidi.linear_scale_pct(arr.min(), arr.max(), arr[i])

    # Find the note that matches your data point
    note = mymidi.scale_to_note(scale_pct, c_major)
    # note = mymidi.scale_to_note(scale_pct, d_minor)

    # Translate that note to a MIDI pitch
    midi_pitch = mymidi.note_to_midi_pitch(note)

    note_list.append([i, midi_pitch, 100, 1])

# Add a track with those notes
mymidi.add_track(note_list)

# Output the .mid file
mymidi.save_midi()


# %% Encode more information into from velocity and acceleration

j, k = 3, 0
pos = bup[:, j, k]
vel = buv[:, j, k]
acc = bua[:, j, k]

c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
d_minor = ['D', 'E', 'F', 'G', 'A', 'Bb', 'C']

from miditime.MIDITime import MIDITime

mymidi = MIDITime(tempo=120, outfile='413_91_test.mid',
                  seconds_per_year=dt, octave_range=4)

note_list = []
for i in np.arange(ntime):
    scale_pos = mymidi.linear_scale_pct(pos.min(), pos.max(), pos[i])
    scale_vel = mymidi.linear_scale_pct(-vel.std(), vel.std(), vel[i])
    scale_acc = mymidi.linear_scale_pct(acc.min(), acc.max(), acc[i], True)

    note = mymidi.scale_to_note(scale_pos, c_major)
    midi_pitch = mymidi.note_to_midi_pitch(note)

    attack = 100  # np.int(200 + 100 * scale_vel)
    duration = scale_acc
    note_list.append([times[i], midi_pitch, attack, duration])

mymidi.add_track(note_list)
mymidi.save_midi()


# %% Try with more than one track...doesn't work yet

j, k = 0, 0
pos = bup[:, j, k]
vel = buv[:, j, k]
acc = bua[:, j, k]

c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
d_minor = ['D', 'E', 'F', 'G', 'A', 'Bb', 'C']

from miditime.MIDITime import MIDITime

mymidi = MIDITime(tempo=120 * 100, outfile='413_91_test.mid',
                  seconds_per_year=dt, octave_range=4)

for i in np.arange(ntime):
    for track_num, arr in enumerate([pos, vel, acc]):

        # Where does this data point sit in the domain of your data?
        scale_pct = mymidi.linear_scale_pct(arr.min(), arr.max(), arr[i])

        # Find the note that matches your data point
        note = mymidi.scale_to_note(scale_pct, c_major)
        # note = mymidi.scale_to_note(scale_pct, d_minor)

        # Translate that note to a MIDI pitch
        midi_pitch = mymidi.note_to_midi_pitch(note)

        mymidi.add_note(track_num, track_num, [i, midi_pitch, 100, 1])

        # note_list.append([i, midi_pitch, 100, 1])

# Add a track with those notes
#mymidi.add_track(note_list)

# Output the .mid file
mymidi.save_midi()


# %%


from miditime.MIDITime import MIDITime

# Instantiate the class with a tempo (120bpm is the default) and an output file destination.
mymidi = MIDITime(120, 'myfile.mid')

# Create a list of notes. Each note is a list: [time, pitch, attack, duration]
midinotes = [
    [0, 60, 200, 3],  #At 0 beats (the start), Middle C with attack 200, for 3 beats
    [10, 61, 200, 4]  #At 10 beats (12 seconds from start), C#5 with attack 200, for 4 beats
]

# Add a track with those notes
mymidi.add_track(midinotes)

# Output the .mid file
mymidi.save_midi()



# %%

colors = sns.husl_palette(n_colors=nmark)
fig, ax = plt.subplots()
ax.plot(times, bup[:, 0, 0], c=colors[0])
ax.plot(times, bup[:, 1, 0], c=colors[1])
ax.plot(times, bup[:, nmark - 1, 0], c=colors[nmark - 1])
ax.plot(times, pr[:, 0, 0], c=colors[0])
ax.plot(times, pr[:, 1, 0], c=colors[1])
ax.plot(times, pr[:, nmark - 1, 0], c=colors[nmark - 1])
sns.despine()
fig.set_tight_layout(True)


# %%

colors = sns.husl_palette(n_colors=nmark)
fig, ax = plt.subplots()
for i in np.arange(nmark - 1 - 2):
    alpha = 1 - .75 * i / nmark  # penalize markers further from the head
    ax.plot(bup[:, i + 1, 0], bup[:, i, 0], c=colors[i], alpha=alpha)
ax.set_xlabel('marker X position along the body')
ax.set_ylabel('head X position')
ax.set_aspect('equal', adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %% Plot individual smoothing

k = 0
fig, ax = plot_indiv(bua, times, k, markers=False, title='Calculated X accelerations')
fig, ax = plot_indiv(buv, times, k, markers=False, title='Calculated X velocities')
fig, ax = plot_indiv(bup, times, k, markers=False, title='Filtered X positions')

# plot all of the accelerations
fig, axs = plot_xyz(bua, times, title='Calculated accelerations')
fig, axs = plot_xyz(buv, times, title='Calculated velocities')
fig, axs = plot_xyz(bup, times, title='Filtered positions')


# %%

# remove some data
j = nmark - 1
j = 0
pgap = pr.copy()
pgap[90:130, j] = np.nan

ixb = np.where(np.isnan(pgap[:, j, 0]))[0]

pfill, nans, pfill0 = ukf_filter.fill_gaps_ukf(pgap, fs, meas_noise=3)

R_fill, fcs_fill = smoothing.residual_butter(pfill, fs, df=.5, fmin=1, fmax=35)
inter_fill, fcopt_fill, rsq_fill, flinreg_fill = \
    smoothing.opt_cutoff(R_fill, fcs_fill, rsq_cutoff=.95)

fig, ax = plot_residuals(R_fill, fcs_fill, inter_fill, fcopt_fill)

but_fill = smoothing.but_fcs(pfill, fs, fcopt_fill)
bfp, bfv, bfa = but_fill['p'], but_fill['v'], but_fill['a']



# %% Plot the raw and filled positions

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
ax1.plot(times[ixb], pr[ixb, j, 0], 'o', ms=4)
ax1.plot(times[ixb], pfill[ixb, j, 0], 'o', ms=4)
ax1.plot(times, pfill[:, j, 0], lw=1)
ax1.plot(times, pfill0[:, j, 0], lw=1)

ax2.plot(times[ixb], pr[ixb, j, 1], 'o', ms=4)
ax2.plot(times[ixb], pfill[ixb, j, 1], 'o', ms=4)
ax2.plot(times, pfill[:, j, 1], lw=1)

ax3.plot(times[ixb], pr[ixb, j, 2], 'o', ms=4)
ax3.plot(times[ixb], pfill[ixb, j, 2], 'o', ms=4)
ax3.plot(times, pfill[:, j, 2], lw=1)

ax1.set_ylabel('X')
ax2.set_ylabel('Y')
ax3.set_ylabel('Z')
ax3.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %% See how the Butterworth processing compares

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
for bad in ixb:
    ax1.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax2.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax3.axvline(times[bad], color='gray', lw=1, alpha=.25)

ax1.plot(times, bup[:, j, 0], alpha=.5)
ax1.plot(times, bfp[:, j, 0])

ax2.plot(times, bup[:, j, 1], alpha=.5)
ax2.plot(times, bfp[:, j, 1])

ax3.plot(times, bup[:, j, 2], alpha=.5)
ax3.plot(times, bfp[:, j, 2])

ax1.set_ylabel('X')
ax2.set_ylabel('Y')
ax3.set_ylabel('Z')
ax3.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
for bad in ixb:
    ax1.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax2.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax3.axvline(times[bad], color='gray', lw=1, alpha=.25)

ax1.plot(times, buv[:, j, 0], '-')
ax1.plot(times, bfv[:, j, 0])

ax2.plot(times, buv[:, j, 1], '-')
ax2.plot(times, bfv[:, j, 1])

ax3.plot(times, buv[:, j, 2], '-')
ax3.plot(times, bfv[:, j, 2])

ax1.set_ylabel('vX')
ax2.set_ylabel('vY')
ax3.set_ylabel('vZ')
ax3.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
for bad in ixb:
    ax1.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax2.axvline(times[bad], color='gray', lw=1, alpha=.25)
    ax3.axvline(times[bad], color='gray', lw=1, alpha=.25)

ax1.plot(times, bua[:, j, 0], '-')
ax1.plot(times, bfa[:, j, 0])

ax2.plot(times, bua[:, j, 1], '-')
ax2.plot(times, bfa[:, j, 1])

ax3.plot(times, bua[:, j, 2], '-')
ax3.plot(times, bfa[:, j, 2])

ax1.set_ylabel('aX')
ax2.set_ylabel('aY')
ax3.set_ylabel('aZ')
ax3.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

def rmsd(a, b):
    """Root mean square difference."""
    return np.sqrt(np.mean((a - b)**2, axis=0))


def rsd(a, b):
    """Root square difference."""
    return np.sqrt((a - b)**2)


def ase(a, b):
    """Absolute Scaled Error"""
    ntime, nmark, ncoord = a.shape

    et = a - b
    qt = np.zeros_like(a)
    for j in np.arange(nmark):
        diff = np.abs(np.diff(a[:, j], axis=0))
        norm = 1 / (ntime - 1)
        den = norm * np.sum(diff, axis=0)
        qt[:, j] = et[:, j] / den

    return qt


def mase(a, b):
    """Mean Absolute Scaled Error from Hyndman (2006)"""
    qt = ase(a, b)
    return np.mean(np.abs(qt), axis=0)


# error vs. time
err_p = rsd(bup, bfp)
err_v = rsd(buv, bfv)
err_a = rsd(bua, bfa)

# absolute scaled errors (not sure how to read these yet...)
ase_p = ase(bup, bfp)
ase_v = ase(buv, bfv)
ase_a = ase(bua, bfa)
mase_p = mase(bup, bfp)
mase_v = mase(buv, bfv)
mase_a = mase(bua, bfa)

# absolute errors
ae_p = bup - bfp
ae_v = buv - bfv
ae_a = bua - bfa
mae_p = np.mean(np.abs(ae_p), axis=0)
mae_v = np.mean(np.abs(ae_v), axis=0)
mae_a = np.mean(np.abs(ae_a), axis=0)


fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1, alpha=.25)
ax.plot(times, ae_a[:, j, 0], label='X: {:.3f}'.format(mae_a[j, 0]))
ax.plot(times, ae_a[:, j, 1], label='Y: {:.3f}'.format(mae_a[j, 1]))
ax.plot(times, ae_a[:, j, 2], label='Z: {:.3f}'.format(mae_a[j, 2]))
ax.legend(loc='best')
ax.set_ylabel(r'Acceleration Error ($mm/s^2$)')
ax.set_xlabel('Time (s)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1, alpha=.25)
ax.plot(times, ae_v[:, j, 0], label='X: {:.3f}'.format(mae_v[j, 0]))
ax.plot(times, ae_v[:, j, 1], label='Y: {:.3f}'.format(mae_v[j, 1]))
ax.plot(times, ae_v[:, j, 2], label='Z: {:.3f}'.format(mae_v[j, 2]))
ax.legend(loc='best')
ax.set_ylabel(r'Velocity Error ($mm/s$)')
ax.set_xlabel('Time (s)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1, alpha=.25)
ax.plot(times, ae_p[:, j, 0], label='X: {:.3f}'.format(mae_p[j, 0]))
ax.plot(times, ae_p[:, j, 1], label='Y: {:.3f}'.format(mae_p[j, 1]))
ax.plot(times, ae_p[:, j, 2], label='Z: {:.3f}'.format(mae_p[j, 2]))
ax.legend(loc='best')
ax.set_ylabel(r'Position error ($mm$)')
ax.set_xlabel('Time (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1)
ax.plot(times, np.abs(bua[:, j, 0] - bfa[:, j, 0]))
ax.plot(times, np.abs(bua[:, j, 1] - bfa[:, j, 1]))
ax.plot(times, np.abs(bua[:, j, 2] - bfa[:, j, 2]))
ax.set_ylabel('absolute acceleration difference')
ax.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1)
ax.plot(times, np.abs(buv[:, j, 0] - bfv[:, j, 0]))
ax.plot(times, np.abs(buv[:, j, 1] - bfv[:, j, 1]))
ax.plot(times, np.abs(buv[:, j, 2] - bfv[:, j, 2]))
ax.set_ylabel('absolute velocity difference')
ax.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for bad in ixb:
    ax.axvline(times[bad], color='gray', lw=1)
ax.plot(times, np.abs(bup[:, j, 0] - bfp[:, j, 0]))
ax.plot(times, np.abs(bup[:, j, 1] - bfp[:, j, 1]))
ax.plot(times, np.abs(bup[:, j, 2] - bfp[:, j, 2]))
ax.set_ylabel('absolute position difference')
ax.set_xlabel('time (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
out = ax.xcorr(bua[:, j, 0], bfa[:, j, 0], maxlags=ntime - 1)
lags, c, line, b = out
ax.set_ylim(-1, 1)
sns.despine()
fig.set_tight_layout(True)


# %%

def npxcorr(a, v):
    """http://stackoverflow.com/a/5639626
    """
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)
    return np.correlate(a, v, mode='full')

lags = np.r_[-ntime + 1:ntime]
ix = np.where(lags >= 0)[0]

lags = lags[ix]
cfp = npxcorr(bup[:, j, 0], bfp[:, j, 0])[ix]
cfv = npxcorr(buv[:, j, 0], bfv[:, j, 0])[ix]
cfa = npxcorr(bua[:, j, 0], bfa[:, j, 0])[ix]

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.75)
ax.axhline(0, color='gray', lw=.75)
ax.plot(lags, cfp)
ax.plot(lags, cfv)
ax.plot(lags, cfa)
ax.set_ylim(-1, 1)
sns.despine()
fig.set_tight_layout(True)


# %%

lags = np.r_[-ntime + 1:ntime]
ix = np.where(lags >= 0)[0]

lags = lags[ix]
cfp = np.zeros((len(lags), ncoord))
cfv = np.zeros((len(lags), ncoord))
cfa = np.zeros((len(lags), ncoord))
for k in np.arange(ncoord):
    cfp[:, k] = npxcorr(bup[:, j, k], bfp[:, j, k])[ix]
    cfv[:, k] = npxcorr(buv[:, j, k], bfv[:, j, k])[ix]
    cfa[:, k] = npxcorr(bua[:, j, k], bfa[:, j, k])[ix]

lags = lags * dt
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                    figsize=(7.75, 10))
for ax in (ax1, ax2, ax3):
    ax.axhline(0, color='gray', lw=.75)
ax1.plot(lags, cfp[:, 0])
ax1.plot(lags, cfp[:, 1])
ax1.plot(lags, cfp[:, 2])
ax2.plot(lags, cfv[:, 0])
ax2.plot(lags, cfv[:, 1])
ax2.plot(lags, cfv[:, 2])
ax3.plot(lags, cfa[:, 0])
ax3.plot(lags, cfa[:, 1])
ax3.plot(lags, cfa[:, 2])
ax.set_ylim(-1, 1)
ax1.set_ylabel('rXX')
ax2.set_ylabel('rVV')
ax3.set_ylabel('rAA')
ax3.set_xlabel('time lag (s)')
sns.despine()
fig.set_tight_layout(True)


# %%

def rmsd(a, b):
    """Root mean square difference."""
    return np.sqrt(np.mean((a - b)**2, axis=0))


def rmse(a, b):
    """Root mean standard(?) error."""
    return np.sqrt(np.mean((a - b)**2, axis=0) / len(a))


def rss(a, b):
    """Residual sum of squares."""
    return np.sum((a - b)**2, axis=0)


def mse(a, b):
    """Residual sum of squares."""
    return np.sum((a - b)**2, axis=0) / a.shape[0]


print rmsd(bup, bfp)
print rmsd(buv, bfv)
print rmsd(bua, bfa)
print rss(bup, bfp)
print rss(buv, bfv)
print rss(bua, bfa)
#print
#print rmse(bup[:, j], bfp[:, j])
#print rmse(buv[:, j], bfv[:, j])
#print rmse(bua[:, j], bfa[:, j]) / 9810


# %%

seed = 417
ngaps = 10
max_length = 15

np.random.seed(seed)
start = np.random.randint(0, ntime, (ngaps, nmark))
lengths = np.random.randint(0, max_length, (ngaps, nmark))

stop = start + lengths
stop[stop > ntime - 1] = ntime - 1

pg = pr.copy()
for i in np.arange(ngaps):
    for j in np.arange(nmark):
        pg[start[i, j]:stop[i, j], j] = np.nan
        print len(pg[start[i, j]:stop[i, j]])


# %%

nans = np.isnan(pg[:, :, 0]).astype(np.int)

fig, ax = plt.subplots()
ax.pcolormesh(nans)
sns.despine()
fig.set_tight_layout(True)


# %%

# times
fig, ax = plt.subplots()
ax.plot(times, pgap[:, j], 'o', ms=4)
ax.plot(times, pfill[:, j], '-')
#ax.plot(times, pfill1[:, j], '-')
sns.despine()
fig.set_tight_layout(True)


# %% Try a bunch of different smoothing techniques

rw = smoothing.raw(pr, fs)
s_3_001 = smoothing.spl1d(pr, fs, 3, .001)
s_5_001 = smoothing.spl1d(pr, fs, 5, .001)
s_5_0 = smoothing.spl1d(pr, fs, 5, 0)
b_2_10 = smoothing.but(pr, fs, 10, 2)
b_2_175 = smoothing.but(pr, fs, 17.5, 2)
b_4_175 = smoothing.but(pr, fs, 17.5, 4)
b_2_20 = smoothing.but(pr, fs, 20, 2)
b_2_30 = smoothing.but(pr, fs, 30, 2)
b_2_60 = smoothing.but(pr, fs, 60, 2)
sg_5_2 = smoothing.svg(pr, fs, 5, 2)
sg_21_5 = smoothing.svg(pr, fs, 21, 5)


# %% Plot the Butterworth filtered position data

bt = b_2_175
bt = b_2_10

plot_xyz(bt['p'], times, title='Butterworth, 17.5 Hz, position')
plot_xyz(bt['v'], times, title='Butterworth, 17.5 Hz, velocity')
plot_xyz(bt['a'] / 9810, times,
         title='Butterworth, 17.5 Hz, acceleration in g')


# %% Plot the different smoothing techniques

sp = s_5_0
bt = b_2_175
sg = sg_21_5

plot_xyz(rw['p'], times, title='Raw position')
plot_xyz(bt['p'], times, title='Butterworth, 17.5 Hz, position')
plot_xyz(sg['p'], times, title='SG filter, 21 points, 5th order, position')
# plot_xyz(sp['p'], times, 0, markers=False)

plot_xyz(rw['v'], times, title='Raw velocities')
plot_xyz(bt['v'], times, title='Butterworth, 17.5 Hz, velocity')
plot_xyz(sg['v'], times, title='SG filter, 21 points, 5th order, velocity')
# plot_xyz(sp['v'], times, 0, markers=False)

plot_xyz(rw['a'], times, title='Raw accelerations')
plot_xyz(bt['a'], times, title='Butterworth, 17.5 Hz, acceleration')
plot_xyz(sg['a'], times, title='SG filter, 21 points, 5th order, acceleration')
# plot_xyz(sp['a'], times, 0, markers=False)


# %% Compare single markers

j = 1
k = 0

fig, ax = plt.subplots()
ax.plot(times, rw['p'][:, j, k])
ax.plot(times, sg['p'][:, j, k])
ax.plot(times, bt['p'][:, j, k])
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(times, rw['v'][:, j, k], '-')
ax.plot(times, sg['v'][:, j, k], '-')
ax.plot(times, bt['v'][:, j, k], '-')
sns.despine()
fig.set_tight_layout(True)


norm = 9.81 * 1000  # g in mm/s^2
fig, ax = plt.subplots()
ax.plot(times, sg['a'][:, j, k] / norm, c=bmap[1])
ax.plot(times, bt['a'][:, j, k] / norm, c=bmap[2])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot(times, rw['a'][:, j, k] / norm, c=bmap[0], lw=1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
sns.despine()
fig.set_tight_layout(True)


# %% Compare the 2nd order and 4th order Butterworth filters

fig, ax = plt.subplots()
ax.plot(times, rw['v'][:, j, k], c=bmap[0], lw=.75)
ax.plot(times, b_2_175['v'][:, j, k], c=bmap[1])
ax.plot(times, b_4_175['v'][:, j, k], c=bmap[2])
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(times, b_2_175['a'][:, j, k] / norm, c=bmap[1])
ax.plot(times, b_4_175['a'][:, j, k] / norm, c=bmap[2])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot(times, rw['a'][:, j, k] / norm, c=bmap[0], lw=.75)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
sns.despine()
fig.set_tight_layout(True)


# %% Plot all the residuals

fig, ax = plt.subplots()
ax.plot(fcs, R[:, :, 0])
ax.plot(fcs, R[:, :, 1])
ax.plot(fcs, R[:, :, 2])
ax.set_ylim(ymax=10)
ax.grid(True)
sns.despine()
fig.set_tight_layout(True)


# %% Plot residuals for x, y, and z separately

fig, ax = plt.subplots()
ax.plot(fcs, R[:, :, 0])
ax.set_ylim(ymax=10)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(fcs, R[:, :, 1])
ax.set_ylim(ymax=10)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(fcs, R[:, :, 2])
ax.set_ylim(ymax=10)
sns.despine()
fig.set_tight_layout(True)


# %% Remove points, fill in gaps, and see which is best


def fill_gaps_spl(p, kk=3, ss=1e-5):
    """Locate gaps in a 1D array.

    TODO: deal with the ends (don't extrapolate as the spline fails)
    """

    ntime, nmarks, ncood = p.shape

    bounds = np.zeros((nmark, 2))
    pg = p.copy()

    for j in np.arange(nmarks):
        x, y, z = p[:, j, :].T
        ixb = np.where(np.isnan(x))[0]
        ixg = np.where(~np.isnan(x))[0]

        imin, imax = 0, ntime
        if len(ixb) > 0:

            # fit a 1D spline to the good points
            spx = UnivariateSpline(times[ixg], x[ixg], k=kk, s=ss)
            spy = UnivariateSpline(times[ixg], y[ixg], k=kk, s=ss)
            spz = UnivariateSpline(times[ixg], z[ixg], k=kk, s=ss)

            # fill in the gap regions
            for ib in ixb:
                pg[ib, j, 0] = spx(times[ib])
                pg[ib, j, 1] = spy(times[ib])
                pg[ib, j, 2] = spz(times[ib])

            # index so we don't extrapolate
            imin, imax = ixb.min(), ixb.max()

        # save bounding region for later
        bounds[j] = imin, imax

    return pg, bounds


def rms_error(pa, pb):
    """RMS error between two signals, where pa and pb are
    ntime x nmark x 3 arrays. Mean is take over time,
    so the result in nmark x 3.
    """

    return np.sqrt(np.mean((pa - pb)**2, axis=0))


pf, vf, af = b_2_175['p'], b_2_175['v'], b_2_175['a']

imin, imax = 20, 30
pc = pr.copy()  # gap fill on the RAW data
pc[imin:imax, 0, :] = np.nan
pc[imin + 50: imax + 50] = np.nan
pc[235:240] = np.nan
pc[180:181] = np.nan

# gap fill
pg_tmp, bounds = fill_gaps_spl(pc)

# smooth the data
pg_dict = but(pg_tmp, 17.5, fs, 2, len(pg_tmp) - 1)
pgs, vgs, ags = pg_dict['p'], pg_dict['v'], pg_dict['a']


j = 0
k = 0

ixg = np.where(~np.isnan(pc[:, j, 0]))[0]
ixb = np.where(np.isnan(pc[:, j, 0]))[0]

fig, ax = plt.subplots()
ax.plot(times[ixg], pr[ixg, j, k], 'o', ms=4)
ax.plot(times[ixg], pf[ixg, j, k], 'o', ms=4)
ax.plot(times[ixb], pg_tmp[ixb, j, k], 'o', ms=4, c=bmap[1])
ax.plot(times, pgs[:, j, k], '-', c=bmap[2])
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(times, pf[:, j, k], c=bmap[0])
ax.plot(times, pgs[:, j, k], '-', c=bmap[1])
ax.plot(times[ixb], pgs[ixb, j, k], 'o', ms=5, c=bmap[1])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, vf[:, j, k], c=bmap[0])
ax.plot(times, vgs[:, j, k], '-', c=bmap[1])
ax.plot(times[ixb], vgs[ixb, j, k], 'o', ms=5, c=bmap[1])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, af[:, j, k], c=bmap[0])
ax.plot(times, ags[:, j, k], '-', c=bmap[1])
ax.plot(times[ixb], ags[ixb, j, k], 'o', ms=5, c=bmap[1])
sns.despine()
fig.set_tight_layout(True)


# %% Interpolation error

errp = np.sqrt((pf - pgs)**2)
errv = np.sqrt((vf - vgs)**2)
erra = np.sqrt((af - ags)**2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(times, errp[:, j, k])
ax2.plot(times, errv[:, j, k])
ax3.plot(times, erra[:, j, k] / 9810)

#for ax in [ax1, ax2, ax3]:
#    ax.axvline(times[ixb[0]], color='gray', lw=.75)
#    ax.axvline(times[ixb[-1]], color='gray', lw=.75)
sns.despine()
fig.set_tight_layout(True)


# %% Play around with smooting the continuous sub arrays and
#    then fitting splines

j = 1
k = 1

arr = pc[:, j]

ixg = np.where(~np.isnan(arr[:, 0]))[0]
ixb = np.where(np.isnan(arr[:, 0]))[0]

good_subindices = np.split(ixg, np.where(np.diff(ixg) > 1)[0] + 1)

arrp, arrv, arra = np.zeros_like(arr), np.zeros_like(arr), np.zeros_like(arr)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
for sub in good_subindices:
    dd = arr[sub]
    out = but1d(dd, 17.5, fs, 2, len(dd) - 1)
    arrp[sub], arrv[sub], arra[sub] = out['p'], out['v'], out['a']

    ax1.plot(times[sub], out['p'][:, k], '-o', c=bmap[0], ms=5)
    ax2.plot(times[sub], out['v'][:, k], '-o', c=bmap[0], ms=5)
    ax3.plot(times[sub], out['a'][:, k], '-o', c=bmap[0], ms=5)

ax1.plot(times, pf[:, j, k], c=bmap[2])
ax2.plot(times, vf[:, j, k], c=bmap[2])
ax3.plot(times, af[:, j, k], c=bmap[2])
sns.despine()
fig.set_tight_layout(True)

# now fit splines to the smoothed position data
arrp[ixb], arrv[ixb], arra[ixb] = np.nan, np.nan, np.nan

arrfull = pr.copy()
arrfull[:, j] = arrp

pgaps, bounds = fill_gaps_spl(arrfull, 3, 1e-3)
outgaps = raw(pgaps, fs)
pg, vg, ag = outgaps['p'], outgaps['v'], outgaps['a']

# put splines back into the missing data, then filter it all
# take the splined regions from pgaps and put into raw
arrfull = pr.copy()
arrfull[ixb, j] = pgaps[ixb, j]
roughgaps = but(arrfull, 17.5, fs, 2, len(arrfull) - 1)
pgg, vgg, agg = roughgaps['p'], roughgaps['v'], roughgaps['a']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
ax1.plot(times, pg[:, j, k], c=bmap[0])
ax2.plot(times, vg[:, j, k], c=bmap[0])
ax3.plot(times, ag[:, j, k], c=bmap[0])

ax1.plot(times, pgg[:, j, k], c=bmap[1])
ax2.plot(times, vgg[:, j, k], c=bmap[1])
ax3.plot(times, agg[:, j, k], c=bmap[1])

ax1.plot(times, pf[:, j, k], c=bmap[2])
ax2.plot(times, vf[:, j, k], c=bmap[2])
ax3.plot(times, af[:, j, k], c=bmap[2])
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# new way
errp = np.sqrt((pf - pg)**2)
errv = np.sqrt((vf - vg)**2)
erra = np.sqrt((af - ag)**2)

ax1.semilogy(times, errp[:, j, k])
ax2.semilogy(times, errv[:, j, k])
ax3.semilogy(times, erra[:, j, k] / 9810)

#ax1.plot(times, errp[:, j, k])
#ax2.plot(times, errv[:, j, k])
#ax3.plot(times, erra[:, j, k] / 9810)

errp = np.sqrt((pf - pgg)**2)
errv = np.sqrt((vf - vgg)**2)
erra = np.sqrt((af - agg)**2)

ax1.semilogy(times, errp[:, j, k])
ax2.semilogy(times, errv[:, j, k])
ax3.semilogy(times, erra[:, j, k] / 9810)

#ax1.plot(times, errp[:, j, k])
#ax2.plot(times, errv[:, j, k])
#ax3.plot(times, erra[:, j, k] / 9810)

sns.despine()
fig.set_tight_layout(True)


## %% Play around with supersmoother
#
#from supersmoother import SuperSmoother
#
#arr = pc[:, j]
#
#ixg = np.where(~np.isnan(arr[:, 0]))[0]
#ixb = np.where(np.isnan(arr[:, 0]))[0]
#
#tg = times[ixg]
#xg, yg, zg = arr[ixg].T
#
#model = SuperSmoother()
#xfit = model.fit(tg, xg, dy=.5).predict(times)
#
#fig, (ax =1, plt.subplots()


# %% Play around with polynomical gap filling

arr = pc[:, j]

ixg = np.where(~np.isnan(arr[:, 0]))[0]
ixb = np.where(np.isnan(arr[:, 0]))[0]

good_subindices = np.split(ixg, np.where(np.diff(ixg) > 1)[0] + 1)
nsubs = len(good_subindices)

arrp, arrv, arra = np.zeros_like(arr), np.zeros_like(arr), np.zeros_like(arr)

for sub in good_subindices:
    dd = arr[sub]
    out = but1d(dd, 17.5, fs, 2, len(dd) - 1)
    arrp[sub], arrv[sub], arra[sub] = out['p'], out['v'], out['a']

# now fit splines to the smoothed position data
arrp[ixb], arrv[ixb], arra[ixb] = np.nan, np.nan, np.nan

order = 3
npts = 2

if nsubs > 1:  # we are not just at the edge
    for i in np.arange(nsubs - 1):
        for k in np.arange(3):
            data = arrp[:, k]
            ix0 = good_subindices[i][-npts:]
            ix1 = good_subindices[i + 1][:npts]
            indices = np.hstack([ix0, ix1])
            xs = times[indices]
            ys = data[indices]

            poly = np.polyfit(xs, ys, order)
            xf = np.polyval(poly, times[ix0[-1] + 1:ix1[0]])
            arrp[ix0[-1] + 1:ix1[0], k] = xf

arrfull = pr.copy()
arrfull[:, j] = arrp

outgaps = raw(arrfull, fs)
pg, vg, ag = outgaps['p'], outgaps['v'], outgaps['a']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
ax1.plot(times, pg[:, j, k], c=bmap[0])
ax2.plot(times, vg[:, j, k], c=bmap[0])
ax3.plot(times, ag[:, j, k], c=bmap[0])

ax1.plot(times, pf[:, j, k], c=bmap[2])
ax2.plot(times, vf[:, j, k], c=bmap[2])
ax3.plot(times, af[:, j, k], c=bmap[2])
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# new way
errp = np.sqrt((pf - pg)**2)
errv = np.sqrt((vf - vg)**2)
erra = np.sqrt((af - ag)**2)

ax1.plot(times, errp[:, j, k])
ax2.plot(times, errv[:, j, k])
ax3.plot(times, erra[:, j, k] / 9810)

sns.despine()
fig.set_tight_layout(True)


# %%

order = 3
npts = 2

a = np.arange(10, dtype=np.float)
#a[:3] = np.nan
#a[-3:] = np.nan
a[3:6] = np.nan

ixg = np.where(~np.isnan(a))[0]
ixb = np.where(np.isnan(a))[0]

good_subindices = np.split(ixg, np.where(np.diff(ixg) > 1)[0] + 1)
nsubs = len(good_subindices)

if nsubs > 1:  # we are not just at the edge
    for i in np.arange(nsubs - 1):
        ix0 = good_subindices[i][-npts:]
        ix1 = good_subindices[i + 1][:npts]
        indices = np.hstack([ix0, ix1])
        xs = times[indices]
        ys = a[indices]
        poly = np.polyfit(xs, ys, order)
        xf = np.polyval(poly, times[ix0[-1] + 1:ix1[0]])





# %%


fig, ax = plt.subplots()
ax.axvline(times[imin - 1], color='gray', lw=.75)
ax.axvline(times[imax], color='gray', lw=.75)
ax.plot(times, pr[:, j, k])
ax.plot(times, pc[:, j, k])
ax.plot(times, pg[:, j, k])
ax.plot(times, pf[:, j, k])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(times[imin - 1], color='gray', lw=.75)
ax.axvline(times[imax], color='gray', lw=.75)
ax.plot(times, pf[:, j, k])
ax.plot(times, pgs[:, j, k])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(times[imin - 1], color='gray', lw=.75)
ax.axvline(times[imax], color='gray', lw=.75)
ax.plot(times, vf[:, j, k])
ax.plot(times, vgs[:, j, k])
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(times[imin - 1], color='gray', lw=.75)
ax.axvline(times[imax], color='gray', lw=.75)
ax.plot(times, af[:, j, k])
ax.plot(times, ags[:, j, k])
sns.despine()
fig.set_tight_layout(True)

print rms_error(pf, pgs)
print rms_error(vf, vgs)
print rms_error(af, ags) / 9810


# %%



# %%



# %% Fit only to "interior points" where have data (don't extrapolate)

j = 13

kk = 1
ss = None

arr = pi[:, j, :]
ixb = np.where(np.isnan(arr[:, 0]))[0]
ixg = np.where(~np.isnan(arr[:, 0]))[0]

# split ixb into subarrays of the individual nan regions
nansps = np.split(ixb, np.where(np.diff(ixb) > 1)[0] + 1)

i0, i1 = 0, ntime
for nansp in nansps:
    if len(nansp) > 0:
        if nansp[0] == 0:
            i0 = nansp[-1] + 1
        elif nansp[-1] == ntime - 1:
            i1 = nansp[0]

print i0, i1

u = np.arange(ntime) / (ntime - 1)

tck, fp = splprep(arr[ixg].T.tolist(), u=u[ixg], k=kk, s=ss)

extrap = splev(u[ixb], tck)
fit = splev(np.linspace(0, 1, 1001), tck)
tfit = np.linspace(times.min(), times.max(), 1001)

xa, ya, za = arr.T
xe, ye, ze = extrap
xf, yf, zf = fit

fig, ax = plt.subplots()
ax.axvline(times[i0], color='gray', lw=.75)
ax.axvline(times[i1 - 1], color='gray', lw=.75)
ax.plot(times, xa, times, ya, times, za)
ax.plot(times[ixb], xe, 'o', c=bmap[0], ms=3)
ax.plot(times[ixb], ye, 'o', c=bmap[1], ms=3)
ax.plot(times[ixb], ze, 'o', c=bmap[2], ms=3)
ax.set_xlim(-.05 * times.min(), 1.05 * times.max())
sns.despine()
fig.set_tight_layout(True)


# %% Gap fill all of the data

P = pi.copy() * np.nan
E = pi.copy() * np.nan
rngix = np.zeros((nmark, 2))
Ph = pi.copy() * np.nan

u = np.arange(ntime) / (ntime - 1)

kk = 3
ss = .001

for j in marks:
    arr = pi[:, j, :]
    ixb = np.where(np.isnan(arr[:, 0]))[0]
    ixg = np.where(~np.isnan(arr[:, 0]))[0]

    i0, i1 = 0, ntime
    if len(ixb) > 0:
        nansps = np.split(ixb, np.where(np.diff(ixb) > 1)[0] + 1)
        for nansp in nansps:
            if nansp[0] == 0:
                i0 = nansp[-1] + 1
            elif nansp[-1] == ntime - 1:
                i1 = nansp[0]
    rngix[j] = i0, i1

    tck, fp = splprep(arr[ixg].T.tolist(), u=u[ixg], k=kk, s=ss)

    complete = splev(u, tck)
    P[:, j, :] = np.array(complete).T

    interior = splev(u[i0:i1], tck)
    Ph[i0:i1, j, :] = np.array(interior).T

    if len(ixb) > 0:
        extrap = splev(u[ixb], tck)
        E[ixb, j, :] = np.array(extrap).T


# %%

k = 0

for j in marks:

    fig, ax = plt.subplots()
    ax.plot(times, P[:, j, k])
    ax.set_xlim(times.min(), times.max())
    ax.set_title(j)
    sns.despine()
    fig.set_tight_layout(True)


# %%

k = 0

fig, ax = plt.subplots()
ax.plot(times, pi[:, :, k])
ax.set_xlim(times.min(), times.max())
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
ax.plot(times, P[:, :, k])
ax.set_xlim(times.min(), times.max())
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.plot(times, E[:, :, k])
ax.set_xlim(times.min(), times.max())
sns.despine()
fig.set_tight_layout(True)

fig, ax = plt.subplots()
for j in marks:
    print j
    ln, = ax.plot(times, pi[:, j, k])
    ax.plot(times, E[:, j, k], 'o', ms=3, c=ln.get_color())
ax.set_xlim(times.min(), times.max())
sns.despine()
fig.set_tight_layout(True)


# %%

ix = 75

tck, fp = splprep(pi[ix].T.tolist(), k=3, s=.001)

ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)


fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.points3d(pi[ix, :, 0], pi[ix, :, 1], pi[ix, :, 2], color=bmap[0],
              scale_factor=16)

mlab.plot3d(pi[ix, :, 0], pi[ix, :, 1], pi[ix, :, 2], color=bmap[1],
            tube_radius=5)

mlab.plot3d(xs, ys, zs, color=bmap[0], tube_radius=3)


# %%

ix = 25
ix = 76

tck, fp = splprep(Ph[ix].T.tolist(), k=3, s=.001)

ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)


fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.points3d(Ph[ix, :, 0], Ph[ix, :, 1], Ph[ix, :, 2], color=bmap[0],
              scale_factor=16)

mlab.plot3d(Ph[ix, :, 0], Ph[ix, :, 1], Ph[ix, :, 2], color=bmap[1],
            tube_radius=5)

mlab.plot3d(xs, ys, zs, color=bmap[0], tube_radius=3)


# %% Plot the unfilled coordinates

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime):
    mlab.points3d(pi[i, :, 0], pi[i, :, 1], pi[i, :, 2], color=bmap[1],
                  scale_factor=10)

    mlab.plot3d(pi[i, :, 0], pi[i, :, 1], pi[i, :, 2], color=bmap[1],
                tube_radius=10)


# %% Plot the fit body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime):
    mlab.points3d(P[i, :, 0], P[i, :, 1], P[i, :, 2], color=bmap[2],
                  scale_factor=12)

    mlab.plot3d(pi[i, :, 0], pi[i, :, 1], pi[i, :, 2], color=bmap[1],
                tube_radius=6)


# %% Plot the fit body

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime):
#
    mlab.plot3d(P[i, :, 0], P[i, :, 1], P[i, :, 2], color=bmap[1],
                tube_radius=4)

for j in marks:
    mlab.plot3d(P[:, j, 0], P[:, j, 1], P[:, j, 2], color=bmap[3],
                tube_radius=4)


# %% Time-body mesh of the snake (with missing points)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

for i in np.arange(ntime):
    mlab.plot3d(Ph[i, :, 0], Ph[i, :, 1], Ph[i, :, 2], color=bmap[1],
                tube_radius=4)

for j in marks:
    mlab.plot3d(Ph[:, j, 0], Ph[:, j, 1], Ph[:, j, 2], color=bmap[3],
                tube_radius=4)


# %%

#xx, yy, zz = [], [], []
#for j in marks:
#    xx.append(Ph[i, :, 0])
#    yy.append(Ph[:, j, 1])
#    zz.append(Ph[:, j, 2])


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(Ph[:, :, 0], Ph[:, :, 1], Ph[:, :, 2], color=bmap[0], opacity=.5)

#for i in np.arange(ntime):
#    mlab.plot3d(Ph[i, :, 0], Ph[i, :, 1], Ph[i, :, 2], color=bmap[1],
#                tube_radius=4)

#for j in marks:
#    mlab.plot3d(Ph[:, j, 0], Ph[:, j, 1], Ph[:, j, 2], color=bmap[3],
#                tube_radius=4)


# %%

coms = np.zeros((ntime, 3))
for i in np.arange(ntime):
    coms[i] = np.nanmean(Ph[i, :, :], axis=0)


T = np.zeros((ntime, nmark))
for i in np.arange(ntime):
    T[i] = times[i]

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(Ph[:, :, 0], Ph[:, :, 1], Ph[:, :, 2], opacity=.5, scalars=T,
          colormap='copper', representation='surface')
mlab.points3d(Ph[:, :, 0], Ph[:, :, 1], Ph[:, :, 2], opacity=.5,
          color=bmap[2], scale_factor=15)

mlab.points3d(coms[:, 0], coms[:, 1], coms[:, 2], color=bmap[0],
              scale_factor=20)


# %%

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.mesh(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], opacity=.5, scalars=T,
          colormap='copper', representation='surface')
mlab.points3d(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], opacity=.5,
          color=bmap[2], scale_factor=15)


# %%



#from tvtk.api import tvtk

#for i in np.arange(ntime):
#    mlab.plot3d(Ph[i, :, 0], Ph[i, :, 1], Ph[i, :, 2], color=bmap[1],
#                tube_radius=4)

#for i in np.arange(ntime):
#    mlab.plot3d(Ph[i, :, 0], Ph[i, :, 1], Ph[i, :, 2], color=bmap[1],
#                tube_radius=4, opacity=.5)
#    mlab.points3d(Ph[i, 0, 0], Ph[i, 0, 1], Ph[i, 0, 2], color=bmap[1],
#                scale_factor=20)

# fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


# %%

ix = 25
ix = 76

tck, fp = splprep(Ph[ix].T.tolist(), k=3, s=.001)

ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)


fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

mlab.points3d(Ph[ix, :, 0], Ph[ix, :, 1], Ph[ix, :, 2], color=bmap[0],
              scale_factor=16)

mlab.plot3d(Ph[ix, :, 0], Ph[ix, :, 1], Ph[ix, :, 2], color=bmap[1],
            tube_radius=5)

mlab.plot3d(xs, ys, zs, color=bmap[0], tube_radius=3)



# %%

fig = mlab.figure(bgcolor=(1, 1, 1), size=(750, 750))

ix = 25
xx, yy, zz = (Ph[ix, :, :] - Ph[ix, :, :].mean(axis=0)).T
tck, fp = splprep([xx, yy, zz], k=3, s=.001)
ts = np.linspace(0, 1, 101)
xs, ys, zs = splev(ts, tck)

pts = mlab.points3d(xx, yy, zz, color=bmap[1], scale_factor=16)
# bdy = mlab.plot3d(xs, ys, zs, color=bmap[1], tube_radius=3)
bdy = mlab.plot3d(xx, yy, zz, color=bmap[1], tube_radius=3)


@mlab.animate(delay=100)
def anim():
    for k in np.arange(20):
        for ix in np.arange(25, 75):
            xx, yy, zz = (Ph[ix, :, :] - Ph[ix, :, :].mean(axis=0)).T
            tck, fp = splprep([xx, yy, zz], k=3, s=.001)
            ts = np.linspace(0, 1, 101)
            xs, ys, zs = splev(ts, tck)

            pts.mlab_source.set(x=xx, y=yy, z=zz)
            # bdy.mlab_source.set(x=xs, y=ys, z=zs)
            bdy.mlab_source.set(x=xx, y=yy, z=zz)

            yield

manim = anim()
mlab.show()


# %%


#TraitError: The 'colormap' trait of a SurfaceFactory instance must be 'Accent' or 'Blues' or 'BrBG' or 'BuGn' or 'BuPu' or 'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 'PRGn' or 'Paired' or 'Pastel1' or 'Pastel2' or 'PiYG' or 'PuBu' or 'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' or 'RdYlBu' or 'RdYlGn' or 'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' or 'YlGn' or 'YlGnBu' or 'YlOrBr' or 'YlOrRd' or 'autumn' or 'binary' or 'black-white' or 'blue-red' or 'bone' or 'cool' or 'copper' or 'file' or 'flag' or 'gist_earth' or 'gist_gray' or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or 'gist_yarg' or 'gray' or 'hot' or 'hsv' or 'jet' or 'pink' or 'prism' or 'spectral' or 'spring' or 'summer' or 'winter', but a value of 'purd' <type 'str'> was specified.

#m = 75
#atom = .01
#A = atom * m
#
#a = np.linspace(0, 80, 100)
#acd = .0975 + .00544 * (a - 10) + .0000844 * (a - 10)**2
#acl = .0975 + .01331 * (a - 10) + .0001969 * (a - 10)**2
#
#cd = acd / A
#cl = acl / A
#
#fig, ax = plt.subplots()
#ax.plot(cd, cl)
#ax.axis('equal')
##ax.set_xlim(0, .9)
##ax.set_ylim(0, .7)
#sns.despine()
#fig.set_tight_layout(True)





