# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:18:02 2016

%reset -f
%clear
%pylab
%load_ext autoreload
%autoreload 2

cd /Volumes/Yeaton_HD6/Code for Manuscripts/Undulation_confers_stability/Experiments/Code

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_all_proc_plots/{}.pdf'
FIGPNG = '../Figures/s_all_proc_plots/{}.png'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}

# notes on Mayavi views
# mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
# mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
# mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
# mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)

# %% Function definitions

def ret_fnames(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = '../Data/Processed Qualisys output/'
    fn_search = fn_proc + fn_trial

    return sorted(glob(fn_search))


def ret_fnames_may(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = '../Data/Processed Qualisys output - March 2016/'
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
X0, Y0 = np.zeros(ntrials), np.zeros(ntrials)
for i, fname in enumerate(ret_fnames()):
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    X, Y, Z = d['Ro_I'].T / 1000  # m
    X0[i] = X[0]
    Y0[i] = Y[0]
Xo = X0.mean()  # 0.44458394864868178
Yo = Y0.mean()  # -4.684127769871548

# for fource calculations
grav = 9.81  # m/s^2 (gravitational acceleration)
rho = 1.17  # kg/m^3 (air density)

# %%

fn_names = ret_fnames(81)

#fn_names = ret_fnames(snake_id)

Wss = []
areas = []
eps_chord_mean = []
eps_chord_max = []
eps_svl = []

for i, fname in enumerate(fn_names):
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)

    SVL = d['SVL_avg'] / 1000

    mass_kg = float(d['mass']) / 1000
    mg = mass_kg * grav  # weight (N)

    chord_len_m = d['chord_spl'][0] / 1000
    dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
    snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
    snake_area_cm = snake_area * 100**2

    Ws = mg / snake_area

    eps_chord_mean_indiv = (rho * grav / 2) * (chord_len_m.mean() / Ws)
    eps_chord_max_indiv = (rho * grav / 2) * (chord_len_m.max() / Ws)
    eps_svl_indiv = (rho * grav / 2) * (SVL / Ws)

    to_non_re_vel = np.sqrt(2 * Ws / rho)
#    v_non_re = v / to_non_re_vel
    v_non_re = d['dRo_I'] / to_non_re_vel

    Wss.append(Ws)
    areas.append(snake_area_cm)
    eps_chord_mean.append(eps_chord_mean_indiv)
    eps_chord_max.append(eps_chord_max_indiv)
    eps_svl.append(eps_svl_indiv)

Ws = np.array(Wss)
areas = np.array(areas)
eps_chord_mean = np.array(eps_chord_mean)
eps_chord_max = np.array(eps_chord_max)
eps_svl = np.array(eps_svl)


# %% Y-Z trajectorires with four subplots

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.8, 9))

axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        axs[col].plot(Y, Z, c=snake_colors[snake_id])
        if i == 0:
            tit = 'Snake {0} ({1} g)'.format(snake_id, float(d['mass']))
            axs[col].set_title(tit)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        if i > 0:
            axs[col].plot(Y, Z, c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, Z, c=snake_colors[snake_id], label=snake_id)
axs[col].set_title('Remaining snakes')
axs[col].legend(loc='upper right')

plt.setp(axs, aspect=1.0, adjustable='box')
axs[0].set_xlim(xmin=-1)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('1_side_view_pos.pdf'), bbox_inches='tight')


# %% Y-Z trajectorires on one plot

fig, ax = plt.subplots(figsize=(7.8, 9))

#ax.axhline(8.3, color='gray', lw=1)

for snake_id in snake_ids:
    fn_names = ret_fnames(snake_id)
    print(snake_id)

    cnt = 0
    for fname in fn_names:
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000

        if Z[-1] < 2:
            if cnt > 0:
                ax.plot(Y, Z, c=snake_colors[snake_id])
            else:
                cnt += 1
                ax.plot(Y, Z, c=snake_colors[snake_id], label=snake_id)

#        if Z[-1] > 2:
#            label = '{0}_{1}'.format(snake_id, trial_id)
#            ax.plot(Y, Z, c=snake_colors[snake_id], label=label)

ax.legend(loc='upper right')

ax.set_aspect('equal', adjustable='box')
#plt.setp(axs, aspect=1.0, adjustable='box')
ax.set_xlim(-1, 5)
ax.set_ylim(0, 9)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('1_side_view_pos.pdf'), bbox_inches='tight')


# %% Y-Z trajectorires on one plot

fig, ax = plt.subplots(figsize=(7.8, 9))

#ax.axhline(8.3, color='gray', lw=1)

for snake_id in snake_ids:
    fn_names = ret_fnames(snake_id)
    print(snake_id)

    cnt = 0
    for fname in fn_names:
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000

#        if Z[-1] < 2:
#            if cnt > 0:
#                ax.plot(Y, Z, c=snake_colors[snake_id])
#            else:
#                cnt += 1
#                ax.plot(Y, Z, c=snake_colors[snake_id], label=snake_id)

        if Z[-1] > 2:
            label = '{0}_{1}'.format(snake_id, trial_id)
            ax.plot(Y, Z, c=snake_colors[snake_id], label=label)

ax.legend(loc='upper right')

ax.set_aspect('equal', adjustable='box')
#plt.setp(axs, aspect=1.0, adjustable='box')
ax.set_xlim(-1, 5)
ax.set_ylim(0, 9)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('1_side_view_pos.pdf'), bbox_inches='tight')


# %% Y-Z trajectorires with four subplots - LABELS

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.8, 9))

axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        # _label = '{0}_{1}'.format(trial_id, snake_id)
        _label = '{0}'.format(trial_id)
        axs[col].plot(Y, Z, c=colors_trial_id[i], lw=2, label=_label)
#        axs[col].plot(Y[-1], Z[-1], 'o', c=colors_trial_id[i])
        if i == 0:
            tit = 'Snake {0} ({1} g)'.format(snake_id, float(d['mass']))
            axs[col].set_title(tit)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        if i > 0:
            axs[col].plot(Y, Z, c=snake_colors[snake_id], lw=2)
        else:
            label = '{}_{}'.format(snake_id, trial_id)
#            label = snake_id
            if snake_id == 88:
                lw = 4
            else:
                lw = 2
            axs[col].plot(Y, Z, c=snake_colors[snake_id], lw=lw, label=label)

#_leg_args = dict(fontsize='xx-small', ncol=2, columnspacing=1, handlelength=1.3, handletextpad=.25)
_leg_args = dict(fontsize='xx-small', ncol=2, columnspacing=.75, handlelength=1, handletextpad=.2, borderaxespad=0)
axs[0].legend(loc='upper right', **_leg_args)
axs[1].legend(loc='upper right', **_leg_args)
axs[2].legend(loc='upper right', **_leg_args)
axs[3].set_title('Remaining snakes')
axs[3].legend(loc='upper right', **_leg_args)

plt.setp(axs, aspect=1.0, adjustable='box')
axs[0].set_xlim(xmin=-1)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Ro_S-side'), **FIGOPT)


# %% Y-Z trajectories, non-dim and rescaled

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.8, 9))

axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000


        mg = d['weight']
        dA = d['spl_ds'] * d['chord_spl'] / 1000**2
        snake_area = dA.sum(axis=1).mean()
        snake_area_cm = snake_area * 100**2
        Ws = mg / snake_area
        to_non_re_pos = (2 * Ws) / (rho * 9.81)

        # non-dim and rescale
        Y = Y / to_non_re_pos
        Z = Z / to_non_re_pos
        Y -= Y[0]
        Z = Z - Z[0] + 1.4

        _label = '{0}'.format(trial_id)
        axs[col].plot(Y, Z, c=colors_trial_id[i], label=_label)
        if i == 0:
            tit = 'Snake {0} ({1} g)'.format(snake_id, float(d['mass']))
            axs[col].set_title(tit)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000

        mg = d['weight']
        dA = d['spl_ds'] * d['chord_spl'] / 1000**2
        snake_area = dA.sum(axis=1).mean()
        snake_area_cm = snake_area * 100**2
        Ws = mg / snake_area
        to_non_re_pos = (2 * Ws) / (rho * 9.81)

        # non-dim and rescale
        Y = Y / to_non_re_pos
        Z = Z / to_non_re_pos
        Y -= Y[0]
        Z = Z - Z[0] + 1.4

        if i > 0:
            axs[col].plot(Y, Z, c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, Z, c=snake_colors[snake_id], label=snake_id)
axs[col].set_title('Remaining snakes')
axs[col].legend(loc='upper right')
#_leg_args = dict(fontsize='xx-small', ncol=2, columnspacing=1, handlelength=1.3, handletextpad=.25)
_leg_args = dict(fontsize='x-small', ncol=2, columnspacing=.75, handlelength=1, handletextpad=.2, borderaxespad=0)
axs[0].legend(loc='upper right', **_leg_args)
axs[1].legend(loc='upper right', **_leg_args)
axs[2].legend(loc='upper right', **_leg_args)

plt.setp(axs, aspect=1.0, adjustable='box')
#axs[0].set_xlim(xmin=-1)
axs[0].set_ylim(0, 1.4)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Ro_S-side_non-dim'), **FIGOPT)


# %% Top view of Y-X landing positions (four subplots)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

_lw = .15
radii = np.r_[1, 2, 3, 4, 5, 5]
angles = np.r_[0, 30, 60, 90]
angles = np.r_[angles, -angles]

for ax in axs:
    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_I'].T / 1000
        X -= Xo
        Y -= Yo
        axs[col].plot(X, Y, c=snake_colors[snake_id])

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_I'].T / 1000
        X -= Xo
        Y -= Yo
        if i > 0:
            axs[col].plot(X, Y, c=snake_colors[snake_id])
        else:
            axs[col].plot(X, Y, c=snake_colors[snake_id], label=snake_id)
axs[col].legend(loc='upper left', frameon=True)

plt.setp(axs, aspect=1.0, adjustable='box')
#ax.set_xlim(-3.1, 3.1)
ax.set_xlim(-3, 2)
ax.set_ylim(0, 5.5)
ax.set_yticks([])
sns.despine(left=True)
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('2_top_view_pos.pdf'), bbox_inches='tight')


# %% Top view of Y-X landing positions (four subplots) - LABELS

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

_lw = .15
radii = np.r_[1, 2, 3, 4, 5, 5]
angles = np.r_[0, 30, 60, 90]
angles = np.r_[angles, -angles]

for ax in axs:
    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
#    colors_trial_id = sns.color_palette("cubehelix", len(fn_names))
#    colors_trial_id = sns.cubehelix_palette(len(fn_names), start=.5, rot=-.75)
#    colors_trial_id = sns.dark_palette("purple", len(fn_names))

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
#        X, Y, Z = d['Ro_I'].T / 1000
        X, Y, Z = d['Ro_I_raw'].T / 1000
        X -= Xo
        Y -= Yo
        _label = '{0}'.format(trial_id)
        axs[col].plot(X, Y, '-', c=colors_trial_id[i], label=_label)
#        axs[col].plot(Y[-1], Z[-1], 'o', c=colors_trial_id[i])

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_I'].T / 1000
        X -= Xo
        Y -= Yo
        if i > 0:
            axs[col].plot(X, Y, c=snake_colors[snake_id])
        else:
            axs[col].plot(X, Y, c=snake_colors[snake_id], label=snake_id)
_leg3 = axs[col].legend(loc='upper right', frameon=True)
_leg_args = dict(fontsize='x-small', ncol=4, columnspacing=.75, handlelength=1, handletextpad=.2, borderaxespad=0, frameon=True)
#axs[0].legend(loc='upper right', fontsize='x-small', ncol=4, frameon=False)
#axs[1].legend(loc='upper right', fontsize='x-small', ncol=4, frameon=False)
#axs[2].legend(loc='upper right', fontsize='x-small', ncol=4, frameon=False)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

plt.setp(axs, aspect=1.0, adjustable='box')
#ax.set_xlim(-3.1, 3.1)
ax.set_xlim(-3, 2)
ax.set_ylim(0, 5.5)
ax.set_yticks([])
sns.despine(left=True)
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Ro_S-top'), bbox_inches='tight')


# %% YRESULT -Z trajectorires with four subplots - LABELS

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.6, 8.3))

axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names), l=.55)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        _label = '{0}'.format(trial_id)
        axs[col].plot(Y, Z, c=colors_trial_id[i], lw=2, label=_label)

        if i == 0:
            tit = 'Snake {0} ({1} g)'.format(snake_id, float(d['mass']))
            axs[col].set_title(tit, fontsize='small')

# get colors for the last plot
ntrials = 0
for snake_id in [86, 88, 90, 94]:
    ntrials += len(ret_fnames(snake_id))

colors_trial_id = sns.husl_palette(n_colors=ntrials, l=.55)

col = 3
cnt = 0
for snake_id in [86, 88, 90, 94]:
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        label = '{0}, {1}'.format(snake_id, trial_id)
        axs[col].plot(Y, Z, c=colors_trial_id[cnt], label=label)
        cnt += 1

_leg_args = dict(fontsize='xx-small', ncol=2, columnspacing=.75,
                 handlelength=1, handletextpad=.2, borderaxespad=0)
axs[0].legend(loc='upper right', **_leg_args)
axs[1].legend(loc='upper right', **_leg_args)
axs[2].legend(loc='upper right', **_leg_args)
axs[3].legend(loc='upper right', **_leg_args)

axs[3].set_title('Remaining snakes', fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box')
axs[0].set_xlim(-.75, 5.5)
axs[0].set_ylim(0, 8.4)
axs[0].set_xticks([0, 2, 4])
axs[0].set_yticks([0, 2, 4, 6, 8])

axs[2].set_xlabel('Y (m)', fontsize='small')
axs[2].set_ylabel('Z (m)', fontsize='small')

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('result Ro_S - side'), **FIGOPT)


# %% RESULT Top view of Y-X landing positions (four subplots) - LABELS

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(9, 7.075))
axs = axs.flatten()

_lw = .15
radii = np.r_[1, 2, 3, 4, 5, 5]
angles = np.r_[0, 30, 60, 90]
angles = np.r_[angles, -angles]

for ax in axs:
    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names), l=.55)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_I_raw'].T / 1000
        X -= Xo
        Y -= Yo

        _label = '{0}'.format(trial_id)
        axs[col].plot(X, Y, '-', c=colors_trial_id[i], label=_label)


# get colors for the last plot
ntrials = 0
for snake_id in [86, 88, 90, 94]:
    ntrials += len(ret_fnames(snake_id))

colors_trial_id = sns.husl_palette(n_colors=ntrials, l=.55)

col = 3
cnt = 0
for snake_id in [86, 88, 90, 94]:
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_I_raw'].T / 1000
        X -= Xo
        Y -= Yo

        _label = '{0}, {1}'.format(snake_id, trial_id)
        axs[col].plot(X, Y, c=colors_trial_id[cnt], label=_label)
        cnt += 1

#_leg3 = axs[col].legend(loc='upper right', frameon=True)
_leg_args = dict(fontsize='xx-small', ncol=4, columnspacing=.75,
                 handlelength=1, handletextpad=.2, borderaxespad=0,
                 frameon=True)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)
_leg3 = axs[3].legend(loc='upper right', **_leg_args)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

plt.setp(axs, aspect=1.0, adjustable='box')
#ax.set_xlim(-3.1, 3.1)
ax.set_xlim(-3.05, 3.05)
ax.set_xticks(np.r_[-3:3.1:1])
ax.set_ylim(-.1, 5.5)
ax.set_yticks([])

axs[2].set_xlabel('X (m)', fontsize='small')
axs[3].set_xlabel('X (m)', fontsize='small')

sns.despine(left=True)
fig.set_tight_layout(True)

#fig.savefig(FIG.format('results Ro_I_raw - top'),**FIGOPT)


# %% LANDING LOCATIONS FROM SOCHA (2005)

import pandas as pd


fname = '../Data/Socha2005/Data from Socha et al 2005 JEB top of tower only.xlsx'
df_2005 = pd.read_excel(fname, header=3, index_col=None, cols='A:F',
                        na_values='\\')

info = '''28 26.16 1.5
29 62.9 83.3
30 76.3 86.5
31 10.5 47.0
32 37.5 69.3
33 35.9 68.5
34 24.2 66.0
35 27.4 62.8
36 30.8 66.0
37 14.6 52.5
38 27.5 63.3
39 40.6 70.7
40 27.4 63.5
41 3.0 31.0
42 82.7 85.3
43 53.2 84.0
44 16.3 54.2
45 68.2 81.8
46 50.7 76.9
48 52.3 77.5
49 22.5 60.5
50 40.5 75.0'''

info = info.split('\n')
infos = []
for line in info:
    sn, mm, ll = line.split(' ')
    infos.append([int(sn), float(mm), float(ll)])

df_2005_info = pd.DataFrame(infos, columns=['Snake', 'mass (g)', 'SVL (cm)'])

Ws_2005 = np.e ** (.37 * np.log(df_2005_info['mass (g)']) + 2.03)
df_2005_info['Ws (N/m2)'] = Ws_2005

df_2005_info['Study'] = 'Socha (2005)'

# add columns for extras
df_2005['m'] = 0
df_2005['SVL'] = 0
df_2005['Ws'] = 0
df_2005_arr = df_2005.values
df_2005_info_arr = df_2005_info.values

sn_ids = np.unique(df_2005['Snake'].values)
#for sn_id in sn_ids:
for i in np.arange(len(df_2005_info_arr)):
    sn_id, m, SVL, Ws, _ = df_2005_info_arr[i]
    idx = np.where(df_2005_arr[:, 1] == sn_id)
    df_2005_arr[idx, 6] = m
    df_2005_arr[idx, 7] = SVL
    df_2005_arr[idx, 8] = Ws


# convert back to dataframe
df_2005 = pd.DataFrame(df_2005_arr, columns=['Trial', 'Snake ID', 'Flight',
                       'X', 'Y', 'Dist. travelled', 'm', 'SVL', 'Ws'])

# glide ratio
df_2005['GR'] = df_2005['Dist. travelled'] / 9.6


# %% CUBE DATA, HEAD POSITION AT LAST RECORDED POSITION

from glob import glob
import m_data_utils as data_utils

fnames = sorted(glob('../Data/Raw Qualisys output/*.tsv'))

ids, trial, xland, yland, zland, dist, dz = [], [], [], [], [], [], []
for fname in fnames:
    ending = fname.split('.tsv')[0]
    if ending.endswith('filled') or ending.endswith('complete'):
        continue

#    print fname
    df_tmp = data_utils.load_qtm_tsv(fname)
    pr, out = data_utils.reconfig_raw_data(df_tmp)

    head = pr[:, 0]
    good_idx = np.where(~np.isnan(head[:, 0]))[0]
    X, Y, Z = head[good_idx[-1]] / 1000

    X -= Xo
    Y -= Yo

    X0, Y0, Z0 = head[good_idx[0]] / 1000

    trial_snake = fname.split('/')[-1].split('.')[0]
    tr, sn = trial_snake.split('_')
    tr, sn = int(tr), int(sn)

    ids.append(sn)
    trial.append(tr)
    xland.append(X)
    yland.append(Y)
    zland.append(Z)
    dist.append(np.sqrt(X**2 + Y**2))
    dz.append(Z0 - Z)

df_cube = pd.DataFrame({'Snake ID': ids, 'Trial': trial,
                        'X': xland, 'Y': yland, 'Z': zland,
                        'Dist. travelled': dist,
                        'dZ': dz})

# (Y[0] - Y[-1]) / (Z[0] - Z[-1])
df_cube['GR'] = df_cube['Dist. travelled'] / df_cube['dZ']


# %%

Ws_dict = dict()
to_non_dict = dict()

sn_arr = []
Ws_mean = []
Ws_std = []
mass_arr = []
SVL_avg = []

for snake_id in snake_ids:
    fn_names = ret_fnames(snake_id)

    Ws_arr = []
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        mg = d['weight']
        dA = d['spl_ds'] * d['chord_spl'] / 1000**2
        snake_area = dA.sum(axis=1).mean()
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        sn_arr.append(snake_id)
        Ws_arr.append(Ws)
#        to_non_arr.append(to_non_re_vel)

    sn_arr.append(snake_id)
    Ws_mean.append(np.array(Ws_arr).mean())
    Ws_std.append(np.array(Ws_arr).std())
    mass_arr.append(float(d['mass']))
    SVL_avg.append(float(d['SVL_avg']))


#info = '''30	132.3	78.9
#31	164.3	75.0
#32	52.5	66.2
#33	272.5	91.8
#35	141.0	82.8
#81	107.2	85.0
#86	54.5	81.0
#88	71.9	88.8
#90	39.7	67.2
#91	71.0	76.6
#94	49.9	73.1
#95	37.3	64.4'''

info = '''81	107.2	85.0
86	54.5	81.0
88	71.9	88.8
90	39.7	67.2
91	71.0	76.6
94	49.9	73.1
95	37.3	64.4'''

info = info.split('\n')
infos = []
for line in info:
    sn, mm, ll = line.split('\t')
    infos.append([int(sn), float(mm), float(ll)])

df_cube_info = pd.DataFrame(infos, columns=['Snake', 'mass (g)', 'SVL (cm)'])

Ws_cube = np.e ** (.37 * np.log(df_cube_info['mass (g)']) + 2.03)
#Ws_cube = np.array(Ws_mean)
df_cube_info['Ws (N/m2)'] = Ws_cube

df_cube_info['Study'] = 'Cube'


# %% Add mass, SVL, and Ws to the dataframes

d2005i = df_2005_info.values
for i in np.arange(len(df_2005_info)):
    sn_id, m, SVL, Ws, _ = d2005i[i]

    df_2005[df_2005['Snake'] == sn_id]['m'] = m
    df_2005[df_2005['Snake'] == sn_id]['SVL'] = SVL
    df_2005[df_2005['Snake'] == sn_id]['Ws'] = Ws


# %% RESULT: MORPHOMETRICS COMPARE SOCHA (2005)

df_info = pd.concat([df_2005_info, df_cube_info])


#fig, ax = plt.subplots()
#sns.swarmplot(x='Study', y='Ws (N/m2)', data=df_info, ax=ax)
#sns.despine()
#fig.set_tight_layout(True)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4.7))
sns.swarmplot(x='Study', y='mass (g)', data=df_info, ax=ax1)
sns.swarmplot(x='Study', y='SVL (cm)', data=df_info, ax=ax2)
sns.swarmplot(x='Study', y='Ws (N/m2)', data=df_info, ax=ax3)
ax1.set_ylim(ymin=0)
ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('compare_m_SVL_Ws'), **FIGOPT)


# %%

## %% WING LOADING AND SCALING PARAMETER DISTRIBUTIONS
#
#import pandas as pd
#
#data = {'ids': sn_arr, 'Ws': Ws_arr, 'to_non': to_non_arr}
#df = pd.DataFrame(data=data)
#
#
#fig, ax = plt.subplots(figsize=(6, 5))
#
#ax.axhline(29, color='gray', lw=1)
#ax.axhspan(29 - 9, 29 + 9, color='gray', alpha=.2)
#ax.axhspan(12, 46, color='gray', alpha=.1)
#
#sns.swarmplot(x='ids', y='Ws', data=df, ax=ax, marker='o')
#
#ax.set_ylim(ymin=0)
#ax.set_xlabel('Snake ID')
#ax.set_ylabel(r'Wing loading (N/m$^\mathrm{2}$)')
#sns.despine()
#fig.set_tight_layout(True)


# %%

g = sns.lmplot('X', 'Y', data=df_2005, hue='Snake', fit_reg=False,
               legend=False)
fig, ax = g.fig, g.ax

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]
_lw = .15

for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.axis('off')

sns.despine()
fig.set_tight_layout(True)


# %%

g = sns.lmplot('X', 'Y', data=df_cube, hue='Snake ID', fit_reg=False,
               legend=False)
fig, ax = g.fig, g.ax

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)


# %% RESULT: COMPARE SOCHA (2005) GLIDE PERFORMANCE

g = sns.lmplot('X', 'Y', data=df_2005, hue='Ws', fit_reg=False,
               legend=False)
fig, ax = g.fig, g.ax

#fig, ax = plt.subplots()
#ax.axvline(0, color='gray', lw=1)
#ax.axhline(0, color='gray', lw=1)

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

_lw = .15

for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

#ax.plot(x, y, 'o')
#sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
#            scatter_kws={'c': df['Snake']})

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
#ax.set_xlim(-20, 20)
#ax.set_ylim(-5, 15)
ax.set_xlim(-22.5, 22.5)
ax.set_ylim(-22.5, 22.5)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('compare_Socha2005'), **FIGOPT)


# %% RESULT: MORPHOMETRICS COMPARE SOCHA (2005)

df_info = pd.concat([df_2005_info, df_cube_info])


#fig, ax = plt.subplots()
#sns.swarmplot(x='Study', y='Ws (N/m2)', data=df_info, ax=ax)
#sns.despine()
#fig.set_tight_layout(True)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4.7))
sns.swarmplot(x='Study', y='mass (g)', data=df_info, ax=ax1,
              hue='mass (g)', palette='Reds_r')
sns.swarmplot(x='Study', y='SVL (cm)', data=df_info, ax=ax2,
              hue='SVL (cm)', palette='Greens_r')
sns.swarmplot(x='Study', y='Ws (N/m2)', data=df_info, ax=ax3,
              hue='Ws (N/m2)', palette='Blues_r')

#fig.canvas.draw()

ax1.legend_.remove()
ax2.legend_.remove()
ax3.legend_.remove()

ax1.set_ylim(ymin=0)
ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('compare_m_SVL_Ws_colors'), **FIGOPT)


# %% RESULT: COMPARE WITH COLORS

_, mass_min, SVL_min, Ws_min, _ = df_info.min()
_, mass_max, SVL_max, Ws_max, _ = df_info.max()

import matplotlib.colors as colors
import matplotlib.cm as cm

cnorm = colors.Normalize(vmin=mass_min, vmax=mass_max)
mass_cmap = cm.ScalarMappable(norm=cnorm, cmap=plt.cm.Reds_r)

cnorm = colors.Normalize(vmin=SVL_min, vmax=SVL_max)
SVL_cmap = cm.ScalarMappable(norm=cnorm, cmap=plt.cm.Greens_r)

cnorm = colors.Normalize(vmin=Ws_min, vmax=Ws_max)
Ws_cmap = cm.ScalarMappable(norm=cnorm, cmap=plt.cm.Blues_r)


fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(11, 8))
axs_2005, axs_cube = axs[0], axs[1]

sn_ids = np.unique(df_2005_info['Snake'].values)
for sn_id in sn_ids:
    df_sel = df_2005[df_2005['Snake ID'] == sn_id]
    X, Y = df_sel['X'], df_sel['Y']

    df_sel_info = df_info[df_info['Snake'] == sn_id].values[0]
    _, mass_sel, SVL_sel, Ws_sel, _ = df_sel_info

    mass_color = mass_cmap.to_rgba(mass_sel)
    SVL_color = SVL_cmap.to_rgba(SVL_sel)
    Ws_color = Ws_cmap.to_rgba(Ws_sel)

    axs_2005[0].plot(X, Y, 'o', c=mass_color, zorder=100)
    axs_2005[1].plot(X, Y, 'o', c=SVL_color, zorder=100)
    axs_2005[2].plot(X, Y, 'o', c=Ws_color, zorder=100)


sn_ids = np.unique(df_cube_info['Snake'].values)
for sn_id in sn_ids:
    df_sel = df_cube[df_cube['Snake ID'] == sn_id]
    X, Y = df_sel['X'], df_sel['Y']

    df_sel_info = df_info[df_info['Snake'] == sn_id].values[0]
    _, mass_sel, SVL_sel, Ws_sel, _ = df_sel_info

    mass_color = mass_cmap.to_rgba(mass_sel)
    SVL_color = SVL_cmap.to_rgba(SVL_sel)
    Ws_color = Ws_cmap.to_rgba(Ws_sel)

    axs_cube[0].plot(X, Y, 'o', c=mass_color, zorder=100)
    axs_cube[1].plot(X, Y, 'o', c=SVL_color, zorder=100)
    axs_cube[2].plot(X, Y, 'o', c=Ws_color, zorder=100)


#g = sns.lmplot('X', 'Y', data=df_2005, hue='Snake', fit_reg=False,
#               legend=False)
#fig, ax = g.fig, g.ax

#fig, ax = plt.subplots()
#ax.axvline(0, color='gray', lw=1)
#ax.axhline(0, color='gray', lw=1)

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

_lw = .15

for ax in axs.flatten():

    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

axs[1, 0].set_xlabel('X (m)')
axs[1, 0].set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
ax.set_xticks([-20, -10, 0, 10, 20])
ax.set_yticks([-20, -10, 0, 10, 20])
ax.set_xlim(-22.5, 22.5)
ax.set_ylim(-22.5, 22.5)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('compare_with_colors'), **FIGOPT)


# %%

fig, ax = plt.subplots()

sn_ids = np.unique(df_2005_info['Snake'].values)
for sn_id in sn_ids:
    df_sel = df_2005[df_2005['Snake ID'] == sn_id]
    Ws_sel = df_sel['Ws'].mean()  # they are all the same, select it out...
    X, Y = df_sel['X'], df_sel['Y']
    color = scalar_map.to_rgba(Ws_sel)
    ax.plot(1, Ws_sel, 'o', c=color)

sns.despine()


# %% RESULT: COMPARE SOCHA (2005) GLIDE PERFORMANCE

g = sns.lmplot('X', 'Y', data=df_cube, hue='Snake ID', fit_reg=False,
               legend=False)
fig, ax = g.fig, g.ax

#fig, ax = plt.subplots()
#ax.axvline(0, color='gray', lw=1)
#ax.axhline(0, color='gray', lw=1)

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

#ax.plot(x, y, 'o')
#sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
#            scatter_kws={'c': df['Snake']})

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
ax.set_xlim(-22.5, 22.5)
ax.set_ylim(-22.5, 22.5)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('compare_Cube'), **FIGOPT)


# %%

df = pd.read_csv('../Data/September 2017/September 2017.csv')

snake = df['Snake']
#X = df['X'] / 100
#Y = df['Y'] / 100

df['X'] = df['X'] / 100
df['Y'] = df['Y'] / 100
height = df['Height']

df['Snake ID'] = df['Snake']

df_sept = df


# %%

g = sns.lmplot('X', 'Y', data=df_sept, hue='Snake ID', fit_reg=False,
               legend=False)
fig, ax = g.fig, g.ax

#fig, ax = plt.subplots()
#ax.axvline(0, color='gray', lw=1)
#ax.axhline(0, color='gray', lw=1)

radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

#ax.plot(x, y, 'o')
#sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
#            scatter_kws={'c': df['Snake']})

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal', adjustable='box')
ax.set_xlim(-20, 20)
ax.set_ylim(-5, 15)
sns.despine()
fig.set_tight_layout(True)


# %%

g = sns.lmplot('X', 'Y', data=df_2005, hue='Snake', col='Snake',
               fit_reg=False, col_wrap=6)
fig, axs = g.fig, g.axes


radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

for ax in axs:
    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

#ax.plot(x, y, 'o')
#sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
#            scatter_kws={'c': df['Snake']})

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal', adjustable='box')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-5, 15)
sns.despine()
fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.axhline(df_2005['GR'].mean(), color='b', lw=1.5)
ax.axhline(df_cube['GR'].mean(), color='r', lw=1.5)
ax.plot(df['GR'], 'bo')
ax.plot(df_cube['GR'], 'rs')
sns.despine()
fig.set_tight_layout(True)


# %%

g = sns.lmplot('X', 'Y', data=df_cube, hue='Snake ID', col='Snake ID',
               fit_reg=False, col_wrap=6)
fig, axs = g.fig, g.axes


radii = np.r_[5, 10, 15, 20]
angles = np.r_[0, 30, 60, 90, 120, 150, 180]
angles = np.r_[angles, -angles[1:-1]]

for ax in axs:
    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)
        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.sin(ang)]
                yy = np.r_[0, radius * np.cos(ang)]
                ax.plot(xx, yy, color='gray', lw=_lw)

#ax.plot(x, y, 'o')
#sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
#            scatter_kws={'c': df['Snake']})

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal', adjustable='box')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-5, 15)
sns.despine()
fig.set_tight_layout(True)


# %%

#g = sns.lmplot('Dist. travelled', 'Z', data=df, hue='Snake ID',
#               fit_reg=False)#, col_wrap=6)
#fig, axs = g.fig, g.axes

g = sns.lmplot('Dist. travelled', 'Z', data=df_cube, hue='Snake ID',
               fit_reg=False, legend=False)
fig, ax = g.fig, g.ax

ax.axhline(1, color='gray', lw=1)
ax.set_ylim(0, 8)


# %%

fig, ax = plt.subplots()
sns.swarmplot('Snake', 'Dist. travelled', data=df_2005, ax=ax)
ax.set_ylim(0, 21.1)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
sns.swarmplot('Snake ID', 'Dist. travelled', data=df_cube, ax=ax)
ax.set_ylim(0, 21.1)
sns.despine()
fig.set_tight_layout(True)


# %%

#radii = np.r_[5, 10, 15, 20]
#angles = np.r_[0, 30, 60, 90, 120, 150, 180]
#angles = np.r_[angles, -angles[1:-1]]
#
#for ax in axs:
#    for radius in radii:
#        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
#        ax.add_artist(circ)
#        if radius == radii[-1]:
#            for ang in np.deg2rad(angles):
#                xx = np.r_[0, radius * np.sin(ang)]
#                yy = np.r_[0, radius * np.cos(ang)]
#                ax.plot(xx, yy, color='gray', lw=_lw)
#
##ax.plot(x, y, 'o')
##sns.regplot('X', 'Y', data=df, fit_reg=False, ax=ax,
##            scatter_kws={'c': df['Snake']})
#
#    ax.set_xlabel('X (m)')
#    ax.set_ylabel('Y (m)')
#    ax.axis('equal', adjustable='box')
#    ax.set_xlim(-20, 20)
#    ax.set_ylim(-5, 15)
#sns.despine()
#fig.set_tight_layout(True)


# %%

g = 9.81
lb2kg = 0.453592
w_lb = np.r_[2.5, 5, 10, 25, 35, 45]
m_kg = lb2kg * w_lb
w_N = g * m_kg

#
PE_rest = 50  # J
h_rest = PE_rest / w_N


# %% RESULT Top view of Y-X landing positions (four subplots) - LABELS

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names), l=.55)


# get colors for the last plot
ntrials = 0
for snake_id in [86, 88, 90, 94]:
    ntrials += len(ret_fnames(snake_id))

colors_trial_id = sns.husl_palette(n_colors=ntrials, l=.55)

col = 3
cnt = 0
for snake_id in [86, 88, 90, 94]:

        _label = '{0}, {1}'.format(snake_id, trial_id)
        axs[col].plot(X, Y, c=colors_trial_id[cnt], label=_label)
        cnt += 1

#_leg3 = axs[col].legend(loc='upper right', frameon=True)
_leg_args = dict(fontsize='xx-small', ncol=4, columnspacing=.75,
                 handlelength=1, handletextpad=.2, borderaxespad=0,
                 frameon=True)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)
_leg3 = axs[3].legend(loc='upper right', **_leg_args)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

fig.set_tight_layout(True)

#fig.savefig(FIG.format('results Ro_I_raw - top'),**FIGOPT)


# %% Velocity polar diagram, non-dim and rescaled coordiantes

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


# %% Velocity polar diagram for all snakes on four different axes

fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(6.75, 6.55))
axs = axs.flatten()

_lw = .5
radii = np.r_[2, 4, 6, 8, 10]
angles = np.r_[0, 15, 30, 45, 60, 75, 90]
for ax in axs:

    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)

        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.cos(ang)]
                yy = np.r_[0, radius * np.sin(ang)]
                if ang == 0:
                    lw = .1
                else:
                    lw = 0.05
                ax.plot(xx, -yy, color='gray', lw=_lw)

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names), l=.55)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['dRo_S'].T / 1000
        _label = '{0}'.format(trial_id)
        axs[col].plot(vy, vz, c=colors_trial_id[i], label=_label)

# get colors for the last plot
ntrials = 0
for snake_id in [86, 88, 90, 94]:
    ntrials += len(ret_fnames(snake_id))

colors_trial_id = sns.husl_palette(n_colors=ntrials, l=.55)

col = 3
cnt = 0
for snake_id in [86, 88, 90, 94]:
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['dRo_S'].T / 1000

        _label = '{0}, {1}'.format(snake_id, trial_id)
        axs[col].plot(vy, vz, c=colors_trial_id[cnt], label=_label)
        cnt += 1

_leg_args = dict(fontsize='xx-small', ncol=3, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg_args_3 = dict(fontsize='xx-small', ncol=2, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)
_leg3 = axs[3].legend(loc='upper right', **_leg_args_3)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

plt.setp(axs.flat, aspect=1.0, adjustable='box')
for ax in axs:
    ax.grid(False)
    ax.set_xlim(0, 10.025)
    ax.set_ylim(-10.025, 0)
    ax.set_xticks(np.r_[0:10.1:2])
    ax.set_yticks(np.r_[-10:.1:2])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    sns.despine(ax=ax, top=False, bottom=True)
    ax.tick_params(axis='both', labelsize='x-small')

axs[0].set_xlabel('Forward velocity (m/s)', fontsize='x-small')
axs[0].set_ylabel('Vertical velocity (m/s)', fontsize='x-small')

for ax in axs[1:]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

#axs[2].set_xticklabels([])
#axs[3].set_xticklabels([])
#axs[1].set_yticklabels([])
#axs[3].set_yticklabels([])
fig.set_tight_layout(True)

#fig.savefig(FIG.format('results VPD - dim'), **FIGOPT)


# %%

Ws_dict = dict()
to_non_dict = dict()

sn_arr = []
Ws_arr = []
to_non_arr = []

# performance metric
vy0_hat = []
vy0 = []
y_z175 = []
y_z175_hat = []
GR = []  # glide ratio = (zi - zf) / (yi - yf)


fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(6.75, 6.55))
axs = axs.flatten()

_lw = .5
radii = np.r_[.2, .4, .6, .8, 1, 1.2]
angles = np.r_[0, 15, 30, 45, 60, 75, 90]
for ax in axs:

    for radius in radii:
        circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
        ax.add_artist(circ)

        if radius == radii[-1]:
            for ang in np.deg2rad(angles):
                xx = np.r_[0, radius * np.cos(ang)]
                yy = np.r_[0, radius * np.sin(ang)]
                if ang == 0:
                    lw = .1
                else:
                    lw = 0.05
                ax.plot(xx, -yy, color='gray', lw=_lw)


for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    colors_trial_id = sns.husl_palette(n_colors=len(fn_names), l=.55)

    Ws_dict[snake_id] = []
    to_non_dict[snake_id] = []

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        # mass_kg = float(d['mass']) / 1000
        # mg = mass_kg * grav  # weight (N)
        mg = d['weight']

        # dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        # snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        dA = d['spl_ds'] * d['chord_spl'] / 1000**2
        snake_area = dA.sum(axis=1).mean()
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        sn_arr.append(snake_id)
        Ws_arr.append(Ws)
        to_non_arr.append(to_non_re_vel)
#        Ws_dict[snake_id].append(Ws)
#        to_non_dict[snake_id].append(to_non_re_vel)

        # when the position crosses z = 1.75 m
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        to_non_re_pos = (2 * Ws) / (rho * 9.81)
        idx_z175 = np.where(Z > 1.75)[0]
        if len(idx_z175) == 0:
            y_z175.append(np.nan)
            y_z175_hat.append(np.nan)
        else:
            Y_z175 = Y[idx_z175[-1]]
            y_z175.append(Y_z175)
            y_z175_hat.append(Y_z175 / to_non_re_pos)
        gr = (Y[0] - Y[-1]) / (Z[0] - Z[-1])  # dy / dz
        GR.append(np.abs(gr))

        # initial jump velocity
        vy0_hat.append(vy[0])
        vy0.append(v[1][0])

        _label = '{0}'.format(trial_id)
        axs[col].plot(vy, vz, c=colors_trial_id[i], label=_label)

# get colors for the last plot
ntrials = 0
for snake_id in [86, 88, 90, 94]:
    ntrials += len(ret_fnames(snake_id))

colors_trial_id = sns.husl_palette(n_colors=ntrials, l=.55)

col = 3
cnt = 0
for snake_id in [86, 88, 90, 94]:
    fn_names = ret_fnames(snake_id)

    Ws_dict[snake_id] = []
    to_non_dict[snake_id] = []

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        # mass_kg = float(d['mass']) / 1000
        # mg = mass_kg * grav  # weight (N)
        mg = d['weight']

        # chord_len_m = d['chord_spl'][0] / 1000
        # dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        # snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        dA = d['spl_ds'] * d['chord_spl'] / 1000**2
        snake_area = dA.sum(axis=1).mean()
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        sn_arr.append(snake_id)
        Ws_arr.append(Ws)
        to_non_arr.append(to_non_re_vel)

        # when the position crosses z = 1.75 m
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        Z = d['Ro_S'][:, 2] / 1000
        to_non_re_pos = (2 * Ws) / (rho * 9.81)
        idx_z175 = np.where(Z > 1.75)[0]
        if len(idx_z175) == 0:
            y_z175.append(np.nan)
            y_z175_hat.append(np.nan)
        else:
            Y_z175 = Y[idx_z175[-1]]
            y_z175.append(Y_z175)
            y_z175_hat.append(Y_z175 / to_non_re_pos)
        gr = (Y[0] - Y[-1]) / (Z[0] - Z[-1])  # dy / dz
        GR.append(np.abs(gr))

        # initial jump velocity
        vy0_hat.append(vy[0])
        vy0.append(v[1][0])

#        Ws_dict[snake_id].append(Ws)
#        to_non_dict[snake_id].append(to_non_re_vel)

        _label = '{0}, {1}'.format(snake_id, trial_id)
        axs[col].plot(vy, vz, c=colors_trial_id[cnt], label=_label)
        cnt += 1

_leg_args = dict(fontsize='xx-small', ncol=3, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg_args_3 = dict(fontsize='xx-small', ncol=2, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)
_leg3 = axs[3].legend(loc='upper right', **_leg_args_3)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

plt.setp(axs.flat, aspect=1.0, adjustable='box')
for ax in axs:
#    ax.set_xlim(0, 1.25)
#    ax.set_ylim(-1.25, 0)
    ax.xaxis.set_major_formatter(decimal_formatter)
    ax.yaxis.set_major_formatter(decimal_formatter)
    ax.set_xlim(0, 1.11)
    ax.set_ylim(-1.11, 0)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.set_xticks([0, .2, .4, .6, .8, 1, 1.2])
    ax.set_yticks([0, -.2, -.4, -.6, -.8, -1, -1.2])
    sns.despine(ax=ax, top=False, bottom=True)
    ax.tick_params(axis='both', labelsize='x-small')

axs[0].set_xlabel('Forward velocity', fontsize='x-small')
axs[0].set_ylabel('Vertical velocity', fontsize='x-small')

for ax in axs[1:]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.set_tight_layout(True)

#fig.savefig(FIG.format('results VPD - non-dim'), **FIGOPT)


# %% WING LOADING AND SCALING PARAMETER DISTRIBUTIONS

import pandas as pd

data = {'ids': sn_arr, 'Ws': Ws_arr, 'to_non': to_non_arr}
df = pd.DataFrame(data=data)


fig, ax = plt.subplots(figsize=(6, 5))

ax.axhline(29, color='gray', lw=1)
ax.axhspan(29 - 9, 29 + 9, color='gray', alpha=.2)
ax.axhspan(12, 46, color='gray', alpha=.1)

sns.swarmplot(x='ids', y='Ws', data=df, ax=ax, marker='o')

ax.set_ylim(ymin=0)
ax.set_xlabel('Snake ID')
ax.set_ylabel(r'Wing loading (N/m$^\mathrm{2}$)')
sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('Ws'), **FIGOPT)


fig, ax = plt.subplots(figsize=(6, 5))

def to_non_pref(ws):
    return np.sqrt(2 * ws / rho)

ax.axhline(to_non_pref(29), color='gray', lw=1)
ax.axhspan(to_non_pref(29 - 9), to_non_pref(29 + 9), color='gray', alpha=.2)
ax.axhspan(to_non_pref(12), to_non_pref(46), color='gray', alpha=.1)

sns.swarmplot(x='ids', y='to_non', data=df, ax=ax, marker='o')

ax.set_ylim(ymin=0)
ax.set_xlabel('Snake ID')
ax.set_ylabel('Scaling parameter (m/s)')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('to_non'), **FIGOPT)


# %% PERFORMANCE: LANDING DISTANCE VS. VY0_HAT

data = {'Snake ID': sn_arr, 'y_z175': y_z175, 'y_z175_hat': y_z175_hat,
        'vy0': vy0, 'vy0_hat': vy0_hat, 'GR': GR}
df = pd.DataFrame(data=data)

dfp = df[df['Snake ID'] > 40]


g = sns.lmplot(x='vy0_hat', y='y_z175', data=df, hue='Snake ID', fit_reg=False,
               scatter_kws={'s': 40}, legend_out=True)
fig, ax = g.fig, g.ax
sns.regplot(x='vy0_hat', y='y_z175', data=df, scatter=False, ax=ax, label=True,
            color='k')
#sns.regplot(x='vy0_hat', y='y_z175', data=dfp, scatter=False, ax=ax, label=True,
#            color='b')
ax.set_xlim(xmin=0)
ax.set_ylim(0, 4.25)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xticks([0, .1, .2, .3])
ax.set_xlabel('Initial horizontal velocity (-)')
ax.set_ylabel('Glide distance (m) at z=1.75m')

fig.savefig(FIG.format('y_z175 vs vy0_hat'), **FIGOPT)
#fig.savefig(FIG.format('y_z175 vs vy0_hat no reg'), **FIGOPT)


# %% PERFORMANCE: GLIDE RATIO VS. VY0_HAT

df_para = df[df['Snake ID'] > 70]
df_para = df

g = sns.lmplot(x='vy0_hat', y='GR', data=df_para, hue='Snake ID', fit_reg=False,
               scatter_kws={'s': 40}, legend_out=True)
fig, ax = g.fig, g.ax
sns.regplot(x='vy0_hat', y='GR', data=df_para, scatter=False, ax=ax, label=True,
            color='k')
ax.set_xlim(xmin=0)
ax.set_ylim(0, .8)
#ax.set_ylim(0, 4.25)
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xticks([0, .1, .2, .3])
#ax.set_xlabel('Initial horizontal velocity (-)')
#ax.set_ylabel('Glide distance (m) at z=1.75m')

#fig.savefig(FIG.format('GR vs vy0_hat'), **FIGOPT)
#fig.savefig(FIG.format('GR vs vy0_hat no reg'), **FIGOPT)


# %% Linear regression for performance vs launch velocity

from scipy.stats import linregress

m, b, rvalue, pvalue, stderr = linregress(df['vy0_hat'], df['y_z175'])


# %% Velocity polar diagram for all snakes on four different axes

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9.8, 11))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['dRo_S'].T / 1000
        _label = '{0}'.format(trial_id)
        axs[col].plot(vy, vz, c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['dRo_S'].T / 1000

        if i > 0:
            axs[col].plot(vy, vz, c=snake_colors[snake_id])
        else:
            axs[col].plot(vy, vz, c=snake_colors[snake_id], label=snake_id)

_leg3 = axs[col].legend(loc='upper right', frameon=True)
_leg_args = dict(fontsize='x-small', ncol=4, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg0 = axs[0].legend(loc='upper right', **_leg_args)
_leg1 = axs[1].legend(loc='upper right', **_leg_args)
_leg2 = axs[2].legend(loc='upper right', **_leg_args)

for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)

plt.setp(axs.flat, aspect=1.0, adjustable='box')
for ax in axs:
    ax.grid(True)
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 0)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
#    ax.set_xticks([0, 2, 4, 6, 8])
    sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)

#fig.savefig(FIG.format('VPD'), **FIGOPT)


# %% Glide angles - GLIDE vs. Z

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
#                        figsize=(12, 10.7))
axs = axs.flatten()

for ax in axs:
    ax.axhline(57, color='gray', lw=1)
    ax.axhspan(52, 62, color='gray', alpha=.1)
    ax.axhspan(54, 60, color='gray', alpha=.2)

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        axs[col].plot(Z, gamma, c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        if i > 0:
            axs[col].plot(Z, gamma,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(Z, gamma,
                          c=snake_colors[snake_id], label=snake_id)
_leg3 = axs[col].legend(loc='lower right', frameon=True,
                        fontsize='x-small', ncol=4, columnspacing=.75, handlelength=1,
                        handletextpad=.2)
_leg_args = dict(fontsize='x-small', ncol=4, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=True)
_leg0 = axs[0].legend(loc='lower right', **_leg_args)
_leg1 = axs[1].legend(loc='lower right', **_leg_args)
_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(8.5, 0)
axs[0].set_ylim(-10, 90)
axs[0].set_yticks([0, 30, 60, 90])
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('glide_angle_vs_Z'), **FIGOPT)


# %% Yaw angles - LABELS vs. Z_S

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(10, 8))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
#        yaw = np.rad2deg(d['mus']).cumsum()
        yaw -=yaw[0]
#        yaw -= yaw[0]
        axs[col].plot(Z, yaw,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
#        yaw = np.rad2deg(d['mus']).cumsum()
        yaw -=yaw[0]
#        yaw -= yaw[0]
        if i > 0:
            axs[col].plot(Z, yaw,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(Z, yaw,
                          c=snake_colors[snake_id], label=snake_id)
#_leg3 = axs[col].legend(loc='lower right', frameon=True)
#_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
#                 handletextpad=.2, borderaxespad=0, frameon=True)
#_leg0 = axs[0].legend(loc='lower right', **_leg_args)
#_leg1 = axs[1].legend(loc='lower right', **_leg_args)
#_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(8.5, 0)
#axs[0].set_ylim(-40, 40)
axs[0].set_ylim(-60, 60)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('yaw_angle_vs_Z'), **FIGOPT)


# %% Yaw angles - LABELS vs. Y_S

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        axs[col].plot(Y, yaw,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        if i > 0:
            axs[col].plot(Y, yaw,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, yaw,
                          c=snake_colors[snake_id], label=snake_id)
_leg3 = axs[col].legend(loc='lower right', frameon=True)
_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
                 handletextpad=.2, borderaxespad=0, frameon=True)
_leg0 = axs[0].legend(loc='lower right', **_leg_args)
_leg1 = axs[1].legend(loc='lower right', **_leg_args)
_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(-.2, 5.5)
axs[0].set_ylim(-40, 40)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% HO ANGULAR MOMENTUM AND YAW ANGLES




# %%


#fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
#                        figsize=(12, 10.7))
#axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))

#    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#    ax2.set_title(snake_id)
#    sns.despine()
#    fig1.set_tight_layout(True)
#
#    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
#    ax4.set_title(snake_id)
#    sns.despine()
#    fig2.set_tight_layout(True)
#
#    fig3, ax = plt.subplots()
#    ax.axhline(0, color='gray', lw=1)
#    ax.axvline(0, color='gray', lw=1)
#    sns.despine()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False,
                                        figsize=(6, 12))
    sns.despine()
    fig.set_tight_layout(True)
    ax1.set_title(snake_id)
#
#    fig, ax = plt.subplots()
#    ax.set_title(snake_id)
#    sns.despine()
#    fig.set_tight_layout(True)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        dyaw = np.rad2deg(d['dyaw'])
#        yaw -= yaw[0]

#        R, dR, ddR = d['R_Ic'], d['dR_Ic'], d['ddR_Ic']
        R, dR, ddR = d['R_Sc'], d['dR_Sc'], d['ddR_Sc']
#        R, dR, ddR = d['R_B'], d['dR_Bc'], d['ddR_Bc']
        R, dR, ddR = R / 1000, dR / 1000, ddR / 1000
        mass_spl, times = d['mass_spl'], d['times'] - d['times'][0]
        ntime = len(times)
        ho = np.zeros((ntime, 3))
        dho = np.zeros((ntime, 3))
        L = np.zeros((ntime, 3))
        for k in np.arange(ntime):
            ho_k = np.cross(R[k], (mass_spl[k] * dR[k].T).T).sum(axis=0)
            ho[k] = ho_k

            dho_k = np.cross(R[k], (mass_spl[k] * ddR[k].T).T).sum(axis=0)
            dho[k] = dho_k

            L_k = (mass_spl[k] * dR[k].T).T.sum(axis=0)
            L[k] = L_k

        KE_k = .5 * mass_spl * np.sum(dR**2, axis=2)
        KE = KE_k.sum(axis=1)

        KE_zi = .5 * mass_spl * dR[:, :, 2]**2
        KE_z = KE_zi.sum(axis=1)
#
#        L_i = (mass_spl.T * dR_Sc.T).T
#        L = L_i.sum(axis=1)

#        times2D = d['times2D'] - d['times2D'][0]
#        tc = d['t_coord'] / d['SVL']
#        fig, ax1 = plt.subplots()
#        cax = ax1.pcolormesh(times2D, tc, KE_k, cmap=plt.cm.viridis,
#                             vmin=0)
#        cbar = fig.colorbar(cax, ax=ax1, orientation='vertical', shrink=.5)
#        ax1.set_title('Snake {0}, trial {1}'.format(snake_id, trial_id))
#        ax1.set_ylim(0, tc.max())
#        ax1.set_xlim(0, times2D.max())
#        sns.despine(ax=ax1)
#        fig.set_tight_layout(True)

#        times2D = d['times2D'] - d['times2D'][0]
#        tc = d['t_coord'] / d['SVL']
#        fig, ax1 = plt.subplots()
#        cax = ax1.pcolormesh(times2D, tc, dR[:, :, 0], cmap=plt.cm.coolwarm)
#        cbar = fig.colorbar(cax, ax=ax1, orientation='vertical', shrink=.5)
#        ax1.set_title('Snake {0}, trial {1}'.format(snake_id, trial_id))
#        ax1.set_ylim(0, tc.max())
#        ax1.set_xlim(0, times2D.max())
#        sns.despine(ax=ax1)
#        fig.set_tight_layout(True)

#        ax1.plot(times, ho[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(times, ho[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(times, ho[:, 2], c=colors_trial_id[i], label=_label)

        ax1.plot(Z, ho[:, 0], c=colors_trial_id[i], label=_label)
        ax2.plot(Z, ho[:, 1], c=colors_trial_id[i], label=_label)
        ax3.plot(Z, ho[:, 2], c=colors_trial_id[i], label=_label)
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax1.set_xlim(8.5, 0)
        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax1.plot(Z, ho[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(Z, ho[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(Z, ho[:, 2], c=colors_trial_id[i], label=_label)
#        ax1.set_xlim(-.25, 5.5)
#        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax.plot(Z, KE, c=colors_trial_id[i], label=_label)
#        ax.set_xlim(8.5, 0)
#        ax.plot(times, KE, c=colors_trial_id[i], label=_label)
#        ax.plot(Z, KE_z, c=colors_trial_id[i], label=_label)
#        ax.set_xlim(8.5, 0)
#        ax.legend(loc='best', fontsize='xx-small', ncol=5)
#
#        ax1.plot(Z, L[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(Z, L[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(Z, L[:, 2], c=colors_trial_id[i], label=_label)
#        ax1.grid(True)
#        ax2.grid(True)
#        ax3.grid(True)
#        ax1.set_xlim(8.5, 0)
#        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax1.plot(times, yaw, c=colors_trial_id[i], label=_label)
#        ax2.plot(times, ho[:, 2], c=colors_trial_id[i], label=_label)
#
#        ax3.plot(times, dyaw, c=colors_trial_id[i], label=_label)
#        ax4.plot(times, dho[:, 2], c=colors_trial_id[i], label=_label)
#
#        yaw_mean = np.mean(yaw - yaw[0])
#        ho_mean = np.mean(ho - ho[0], axis=0)
#
#        ax.plot(yaw_mean, ho_mean[2], 'o')
#
#        ax.plot(yaw, ho[:, 2], c=colors_trial_id[i], lw=2)


# %%

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        if i > 0:
            axs[col].plot(Y, yaw,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, yaw,
                          c=snake_colors[snake_id], label=snake_id)
_leg3 = axs[col].legend(loc='lower right', frameon=True)
_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
                 handletextpad=.2, borderaxespad=0, frameon=True)
_leg0 = axs[0].legend(loc='lower right', **_leg_args)
_leg1 = axs[1].legend(loc='lower right', **_leg_args)
_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(-.2, 5.5)
axs[0].set_ylim(-40, 40)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% Yaw angles - LABELS vs. dRo_S_Z

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        dX, dY, dZ = d['dRo_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
#        yaw -= yaw[0]
        axs[col].plot(dZ, yaw,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        dX, dY, dZ = d['dRo_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
#        yaw -= yaw[0]
        if i > 0:
            axs[col].plot(dZ, yaw,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(dZ, yaw,
                          c=snake_colors[snake_id], label=snake_id)
#_leg3 = axs[col].legend(loc='lower right', frameon=True)
#_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
#                 handletextpad=.2, borderaxespad=0, frameon=True)
#_leg0 = axs[0].legend(loc='lower right', **_leg_args)
#_leg1 = axs[1].legend(loc='lower right', **_leg_args)
#_leg2 = axs[2].legend(loc='lower right', **_leg_args)

#axs[0].set_xlim(8.5, 0)
axs[0].set_ylim(-40, 40)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% Yaw angles - YAW vs. GLIDE ANGLE

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        axs[col].plot(gamma, yaw,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        if i > 0:
            axs[col].plot(gamma, yaw,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(gamma, yaw,
                          c=snake_colors[snake_id], label=snake_id)
#_leg3 = axs[col].legend(loc='lower right', frameon=True)
#_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
#                 handletextpad=.2, borderaxespad=0, frameon=True)
#_leg0 = axs[0].legend(loc='lower right', **_leg_args)
#_leg1 = axs[1].legend(loc='lower right', **_leg_args)
#_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(90, 0)
axs[0].set_ylim(-40, 40)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% Glide angles - GLIDE vs. Y

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        axs[col].plot(Y, gamma,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        if i > 0:
            axs[col].plot(Y, gamma,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, gamma,
                          c=snake_colors[snake_id], label=snake_id)
#_leg3 = axs[col].legend(loc='lower right', frameon=True)
#_leg_args = dict(fontsize='x-small', ncol=1, columnspacing=.75, handlelength=1,
#                 handletextpad=.2, borderaxespad=0, frameon=True)
#_leg0 = axs[0].legend(loc='lower right', **_leg_args)
#_leg1 = axs[1].legend(loc='lower right', **_leg_args)
#_leg2 = axs[2].legend(loc='lower right', **_leg_args)

axs[0].set_xlim(-.5, 5.5)
axs[0].set_ylim(-10, 90)
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% Glide angles - GLIDE vs. TIME

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        axs[col].plot(d['times'], gamma,
                      c=colors_trial_id[i], label=_label)

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
        yaw = np.rad2deg(d['yaw'])
        yaw -= yaw[0]
        gamma = np.rad2deg(d['gamma'])
        if i > 0:
            axs[col].plot(d['times'], gamma,
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(d['times'], gamma,
                          c=snake_colors[snake_id], label=snake_id)
_leg3 = axs[col].legend(loc='lower right', frameon=True)
_leg_args = dict(fontsize='x-small', ncol=3, columnspacing=.75, handlelength=1,
                 handletextpad=.2, borderaxespad=0, frameon=True)
_leg0 = axs[0].legend(loc='lower right', **_leg_args)
_leg1 = axs[1].legend(loc='lower right', **_leg_args)
_leg2 = axs[2].legend(loc='lower right', **_leg_args)

#axs[0].set_xlim(8.5, 0)
axs[0].set_ylim(-10, 90)
axs[0].set_yticks([0, 30, 60, 90])
for leg in [_leg0, _leg1, _leg2, _leg3]:
    leg.get_frame().set_linewidth(0)
sns.despine()
fig.set_tight_layout(True)


# %% Fa_Z vs. Y

figsize = (8, 10.25)
figsize = (6, 10)
#figsize=None

grav = 9.81

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)
axs = axs.flatten()

for ax in axs:
    ax.axhline(1, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['Ro_S'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

#        Fa_I = d['Fa_I']
#        Fa_S = np.zeros_like(Fa_I)
#        C_I2S = d['C_I2S']
#        for ii in np.arange(d['ntime']):
#            Fa_S[ii] = np.dot(C_I2S[ii], Fa_I[ii].T).T
        Fa_S = d['Fa_S']
        Faero_Z = Fa_S[:, :, 2].sum(axis=1) / mg

#        idx_cross = np.where(Faero_Z > 1)[0]
#        if len(idx_cross) > 0:
#            idx_cross = idx_cross[0]
#            axs[col].axvline(Y[idx_cross], color='gray', lw=1)
        _label = '{0}'.format(trial_id)
        axs[col].plot(Y, Faero_Z, c=colors_trial_id[i], label=_label)

#_leg3 = axs[col].legend(loc='lower right', frameon=True)
#_leg_args = dict(fontsize='xx-small', ncol=5, columnspacing=.75, handlelength=1,
#                 handletextpad=.2, borderaxespad=0, frameon=True)
#_leg0 = axs[0].legend(loc='upper left', **_leg_args)
#_leg1 = axs[1].legend(loc='upper left', **_leg_args)
#_leg2 = axs[2].legend(loc='upper left', **_leg_args)

for ax in axs:
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(0, 2)
#    ax.set_ylim(0, 1.6)
    ax.set_xlim(xmin=-.2)
    ax.set_yticks([0, .5, 1, 1.5, 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Faero_vs_Y_3sub'), **FIGOPT)


# %% Fa_Z vs. Z

figsize = (9.8, 10)

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)

for ax in axs:
    ax.axhline(1, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Z = d['Ro_S'][:, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

#        Fa_I = d['Fa_I']
#        Fa_S = np.zeros_like(Fa_I)
#        C_I2S = d['C_I2S']
#        for ii in np.arange(d['ntime']):
#            Fa_S[ii] = np.dot(C_I2S[ii], Fa_I[ii].T).T
        Fa_S = d['Fa_S']
        Faero_Z = Fa_S[:, :, 2].sum(axis=1) / mg

#        Faero_Z = d['Faero'][:, :, 2].sum(axis=1) / mg
        _label = '{0}'.format(trial_id)
        axs[col].plot(Z, Faero_Z, c=colors_trial_id[i], label=_label)

_leg3 = axs[col].legend(loc='lower right', frameon=True)
_leg_args = dict(fontsize='small', ncol=5, columnspacing=.75, handlelength=1,
                 handletextpad=.2, frameon=False)
_leg0 = axs[0].legend(loc='upper left', **_leg_args)
_leg1 = axs[1].legend(loc='upper left', **_leg_args)
_leg2 = axs[2].legend(loc='upper left', **_leg_args)

#ax.legend(loc='best', ncol=2)
for ax in axs:
    ax.invert_xaxis()
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(0, 2)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([0, .5, 1, 1.5, 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Faero_vs_Z_3sub'), **FIGOPT)


# %%

import pandas as pd

from scipy.interpolate import interp1d


# %% Fa_Z vs. Z

figsize = (5.5, 7)

figsize = (6., 6)

ZZ = np.linspace(8.5, 0, 200)
Fa_Z_avg = np.zeros((len(ZZ), 3))

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)

for ax in axs:
    ax.axhline(1, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = bmap[col]

#    dfs = []
    data = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Z = d['Ro_S'][:, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

        Fa_S = d['Fa_S']
        Faero_Z = Fa_S[:, :, 2].sum(axis=1) / mg

        Faero_Z_interp = interp1d(Z, Faero_Z, bounds_error=False, fill_value=np.nan)(ZZ)

        # store data for mean
        # dfs.append(pd.DataFrame({'Z': Z, 'Fa': Faero_Z}))
#        dfs.append(pd.DataFrame({'Fa': Faero_Z_interp}, index=ZZ))
        data[i] = Faero_Z_interp

        axs[col].plot(Z, Faero_Z, lw=1.5, c=colors_trial_id)


    # mean force calculations
    df = pd.DataFrame(data=data, index=ZZ)
    df_mean = df.mean(axis=1)
    Fa_Z_avg[:, col] = df_mean

    axs[col].plot(ZZ, df_mean, c='k', lw=2)

for ax in axs:
    ax.invert_xaxis()
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(0, 2)
    ax.set_xlim(8.5, 0)
#    ax.set_yticks([0, .5, 1, 1.5, 2])
    ax.set_yticks([0, 1, 2])
    # ax.set_yticklabels([0, '', 1, '', 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Faero_vs_Z_3sub_SICB2017'), **FIGOPT)

# concat to find the average for each snake
# df = pd.concat(dfs)


# %%

fig, ax = plt.subplots()
ax.axhline(1, color='gray', linestyle='--')
ax.plot(ZZ, Fa_Z_avg)
ax.invert_xaxis()
ax.set_ylim(0, 2)
ax.set_xlim(8.5, 0)
sns.despine()
fig.set_tight_layout(True)


# %%

# %% Fa_Y vs. Z

figsize = (5.5, 7)

figsize = (6., 6)

ZZ = np.linspace(8.5, 0, 200)
Fa_Y_avg = np.zeros((len(ZZ), 3))

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = bmap[col]

#    dfs = []
    data = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Z = d['Ro_S'][:, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

        Fa_S = d['Fa_S']
        # Fa_S = d['Fl_S']
        # Fa_S = d['Fd_S']
        Faero_Y = Fa_S[:, :, 1].sum(axis=1) / mg

        Faero_Y_interp = interp1d(Z, Faero_Y, bounds_error=False, fill_value=np.nan)(ZZ)

        # store data for mean
        # dfs.append(pd.DataFrame({'Z': Z, 'Fa': Faero_Z}))
#        dfs.append(pd.DataFrame({'Fa': Faero_Z_interp}, index=ZZ))
        data[i] = Faero_Y_interp

        axs[col].plot(Z, Faero_Y, lw=1.5, c=colors_trial_id)


    # mean force calculations
    df = pd.DataFrame(data=data, index=ZZ)
    df_mean = df.mean(axis=1)
    Fa_Y_avg[:, col] = df_mean

    axs[col].plot(ZZ, df_mean, c='k', lw=2)

for ax in axs:
    ax.invert_xaxis()
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(-.1, .3)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([-.1, 0, .1, .2, .3])
#    ax.set_yticks([-1, 0, 1])
    # ax.set_yticklabels([0, '', 1, '', 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('Faero_Y_vs_Z_3sub_SICB2017'), **FIGOPT)

# concat to find the average for each snake
# df = pd.concat(dfs)


# %% Fa_Y vs. Z

figsize = (5.5, 7)

figsize = (6., 6)

ZZ = np.linspace(8.5, 0, 200)
Fa_Y_avg = np.zeros((len(ZZ), 3))

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = bmap[col]

#    dfs = []
    data = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Z = d['Ro_S'][:, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav


        ddRo_Z_S = d['ddRo_S'][:, 2] / 1000
        Fa_Z_S = d['Fa_S'][:, :, 2].sum(axis=1)
        # Fg = np.zeros_like(Fa_Z_S)
        # Fg[:, 2] = mg

        F = (Fa_Z_S - mg) / mg
        ma = (mass_kg * ddRo_Z_S) / mg

        # Fa_S = d['Fl_S']
        # Fa_S = d['Fd_S']
#        Faero_Y = Fa_S[:, :, 1].sum(axis=1) / mg
#
#        Faero_Y_interp = interp1d(Z, Faero_Y, bounds_error=False, fill_value=np.nan)(ZZ)

        # store data for mean
        # dfs.append(pd.DataFrame({'Z': Z, 'Fa': Faero_Z}))
#        dfs.append(pd.DataFrame({'Fa': Faero_Z_interp}, index=ZZ))
#        data[i] = Faero_Y_interp

        start = 6

#         axs[col].plot(Z[start:], F[start:], lw=1.5, c=colors_trial_id)
#         axs[col].plot(Z[start:], ma[start:], lw=1.5, c=bmap[col + 1])
        axs[col].plot(Z[start:], ma[start:], lw=1.5, c=bmap[col])


#    # mean force calculations
#    df = pd.DataFrame(data=data, index=ZZ)
#    df_mean = df.mean(axis=1)
#    Fa_Y_avg[:, col] = df_mean
#
#    axs[col].plot(ZZ, df_mean, c='k', lw=2)

for ax in axs:
    ax.invert_xaxis()
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([-1, 0, 1])
#    ax.set_yticks([-1, 0, 1])
    # ax.set_yticklabels([0, '', 1, '', 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('MA_z_vs_Z_3sub_SICB2017'), **FIGOPT)

# concat to find the average for each snake
# df = pd.concat(dfs)


# %% Fa_Y vs. Z

figsize = (5.5, 7)

figsize = (6., 6)

ZZ = np.linspace(8.5, 0, 200)
Fa_Y_avg = np.zeros((len(ZZ), 3))

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = bmap[col]

#    dfs = []
    data = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Z = d['Ro_S'][:, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav


        ddRo_Y_S = d['ddRo_S'][:, 1] / 1000
        Fa_Y_S = d['Fa_S'][:, :, 1].sum(axis=1)
        # Fg = np.zeros_like(Fa_Z_S)
        # Fg[:, 2] = mg

        F = Fa_Y_S / mg
        ma = (mass_kg * ddRo_Y_S) / mg

        # Fa_S = d['Fl_S']
        # Fa_S = d['Fd_S']
#        Faero_Y = Fa_S[:, :, 1].sum(axis=1) / mg
#
#        Faero_Y_interp = interp1d(Z, Faero_Y, bounds_error=False, fill_value=np.nan)(ZZ)

        # store data for mean
        # dfs.append(pd.DataFrame({'Z': Z, 'Fa': Faero_Z}))
#        dfs.append(pd.DataFrame({'Fa': Faero_Z_interp}, index=ZZ))
#        data[i] = Faero_Y_interp

        start = 6

#        axs[col].plot(Z[start:], F[start:], lw=1.5, c=colors_trial_id)
#        axs[col].plot(Z[start:], ma[start:], lw=1.5, c=bmap[col + 1])
        axs[col].plot(Z[start:], ma[start:], lw=1.5, c=bmap[col])
#        axs[col].plot(Z[start:], ma[start:], lw=1.5, c=bmap[col + 1])


#    # mean force calculations
#    df = pd.DataFrame(data=data, index=ZZ)
#    df_mean = df.mean(axis=1)
#    Fa_Y_avg[:, col] = df_mean
#
#    axs[col].plot(ZZ, df_mean, c='k', lw=2)

for ax in axs:
    ax.invert_xaxis()
#    ax.set_ylim(0, 1.8)
    ax.set_ylim(-.3, 1)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([0, .5, 1])
#    ax.set_yticks([-1, 0, 1])
    # ax.set_yticklabels([0, '', 1, '', 2])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('MA_Y_vs_Z_3sub_SICB2017'), **FIGOPT)

# concat to find the average for each snake
# df = pd.concat(dfs)


# %%

fname = ret_fnames(95, 618)[0]
d = np.load(fname)
dt = float(d['dt'])
Ro_S = d['Ro_S']
dRo_S = d['dRo_S']
ddRo_S = d['ddRo_S']


# %%

dRo_S_grad = np.gradient(Ro_S, dt, edge_order=1, axis=0)
ddRo_S_grad = np.gradient(dRo_S_grad, dt, edge_order=1, axis=0)

figure()
plot(ddRo_S[:, 2], 'o-')
plot(ddRo_S_grad[:, 2], 'o-')



figure()
plot(dRo_S[:, 2])
plot(dRo_S_grad[:, 2])


figure()
plot(Ro_S[:, 2], 'o-')


# %% Pitch angles

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(12, 10.7))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        axs[col].plot(d['times'], np.rad2deg(d['pitch']), c=snake_colors[snake_id])

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        if i > 0:
            axs[col].plot(d['times'], np.rad2deg(d['pitch']),
                          c=snake_colors[snake_id])
        else:
            axs[col].plot(d['times'], np.rad2deg(d['pitch']),
                          c=snake_colors[snake_id], label=snake_id)
axs[col].legend(loc='upper left', frameon=True)

#plt.setp(axs, aspect=1.0, adjustable='box')
#ax.set_xlim(-3.1, 3.1)
#ax.set_xlim(-3, 2)
#ax.set_ylim(0, 5.5)
#ax.set_yticks([])
sns.despine()
fig.set_tight_layout(True)


# %% Velocity polar diagram, non-dim and rescaled coordiantes

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9.8, 11))
axs = axs.flatten()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav  # weight (N)

        chord_len_m = d['chord_spl'][0] / 1000
        dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        axs[col].plot(vy, vz, c=snake_colors[snake_id])

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav  # weight (N)

        chord_len_m = d['chord_spl'][0] / 1000
        dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        if i > 0:
            axs[col].plot(vy, vz, c=snake_colors[snake_id])
        else:
            axs[col].plot(vy, vz, c=snake_colors[snake_id], label=snake_id)
axs[col].legend(loc='upper right', ncol=2)

plt.setp(axs.flat, aspect=1.0, adjustable='box')
for ax in axs:
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-1.25, 0)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid(True)
#    ax.set_xticks([0, 2, 4, 6, 8])
    sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('3b_VPD_non_rescaled.pdf'), bbox_inches='tight')


# %% VPD, non-dim and resacled, all on top of each other

alpha = 1#.75

fig, ax = plt.subplots()

for col, snake_id in enumerate([81, 91, 95, 90, 88]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav  # weight (N)

        chord_len_m = d['chord_spl'][0] / 1000
        dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area
#        print snake_area_cm

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        # ax.plot(vy[-1], vz[-1], 'o', c=snake_colors[snake_id])

        if i > 0:
            ax.plot(vy, vz, c=snake_colors[snake_id], alpha=alpha)
        else:
            label = r'{0} ({1:.1f} N/m$^2$)'.format(snake_id, Ws)
            ax.plot(vy, vz, c=snake_colors[snake_id], label=label,
                    alpha=alpha)

ax.legend(loc='upper right', ncol=1, frameon=True)
ax.grid(True)
plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(0, 1.255)
ax.set_ylim(-1.255, 0)
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([-1.25, -1, -.75, -.5, -.25, 0])
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('3c_VPD_on_top.pdf'), bbox_inches='tight')


# %% VPD, non-dim and resacled, all on top of each other

alpha = .25

fig, ax = plt.subplots()

for col, snake_id in enumerate([81, 91, 95, 90, 88]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav  # weight (N)

        chord_len_m = d['chord_spl'][0] / 1000
        dt_coord_m = np.gradient(d['t_coord'][0], edge_order=2) / 1000
        snake_area = (chord_len_m * dt_coord_m).sum()  # m^2
        snake_area_cm = snake_area * 100**2

        Ws = mg / snake_area

        to_non_re_vel = np.sqrt(2 * Ws / rho)
        v = d['dRo_S'].T / 1000
        v_non_re = v / to_non_re_vel
        vx, vy, vz = v_non_re

        ax.plot(vy[-1], vz[-1], 'o', c=snake_colors[snake_id])

        if i > 0:
            ax.plot(vy, vz, c=snake_colors[snake_id], alpha=alpha)
        else:
            ax.plot(vy, vz, c=snake_colors[snake_id], label=snake_id,
                    alpha=alpha)

ax.legend(loc='upper right', ncol=2, frameon=True)
ax.grid(True)
plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(0, 1.255)
ax.set_ylim(-1.255, 0)
ax.set_xticks([0, .25, .5, .75, 1, 1.25])
ax.set_yticks([-1.25, -1, -.75, -.5, -.25, 0])
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('3d_VPD_on_top.pdf'), bbox_inches='tight')


# %% Vertical aerodynamic force (2 x 2 subplots)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11.3, 10.25))
axs = axs.flatten()

for ax in axs:
    ax.axhline(1, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav
        Faero_Z = d['Faero'][:, :, 2].sum(axis=1) / mg

#        idx_cross = np.where(Faero_Z > 1)[0]
#        if len(idx_cross) > 0:
#            idx_cross = idx_cross[0]
#            axs[col].axvline(Y[idx_cross], color='gray', lw=1)
        axs[col].plot(Y, Faero_Z, c=snake_colors[snake_id])

col = 3
for snake_id in snake_ids:
    if snake_id in [81, 91, 95]:
        continue
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav
        Faero_Z = d['Faero'][:, :, 2].sum(axis=1) / mg

        axs[col].plot(Y, Faero_Z, c=snake_colors[snake_id])
        if i > 0:
            axs[col].plot(Y, Faero_Z, c=snake_colors[snake_id])
        else:
            axs[col].plot(Y, Faero_Z, c=snake_colors[snake_id], label=snake_id)
axs[col].legend(loc='lower right', ncol=2)

for ax in axs:
    ax.set_ylim(0, 1.7)
    ax.set_xlim(xmin=-.2)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('4b_Faero_Z_2x2_sub.pdf'), bbox_inches='tight')


# %% Horizontal force in the direction of motion

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 10.25))

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

        Faero_Y = d['Faero'][:, :, 1].sum(axis=1) / mg

        axs[col].plot(Y, Faero_Y, c=snake_colors[snake_id])

for ax in axs:
    ax.set_ylim(-.25, .25)
    ax.set_xlim(xmin=-.2)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('6_Faero_Y_3sub.pdf'), bbox_inches='tight')


# %% Lateral force

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 10.25))

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

        Faero_X = d['Faero'][:, :, 0].sum(axis=1) / mg

        axs[col].plot(Y, Faero_X, c=snake_colors[snake_id])

for ax in axs:
    ax.set_ylim(-.4, .4)
    ax.set_xlim(xmin=-.2)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('7_Faero_X_3sub.pdf'), bbox_inches='tight')


# %%


# %% Amplitude plot

ntime = d['ntime']
spl_c_plane = d['spl_c_plane']
vent_idx_spl = int(d['vent_idx_spl'])

fig, ax = plt.subplots()
for i in np.arange(0, ntime, 20):
    ax.plot(spl_c_plane[i, :, 0], spl_c_plane[i, :, 1], c=bmap[1])
ax.axvline(0, color=bmap[2], lw=1)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
fig.set_tight_layout(True)


fig, ax = plt.subplots()
for i in np.arange(20, ntime, 10):
    ax.plot(spl_c_plane[i, :vent_idx_spl, 0],
            spl_c_plane[i, :vent_idx_spl, 1], c=bmap[1])
ax.axvline(0, color=bmap[2], lw=1)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
fig.set_tight_layout(True)



# %%


# %%

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)
axs = axs.flatten()

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        SVL = float(d['SVL_avg']) / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav
        Maero_X, Maero_Y, Maero_Z = d['Maero'].sum(axis=1).T / (mg * SVL)

#        idx_cross = np.where(Faero_Z > 1)[0]
#        if len(idx_cross) > 0:
#            idx_cross = idx_cross[0]
#            axs[col].axvline(Y[idx_cross], color='gray', lw=1)
        axs[col].plot(Y, Maero_Z, c=snake_colors[snake_id])

for ax in axs:
#    ax.set_ylim(0, 1.7)
    ax.set_xlim(xmin=-.2)
#    ax.set_yticks([0, .5, 1, 1.5])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(fn_plots.format('4a_Faero_Z_3sub.pdf'), bbox_inches='tight')


# %%

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize)
axs = axs.flatten()

for ax in axs:
    ax.axhline(0, color='gray', linestyle='--')

#for col, snake_id in enumerate([81, 91, 95]):
for col, snake_id in enumerate([95]):
    fn_names = ret_fnames(snake_id)
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        Y = d['com'][:, 1] / 1000 - Yo
        SVL = float(d['SVL_avg']) / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav
        Maero_X, Maero_Y, Maero_Z = d['Maero'].sum(axis=1).T / (mg * SVL)

        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav
        SVL = float(d['SVL_avg']) / 1000
        mass_spl = d['mass_spl'] / 1000

        spl_c = d['spl_c']
        a_spl_c = d['a_spl_c']
        ntime, nspl, _ = spl_c.shape

        spl_c_m = spl_c / 1000  # m
        a_spl_m = a_spl_c / 1000  # m/s^2  # local acceleration

        Minert = np.zeros((ntime, nspl, 3))
        for i in np.arange(ntime):
            for j in np.arange(nspl):

                # angular momenum
                tmp_iner = np.cross(spl_c_m[i, j], mass_spl[i, j] * a_spl_m[i, j])
                Minert[i, j] = tmp_iner

        Maero_tot = d['Maero'].sum(axis=1)
        Maero_mag = np.linalg.norm(Maero_tot, axis=1)
        Minert_mag = np.linalg.norm(np.sum(Minert, axis=1), axis=1)
        axs[col].plot(Y, Minert_mag / Maero_mag, c=snake_colors[snake_id])

for ax in axs:
#    ax.set_ylim(0, 1.7)
    ax.set_xlim(xmin=-.2)
#    ax.set_yticks([0, .5, 1, 1.5])

sns.despine()
fig.set_tight_layout(True)



# %% Inertial terms

mass_kg = float(d['mass']) / 1000
mg = mass_kg * grav
SVL = float(d['SVL_avg']) / 1000
mass_spl = d['mass_spl'] / 1000

spl_c = d['spl_c']
a_spl_c = d['a_spl_c']
ntime, nspl, _ = spl_c.shape

spl_c_m = spl_c / 1000  # m
a_spl_m = a_spl_c / 1000  # m/s^2  # local acceleration

inertial_term = np.zeros((ntime, nspl, 3))
Faero_term = np.zeros((ntime, nspl, 3))

for i in np.arange(ntime):
    for j in np.arange(nspl):

        # angular momenum
        tmp_iner = np.cross(spl_c_m[i, j], mass_spl[i, j] * a_spl_m[i, j])
        inertial_term[i, j] = tmp_iner
#
#        tmp_aero = np.cross(spl_c_m[i, j], Faero[i, j])
#        Faero_term[i, j] = tmp_aero

        # newton's linear momentum
#        tmp_iner = mass_spl[i, j] * a_spl_m[i, j]
#        inertial_term[i, j] = tmp_iner

#        tmp_aero = Faero[i, j]
#        Faero_term[i, j] = tmp_aero


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




# %%


# %% Vertical aerodynamic force

fn_names = ret_fnames(81)
#fn_names = ret_fnames(95)
fn_names = ret_fnames()

fig, ax = plt.subplots()
ax.axhline(1, color='gray', linestyle='--')

for snake_id in snake_ids:
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times']
        Y = d['com'][:, 1] / 1000 - Yo
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * grav

        Faero_Z = d['Faero'][:, :, 2].sum(axis=1) / mg

        if i == 0:
            ax.plot(Y, Faero_Z, label=snake_id, c=snake_colors[snake_id])
        else:
            ax.plot(Y, Faero_Z, c=snake_colors[snake_id])

ax.legend(loc='best', ncol=2)
sns.despine()
fig.set_tight_layout(True)


# %% Velocity polar diagram

fn_names = ret_fnames(95)


fig, ax = plt.subplots()

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    vx, vy, vz = d['v_com'].T / 1000
    ax.plot(vy, vz, c=snake_colors[snake_id])

plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

sns.despine(top=False, bottom=True)
fig.set_tight_layout(True)


# %% Velocity polar diagram for all snakes

fig, ax = plt.subplots()

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['v_com'].T / 1000
        ax.plot(vy, vz, c=snake_colors[snake_id])

plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

sns.despine(top=False, bottom=True)
fig.set_tight_layout(True)


# %% Velocity polar diagram for all snakes on three different axes

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14.3, 5.2))

for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        vx, vy, vz = d['v_com'].T / 1000
        axs[col].plot(vy, vz, c=snake_colors[snake_id])

plt.setp(axs.flat, aspect=1.0, adjustable='box')
for ax in axs:
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 0)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    sns.despine(ax=ax, top=False, bottom=True)
fig.set_tight_layout(True)


# %% Sfrac

fn_names = ret_fnames(95)
#fn_names = ret_fnames(33)
#fn_names = ret_fnames()

fig, ax = plt.subplots()
#ax.axhline(1, color='gray', linestyle='--')

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)

    times = d['times']
    Y = d['com'][:, 1]
    Sfrac = 100 * d['Sfrac'][:, 2]  # this is the out-of-plane component

    ax.plot(times, Sfrac, label=trial_id)
#    ax.plot(Y, Sfrac, label=trial_id)

ax.legend(loc='best', ncol=2)
sns.despine()
fig.set_tight_layout(True)


# %% planar_fit_error

fn_names = ret_fnames(81)
#fn_names = ret_fnames(33)
#fn_names = ret_fnames()

fig, ax = plt.subplots()
#ax.axhline(1, color='gray', linestyle='--')

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    SVL = float(d['SVL_avg'])
    Y = d['com'][:, 1] / 1000 - Yo
    planar_fit_error = d['planar_fit_error'] / SVL

    ax.plot(Y, planar_fit_error, label=trial_id)

ax.legend(loc='best', ncol=2)
sns.despine()
fig.set_tight_layout(True)


# %% Y-X trajectorires

fn_names = ret_fnames(95)
fn_names = ret_fnames()

fig, ax = plt.subplots()

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    Y = d['com'][:, 1] / 1000
    Z = d['com'][:, 2] / 1000

    ax.plot(Y, Z, c=snake_colors[snake_id])

plt.setp(ax, aspect=1.0, adjustable='box')
sns.despine()
fig.set_tight_layout(True)


# %% Top view of Y-X landing positions

fn_names = ret_fnames(95)
#fn_names = ret_fnames()

fig, ax = plt.subplots()
ax.axvline(0, color='gray')

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)

    times = d['times']
    X, Y, Z = d['com_3D'].T

    ax.plot(X, Y)
    ax.plot(X[0], Y[0], 'o', ms=6, label=trial_id)
    ax.plot(X[-1], Y[-1], 'o', ms=6, label=trial_id)

plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(-3000, 3000)
ax.set_ylim(-6000, 0)
#ax.legend(loc='best', ncol=2)
sns.despine()
fig.set_tight_layout(True)


# %% Top view of Y-X landing positions

fn_names = ret_fnames(95)
#fn_names = ret_fnames()


fig, ax = plt.subplots()

_lw = .25
radii = np.r_[1, 2, 3, 4, 5, 6]
angles = np.r_[0, 30, 60, 90]
angles = np.r_[angles, -angles]
for radius in radii:
    circ = plt.Circle((0, 0), radius, color='gray', fill=False, lw=_lw)
    ax.add_artist(circ)
    if radius == radii[-1]:
        for ang in np.deg2rad(angles):
            xx = np.r_[0, radius * np.sin(ang)]
            yy = np.r_[0, radius * np.cos(ang)]
            ax.plot(xx, yy, color='gray', lw=_lw)

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    X, Y, Z = d['com_3D'].T / 1000
    X -= Xo
    Y -= Yo
    ax.plot(X, Y, c=snake_colors[snake_id])
#    ax.plot(X[0], Y[0], 'o', ms=6, label=trial_id)
#    ax.plot(X[-1], Y[-1], 'o', ms=6, label=trial_id)

plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(-3.1, 3.1)
ax.set_ylim(0, 6.5)
ax.set_yticks([])
sns.despine(left=True)
fig.set_tight_layout(True)


# %% Top view of Y-X landing positions - hex bin plot

fn_names = ret_fnames(95)
fn_names = ret_fnames()

fig, ax = plt.subplots()
#ax.axvline(0, color='gray')

X1, Y1 = [], []

for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)

    times = d['times']
    X, Y, Z = d['Ro_I'].T
    X1.append(X[-1])
    Y1.append(Y[-1])

    ax.plot(X, Y, alpha=.5)
#    ax.plot(X[-1], Y[-1], 'o', ms=6, label=trial_id)

X1, Y1 = np.array(X1), np.array(Y1)
ax.hexbin(X1, Y1, gridsize=12, cmap=plt.cm.viridis, alpha=.25,
          linewidths=.1, mincnt=1)

plt.setp(ax, aspect=1.0, adjustable='box')
ax.set_xlim(-3000, 3000)
ax.set_ylim(-6000, 0)
#ax.legend(loc='best', ncol=2)
sns.despine()
fig.set_tight_layout(True)
