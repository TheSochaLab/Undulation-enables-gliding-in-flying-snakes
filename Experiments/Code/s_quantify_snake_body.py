"""
cd /Volumes/Yeaton_HD6/Code for Manuscripts/Undulation_confers_stability/Experiments/Code

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_quantify_snake_body/{}.pdf'
FIGPNG = '../Figures/s_quantify_snake_body/{}.png'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}
SAVEFIG = False

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


# %% Select one trial and extract information on it

#fname = ret_fnames(91, 413)[0]
# fname = ret_fnames(95, 618)[0]
#fname = ret_fnames(95, 807)[0]
#fname = ret_fnames(95, 712)[0]
#fname = ret_fnames(81, 303)[0]
#fname = ret_fnames(81, 403)[0]
#fname = ret_fnames(81, 501)[0]
#fname = ret_fnames(81, 309)[0]
#fname = ret_fnames(90, 623)[0]  # pitch down at start
#fname = ret_fnames(90, 629)[0]  # pitch down at start
#fname = ret_fnames(81, 706)[0]  # pitch down at start
#fname = ret_fnames(95, 712)[0]
#fname = ret_fnames(95, 808)[0]  # f_theta = 2.52 Hz -> too large!
#fname = ret_fnames(91, 719)[0]
#fname = ret_fnames(86, 406)[0]  # messed-up angles
#fname = ret_fnames(86, 412)[0]  # fine....
#fname = ret_fnames(86, 803)[0]  # vertical wave little messed-up
fname = ret_fnames(81, 507)[0]  # TRIAL USED IN FIGURE 2 OF PAPER

d = np.load(fname)

dt = float(d['dt'])
times = d['times']
ntime = d['ntime']
vent_loc = d['vent_idx_spl'] + 1
SVL = d['SVL_avg']
start = d['idx_pts'][1]  # 0 is the virtual marker
#start = 0
snon = d['t_coord'][0, start:vent_loc] / SVL
snonf = d['t_coord'][0] / SVL
s_plot = np.arange(vent_loc)
nbody = len(snon)


# body position
R = d['R_Sc']

x, y, z = R[:, 0:vent_loc].T  # TODO
xf, yf, zf = R.T

x, y, z = x.T, y.T, z.T
xf, yf, zf = xf.T, yf.T, zf.T


# bending angles
dRds = d['Tdir_S']
dRds = d['Tdir_I']  #TODO was using _S before 2017-02-13

psi = np.arcsin(dRds[:, start:vent_loc, 2])
psi_f = np.arcsin(dRds[:, :, 2])

theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])
theta_f = np.arctan2(dRds[:, :, 0], -dRds[:, :, 1])

# 2017-02-22 Maybe don't actually ave to unwrap the angles
psi = np.unwrap(psi, axis=1)
psi_f = np.unwrap(psi_f, axis=1)

theta = np.unwrap(theta, axis=1)
theta_f = np.unwrap(theta_f, axis=1)

# mean remove
psi_mean = psi.mean(axis=1)
theta_mean = theta.mean(axis=1)
psi = (psi.T - psi_mean).T
theta = (theta.T - theta_mean).T
psi_f = (psi_f.T - psi_mean).T
theta_f = (theta_f.T - theta_mean).T

# detrent the angles
d_psi_pp = np.zeros((ntime, 2))
d_psi_fit = np.zeros((ntime, nbody))
psi_detrend = np.zeros((ntime, nbody))

d_theta_pp = np.zeros((ntime, 2))
d_theta_fit = np.zeros((ntime, nbody))
theta_detrend = np.zeros((ntime, nbody))

for i in np.arange(ntime):

    pp = np.polyfit(snon, psi[i], 1)
    y_lin = np.polyval(pp, snon)
    y_fit = psi[i] - y_lin
    d_psi_pp[i] = pp
    d_psi_fit[i] = y_lin
    psi_detrend[i] = y_fit

    psi_f[i] = psi_f[i] - np.polyval(pp, snonf)

    pp = np.polyfit(snon, theta[i], 1)
    y_lin = np.polyval(pp, snon)
    y_fit = theta[i] - y_lin
    d_theta_pp[i] = pp
    d_theta_fit[i] = y_lin
    theta_detrend[i] = y_fit

# only remove trend on vertical wave
psi_trend = psi.copy()
psi = psi_detrend.copy()

# find zero crossings of the lateral wave
snon_zr = []
snon_zr_f = []
diff_snon_zr = []
theta_zr, psi_zr = [], []
x_zr, y_zr, z_zr = [], [], []
for i in np.arange(ntime):
    ti = theta[i]

    i0 = np.where(np.diff(np.signbit(theta[i])))[0]
    i1 = i0 + 1
    i0_f, i1_f = i0 + start, i1 + start
    frac = np.abs(ti[i0] / (ti[i1] - ti[i0]))

    zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
    snon_zr.append(zrs_i)
    snon_zr_f.append(zrs_i + start)
    diff_snon_zr.append(np.diff(zrs_i))

    theta_zr.append(ti[i0] + frac * (ti[i1] - ti[i0]))
    psi_zr.append(psi[i][i0] + frac * (psi[i][i1] - psi[i][i0]))

    x_zr.append(x[i][i0_f] + frac * (x[i][i1_f] - x[i][i0_f]))
    y_zr.append(y[i][i0_f] + frac * (y[i][i1_f] - y[i][i0_f]))
    z_zr.append(z[i][i0_f] + frac * (z[i][i1_f] - z[i][i0_f]))

# %% FIGURE 2A: Bending angles vs distance along body

i = 116  # NOTE: i=116 for trial 507, snake 81 is the example used in the text

figsize = (5.5, 4)
fig, ax = plt.subplots(figsize=figsize)
ax.axhline(0, color='gray', lw=1)
ax.set_ylim(-120, 120)
ax.set_yticks([-120, -80, -40, 0, 40, 80, 120])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
sns.despine()
fig.set_tight_layout(True)

theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='Lateral')
phi_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, c='y', label='Vertical')

o_zr_theta, = ax.plot(100 * snon_zr[i], np.rad2deg(theta_zr[i]), 'ro',
                      mfc='none', mew=2, mec='r')
o_zr_psi, = ax.plot(100 * snon_zr[i], np.rad2deg(psi_zr[i]), 'ko',
                    mfc='none', mew=2, mec='k')

ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Bending angles')

if SAVEFIG:
    fig.savefig(FIG.format('507_81_i116_phi_theta small'), **FIGOPT)


# %% FIGURE 2B: Top view of snake's body

fig, ax = plt.subplots()
ax.axis('equal', adjustable='box')
ax.plot([0, 0], 1.05 * np.r_[y.min(), y.max()], c='gray', lw=1)
ax.plot(1.05 * np.r_[x.min(), x.max()], [0, 0], c='gray', lw=1)
ax.axis('off')
fig.set_tight_layout(True)

body_line, = ax.plot(x[i], y[i], 'go-', lw=5, ms=10, markevery=1000)

o_zr, = ax.plot(x_zr[i], y_zr[i], 'ro',
                mfc='none', mew=2, mec='r')

if SAVEFIG:
    fig.savefig(FIG.format('507_81_i116_phi_body small'), **FIGOPT)

# %% FIGURE 2K: Detrended psi

i = 217

figsize = (5.5, 4)
fig, ax = plt.subplots(figsize=figsize)
ax.set_ylim(-40, 40)
ax.set_yticks([-30, 0, 30])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])

ax.plot(100 * snon, np.rad2deg(psi_trend[i]), 'y', lw=3)

fit = d_psi_pp[0] * np.r_[0, 100] + d_psi_pp[1]
ax.plot(100 * snon, np.rad2deg(d_psi_fit[i]), 'k-', lw=2)

ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Distance along body (%SVL)', fontsize='small')
ax.set_ylabel('Vertical bending angle', fontsize='small')

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('507_81_i227_d_psi fit small'), **FIGOPT)

# %% FIGURE 2C,D: Heat map of bending

i = 116

Ss, Tt = np.meshgrid(snon, times)
Ss, Tt = np.meshgrid(100 * snonf, times - times[0])
Ss, Tt = np.meshgrid(100 * snonf, times)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               figsize=(9.5, 4))

cax1 = ax1.pcolormesh(Tt, Ss, np.rad2deg(theta_f), cmap=plt.cm.coolwarm,
              vmin=-120, vmax=120, linewidth=0, rasterized=True)
ax1.plot(Tt[i, start:vent_loc + 1], Ss[i, start:vent_loc + 1],
        c='b', lw=3)
cax1.set_edgecolor('face')
cbar1 = fig.colorbar(cax1, ax=ax1, shrink=.8, orientation='vertical')
cbar1.set_ticks(np.r_[-120:121:60])

cax2 = ax2.pcolormesh(Tt, Ss, np.rad2deg(psi_f), cmap=plt.cm.coolwarm,
              vmin=-60, vmax=60, linewidth=0, rasterized=True)
ax2.plot(Tt[i, start:vent_loc + 1], Ss[i, start:vent_loc + 1],
        c='y', lw=3)
cax2.set_edgecolor('face')
cbar2 = fig.colorbar(cax2, ax=ax2, shrink=.8, orientation='vertical')
cbar2.set_ticks(np.r_[-60:61:30])

ax1.plot(len(snon_zr[i]) * [Tt[i, 0]], 100 * snon_zr[i],
         'ro', mfc='none', mew=2, mec='r', zorder=1000)
ax2.plot(len(snon_zr[i]) * [Tt[i, 0]], 100 * snon_zr[i],
         'ko', mfc='none', mew=2, mec='k', zorder=1000)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Body coordinate (%SVL)')
ax2.set_xlabel('Time (s)')

ax1.set_xlim(Tt.min(), Tt.max())
ax1.set_ylim(0, 130)
ax1.set_yticks([0, 25, 50, 75, 100, 130])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
fig.set_tight_layout(True)

# add degree symbol to angles
fig.canvas.draw()
for cbar in [cbar1, cbar2]:
    ax = cbar.ax

    ticks = ax.yaxis.get_ticklabels()
    newticks = []
    for tick in ticks:
        text = tick.get_text()
        newticks.append(text + u'\u00B0')
    cbar.ax.yaxis.set_ticklabels(newticks)
    cbar.ax.tick_params(labelsize='x-small')

if SAVEFIG:
    fig.savefig(FIG.format('507_81 heatmap - pub'), **FIGOPT)

# %% FIGURE 2I: Top view of snake body and tangent vector

# extract data
foils_Ic = d['foils_Ic']
pfe_Ic = d['pfe_Ic']

foils_I = d['foils_I']
pfe_I = d['pfe_I']

foils_S = d['foils_S']
pfe_S = d['pfe_S']

foils_Sc = d['foils_Sc']
pfe_Sc = d['pfe_Sc']
foil_color = d['foil_color']

Tdir_S, Cdir_S, Bdir_S = d['Tdir_S'], d['Cdir_S'], d['Bdir_S']
R_Sc = d['R_Sc']

# make figure
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

i = 116

pts = mlab.points3d(pfe_Sc[i, 1:, 0], pfe_Sc[i, 1:, 1], pfe_Sc[i, 1:, 2],
                    color=(.85, .85, .85), scale_factor=9, resolution=64)

body = mlab.mesh(foils_Sc[i, :, :, 0],
                 foils_Sc[i, :, :, 1],
                 foils_Sc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

scale = 30
j = 50

sf = 80

xx, yy, zz = R_Sc[i, j, 0], R_Sc[i, j, 1], R_Sc[i, j, 2]

# for horizontal wave
mlab.quiver3d(xx, yy, zz,
              np.cos(psi_f[i, j]) * Tdir_S[i, j, 0],
              np.cos(psi_f[i, j]) * Tdir_S[i, j, 1],
              0 * Tdir_S[i, j, 2],
              color=(0, 0, 0), mode='arrow', resolution=64,
              scale_factor=sf)

mlab.quiver3d(xx, yy, zz,
              0,
              -1,
              0,
              color=bmap[0], mode='arrow', resolution=64,
              scale_factor=sf)

fig.scene.isometric_view()
fig.scene.parallel_projection = True

mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)

if SAVEFIG:
    mlab.savefig(FIGPNG.format('507_81 i116 theta top'),
                 size=(5 * 750, 5 * 708), figure=fig)


# %% FIGURE 2J: Rear view of snake body and tangent vector

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 116

pts = mlab.points3d(pfe_Sc[i, 1:, 0], pfe_Sc[i, 1:, 1], pfe_Sc[i, 1:, 2],
                    color=(.85, .85, .85), scale_factor=9, resolution=64)

body = mlab.mesh(foils_Sc[i, :, :, 0],
                 foils_Sc[i, :, :, 1],
                 foils_Sc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1,
                 representation='surface')

scale = 30
j = 62
sf = 80

xx, yy, zz = R_Sc[i, j, 0], R_Sc[i, j, 1], R_Sc[i, j, 2]

# for vertical wave
mlab.quiver3d(xx, yy, zz,
              Tdir_S[i, j, 0],
              Tdir_S[i, j, 1],
              Tdir_S[i, j, 2],
              color=bmap[8], mode='arrow', resolution=64,
              scale_factor=sf, scale_mode='vector')

mlab.quiver3d(xx, yy, zz,
              0,
              -1,
              0,
              color=bmap[0], mode='arrow', resolution=64,
              scale_factor=sf)

mlab.quiver3d(xx, yy, zz,
              1 * Tdir_S[i, j, 0],
              1 * Tdir_S[i, j, 1],
              0 * Tdir_S[i, j, 2],
              color=(0 ,0, 0), mode='arrow', resolution=64,
              scale_factor=sf, scale_mode='vector')

fig.scene.parallel_projection = True

view = (-83.688632659425451,
 68.089247773304535,
 1312.9895557482814,
 np.array([  5.97264426, -41.87065119,  10.99942483]))

mlab.view(*view)

if SAVEFIG:
    mlab.savefig(FIGPNG.format('507_81 i116 psi side'),
                 size=(5 * 750, 5 * 708), figure=fig)
    #
    # mlab.savefig(FIGPNG.format('507_81 i116 theta top'),
    #              size=(5 * 750, 5 * 708), figure=fig)

#### Extra material

# %% Cool: Movies of the angles, with zero crossings

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.set_ylim(-120, 120)
ax.set_yticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
#ax.grid(True, axis='y')
sns.despine()
fig.set_tight_layout(True)

i = 0
theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='Lateral')
phi_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, label='Vertical')
_leg = ax.legend(loc='upper right', frameon=True, ncol=2)
_leg.get_frame().set_linewidth(0)

o_zr_theta, = ax.plot(100 * snon_zr[i], np.rad2deg(theta_zr[i]), 'ro',
                      mfc='none', mew=2, mec='r')
o_zr_psi, = ax.plot(100 * snon_zr[i], np.rad2deg(psi_zr[i]), 'ko',
                    mfc='none', mew=2, mec='k')

title_str = '{0:.0f}%, {1:.2f} sec'
title_text = ax.set_title(title_str.format(100 * i / ntime, times[i]))

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Body angles')

def animate(i):
    theta_line.set_ydata(np.rad2deg(theta[i]))
    phi_line.set_ydata(np.rad2deg(psi[i]))

    o_zr_theta.set_data((100 * snon_zr[i], np.rad2deg(theta_zr[i])))
    o_zr_psi.set_data((100 * snon_zr[i], np.rad2deg(psi_zr[i])))

    title_text.set_text(title_str.format(100 * i / ntime, times[i]))

    return theta_line, phi_line, o_zr_theta, o_zr_psi, title_text

slowed = 10
ani = FuncAnimation(fig, animate, frames=int(d['ntime']),
                    interval=float(d['dt']) * 1000 * slowed,
                    repeat=1, blit=False)

save_movie = False
if save_movie:
    movie_name = '../Movies/s_all_proc_plots/{0}_{1} 10x bending ZR.mp4'
    movie_name = movie_name.format(d['trial_id'], d['snake_id'])
    ani.save(movie_name,
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

# %% Cool: Movies of the angles, with zero crossings, including tail

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.set_ylim(-150, 150)
ax.set_yticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
#ax.set_xlim(0, 100)
ax.set_xlim(0, 135)
ax.set_xticks([0, 25, 50, 75, 100, 135])
#ax.grid(True, axis='y')
sns.despine()
fig.set_tight_layout(True)

i = 0
theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='Lateral')
phi_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, label='Vertical')
_leg = ax.legend(loc='upper right', frameon=True, ncol=2)
_leg.get_frame().set_linewidth(0)

theta_line_full, = ax.plot(100 * snonf, np.rad2deg(theta_f[i]), 'b', lw=1.5)
phi_line_full, = ax.plot(100 * snonf, np.rad2deg(psi_f[i]), 'g', lw=1.5)

o_zr_theta, = ax.plot(100 * snon_zr[i], np.rad2deg(theta_zr[i]), 'ro')
o_zr_psi, = ax.plot(100 * snon_zr[i], np.rad2deg(psi_zr[i]), 'ko')

title_str = '{0:.0f}%, {1:.2f} sec'
title_text = ax.set_title(title_str.format(100 * i / ntime, times[i]))

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Body angles')

def animate(i):
    theta_line.set_ydata(np.rad2deg(theta[i]))
    phi_line.set_ydata(np.rad2deg(psi[i]))
    theta_line_full.set_ydata(np.rad2deg(theta_f[i]))
    phi_line_full.set_ydata(np.rad2deg(psi_f[i]))

    o_zr_theta.set_data((100 * snon_zr[i], np.rad2deg(theta_zr[i])))
    o_zr_psi.set_data((100 * snon_zr[i], np.rad2deg(psi_zr[i])))

    title_text.set_text(title_str.format(100 * i / ntime, times[i]))

    return theta_line, phi_line, theta_line_full, phi_line_full, o_zr_theta, o_zr_psi, title_text

slowed = 10
ani = FuncAnimation(fig, animate, frames=int(d['ntime']),
                    interval=float(d['dt']) * 1000 * slowed,
                    repeat=1, blit=False)

save_movie = False
if save_movie:
    movie_name = '../Movies/s_all_proc_plots/{0}_{1} 10x bending ZR.mp4'
    movie_name = movie_name.format(d['trial_id'], d['snake_id'])
    ani.save(movie_name,
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

# %% Cool: 3D surface of the angles

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

surf_psi = mlab.surf(np.rad2deg(psi), colormap='PuOr', vmin=-10, vmax=10)
surf_theta = mlab.surf(np.rad2deg(theta), colormap='RdBu', vmin=-90, vmax=90)

surf_psi.module_manager.scalar_lut_manager.reverse_lut = True
surf_theta.module_manager.scalar_lut_manager.reverse_lut = True

# %% Movies of the angles, with the psi_trend removed

fig, ax = plt.subplots()
ax.set_ylim(-150, 150)
ax.set_yticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
ax.set_xlim(0, 100)
ax.grid(True, axis='y')
sns.despine()
fig.set_tight_layout(True)

i = 0
theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='lateral')
phi_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, label='vertical')
_leg = ax.legend(loc='lower right', frameon=True, ncol=2)
_leg.get_frame().set_linewidth(0)

d_psi_line, = ax.plot(100 * snon, np.rad2deg(d_psi_fit[i]), ls='--')

title_str = '{0:.0f}%, {1:.2f} sec'
title_text = ax.set_title(title_str.format(100 * i / ntime, times[i]))

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Body angles')

def animate(i):
    theta_line.set_ydata(np.rad2deg(theta[i]))
    phi_line.set_ydata(np.rad2deg(psi[i]))
    d_psi_line.set_ydata(np.rad2deg(d_psi_fit[i]))
    title_text.set_text(title_str.format(100 * i / ntime, times[i]))
    return theta_line, phi_line, d_psi_line, title_text

slowed = 10
ani = FuncAnimation(fig, animate, frames=int(d['ntime']),
                    interval=d['dt'] * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False)#, init_func=init)

save_movie = False
if save_movie:
    movie_name = '../Movies/s_all_proc_plots/{0}_{1} 10x bending MR.mp4'
    movie_name = movie_name.format(d['trial_id'], d['snake_id'])
    ani.save(movie_name,
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])

# %% Movies of the angles, without the trends removed

fig, ax = plt.subplots()
ax.set_ylim(-120, 120)
ax.set_yticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])
ax.axhline(-90, color='k', lw=1)
ax.axhline(90, color='k', lw=1)
#ax.set_xlim(-.05, 1.35)
ax.set_xlim(0, 100)
ax.grid(True)
sns.despine()
fig.set_tight_layout(True)

i = 0
theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='lateral')
phi_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, label='vertical')
_leg = ax.legend(loc='lower right', frameon=True, ncol=2)
_leg.get_frame().set_linewidth(0)

d_theta_line, = ax.plot(100 * snon, np.rad2deg(d_theta_fit[i]), ls='--')
d_psi_line, = ax.plot(100 * snon, np.rad2deg(d_psi_fit[i]), ls='--')

title_text = ax.set_title('{0:.2f}'.format(i / ntime))

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Body angles')

def animate(i):
    theta_line.set_ydata(np.rad2deg(theta[i]))
    phi_line.set_ydata(np.rad2deg(psi[i]))

    d_theta_line.set_ydata(np.rad2deg(d_theta_fit[i]))
    d_psi_line.set_ydata(np.rad2deg(d_psi_fit[i]))

    title_text.set_text('{0:.2f}'.format(i / ntime))
    return theta_line, phi_line, d_theta_line, d_psi_line, title_text

slowed = 10
ani = FuncAnimation(fig, animate, frames=d['ntime'],
                    interval=d['dt'] * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=False)#, init_func=init)


save_movie = False
if save_movie:
    movie_name = '../Movies/s_all_proc_plots/{0}_{1} 10x bending.mp4'
    movie_name = movie_name.format(d['trial_id'], d['snake_id'])
    ani.save(movie_name,
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])