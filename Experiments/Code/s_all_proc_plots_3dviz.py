# -*- coding: utf-8 -*-
"""
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

# %% Load in a trial and make 3D visualizations

fname = ret_fnames(91, 413)[0]
fname = ret_fnames(95, 618)[0]  # best performance
#fname = ret_fnames(95, 807)[0]
#fname = ret_fnames(81, 507)[0]  # use in talks + paper
#fname = ret_fnames(81, 303)[0]
#fname = ret_fnames(81, 501)[0]
#fname = ret_fnames(88, 505)[0]  # 2nd best performance

d = np.load(fname)

# extract values for 3D Plot
foil_color = d['foil_color']
dR_S = d['dR_S']
dR_I = d['dR_I']
dR_BC_I = d['dR_BC_I']
Cdir_I, Bdir_I = d['Cdir_I'], d['Bdir_I']
pfe_Ic = d['pfe_Ic']

foils_Ic = d['foils_Ic']
pfe_Ic = d['pfe_Ic']

foils_I = d['foils_I']
pfe_I = d['pfe_I']

foils_S = d['foils_S']
pfe_S = d['pfe_S']

foils_Sc = d['foils_Sc']
pfe_Sc = d['pfe_Sc']
foil_color = d['foil_color']

foils_B = d['foils_B']
pfe_B = d['pfe_B']

YZ_S, XZ_S, XY_S = d['YZ_S'], d['XZ_S'], d['XY_S']

# %% Check rotatation to the flow frame

C_I2S = d['C_I2S']
gamma = d['gamma']
R_Ic, R_Sc = d['R_Ic'], d['R_Sc']
dRo_I = d['dRo_I']
foils_Ic = d['foils_Ic']
foils_Sc = d['foils_Sc']
R_Fc = np.zeros_like(R_Sc)
R_Fcheck = np.zeros_like(R_Sc)
C_I2F = np.zeros_like(C_I2S)
C_S2F = np.zeros_like(C_I2S)
foils_Fc = np.zeros_like(foils_Ic)
foils_Fcheck = np.zeros_like(foils_Sc)
dRo_F = np.zeros_like(dRo_I)
ntime = d['ntime']

for i in np.arange(ntime):
    gamma_i = -gamma[i]
    C_S2F[i] = np.array([[1, 0, 0],
                         [0, np.cos(gamma_i), np.sin(gamma_i)],
                         [0, -np.sin(gamma_i), np.cos(gamma_i)]])

    C_I2F[i] = np.dot(C_S2F[i], C_I2S[i])
    R_Fc[i] = np.dot(C_I2F[i], R_Ic[i].T).T
    R_Fcheck[i] = np.dot(C_S2F[i], R_Sc[i].T).T
    dRo_F[i] = np.dot(C_I2F[i], dRo_I[i].T).T

    for j in np.arange(foils_Ic.shape[1]):
        foils_Fc[i, j] = np.dot(C_I2F[i], foils_Ic[i, j].T).T
        foils_Fcheck[i, j] = np.dot(C_S2F[i], foils_Sc[i, j].T).T

assert(np.allclose(R_Fc, R_Fcheck))
assert(np.allclose(foils_Fc, foils_Fcheck))

# %% Plot body in inertial and flow frames together

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 152

# inertial axes
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

mlab.plot3d([0, 400], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([200, 200], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([200, 200], [0, 0], [-75, 75], color=frame_c[2], **_args)

pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(200 + foils_Fc[i, :, :, 0],
                 foils_Fc[i, :, :, 1],
                 foils_Fc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='RdBu', opacity=1,
                 vmin=0, vmax=1)

body = mlab.mesh(foils_Ic[i, :, :, 0],
                 foils_Ic[i, :, :, 1],
                 foils_Ic[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

mlab.quiver3d([0], [0], [0],
              [dRo_I[i, 0]], [dRo_I[i, 1]], [dRo_I[i, 2]],
              color=(0, 0, 0), scale_factor=.01,
              mode='arrow')

mlab.quiver3d([200], [0], [0],
              [dRo_F[i, 0]], [dRo_F[i, 1]], [dRo_F[i, 2]],
              color=(0, 0, 0), scale_factor=.01,
              mode='arrow')

sk = 2

fig.scene.isometric_view()

# %% Plot invidual snake cross sections

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 116

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

pts = mlab.points3d(pfe_Ic[i, 1:, 0], pfe_Ic[i, 1:, 1], pfe_Ic[i, 1:, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

fc_minmax = foil_color[i].copy()
fc_minmax[-1, -1] = 0
fc_minmax[-1, -2] = 1
body = mlab.mesh(foils_Ic[i, :, :, 0],
                 foils_Ic[i, :, :, 1],
                 foils_Ic[i, :, :, 2],
                 scalars=fc_minmax,
                 colormap='YlGn', representation='points')

snake_green = tuple(np.array([34, 139, 34]) / 255)

fig.scene.isometric_view()

# %% A) Setup figure to animate

# down sample along the body
skip_body = 2

nbody = foils_Sc[0].shape[0]
vent_idx_spl = int(d['vent_idx_spl'])
idx = np.arange(nbody)

tail_idx = idx[vent_idx_spl:-1][::4]
body_idx = idx[1:vent_idx_spl][::2]
idx = np.r_[body_idx, tail_idx]


fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
#                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 0

#pts = mlab.points3d(pfe_Sc[i, 1:, 0], pfe_Sc[i, 1:, 1], pfe_Sc[i, 1:, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=3)

#body = mlab.mesh(foils_Sc[i, 1:-1:skip_body, :, 0],
#                 foils_Sc[i, 1:-1:skip_body, :, 1],
#                 foils_Sc[i, 1:-1:skip_body, :, 2],
#                 color=snake_green)

body = mlab.mesh(foils_Sc[i, idx, :, 0],
                 foils_Sc[i, idx, :, 1],
                 foils_Sc[i, idx, :, 2],
                 color=snake_green)

fig.scene.isometric_view()

# %% B) Export data as obj

ntime = pfe_Sc.shape[0]

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
#        pts.mlab_source.set(x=pfe_Sc[i, 1:, 0],
#                            y=pfe_Sc[i, 1:, 1],
#                            z=pfe_Sc[i, 1:, 2])
        body.mlab_source.set(x=foils_Sc[i, idx, :, 0],
                             y=foils_Sc[i, idx, :, 1],
                             z=foils_Sc[i, idx, :, 2])

        # SAVE OBJ FOR 3D MOVIE
#        mlab.savefig('../Movies/s_all_proc_plots/507_81_Sc_obj_no_pts/frame{0:04d}.obj'.format(i))
#        mlab.savefig('../Movies/s_all_proc_plots/frame{0:04d}.obj'.format(i))

        # SAVE OBJ FOR 3D MOVIE
        # mlab.savefig('../Movies/s_all_proc_plots/618_95_Sc_obj_no_pts/frame{0:04d}.obj'.format(i))
        yield
manim = anim()
mlab.show()


# %% A) Down-sample the mesh

skip_body = 2  # along the body
#skip_circ = 4  # around circumference
skip_circ = 1  # around circumference  # 2017-10-05

# ignore the end caps
foils_tmp = foils_Ic[:, 1:-1:skip_body, ::skip_circ].copy()
shape = list(foils_tmp.shape)
shape[2] += 1  # add to the circum dir.

foils = np.zeros(shape)
foils[:, :, :-1] = foils_tmp
foils[:, :, -1] = foils_tmp[:, :, 0]

# shorten the colors array
foil_color_tmp = foil_color[:, 1:-1:skip_body, ::skip_circ].copy()
shape = list(foil_color_tmp.shape)
shape[2] += 1  # add to the circum dir.

foil_color_minmax = np.zeros(shape)
foil_color_minmax[:, :, :-1] = foil_color_tmp
foil_color_minmax[:, :, -1] = foil_color_tmp[:, :, 0]
foil_color_minmax[:, -1, -1] = 0
foil_color_minmax[:, -1, -2] = 1

# color without mesh
snake_green = tuple(np.array([34, 139, 34]) / 255)

# %% A) Setup figure for saving

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
#                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 0

#pts = mlab.points3d(pfe_Ic[i, 1:, 0], pfe_Ic[i, 1:, 1], pfe_Ic[i, 1:, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=64)

#body = mlab.mesh(foils[i, :, :, 0],
#                 foils[i, :, :, 1],
#                 foils[i, :, :, 2],
#                 scalars=foil_color_minmax[i], colormap='YlGn')

body = mlab.mesh(foils[i, :, :, 0],
                 foils[i, :, :, 1],
                 foils[i, :, :, 2],
                 color=snake_green)

#body = mlab.mesh(foils_Ic[i, :, :, 0],
#                 foils_Ic[i, :, :, 1],
#                 foils_Ic[i, :, :, 2])

fig.scene.isometric_view()

# %% B) Export 3D mesh to obj

ntime = pfe_Ic.shape[0]

@mlab.animate(delay=100)
def anim():
    for i in np.arange(ntime):
#        body.mlab_source.set(x=foils_Ic[i, :, :, 0],
#                             y=foils_Ic[i, :, :, 1],
#                             z=foils_Ic[i, :, :, 2])
#        fig.scene.isometric_view()
#        pts.mlab_source.set(x=pfe_Ic[i, 1:, 0],
#                            y=pfe_Ic[i, 1:, 1],
#                            z=pfe_Ic[i, 1:, 2])
#        mlab.savefig('../Movies/s_all_proc_plots/807_95/frame_{0:03d}.wrl'.format(i))
#        mlab.savefig('../Movies/s_all_proc_plots/618_95_obj/frame_{0:03d}.obj'.format(i))


#        body.mlab_source.set(x=foils_Ic[i, 1:-1:sk_body, ::sk_circ, 0],
#                             y=foils_Ic[i, 1:-1:sk_body, ::sk_circ, 1],
#                             z=foils_Ic[i, 1:-1:sk_body, ::sk_circ, 2])

#        body.mlab_source.set(x=foils[i, :, :, 0],
#                             y=foils[i, :, :, 1],
#                             z=foils[i, :, :, 2],
#                             scalars=foil_color_minmax[i])
#        mlab.savefig('../Movies/s_all_proc_plots/618_95_small/frame{0:04d}.obj'.format(i))


        body.mlab_source.set(x=foils[i, :, :, 0],
                             y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2])
        # mlab.savefig('../Movies/s_all_proc_plots/807_95_obj/frame{0:04d}.obj'.format(i))
        yield
manim = anim()
mlab.show()

# %% Plot every 10 frames, inertial and straightened trajectories

ntime = foils_I.shape[0]

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 0

snake_green = tuple(np.array([34, 139, 34]) / 255)

for i in np.arange(0, ntime, 10):
    body = mlab.mesh(foils_I[i, :, :, 0],
                     foils_I[i, :, :, 1],
                     foils_I[i, :, :, 2],
                     color=snake_green)

    body = mlab.mesh(foils_S[i, :, :, 0],
                     foils_S[i, :, :, 1],
                     foils_S[i, :, :, 2],
                     color=bmap[2])

fig.scene.isometric_view()

# %% Plot body with IR markers

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

i = 190

# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1)
mlab.plot3d([-200, 200], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-200, 200], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-75, 75], color=frame_c[2], **_args)

pts = mlab.points3d(pfe_Sc[i, 1:, 0], pfe_Sc[i, 1:, 1], pfe_Sc[i, 1:, 2],
                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_Sc[i, :, :, 0],
                 foils_Sc[i, :, :, 1],
                 foils_Sc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#YZ_mesh = mlab.mesh(YZ_S[i, :, :, 0], YZ_S[i, :, :, 1], YZ_S[i, :, :, 2],
#                    color=bmap[2], opacity=.25)
#XZ_mesh = mlab.mesh(XZ_S[i, :, :, 0], XZ_S[i, :, :, 1], XZ_S[i, :, :, 2],
#                    color=bmap[1], opacity=.25)
#XY_mesh = mlab.mesh(XY_S[i, :, :, 0], XY_S[i, :, :, 1], XY_S[i, :, :, 2],
#                    color=bmap[0], opacity=.25)

#mlab.orientation_axes()
fig.scene.isometric_view()

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)

