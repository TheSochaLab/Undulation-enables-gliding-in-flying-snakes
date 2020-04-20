# -*- coding: utf-8 -*-
"""
%load_ext autoreload
%autoreload 2

cd /Volumes/Yeaton_HD6/Code for Manuscripts/Undulation_confers_stability/Experiments/Code

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import linregress

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_cod/{}.pdf'
FIGPNG = '../Figures/s_cod/{}.png'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}
SAVEFIG = True
SAVEDATA = False

# %% Functions definitions

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

    Zi = hilbert(X, axis=1)
    Z = np.zeros((m_nbody, n_ntime), dtype=np.complex)
    for i in np.arange(m_nbody):
        Z[i] = hilbert(X[i])  # rows of Z, how each location the spline varies

    assert(np.allclose(Zi, Z))

    R = np.dot(Z, Z.conj().T) / n_ntime
    lamb, W = np.linalg.eig(R)
    lamb = lamb.real
    idx = np.argsort(lamb)[::-1]
    W = W[:, idx]  # columns are the eigenvectors

    lamb_norm = lamb / lamb.sum()
    A = np.sqrt(lamb)

    Q = np.dot(W.conj().T, Z)  # modal coordinates (length of ntime)

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

        Ws[:, i] = w_s
        Wt[:, i] = w_t

    return Ws, Wt, traveling_index


def cod_reanimate(cod_dict, mode_number, times):

    a = cod_dict['A'][mode_number]
    w = cod_dict['W'][:, mode_number]
    f = cod_dict['f'][mode_number]

    ntime = len(times)
    nbody = len(w)
    Y = np.zeros((ntime, nbody))

    for i in np.arange(ntime):
        t = times[i]
        amp = a * np.exp(1j * 2 * np.pi * f * t)
        zz = amp * w
        yy = zz.real
        Y[i] = yy

    return Y


def whirl_rate(cod_dict, times, snon, nmodes=5):
    """Complex whirl rate for the modal coordiante Q (frequency) and
    complex orthogonal mode W (number of waves) for all modes.
    """

    ntime, nbody = len(times), len(snon)

    f = np.zeros(nmodes)
    nu = np.zeros(nmodes)

    Qang = np.zeros((nmodes, ntime))
    Qfit = np.zeros((nmodes, 2))
    linear_fit_Q = np.zeros((nmodes, ntime))

    Wang = np.zeros((nmodes, nbody))
    Wfit = np.zeros((nmodes, 2))
    linear_fit_W = np.zeros((nmodes, nbody))

    for i in np.arange(nmodes):
        Q = cod_dict['Q'][i]
        W = cod_dict['W'][:, i]

        # temporal frequency
        Qang[i] = np.unwrap(np.angle(Q))
        Qfit[i] = np.polyfit(times, Qang[i], 1)  # rad / sec
        f[i] = Qfit[i, 0] / (2 * np.pi)  # slope of the best fit
        linear_fit_Q[i] = np.polyval(Qfit[i], times)

        # spatial frequency
        Wang[i] = np.unwrap(np.angle(W))
        Wfit[i] = np.polyfit(snon, Wang[i], 1)  # rad
        nu[i] = Wfit[i, 0] / (2 * np.pi)
        linear_fit_W[i] = np.polyval(Wfit[i], snon)

    Qs = (Qang, Qfit, linear_fit_Q)
    Ws = (Wang, Wfit, linear_fit_W)
    return f, nu, Qs, Ws

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

# %% Iterate through trials, performing COD

snake_ids_cod = []
trial_ids_cod = []
f_ratios, f_ratios_xz = [], []
nu_ratios, nu_ratios_xz = [], []
f_theta, nu_theta = [], []
f_psi, nu_psi = [], []
vp_theta, vp_psi = [], []
f_x, nu_x = [], []
theta_var0, theta_var1 = [], []
psi_var0, psi_var1 = [], []
theta_max, psi_max = [], []
theta_std, psi_std = [], []
theta_sem, psi_sem = [], []
theta_ci, psi_ci = [], []
d_psi_avg, d_psi_std, d_psi_sem, d_psi_ci = [], [], [], []
theta_nmodes, psi_nmodes = [], []
Xf, Yf, Zf = [], [], []  # landing locations
theta_max_data, psi_max_data = [], []
masses, SVLs = [], []

theta_cods, psi_cods = {}, {}
x_cods, z_cods = {}, {}

fnames = ret_fnames()

for i, fname in enumerate(fnames):
    print(i, fname.split('/')[-1].split('.')[0])

    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)

    times = d['times'] - d['times'][0]
    vent_loc = d['vent_idx_spl'] + 1
    start = d['idx_pts'][1]  # 0 is the virtual marker
    SVL = d['SVL_avg']
    snon = d['t_coord'][0, start:vent_loc] / SVL
    nbody = len(snon)

    X, Y, Z = d['Ro_S'].T / 1000
    X -= Xo
    Y -= Yo
    idx = np.where(Z < 7.25)[0]

    # # trial 808_95: truncate digitization error
    # if trial_id == 808:
    #     print(idx)

    times = times[idx]
    ntime = len(times)

    # x and z positions
    R = d['R_B']
    x, y, z = R[idx, start:vent_loc].T
    x, y, z = x.T, y.T, z.T

    # lateral and vertical bending angles
    dRds = d['Tdir_I']  # 2017-02-13 d['Tdir_S']
    psi = np.arcsin(dRds[idx, start:vent_loc, 2])
    psi = np.unwrap(psi, axis=1)
    theta = np.arctan2(dRds[idx, start:vent_loc, 0], -dRds[idx, start:vent_loc, 1])
    theta = np.unwrap(theta, axis=1)

    # mean remove
    psi = (psi.T - psi.mean(axis=1)).T
    theta = (theta.T - theta.mean(axis=1)).T

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

        pp = np.polyfit(snon, theta[i], 1)
        y_lin = np.polyval(pp, snon)
        y_fit = theta[i] - y_lin
        d_theta_pp[i] = pp
        d_theta_fit[i] = y_lin
        theta_detrend[i] = y_fit

    # store values with a trend
    #theta_trend = theta.copy()
    #theta = theta_detrend.copy()

    # only remove trend on vertical wave
    psi_trend = psi.copy()
    psi = psi_detrend.copy()

    # perform cod analysis
    x_cod = cod(x)
    z_cod = cod(z)
    theta_cod = cod(theta)
    psi_cod = cod(psi)

    # decompose modes into standing and traveling waves
    _x_out = cod_mode_decomp(x_cod['W'])
    x_cod['Ws'], x_cod['Wt'], x_cod['traveling_index'] = _x_out

    _z_out = cod_mode_decomp(z_cod['W'])
    z_cod['Ws'], z_cod['Wt'], z_cod['traveling_index'] = _z_out

    _theta_out = cod_mode_decomp(theta_cod['W'])
    theta_cod['Ws'], theta_cod['Wt'], theta_cod['traveling_index'] = _theta_out

    _psi_out = cod_mode_decomp(psi_cod['W'])
    psi_cod['Ws'], psi_cod['Wt'], psi_cod['traveling_index'] = _psi_out

    # whirl rate using standard linear regression
    # temporal frequency and number of body waves
    _x_out = whirl_rate(x_cod, times, snon)
    x_cod['f'], x_cod['nu'] = _x_out[0], _x_out[1]
    x_cod['Qang'], x_cod['Qfit'], x_cod['linear_fit_Q'] = _x_out[2]
    x_cod['Wang'], x_cod['Wfit'], x_cod['linear_fit_W'] = _x_out[3]

    _z_out = whirl_rate(z_cod, times, snon)
    z_cod['f'], z_cod['nu'] = _z_out[0], _z_out[1]
    z_cod['Qang'], z_cod['Qfit'], z_cod['linear_fit_Q'] = _z_out[2]
    z_cod['Wang'], z_cod['Wfit'], z_cod['linear_fit_W'] = _z_out[3]

    _theta_out = whirl_rate(theta_cod, times, snon)
    theta_cod['f'], theta_cod['nu'] = _theta_out[0], _theta_out[1]
    theta_cod['Qang'], theta_cod['Qfit'], theta_cod['linear_fit_Q'] = _theta_out[2]
    theta_cod['Wang'], theta_cod['Wfit'], theta_cod['linear_fit_W'] = _theta_out[3]

    _psi_out = whirl_rate(psi_cod, times, snon)
    psi_cod['f'], psi_cod['nu'] = _psi_out[0], _psi_out[1]
    psi_cod['Qang'], psi_cod['Qfit'], psi_cod['linear_fit_Q'] = _psi_out[2]
    psi_cod['Wang'], psi_cod['Wfit'], psi_cod['linear_fit_W'] = _psi_out[3]

    f_ratio = psi_cod['f'] / theta_cod['f']
    f_ratio_xz = z_cod['f'] / x_cod['f']

    nu_ratio = psi_cod['nu'] / theta_cod['nu']
    nu_ratio_xz = z_cod['nu'] / x_cod['nu']

    # reanimate waves, using enough modes for 95% var
    theta_nmodes_95 = np.where(theta_cod['lamb_norm'].cumsum() > .95)[0][0] + 1
    theta_recon = np.zeros((ntime, nbody))
    for k in np.arange(theta_nmodes_95):
        recon = cod_reanimate(theta_cod, k, times)
        theta_recon += recon

    psi_nmodes_95 = np.where(psi_cod['lamb_norm'].cumsum() > .95)[0][0] + 1
    psi_recon = np.zeros((ntime, nbody))
    for k in np.arange(psi_nmodes_95):
        recon = cod_reanimate(psi_cod, k, times)
        psi_recon += recon

    theta_cod['nmodes_95'] = theta_nmodes_95
    theta_cod['recon'] = theta_recon
    psi_cod['nmodes_95'] = psi_nmodes_95
    psi_cod['recon'] = psi_recon

    theta_cod['snon'] = snon
    theta_cod['times'] = times
    psi_cod['snon'] = snon
    psi_cod['times'] = times

    # store the data
    snake_ids_cod.append(snake_id)
    trial_ids_cod.append(trial_id)

    f_ratios.append(f_ratio[0])
    f_ratios_xz.append(f_ratio_xz[0])
    nu_ratios.append(nu_ratio[0])
    nu_ratios_xz.append(nu_ratio_xz[0])

    f_theta.append(theta_cod['f'][0])
    nu_theta.append(np.abs(theta_cod['nu'][0]))

    f_psi.append(psi_cod['f'][0])
    nu_psi.append(np.abs(psi_cod['nu'][0]))

    # wave speed in SVL/sec
    SVL_cm = float(d['SVL']) / 10
    SVL_m = float(d['SVL']) / 1000
    SVL_m = 1  # non-dimensional phase speed
    vp_theta_i = theta_cod['f'][0] / (np.abs(theta_cod['nu'][0]) * SVL_m)
    vp_psi_i = psi_cod['f'][0] / (np.abs(psi_cod['nu'][0]) * SVL_m)
    vp_theta.append(vp_theta_i)
    vp_psi.append(vp_psi_i)

    f_x.append(x_cod['f'][0])
    nu_x.append(np.abs(x_cod['nu'][0]))

    theta_var0.append(theta_cod['lamb_norm'].cumsum()[0])
    theta_var1.append(theta_cod['lamb_norm'].cumsum()[1])
    psi_var0.append(psi_cod['lamb_norm'].cumsum()[0])
    psi_var1.append(psi_cod['lamb_norm'].cumsum()[1])

    theta_max_data_i = np.rad2deg(np.ptp(theta, axis=1)) / 2
    psi_max_data_i = np.rad2deg(np.ptp(psi, axis=1)) / 2
    theta_max_data.append(theta_max_data_i.mean())
    psi_max_data.append(psi_max_data_i.mean())

    theta_0 = np.rad2deg(np.ptp(theta_recon, axis=1)) / 2  # length ntime
    theta_0_avg = np.mean(theta_0)
    theta_0_std = np.std(theta_0)
    theta_0_sem = sem(theta_0)
    theta_0_ci = 1.96 * theta_0_sem

    psi_0 = np.rad2deg(np.ptp(psi_recon, axis=1)) / 2
    psi_0_avg = np.mean(psi_0)
    psi_0_std = np.std(psi_0)
    psi_0_sem = sem(psi_0)
    psi_0_ci = 1.96 * psi_0_sem

    theta_max.append(theta_0_avg)
    psi_max.append(psi_0_avg)

    theta_std.append(theta_0_std)
    psi_std.append(psi_0_std)

    theta_sem.append(theta_0_sem)
    psi_sem.append(psi_0_sem)

    theta_ci.append(theta_0_ci)
    psi_ci.append(psi_0_ci)

    d_psi_avg.append(np.rad2deg(d_psi_pp[:, 0].mean()))
    d_psi_std.append(np.rad2deg(d_psi_pp[:, 0].std()))
    d_psi_sem.append(np.rad2deg(sem(d_psi_pp[:, 0])))
    d_psi_ci.append(1.96 * np.rad2deg(sem(d_psi_pp[:, 0])))

    theta_nmodes.append(theta_nmodes_95)
    psi_nmodes.append(psi_nmodes_95)

    Xf.append(X[-1])
    Yf.append(Y[-1])
    Zf.append(Z[-1])

    masses.append(d['mass_spl'][0].sum())
    SVLs.append(SVL_cm)

    key = '{}_{}'.format(trial_id, snake_id)
    theta_cods[key] = theta_cod
    psi_cods[key] = psi_cod
    x_cods[key] = x_cod
    z_cods[key] = z_cod

# %% SI figure on Q and W

trial = '507_81'

tc, pc = theta_cods[trial], psi_cods[trial]

fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
ax1, ax2 = axs

for ax in axs:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

At = np.rad2deg(tc['A'])
Ap = np.rad2deg(pc['A'])

i = 0
ax1.plot(At[i] * tc['W'][:, i].real, At[i] * tc['W'][:, i].imag, 'b',
         label=r'$\theta$')
ax1.plot(Ap[i] * pc['W'][:, i].real, Ap[i] * pc['W'][:, i].imag, 'g',
         label=r'$\psi$')

ax2.plot(tc['Q'][i].real, tc['Q'][i].imag, 'b')
ax2.plot(pc['Q'][i].real, pc['Q'][i].imag, 'g')

ax1.legend(loc='best')

ax1.set_xlim(-110, 110)
ax1.set_ylim(-110, 110)
ax1.xaxis.set_major_formatter(degree_formatter)
ax1.yaxis.set_major_formatter(degree_formatter)
ax1.set_xlabel('Re(W) (deg)', fontsize='x-small')
ax1.set_ylabel('Im(W) (deg)', fontsize='x-small')


ax2.set_xlim(-25, 25)
ax2.set_ylim(-25, 25)
ax2.set_xlabel('Re(Q) (sec)', fontsize='x-small')
ax2.set_ylabel('Im(Q) (sec)', fontsize='x-small')

plt.setp(axs, aspect=1.0, adjustable='box')
sns.despine()

if SAVEFIG:
    fig.savefig(FIG.format('507_81 W Q spectrum'), **FIGOPT)

# %% SI figure on Q and W

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.25))

ax1.plot(100 * tc['snon'], tc['Wang'][0], 'b', lw=3, label=r'$\theta$')
ax1.plot(100 * tc['snon'], tc['linear_fit_W'][0], '--', c='r')
ax1.plot(100 * pc['snon'], pc['Wang'][0], 'g', lw=3, label=r'$\psi$')
ax1.plot(100 * pc['snon'], pc['linear_fit_W'][0], '--', c='r')

ax2.plot(tc['times'], tc['Qang'][0], lw=3)
ax2.plot(tc['times'], tc['linear_fit_Q'][0], '--', c='r')
ax2.plot(tc['times'], pc['Qang'][0], lw=3)
ax2.plot(tc['times'], pc['linear_fit_Q'][0], '--', c='r')

ax1.legend(loc='best')

ax1.set_ylabel(r'$\angle$ W', fontsize='x-small')
ax2.set_ylabel(r'$\angle$ Q', fontsize='x-small')

ax1.set_xlabel('Distance along body (%SVL)', fontsize='x-small')
ax2.set_xlabel('Time (sec)', fontsize='x-small')

ax1.set_xlim(0, 100)
ax2.set_xlim(xmax=1.5)

sns.despine()

if SAVEFIG:
    fig.savefig(FIG.format('507_81 W Q unwind'), **FIGOPT)

# %% Store data COD data into DataFrame

import pandas as pd

data = {'Snake ID': snake_ids_cod, 'Trial ID': trial_ids_cod,
        'f_ratio': f_ratios, 'f_ratio_xz': f_ratios_xz,
        'nu_ratio': nu_ratios, 'nu_ratio_xz': nu_ratios_xz,
        'f_theta': f_theta, 'nu_theta': nu_theta,
        'f_psi': f_psi, 'nu_psi': nu_psi,
        'vp_theta': vp_theta, 'vp_psi': vp_psi,
        'f_x': f_x, 'nu_x': nu_x,
        'theta_var0': theta_var0, 'theta_var1': theta_var1,
        'psi_var0': psi_var0, 'psi_var1': psi_var1,
        'theta_max': theta_max, 'psi_max': psi_max,
        'theta_std': theta_std, 'psi_std': psi_std,
        'theta_sem': theta_sem, 'psi_sem': psi_sem,
        'theta_ci': theta_ci, 'psi_ci': psi_ci,
        'd_psi_avg': d_psi_avg, 'd_psi_std': d_psi_std,
        'd_psi_sem': d_psi_sem, 'd_psi_ci': d_psi_ci,
        'theta_nmodes': theta_nmodes, 'psi_nmodes': psi_nmodes,
        'Xf': Xf, 'Yf': Yf, 'Zf': Zf,
        'theta_max_data': theta_max_data, 'psi_max_data': psi_max_data,
        'mass': masses, 'SVL': SVLs}
df = pd.DataFrame(data=data)

# prune out bad trials
b1 = df['Trial ID'] == 623  # 90
b2 = df['Trial ID'] == 629  # 90
b3 = df['Trial ID'] == 706  # 81
b5 = df['Trial ID'] == 712  # 95; just short
b6 = df['Trial ID'] == 808  # 95; digitization error
b7 = df['Trial ID'] == 406  # 86; angle flips, short
b8 = df['Trial ID'] == 803  # 86; short, nu_theta low
ornata = df['Snake ID'] < 40

# pruned dataframe
dfp = df[~ornata & ~b1 & ~b2 & ~b3 & ~b5 & ~b6 & ~b7 & ~b8]

if SAVEDATA:
    df.to_csv("../Data/COD_52_all_trials.csv")
    dfp.to_csv("../Data/COD_36_paper_trial.csv")

# %% Frequency ratios reported in the paper

print(dfp[['nu_ratio', 'f_ratio']].describe())
print()
print(dfp[['theta_var0', 'psi_var0']].describe())
print()
print(dfp[['theta_nmodes', 'psi_nmodes']].describe())
print()
print(dfp['nu_theta'].describe())
print()
print(dfp['f_theta'].describe())
print()
print(dfp['theta_max'].describe())
print()
print(dfp['psi_max'].describe())
print()
print(dfp['d_psi_avg'].describe())

# %% GROUP TOGETHER BY SNAKE --- summary for slopes

sn = dict(list(dfp.groupby('Snake ID')))

cols_avg = ['f_ratio', 'nu_ratio', 'f_theta', 'nu_theta',
            'f_psi', 'nu_psi', 'vp_theta', 'vp_psi',
            'theta_max', 'psi_max', 'd_psi_avg',
            'theta_nmodes', 'psi_nmodes',
            'theta_max_data', 'psi_max_data']

cols_std = ['f_ratio', 'nu_ratio', 'f_theta', 'nu_theta',
            'f_psi', 'nu_psi',
            'vp_theta', 'vp_psi',
            'theta_max', 'psi_max',
            'd_psi_avg', 'd_psi_std']
cols_std_name = [name + '_std' for name in cols_std]


grp = dfp.groupby('Snake ID')

dfpg = grp.apply(np.mean)[cols_avg]
dfpg[cols_std_name] = grp.apply(np.std)[cols_std]

dfpg['ntrials'] = grp.count()['theta_max']
dfpg['Snake ID'] = dfpg.index
dfpg['mass'] = grp.apply(np.mean)['mass']
dfpg['colors'] = sns.husl_palette(n_colors=len(dfpg))

# %% Figures: Spatial and temporal frequency ratios

fig, ax = plt.subplots(figsize=(5, 5))

grp_nu_r_avg, grp_nu_r_std = dfpg['nu_ratio'].mean(), dfpg['nu_ratio'].std()
grp_f_r_avg, grp_f_r_std = dfpg['f_ratio'].mean(), dfpg['f_ratio'].std()

# just the mean
ax.plot(grp_nu_r_avg, grp_f_r_avg, 'k+', mew=2.5, ms=13)

# now plot all of the data
for i in np.arange(len(dfpg)):
    dfpg_i = dfpg.iloc[i]
    dfp_i = dfp[dfp['Snake ID'] == dfpg_i['Snake ID']]

    ax.scatter(dfpg_i['nu_ratio'], dfpg_i['f_ratio'], s=1.5 * dfpg_i['mass'],
               c=np.array([dfpg_i['colors']]), label=dfpg_i['Snake ID'])

    ax.scatter(dfp_i['nu_ratio'].values, dfp_i['f_ratio'].values, s=35,
               c='none', edgecolors=dfpg_i['colors'], linewidths=1.5,
               alpha=.6, zorder=1)

    plt.errorbar(dfpg_i['nu_ratio'], dfpg_i['f_ratio'], yerr=dfpg_i['f_ratio_std'],
                 c=dfpg_i['colors'])

    plt.errorbar(dfpg_i['nu_ratio'], dfpg_i['f_ratio'], xerr=dfpg_i['nu_ratio_std'],
                 c=dfpg_i['colors'])

ax.set_xlabel('Spatial frequency ratio')
ax.set_ylabel('Temporal frequency ratio')

ax.axis('equal', adjustable='box-forced')

ax.set_xlim(.75, 3.25)
ax.set_ylim(.75, 3.25)
ax.set_xticks([1, 2, 3])
ax.set_yticks([1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('nu and f ratio 3-35 means - pub - +'), **FIGOPT)

# %% Figure: Horizontal wave amplitude vs Number of spatial periods

nu_fit = np.r_[1.05, 1.5]
nt_grp_reg = linregress(dfpg['nu_theta'], dfpg['theta_max'])
theta_grp_fit = nt_grp_reg.slope * nu_fit + nt_grp_reg.intercept

# values reported in the manuscript
# nt_grp_reg.slope = -66 (in figure)
# nt_grp_reg.intercept = 188 (in figure)
# nt_grp_reg.rvalue = -0.9079550255837797
# nt_grp_reg.rvalue**2 = 0.8243823284828421
# nt_grp_reg.pvalue = 0.004695651661365277

fig, ax = plt.subplots(figsize=(5, 5))

for i in np.arange(len(dfpg)):
    dfpg_i = dfpg.iloc[i]
    dfp_i = dfp[dfp['Snake ID'] == dfpg_i['Snake ID']]

    ax.scatter(dfpg_i['nu_theta'], dfpg_i['theta_max'], s=1.5 * dfpg_i['mass'],
               c=np.array([dfpg_i['colors']]), label=dfpg_i['Snake ID'])

    ax.scatter(dfp_i['nu_theta'].values, dfp_i['theta_max'].values, s=35,
               c='none', edgecolors=dfpg_i['colors'], linewidths=1.5)

    plt.errorbar(dfpg_i['nu_theta'], dfpg_i['theta_max'],
                 yerr=dfpg_i['theta_max_std'],
                 c=dfpg_i['colors'])

    plt.errorbar(dfpg_i['nu_theta'], dfpg_i['theta_max'],
                 xerr=dfpg_i['nu_theta_std'],
                 c=dfpg_i['colors'])

# fit based on snake average
ax.plot(nu_fit, theta_grp_fit, '-', c='gray', lw=2, zorder=0)

leg = ax.legend(ncol=2, loc='lower left', handletextpad=.05, columnspacing=.5,
                title='Snake ID', fontsize='small')
leg.get_title().set_fontsize('small')

ax.set_xticks(np.r_[1:1.6:.1])
ax.set_xlim(1, 1.525)
ax.set_ylim(85, 122)
ax.set_xlabel(r'Number of spatial periods')
ax.set_ylabel(r'Horizontal wave amplitude (deg)')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('nu theta snake avg std bars - pub - gray'), **FIGOPT)

# %% Figure 4A: Histogram of bodyshapes in simulation grid

m = -56.6005
b = 175.72823

nu_thetas = np.arange(1, 1.51, .05)
theta_maxs = m * nu_thetas + b

# switch theta_maxs so that increasing
theta_maxs = theta_maxs[::-1]

dnut = np.diff(nu_thetas)[0]
dthm = np.diff(theta_maxs)[0]

binx = np.r_[nu_thetas - dnut / 2, nu_thetas[-1] + dnut / 2]
biny = np.r_[theta_maxs - dthm / 2, theta_maxs[-1] + dthm / 2]
# binx = np.arange(nu_thetas[0] - .05/2, nu_thetas[-1] + .05, .05)

shape_cnts, _, _ = np.histogram2d(dfp['nu_theta'], dfp['theta_max'],
                                  bins=(binx, biny))

# flip back the theta_maxs direction
theta_maxs = theta_maxs[::-1]
shape_cnts = shape_cnts[::-1]

N, T = np.meshgrid(nu_thetas, theta_maxs.astype(np.int))
shape_cnts = pd.DataFrame(data=shape_cnts, index=T[:, 0], columns=N[0])
shape_cnts_label = shape_cnts.replace(np.arange(4), ['', '1', '2', '3'])

figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)

sns.heatmap(shape_cnts, ax=ax, cmap=plt.cm.gray,
            annot=shape_cnts_label, fmt='', annot_kws={'fontsize': 'xx-small'},
            cbar=i == 2, cbar_kws={'shrink': .75},
            rasterized=True)

ax.set_xticklabels(N[0])
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

if SAVEFIG:
    fig.savefig(FIG.format('nu theta 2d histogram'), **FIGOPT)

# %% Figure: DORSO-VENTRAL FLEXION VS. NU_THETA

nd_grp_reg = linregress(dfpg['nu_theta'], dfpg['d_psi_avg'])
nu_fit = np.r_[1, 1.6]
d_psi_avg_grp_fit = nd_grp_reg.slope * nu_fit + nd_grp_reg.intercept

fig, ax = plt.subplots(figsize=(5, 5))

for i in np.arange(len(dfpg)):
    dfpg_i = dfpg.iloc[i]
    dfp_i = dfp[dfp['Snake ID'] == dfpg_i['Snake ID']]

    ax.scatter(dfpg_i['nu_theta'], dfpg_i['d_psi_avg'], s=1.5 * dfpg_i['mass'],
               c=np.array([dfpg_i['colors']]), label=dfpg_i['Snake ID'])

    ax.scatter(dfp_i['nu_theta'].values, dfp_i['d_psi_avg'].values, s=35,
               c='none', edgecolors=dfpg_i['colors'], linewidths=1.5)

    plt.errorbar(dfpg_i['nu_theta'], dfpg_i['d_psi_avg'], yerr=dfpg_i['d_psi_avg_std'],
                 c=dfpg_i['colors'])

    plt.errorbar(dfpg_i['nu_theta'], dfpg_i['d_psi_avg'], xerr=dfpg_i['nu_theta_std'],
                 c=dfpg_i['colors'])

ax.set_xticks(np.r_[1:1.6:.1])
ax.set_xlim(1, 1.525)
ax.set_xlabel(r'Number of spatial periods')
ax.set_ylabel(r'Dorsoventral flexion (deg)')

# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('nu d_psi avg std bars - pub'), **FIGOPT)

# %% SI Figure: EXPLAINED VARIANCE IN FIRST TWO MODES

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

sns.swarmplot(x='Snake ID', y='theta_var0', data=dfp, ax=ax1, color='b')
sns.swarmplot(x='Snake ID', y='theta_var1', data=dfp, ax=ax1, color='r')

sns.swarmplot(x='Snake ID', y='psi_var0', data=dfp, ax=ax2, color='b')
sns.swarmplot(x='Snake ID', y='psi_var1', data=dfp, ax=ax2, color='r')

ax1.set_title('Horizontal wave', fontsize=17)
ax2.set_title('Vertical wave', fontsize=17)
ax1.grid(True)
ax2.grid(True)
ax1.set_ylim(0.45, 1.025)
ax1.set_ylabel('Explained variance fraction')
ax2.set_ylabel('')
sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('cod explained variance'), **FIGOPT)

# %% SI Figure: MODES REQUIRED TO CAPTURE 95% OF THE VARIANCE

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

sns.swarmplot(x='Snake ID', y='theta_nmodes', data=dfp, ax=ax1, color='b')

sns.swarmplot(x='Snake ID', y='psi_nmodes', data=dfp, ax=ax2, color='b')

ax1.set_title('Horizontal wave', fontsize=17)
ax2.set_title('Vertical wave', fontsize=17)
ax1.grid(True)
ax2.grid(True)
ax1.set_ylabel('Modes needed to capture\n95% of variance')
ax2.set_ylabel('')
sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('cod required modes'), **FIGOPT)