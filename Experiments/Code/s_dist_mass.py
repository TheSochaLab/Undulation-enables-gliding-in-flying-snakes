# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:59:07 2015

%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

cd /Volumes/Yeaton_HD6/Code for Manuscripts/Undulation_confers_stability/Experiments/Code

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import cumtrapz

import seaborn as sns

#rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm'}
#sns.set('notebook', 'ticks', font_scale=1.5, color_codes=True, rc=rc)
#bmap = sns.color_palette()

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_dist_mass/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %% DENSITY DATA


# %% Load in mass measurements

base = '../Data/Mass distribution/mass-distribution_snake-{}.csv'

df80 = pd.read_csv(base.format(80), skiprows=5)
df87 = pd.read_csv(base.format(87), skiprows=5)
df92 = pd.read_csv(base.format(92), skiprows=5)


# %% Iterate through each snake, calcuate density

from scipy.integrate import cumtrapz

data = {}

for df, snake_id in zip([df80, df87, df92], [80, 87, 92]):

    # which segment contains the vent
    vent_idx = np.where(df['body part'] == 'vent')[0][0]

    # individual segment lengths
    seg_lengths = df['length (mm)'].values  # of individual segments

    # position along the body in mm
    s_mm = cumtrapz(seg_lengths, initial=0)

    # body and tail lengths
    SVL = s_mm[vent_idx]  # snout-vent length
    VTL = s_mm[-1] - SVL  # vent-tail length

    # normalized position
    s = s_mm / SVL

    # masses of each segment
    mass = df['mass (mg)'].values  # mg
    mtot = mass.sum()

    rho_bar = mtot / SVL

    # linear mass density
    rho = mass / seg_lengths  # mg / mm

    # this normalization isn't as good (don't line-up as well)
    # normalized (multiply my total mass to recover the distribution)
    # rho_norm2 = rho / rho.sum()

    # divide by total mass and SVL to make it non-dim density
    rho_non = rho / rho_bar

    # save the data
    snake = dict(vent_idx=vent_idx,
                 seg_lengths=seg_lengths, s_mm=s_mm, s=s,
                 SVL=SVL, VTL=VTL,
                 mass=mass, mtot=mtot,
                 rho=rho, rho_non=rho_non, rho_bar=rho_bar)

    data[snake_id] = snake


# %% Linear intrpolation of the density and non-dim density

# force the end to be at 1.35 * SVL
# 1.34843297975, 1.30863350668, 1.34291839703  # s[-1]
s_tail_tip = 1.35

# x to interpolate agains
npts = 136
s_interp = np.linspace(0, s_tail_tip, npts)
vent_idx_interp = 100

rho_interps = np.zeros((npts, 3))
rho_non_interps = np.zeros((npts, 3))
VTL_to_SVL = np.zeros(3)

for i, sn in enumerate([80, 87, 92]):
    d = data[sn]
    s = d['s']
    rho, rho_non = d['rho'], d['rho_non']

    rho_interps[:, i] = np.interp(s_interp, s, rho)
    rho_non_interps[:, i] = np.interp(s_interp, s, rho_non)

    VTL_to_SVL[i] = d['VTL'] / d['SVL']

rho_avg = rho_interps.mean(axis=1)
rho_non_avg = rho_non_interps.mean(axis=1)

VTL_to_SVL_avg = VTL_to_SVL.mean()

# add the data to the dictionary
data['s_interp'] = s_interp
data['rho_interps'] = rho_interps
data['rho_non_interps'] = rho_non_interps
data['rho_avg'] = rho_avg
data['rho_non_avg'] = rho_non_avg
data['VTL_to_SVL'] = VTL_to_SVL
data['VTL_to_SVL_avg'] = VTL_to_SVL_avg


# %% Save the average density values

columns = ['arc length (SVL)', 'density (rho/rho_bar)']
df = pd.DataFrame(data=np.c_[s_interp, rho_non_avg], columns=columns)

if False:
    save_folder = '../Data/Mass distribution/'
    data_name = save_folder + 'snake_density.csv'
    df.to_csv(data_name)


# %% Select out the head and body (trunk) to fit with a parabola

# fiddle with this to select different neck lengths
sneck = .04

# stack all of the data together, separated into head and body
s_head = np.array([])
s_body = np.array([])
rho_non_head = np.array([])
rho_non_body = np.array([])

for sn in [80, 87, 92]:
    d = data[sn]

    s, rho_non = d['s'], d['rho_non']

    idx_head = np.where(s <= sneck)[0]
    idx_body = np.where((s > sneck) & (s <= 1))[0]

    s_head = np.r_[s_head, s[idx_head]]
    rho_non_head = np.r_[rho_non_head, rho_non[idx_head]]

    s_body = np.r_[s_body, s[idx_body]]
    rho_non_body = np.r_[rho_non_body, rho_non[idx_body]]


# %% Parabolic fit to the RAW normalized densities

from scipy.optimize import curve_fit

def para_fit_old(x, *params):
    h, k, p = params
    return (x - h)**2 / (4 * p) + k


def para_fit(x, *params):
    # http://tutorial.math.lamar.edu/Classes/Alg/Parabolas.aspx
    h, k, a = params
    return a * (x - h)**2 + k


    # initial guess
hkp0_head = (s_head.mean(), rho_non_head.max(), -.0001)
hkp0_body = (s_body.mean(), rho_non_body.max(), -.0001)
#hkp0_head = (s_head.mean(), rho_non_head.max(), -1)
#hkp0_body = (s_body.mean(), rho_non_body.max(), -1)

# optimal values to fit a paraboa
popt_head, pcov_head = curve_fit(para_fit, s_head, rho_non_head, hkp0_head)
popt_body, pcov_body = curve_fit(para_fit, s_body, rho_non_body, hkp0_body)

# rms errors in %total mass
rho_head_fit_check = para_fit(s_head, *popt_head)
rho_body_fit_check = para_fit(s_body, *popt_body)
err_head = np.sqrt(np.mean((rho_head_fit_check - rho_non_head)**2))  # * 100
err_body = np.sqrt(np.mean((rho_body_fit_check - rho_non_body)**2))  # * 100


# %% Evaluate the fit at a high resolution for plotting

# cutoff for the "neck" to make parabolas meet
#sneck_fit = .036975
sneck_fit = .037

# fit lots of points for plotting
s_fit_head, ds_fit_head = np.linspace(0, sneck_fit, 100, retstep=True)
s_fit_body, ds_fit_body = np.linspace(sneck_fit, 1, 1000, retstep=True)
rho_non_fit_head = para_fit(s_fit_head, *popt_head)
rho_non_fit_body = para_fit(s_fit_body, *popt_body)

# combine the fit into one array
s_fit = np.r_[s_fit_head, s_fit_body]
rho_non_fit = np.r_[rho_non_fit_head, rho_non_fit_body]
rho_non_fit_bar = rho_non_fit.mean()
rho_non_fit_total = rho_non_fit.sum()


# %% PLOTS

# %% FIGURE in SI: Normalized density with parabola

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 100 * 1.2], color='gray', lw=1)

snakes = [80, 87, 92]
markers = ['o', '^', 's']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, 100 * rho_non, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_fit, 100 * rho_non_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5, fontsize='x-small')
ax.set_ylabel('Density (% mean density)')
ax.set_xlabel('Length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('4 data with parabolic fit'), **FIGOPT)


# %% Raw data

snakes = [80, 87, 92]
markers = ['o', '^', 's']

fig, ax = plt.subplots(figsize=(7, 3.5))
#ax.plot([100, 100], [0, 50], color='gray', lw=1)

for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s = d['s_mm'] / 10
    mass = d['mass'] / 1000

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(s, mass, mk, ms=6, mec=bmap[i], label=label)

#ax.plot(100 * s_interp, rho_avg, 'k', lw=3)

ax.legend(loc='upper right', fontsize='x-small')
ax.set_ylabel(r'Mass (g)')
ax.set_xlabel('Length (cm)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('0 raw data'), **FIGOPT)


# %% Raw linear mass density

snakes = [80, 87, 92]
markers = ['o', '^', 's']

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 50], color='gray', lw=1)

for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho = d['rho']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, rho, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_interp, rho_avg, 'k', lw=3, label='average')

ax.legend(loc='upper right', fontsize='x-small')
ax.set_ylabel(r'Density (mg/mm)')
ax.set_xlabel('Length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('1 raw density'), **FIGOPT)


# %% Normalized mass density

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 100 * 1.2], color='gray', lw=1)

snakes = [80, 87, 92]
markers = ['o', '^', 's']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, 100 * rho_non, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_interp, 100 * rho_non_avg, 'k', lw=3, label='average')

ax.legend(loc='upper right', fontsize='x-small')
#ax.set_ylabel(r'density ((g/cm) / (m$_\mathsf{tot}$/SVL))')
ax.set_ylabel(r'Density (% mean density)')
#ax.set_ylabel(r'density (%(m$_\mathsf{tot}$/SVL)')
ax.set_xlabel('Length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('2 normalized density'), **FIGOPT)


# %% Data used for parabolic fit

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(100 * s_head, 100 * rho_non_head, 'o', label='head')
ax.plot(100 * s_body, 100 * rho_non_body, 'o', label='body')
ax.plot(100 * s_fit_head, 100 * rho_non_fit_head, 'k', lw=3)
ax.plot(100 * s_fit_body, 100 * rho_non_fit_body, 'k', lw=3)
ax.legend(loc='best', fontsize='x-small')
ax.set_ylabel(r'Density (% mean density)')
ax.set_xlabel('Length (%SVL)')
#ax.set_ylabel(r'mass (%m$_\mathsf{total}$)')
ax.margins(x=.01)
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('3 data for parabolic fit'), **FIGOPT)


# %% Density distribution with parabola

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 50], color='gray', lw=1)

snakes = [80, 87, 92]
markers = ['o', '^', 's']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL, mtot = d['s_mm'], d['SVL'], d['mtot']
    rho, rho_bar = d['rho'], d['rho_bar']

    # parabolic fit back to dimensional units
    rho_fit_head = rho_non_fit_head * rho_bar
    rho_fit_body = rho_non_fit_body * rho_bar

    mk = markers[i]  # + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, rho, mk, ms=6, c=bmap[i], label=label)
    ax.plot(100 * s_fit_head, rho_fit_head, c=bmap[i])
    ax.plot(100 * s_fit_body, rho_fit_body, c=bmap[i])

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5, fontsize='x-small')
ax.set_ylabel(r'Density (mg/mm)')
ax.set_xlabel('Length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('5 parabolic fit on raw density'), **FIGOPT)


# %% Normalized density with parabola as %rho_bar

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 100], color='gray', lw=1)

snakes = [80, 87, 92]
markers = ['o', '^', 's']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

#    ax.plot(100 * s / SVL, 100 * rho_non, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_fit, 100 * rho_non_fit, 'k', lw=3, label='fit')
#ax.plot(100 * s_interp, 100 * rho_non_avg, 'om', lw=3, label='average')
ax.plot(100 * s_interp, 100 * rho_non_avg, 'om', mfc='none', mec='m',
        mew=2, label='average')
# ax.axhline(100 * rho_non_fit_bar, color='gray')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5, fontsize='x-small')
#xticks = ax.get_xticks()
#xticks = np.r_[xticks, 135]
#ax.set_xticks(xticks)
ax.set_xticks([0, 25, 50, 75, 100, 135])
#ax.set_ylabel(r'density (%$\bar{\rho}$)')
#ax.set_ylabel(r'Density (%$\rho_\mathsf{mean}$)')
ax.set_ylabel('Density (% mean density)')
#ax.set_ylabel(r'density (%mean density)')
ax.set_xlabel('Length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('6 fit and average'), **FIGOPT)
