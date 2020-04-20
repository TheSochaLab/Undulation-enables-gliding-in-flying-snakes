# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:45:08 2016

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
import pandas as pd

from glob import glob

import seaborn as sns

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_dist_chord/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %% CHORD DATA

base_folder = '../Data/Snake silhouettes/'
processed_folder = base_folder + 'Processed/'
meta = pd.read_csv(base_folder + 'selected_chord_images.csv')
files = sorted(glob(processed_folder + '*.csv'))

# extract dataframe and combine into a dictionary
keys = ['44a', '44b', '33a', '33b', '40a', '40b']
snakes = ['44', '33', '40']
d = dict()
for i, fname in enumerate(files):
    df = pd.read_csv(fname, index_col=0)
    df.columns = ['s', 'body', 'tail']
    d[keys[i]] = df

# arc length parameter
s = df['s'].values

# separate into the body and tail
body = np.zeros((len(s), 6))
tail = np.zeros((len(s), 6))
for i, key in enumerate(keys):
    b, t = d[key]['body'], d[key]['tail']
    body[:, i] = b
    tail[:, i] = t

# average two measurments per snake
body_avg = np.zeros((len(s), 3))
tail_avg = np.zeros((len(s), 3))
for i, snake in enumerate(snakes):
    ba, ta = d[snake + 'a']['body'], d[snake + 'a']['tail']
    bb, tb = d[snake + 'b']['body'], d[snake + 'b']['tail']
    body_avg[:, i] = (ba + bb) / 2
    tail_avg[:, i] = (ta + tb) / 2


# %% Combine the body and tail segments into one array

VTL_to_SVL = .35  # we know this from the mass distribution measurements
s_chord = np.r_[s, s[-1] + VTL_to_SVL * s[1:]]  # [0, 1.35]

# body chord normalized by SVL, which is good
# tail chord normalized by VTL, but convert to SVL
chord = np.r_[body, tail[1:] * VTL_to_SVL]
chord_raw = chord.copy()  # this will not be corrected later on

chord_avg = np.r_[body_avg, tail_avg[1:] * VTL_to_SVL]
chord_avg_raw = chord_avg.copy()

chord_avg_max = chord_avg.max(axis=0)


# %% Plot the uncorrected chord lengths

marker = ['-o', '-o', '-s', '-s', '-^', '-^']
colors = [bmap[0], bmap[0], bmap[1], bmap[1], bmap[2], bmap[2]]
labels = ['44', '44', '33', '33', '40', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
#ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(6):
    if i % 2:
        label = 'snake {0}'.format(labels[i])
    else:
        label = ''
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_raw[:, i], marker[i], **kwargs)

ax.legend(loc='upper right')
ax.set_xlabel('Length (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('1_uncorrected_chord'), **FIGOPT)


# %% Plot the uncorrected chord lengths, average for each animal

marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['44', '33', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
#ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('Length (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('2_uncorrected_avg'), **FIGOPT)


# %% Fit a parabola to the average chord width for each snake

from scipy.optimize import curve_fit

def para_fit(x, *params):
    # http://tutorial.math.lamar.edu/Classes/Alg/Parabolas.aspx
    h, k, a = params
    return a * (x - h)**2 + k

def para_fit_old(x, *params):
    h, k, p = params
    return (x - h)**2 / (4 * p) + k

# stack all of the average data together
s_chord_para = np.r_[s, s, s]
chord_para = body_avg.T.flatten()

# initial guess
#hkp0_chord = (s_chord_para.mean(), chord_para.max(), -.001)
hka0_chord = (s_chord_para.mean(), chord_para.max(), -1)

# optimal values to fit a paraboa
popt_chord, pcov_chord = curve_fit(para_fit, s_chord_para, chord_para,
                                   hka0_chord)
# rms errors in %total mass
chord_fit_check = para_fit(s_chord_para, *popt_chord)
err_chord = np.sqrt(np.mean((chord_fit_check - chord_para)**2))  # * 100


# %% Fit the parabola to the chord parabola at many points

s_chord_fit = np.linspace(0, 1, 1000)
chord_fit_uncorrected = para_fit(s_chord_fit, *popt_chord)


# %%

marker = ['-o', '-o', '-s', '-s', '-^', '-^']
colors = [bmap[0], bmap[0], bmap[1], bmap[1], bmap[2], bmap[2]]
labels = ['44', '44', '33', '33', '40', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
#ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(6):
    if i % 2:
        label = 'snake {0}'.format(labels[i])
    else:
        label = ''
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_raw[:, i], marker[i], **kwargs)

ax.plot(100 * s_chord_fit, 100 * chord_fit_uncorrected, 'k', lw=3,
        label='Fit')

ax.legend(loc='upper right', handlelength=1.)
ax.set_xlabel('Length (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('3_uncorrected_parabola'), **FIGOPT)


# %% Correct for bias in image processing

# for some reason, the image process consistently under estimates the
# chord. see m_morphometrics, but from the chord that jake has measured for
# given SVLs, we need to boost our values about .13
# this was found from the slope of the parabolic fit and its max chord
# for different legnths vs. a linear fit from jake's data


# there is a bias in the image processing, in which we underestimate the
# chord with; correct for this
# from linear regression of chord vs. svl: slope = 0.028951012334765807
chord_vs_svl_slope = 0.028951012334765807
chord_correction = chord_vs_svl_slope / popt_chord[1]

chord = chord_raw * chord_correction
chord_avg = chord_avg * chord_correction

popt_chord[1] = popt_chord[1] * chord_correction


# %% Average and interpolate for the csv file

# after correcting the length, average all of trials together
chord_avg_all = chord_avg.mean(axis=1)

# make a data array to save
s_chord_save = np.linspace(0, 1.35, 136)  # 1% units
chord_avg_save = np.interp(s_chord_save, s_chord, chord_avg_all)

columns = ['arc length (SVL)', 'length (SVL)']
df = pd.DataFrame(data=np.c_[s_chord_save, chord_avg_save], columns=columns)

data_name = base_folder + 'snake_width.csv'
#df.to_csv(data_name)


# %% Plot the average chord length

marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['44', '33', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)
#    ax.plot(100 * s_chord, 100 * chord[:, i], marker[i], **kwargs)

ax.plot(100 * s_chord_save, 100 * chord_avg_save, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('Length (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('4_corrected_final'), **FIGOPT)


# %% Fit the parabola to the chord parabola at many points

s_chord_fit = np.linspace(0, 1, 1000)
chord_fit = para_fit(s_chord_fit, *popt_chord)


# %% Plot all the chord lengths and parabolic fit

marker = ['-o', '-o', '-s', '-s', '-^', '-^']
colors = [bmap[0], bmap[0], bmap[1], bmap[1], bmap[2], bmap[2]]
labels = ['44', '44', '33', '33', '40', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(6):
    b, t = d[key]['body'], d[key]['tail']

    if i % 2:
        label = 'snake {0}'.format(labels[i])
    else:
        label = ''
    kwargs = dict(c=colors[i], label=label, ms=3)
    ax.plot(100 * s_chord, 100 * chord[:, i], marker[i], **kwargs)

ax.plot(100 * s_chord_fit, 100 * chord_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('Length (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('5_corrected_parabola'), **FIGOPT)


# %% FIGURE in SI: Plot the corrected chord lengths, average for each animal

marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['44', '33', '40']

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)
#    ax.plot(100 * s_chord, 100 * chord[:, i], marker[i], **kwargs)

ax.plot(100 * s_chord_fit, 100 * chord_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Width (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format('6_average_snake_corrected_parabola'), **FIGOPT)