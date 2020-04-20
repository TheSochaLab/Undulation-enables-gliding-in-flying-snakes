# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:31:31 2016

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
from scipy.stats import linregress

from glob import glob

np.set_printoptions(suppress=True)

import seaborn as sns
rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=13/11, color_codes=True, rc=rc)
bmap = sns.color_palette()


# %% Where to save figures

FIG = '../Figures/s_morphometrics/{}.pdf'
FIGOPT = {'transparent': True}
SAVEFIG = False


# %%

def MLE_log(x, y):
    """Maximum likelihood estimate for regression of log data.

    See Brodziak (2012) "Fitting Length-Weight Relationships with Linear Regression
    Using the Log-Transformed Allometric Model with Bias-Correction"

    y = ax^c -> log(y) = b0 + b1 * log(x) + eps

    Parameters
    ----------
    x : array, independent variable in regular units
    y : array, dependent variable in regular units

    Returns
    -------
    B : array, size(2)
        log(y) = B[0] + B[1] * log(x)
    P : array, size(2)
        y = P[0] * x^P[1]
    R2 :
        coefficient of determination

    Notes
    -----
    B[1] = P[1] (exponent parameters)

    We also calculate the std of P, which can be used for a confidence interval
    """

    n = len(x)
    logx, logy = np.log(x), np.log(y)  # use natural logarithm
    E_logx, E_logy = np.mean(logx), np.mean(logy)

    # intercept
    b1_num = np.sum((logx - E_logx) * (logy - E_logy))
    b1_den = np.sum((logx - E_logx)**2)
    b1 = b1_num / b1_den

    # intercept
    b0 = E_logy - b1 * E_logx

    # slope
    eps = logy - (b0 + b1 * logx)

    # bias-corrected maximum likelihood estimate of variance
    var = 1 / (n - 2) * np.sum(eps**2)

    # variance and standard deviations
    var_b1 = var / np.sum((logx - E_logx)**2)
    var_b0 = (var * np.sum(logx**2)) / (n * np.sum((logx - E_logx)**2))
    std_b1 = np.sqrt(var_b1)
    std_b0 = np.sqrt(var_b0)

    # MLE of exponent parameter
    c = b1
    std_c = std_b1

    # bias-corrected MLE scaling parameter
    a = np.e**b0 * np.e**(var / 2)
    std_a = np.e**(std_b0) * np.e**(var / 2)

    # coefficient of determination
    R2 = 1 - np.sum(eps**2) / np.sum((logy - E_logy)**2)

    B = np.r_[b0, b1]
    P = np.r_[a, c]

    return B, P, R2


# %% MORPHOMETRICS


# %% Socha (2005): length, chord, mass, wing loading

d = pd.read_csv('../Data/Morphometrics/Socha2005_morphometrics.csv')

# chord vs. svl does better with this
ix = d['Common name'] == 'flying snake'  # 17 snakes

svl = d[u'SVL snake (cm)'][ix].values
chord = d[u'Chord length (cm)'][ix].values
chord_mm = chord * 10
mass = d[u'Mass (g)'][ix].values
Ws = d[u'Wing loading (N/m^2)'][ix].values
avg_den = mass / svl * 100  # mg/mm

# c = (m * g) / (Ws * SVL)
chord_from_Ws = (mass / 1000 * 9.81) / (Ws * svl / 100) * 100

# log transform the variables
log_svl = np.log(svl)
log_chord = np.log(chord)
log_chord_mm = np.log(chord_mm)
log_chord_from_Ws = np.log(chord_from_Ws)
log_mass = np.log(mass)
log_Ws = np.log(Ws)
log_avg_den = np.log(avg_den)


# %% Predicted chord length from wing loading

fig, ax = plt.subplots()
ax.plot(svl, chord_from_Ws, 'o', label='Avg. chord from Ws')
ax.plot(svl, chord, 'o', label='Chord from column S')
ax.legend(loc='best')
ax.set_xlabel('svl (cm)')
ax.set_ylabel('chord (cm)')
sns.despine()
fig.set_tight_layout(True)


# %% Cube: mass and mean length

d = pd.read_csv('../Data/Morphometrics/Cube_morphometrics.csv')

# select out only C. paradisi (30's are ornanta)
ix = d['Snake'] > 50

mass_cube = d[u'Mass (g)'][ix].values
svl_cube = d[u'Avg SVL (cm)'][ix].values
avg_den_cube = mass_cube / svl_cube

log_mass_cube = np.log(mass_cube)
log_svl_cube = np.log(svl_cube)
log_avg_den_cube = np.log(avg_den_cube)


# %% Combine Cube and Socha (2005) data

svl_comb = np.r_[svl, svl_cube]
mass_comb = np.r_[mass, mass_cube]
avg_den_comb = np.r_[avg_den, avg_den_cube]

log_svl_comb = np.log(svl_comb)
log_mass_comb = np.log(mass_comb)
log_avg_den_comb = np.log(avg_den_comb)


# %% Regressions of the morphometrics data

#svl_rng = np.r_[np.e**3.4, 92.5]  # cm from Socha (2005): 31 - 86.5
#svl_rng = np.r_[27.5, 90]  # cm from Socha (2005): 31 - 86.5
#svl_rng = np.r_[30, 90]  # cm from Socha (2005): 31 - 86.5
svl_rng = np.r_[np.e**3.4, 90]  # cm from Socha (2005): 31 - 86.5
log_svl_rng = np.log(svl_rng)

#mass_rng = np.r_[3, 90]  # g
#log_mass_rng = np.log(mass_rng)  # np.r_[1, 4.5]
log_mass_rng = np.r_[1, 4.5]
mass_rng = np.e**log_mass_rng

# linear fit of chord vs. SVL
chord_vs_svl = linregress(svl, chord)
chord_vs_svl_fit = chord_vs_svl.slope * svl_rng + chord_vs_svl.intercept

chord_from_Ws_vs_svl = linregress(svl, chord_from_Ws)
chord_from_Ws_vs_svl_fit = chord_from_Ws_vs_svl.slope * svl_rng + chord_from_Ws_vs_svl.intercept

# linear fit of log variables vs. MASS
# X_dependent_mass
B_Ws_mass, P_Ws_mass, R2_Ws_mass = MLE_log(mass, Ws)
B_svl_mass, P_svl_mass, R2_svl_mass = MLE_log(mass, svl)
B_chord_mass, P_chord_mass, R2_chord_mm_mass = MLE_log(mass, chord)
B_chord_mm_mass, P_chord_mm_mass, R2_chord_mm_mass = MLE_log(mass, chord_mm)
B_avg_den_mass, P_avg_den_mass, R2_avg_den_mass = MLE_log(mass, avg_den)

# fit model of log variables vs. MASS
Ws_mass_fit = B_Ws_mass[0] + B_Ws_mass[1] * log_mass_rng
svl_mass_fit = B_svl_mass[0] + B_svl_mass[1] * log_mass_rng
chord_mass_fit = B_chord_mass[0] + B_chord_mass[1] * log_mass_rng
chord_mm_mass_fit = B_chord_mm_mass[0] + B_chord_mm_mass[1] * log_mass_rng
avg_den_mass_fit = B_avg_den_mass[0] + B_avg_den_mass[1] * log_mass_rng

# linear fit of log variables vs. SVL
B_mass_svl, P_mass_svl, R2_mass_svl = MLE_log(svl, mass)
B_Ws_svl, P_Ws_svl, R2_Ws_svl = MLE_log(svl, Ws)
B_chord_svl, P_chord_svl, R2_chord_svl = MLE_log(svl, chord)
B_chord_mm_svl, P_chord_mm_svl, R2_chord_mm_svl = MLE_log(svl, chord_mm)
B_chord_from_Ws_svl, P_chord_from_Ws_svl, R2_chord_from_Ws_svl = MLE_log(svl, chord_from_Ws)
B_avg_den_svl, P_avg_den_svl, R2_avg_den_svl = MLE_log(svl, avg_den)

# fit model of log variables vs. SVL
mass_svl_fit = B_mass_svl[0] + B_mass_svl[1] * log_svl_rng
Ws_svl_fit = B_Ws_svl[0] + B_Ws_svl[1] * log_svl_rng
chord_svl_fit = B_chord_svl[0] + B_chord_svl[1] * log_svl_rng
chord_mm_svl_fit = B_chord_mm_svl[0] + B_chord_mm_svl[1] * log_svl_rng
chord_from_Ws_svl_fit = B_chord_from_Ws_svl[0] + B_chord_from_Ws_svl[1] * log_svl_rng
avg_den_svl_fit = B_avg_den_svl[0] + B_avg_den_svl[1] * log_svl_rng


# %% Plot log data vs. log mass

fig, ax = plt.subplots(figsize=(4, 4))

# ax.plot(log_mass_cube, log_svl_cube, 'ks')
ax.plot(log_mass, log_svl, 'bo', label='log SVL (cm)')
ax.plot(log_mass, log_Ws, 'g^', label=r'log $W_S$ (N/m$^\mathrm{2}$)')
#ax.plot(log_mass, log_chord, 'rs', label='log c (cm)')
ax.plot(log_mass, log_chord_mm, 'rs', label='log c (mm)')
ax.plot(log_mass, log_avg_den, 'mD', label=r'log $\bar{\rho}$ (mg/mm)')

ax.plot(log_mass_rng, svl_mass_fit, 'b-')
ax.plot(log_mass_rng, Ws_mass_fit, 'g-')
#ax.plot(log_mass_rng, chord_mass_fit, 'r-')
ax.plot(log_mass_rng, chord_mm_mass_fit, 'r-')
ax.plot(log_mass_rng, avg_den_mass_fit, 'm-')

ax.legend(loc='upper left', fontsize=11, handletextpad=0, borderaxespad=0)
ax.set_xlabel('log mass (g)')
ax.set_xlim(log_mass_rng)
ax.set_ylim(2, 5)
ax.set_yticks([2, 3, 4, 5])

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('data vs log mass'), **FIGOPT)


# %% Plot log data vs. log svl

fig, ax = plt.subplots(figsize=(4, 4))

# ax.plot(log_svl_cube, log_mass_cube, 'ks')
ax.plot(log_svl, log_mass, 'bo', label='log m (g)')
ax.plot(log_svl, log_Ws, 'g^', label=r'log $W_S$ (N/m$^\mathrm{2}$)')
#ax.plot(log_svl, log_chord, 'rs', label='log c (cm)')
ax.plot(log_svl, log_chord_mm, 'rs', label='log c (mm)')
ax.plot(log_svl, log_avg_den, 'mD', label=r'log $\bar{\rho}$ (mg/mm)')

ax.plot(log_svl_rng, mass_svl_fit, 'b-')
ax.plot(log_svl_rng, Ws_svl_fit, 'g-')
#ax.plot(log_svl_rng, chord_svl_fit, 'r-')
ax.plot(log_svl_rng, chord_mm_svl_fit, 'r-')
ax.plot(log_svl_rng, avg_den_svl_fit, 'm-')

ax.legend(loc='upper left', fontsize=11, handletextpad=0, borderaxespad=0)
ax.set_xlabel('log SVL (cm)')
ax.set_xlim(log_svl_rng)
ax.set_ylim(1, 5)
ax.set_yticks([1, 2, 3, 4, 5])

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('1 - data vs log SVL'), **FIGOPT)


# %% Plot log data vs. log svl with two subplots

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

ax1.plot(log_svl, log_mass, 'bo', label='log m (g)')
ax2.plot(log_svl, log_Ws, 'g^', label=r'log $W_S$ (N/m$^\mathrm{2}$)')
ax1.plot(log_svl, log_chord_mm, 'rs', label='log c (mm)')
ax2.plot(log_svl, log_avg_den, 'mD', label=r'log $\bar{\rho}$ (mg/mm)')

ax1.plot(log_svl_rng, mass_svl_fit, 'b-')
ax2.plot(log_svl_rng, Ws_svl_fit, 'g-')
ax1.plot(log_svl_rng, chord_mm_svl_fit, 'r-')
ax2.plot(log_svl_rng, avg_den_svl_fit, 'm-')

ax1.legend(loc='upper left', handletextpad=0, borderaxespad=0)
ax2.legend(loc='upper left', handletextpad=0, borderaxespad=0)
ax1.set_xlabel('log SVL (cm)')
ax2.set_xlabel('log SVL (cm)')
ax1.set_xlim(log_svl_rng)
ax1.set_ylim(1, 5)
ax1.set_yticks([1, 2, 3, 4, 5])

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('data vs log SVL subplots'), **FIGOPT)


# %% Chord, mass and Ws vs SVL

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.plot(svl, chord, 'rs', label='chord (cm)')
ax1.plot(svl_rng, chord_vs_svl_fit, 'gray')
ax1.plot(svl, chord_from_Ws, 'ms', label='chord from Ws (cm)')
ax1.plot(svl_rng, np.e**chord_from_Ws_svl_fit, 'gray')
ax1.set_xlim(svl_rng)
ax1.set_xlabel('SVL (cm)', fontsize='small')
ax1.legend(loc='upper left', fontsize='small',
           handletextpad=0, borderaxespad=0)
#ax1.set_ylabel('chord (cm)')
#ax2.plot(log_svl_cube, log_mass_cube, 'ks')
ax2.plot(log_svl, log_mass, 'bo', label='log m (g)')
ax2.plot(log_svl, log_Ws, 'g^', label=r'log $W_S$ (N/m$^\mathrm{2}$)')
ax2.plot(log_svl_rng, mass_svl_fit, 'gray')
ax2.plot(log_svl_rng, Ws_svl_fit, 'gray')
ax2.legend(loc='upper left', fontsize='small',
           handletextpad=0, borderaxespad=0)
ax2.set_xlabel('log SVL (cm)', fontsize='small')
ax1.set_xlim(svl_rng)
ax2.set_xlim(log_svl_rng)
sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('data vs SVL'), **FIGOPT)


# %% How do R2 compare b/n mass fit and svl fit

print('We care about the first number more, R2 with SVL as dependent var')
print R2_mass_svl, R2_svl_mass
print R2_Ws_svl, R2_Ws_mass
print R2_chord_svl, R2_chord_mm_mass
print R2_avg_den_svl, R2_avg_den_mass


# %% CHORD DATA

base_folder = '../Data/Morphometrics/Snake silhouettes/'
processed_folder = base_folder + 'Processed/'
meta = pd.read_csv(base_folder + 'selected_chord_images.csv')
files = sorted(glob(processed_folder + '*.csv'))

# extract dataframe and combine into a dictionary
keys = ['33a', '33b', '40a', '40b', '44a', '44b']
snakes = ['33', '40', '44']
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

chord_avg_max = chord_avg.max(axis=0)


# %% How to correct the raw data using mean chord witdh

chord_body_avg = body_avg.mean(axis=0).mean()




# %% Fit a parabola to the average chord width for each snake

from scipy.optimize import curve_fit

def para_fit(x, *params):
    h, k, p = params
    return (x - h)**2 / (4 * p) + k

# stack all of the average data together
s_chord_para = np.r_[s, s, s]
chord_para = body_avg.T.flatten()

# initial guess
hkp0_chord = (s_chord_para.mean(), chord_para.max(), -.001)

# optimal values to fit a paraboa
popt_chord, pcov_chord = curve_fit(para_fit, s_chord_para, chord_para,
                                   hkp0_chord)
# rms errors in %total mass
chord_fit_check = para_fit(s_chord_para, *popt_chord)
err_chord = np.sqrt(np.mean((chord_fit_check - chord_para)**2))  # * 100


# %% Correct for bias in image processing

# for some reason, the image process consistently under estimates the
# chord. see m_morphometrics, but from the chord that jake has measured for
# given SVLs, we need to boost our values about .13
# this was found from the slope of the parabolic fit and its max chord
# for different legnths vs. a linear fit from jake's data


# there is a bias in the image processing, in which we underestimate the
# chord with; correct for this
# from linear regression of chord vs. svl: slope = 0.028951012334765807
# chord_vs_svl.slope = 0.028951012334765807
chord_vs_svl_slope = 0.028951012334765807
chord_correction = chord_vs_svl_slope / popt_chord[1]

# correct the image processing data
chord = chord * chord_correction
chord_avg = chord_avg * chord_correction

# correct the parablic fit peak location
popt_chord[1] = popt_chord[1] * chord_correction


# %% Fit the parabola to the chord parabola at many points

s_chord_fit = np.linspace(0, 1, 1000)
chord_fit = para_fit(s_chord_fit, *popt_chord)


# %% Linear interpolation of average for plotting

s_chord_interp = np.linspace(0, 1.35, 136)
chord_avg_interp = np.interp(s_chord_interp, s_chord, chord_avg.mean(axis=1))


# %% Plot the uncorrected chord lengths

marker = ['-o', '-o', '-s', '-s', '-^', '-^']
colors = [bmap[0], bmap[0], bmap[1], bmap[1], bmap[2], bmap[2]]
labels = ['33', '33', '40', '40', '44', '44']

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(6):
    if i % 2:
        label = 'snake {0}'.format(labels[i])
    else:
        label = ''
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_raw[:, i], marker[i], **kwargs)

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('length (%SVL)')
ax.set_ylabel('chord (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

fig.savefig(FIG.format('c(s) raw data'), **FIGOPT)


# %% Plot the uncorrected chord lengths, average for each animal

marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['33', '40', '44']

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('length (%SVL)')
ax.set_ylabel('chord (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('c(s) avg data'), **FIGOPT)


# %% Plot all the chord lengths and parabolic fit

marker = ['-o', '-o', '-s', '-s', '-^', '-^']
colors = [bmap[0], bmap[0], bmap[1], bmap[1], bmap[2], bmap[2]]
labels = ['33', '33', '40', '40', '44', '44']

fig, ax = plt.subplots(figsize=(6, 3))
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
ax.set_xlabel('length (%SVL)')
ax.set_ylabel('chord (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('c(s) raw data with fit'), **FIGOPT)


# %% Plot the uncorrected chord lengths, average for each animal with parabola

marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['33', '40', '44']

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 2.25], color='gray', lw=1)

for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=3)

    ax.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)

ax.plot(100 * s_chord_fit, 100 * chord_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xlabel('length (%SVL)')
ax.set_ylabel('chord (%SVL)')
ax.set_xlim(-1.35, 135)
ax.set_ylim(0, 3.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks([0, 1, 2, 3])

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('c(s) avg data with fit'), **FIGOPT)


# %% DENSITY DATA

# %% Load in mass measurements

base = '../Data/Morphometrics/Mass distribution/mass-distribution_snake-{}.csv'

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

def para_fit(x, *params):
    h, k, p = params
    return (x - h)**2 / (4 * p) + k


# initial guess
hkp0_head = (s_head.mean(), rho_non_head.max(), -.0001)
hkp0_body = (s_body.mean(), rho_non_body.max(), -.0001)

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


# %% DENSITY PLOTS

# %% Linear mass density

snakes = [80, 87, 92]
markers = ['o', '^', 's']

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 50], color='gray', lw=1)

for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho = d['rho']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, rho, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_interp, rho_avg, 'k', lw=3)

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_ylabel(r'density (mg/mm)')
ax.set_xlabel('length (%SVL)')
ax.set_xlim(-1.35, 135)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('rho(s) mg per mm raw data'), **FIGOPT)


# %% Normalized mass density

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 1.2], color='gray', lw=1)

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

ax.plot(100 * s_interp, 100 * rho_non_avg, 'k', lw=3)

ax.legend(loc='upper right')
#ax.set_ylabel(r'density ((g/cm) / (m$_\mathsf{tot}$/SVL))')
ax.set_ylabel(r'density (%$\rho_\mathsf{mean}$)')
#ax.set_ylabel(r'density (%(m$_\mathsf{tot}$/SVL)')
ax.set_xlabel('length (%SVL)')
ax.set_xlim(-1.35, 135)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('rho_norm(s) raw data'), **FIGOPT)


# %% Data used for parabolic fit

fig, ax = plt.subplots(figsize=(6, 3))

ax.plot(100 * s_head, rho_non_head, 'o')
ax.plot(100 * s_body, rho_non_body, 'o')
ax.plot(100 * s_fit_head, rho_non_fit_head, 'k', lw=3)
ax.plot(100 * s_fit_body, rho_non_fit_body, 'k', lw=3)
ax.set_ylabel(r'density (-)')
ax.set_xlabel('%SVL')
#ax.set_ylabel(r'mass (%m$_\mathsf{total}$)')
ax.margins(x=.01)
sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('data for parabolic fit'), **FIGOPT)


# %% Normalized density with parabola

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([100, 100], [0, 1.2], color='gray', lw=1)

snakes = [80, 87, 92]
markers = ['o', '^', 's']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax.plot(100 * s / SVL, rho_non, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_fit, rho_non_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_ylabel('density (-)')
ax.set_xlabel('length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('rho_norm(s) with fit'), **FIGOPT)


# %% Density distribution with parabola

fig, ax = plt.subplots(figsize=(6, 3))
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
          handlelength=1.5)
ax.set_ylabel(r'density (mg/mm)')
ax.set_xlabel('length (%SVL)')
ax.margins(x=.01)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('rho(s) mg per mm with fit'), **FIGOPT)


# %% Normalized density with parabola as %rho_bar

fig, ax = plt.subplots(figsize=(6, 3))
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

    ax.plot(100 * s / SVL, 100 * rho_non, mk, ms=6, c=bmap[i], label=label)

ax.plot(100 * s_fit, 100 * rho_non_fit, 'k', lw=3, label='fit')

ax.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax.set_xticks([0, 25, 50, 75, 100, 135])
ax.set_yticks(np.arange(0, 180, 40))
ax.set_ylabel(r'density (%$\rho_\mathsf{mean}$)')
ax.set_xlabel('length (%SVL)')
ax.set_xlim(-1.35, 135)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('rho_non(s) with fit'), **FIGOPT)


# %% NORMALIZED PLOT FOR PAPER

# %% FIGURE for SI: Plot chord and density distributions on one plot

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5.125))

ax1.plot([100, 100], [0, 2.25], color='gray', lw=1)
ax2.plot([100, 100], [0, 115], color='gray', lw=1)

# plot the chord distribution
marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['33', '40', '44']
for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=4)
    ax1.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)

# plot the density distribution
snakes = [80, 87, 92]
markers = ['o', 's', '^']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax2.plot(100 * s / SVL, 100 * rho_non, mk, ms=4, c=bmap[i], label=label)

# plot the parabolic fits
ax1.plot(100 * s_chord_fit, 100 * chord_fit, 'k', lw=3, label='fit')
ax2.plot(100 * s_fit, 100 * rho_non_fit, 'k', lw=3, label='fit')

# adjust the chord plot
ax1.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax1.set_ylabel('chord (%SVL)')
ax1.set_xlim(-1.35, 135)
ax1.set_ylim(0, 3.5)
ax1.set_xticks([0, 25, 50, 75, 100, 135])
ax1.set_yticks([0, 1, 2, 3])

# adjust the density plot
ax2.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax2.set_xticks([0, 25, 50, 75, 100, 135])
ax2.set_yticks(np.arange(0, 180, 40))
#ax2.set_ylabel(r'density (%$\rho_\mathsf{mean}$)')
ax2.set_ylabel(r'density (%$\bar{\rho}$)')
ax2.set_xlabel('length (%SVL)')
ax2.set_xlim(-1.35, 135)

# align the ylabels
# http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots
ax1.yaxis.set_label_coords(-.1, .5)
ax2.yaxis.set_label_coords(-.1, .5)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('2 - c(s) and rho(s)'), **FIGOPT)


# %% Plot chord and density distributions on one plot (interp values)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5.125))

ax1.plot([100, 100], [0, 2.25], color='gray', lw=1)
ax2.plot([100, 100], [0, 115], color='gray', lw=1)

# plot the chord distribution
marker = ['-o', '-s', '-^']
colors = [bmap[0], bmap[1], bmap[2]]
labels = ['33', '40', '44']
for i in range(3):
    label = 'snake {0}'.format(labels[i])
    kwargs = dict(c=colors[i], label=label, ms=4)
    ax1.plot(100 * s_chord, 100 * chord_avg[:, i], marker[i], **kwargs)

# plot the density distribution
snakes = [80, 87, 92]
markers = ['o', 's', '^']
for i in np.arange(3):
    sn = snakes[i]
    d = data[sn]

    s, SVL = d['s_mm'], d['SVL']
    rho_non = d['rho_non']

    mk = markers[i] + '-'
    label = 'snake ' + str(sn)

    ax2.plot(100 * s / SVL, 100 * rho_non, mk, ms=4, c=bmap[i], label=label)

# plot the parabolic fits
ax1.plot(100 * s_chord_interp, 100 * chord_avg_interp, 'k', lw=3, label='average')
ax2.plot(100 * s_interp, 100 * rho_non_avg, 'k', lw=3, label='average')

# adjust the chord plot
ax1.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax1.set_ylabel('chord (%SVL)')
ax1.set_xlim(-1.35, 135)
ax1.set_ylim(0, 3.5)
ax1.set_xticks([0, 25, 50, 75, 100, 135])
ax1.set_yticks([0, 1, 2, 3])

# adjust the density plot
ax2.legend(loc='upper right', borderaxespad=0, handletextpad=.5,
          handlelength=1.5)
ax2.set_xticks([0, 25, 50, 75, 100, 135])
ax2.set_yticks(np.arange(0, 180, 40))
#ax2.set_ylabel(r'density (%$\rho_\mathsf{mean}$)')
ax2.set_ylabel(r'density (%$\bar{\rho}$)')
ax2.set_xlabel('length (%SVL)')
ax2.set_xlim(-1.35, 135)

# align the ylabels
# http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots
ax1.yaxis.set_label_coords(-.1, .5)
ax2.yaxis.set_label_coords(-.1, .5)

sns.despine()
fig.set_tight_layout(True)

if SAVEFIG:
    fig.savefig(FIG.format('3 - c(s) and rho(s)'), **FIGOPT)