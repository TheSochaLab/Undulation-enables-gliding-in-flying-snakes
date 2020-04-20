# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:20:48 2015

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42}
sns.set('notebook', 'ticks', font_scale=1.35, rc=rc)
bmap = sns.color_palette()

FIG = '../Figures/s_xsection/{}.pdf'
FIGPNG = '../Figures/s_xsection/{}.png'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %% FIGURE SI: Plot the snake cross-section (small multiples)

def rotate_xc(xy, aoa):
    th = np.deg2rad(aoa)
    Rth = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    return np.dot(Rth, xy.T).T


def rv(xy, aoa):
    th = np.deg2rad(aoa)
    Rth = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    return np.dot(Rth, xy).T.flatten()

foil = np.genfromtxt('../Data/Xsection/snake0.004.bdy.txt', skip_header=1)
foil -= foil.mean(axis=0)
foil = rotate_xc(foil, 0)
xx, yy = foil.T

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 6.5))
((ax1, ax2), (ax3, ax4)) = axs
for ax in axs.flatten():
    ax.fill(xx, yy, color='gray', alpha=.25)
    ax.arrow(0, 0, 0, .25, head_width=.02, fc=bmap[0], ec=bmap[0], lw=1.5)
    ax.arrow(0, 0, .25, 0, head_width=.02, fc=bmap[0], ec=bmap[0], lw=1.5)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-.6, .6)
    ax.set_ylim(-.6, .6)

v = np.array([[1], [0]]) * .3
p1a, p1b = rv(v, -30)
p2a, p2b = rv(v, 30)
p3a, p3b = rv(v, 140)
p4a, p4b = rv(v, -140)
kwargs = dict(head_width=.03, fc='gray', ec='gray', lw=1.5)
ax1.arrow(0, 0, p1a, p1b, **kwargs)
ax2.arrow(0, 0, p2a, p2b, **kwargs)
ax3.arrow(0, 0, p3a, p3b, **kwargs)
ax4.arrow(0, 0, p4a, p4b, **kwargs)
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format("foils_2x2"), **FIGOPT)


# %% Extra figures below


# %% Paper: plot the snake cross-section

def rotate_xc(xy, aoa):
    th = np.deg2rad(aoa)
    Rth = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    return np.dot(Rth, xy.T).T

foil = np.genfromtxt('../Data/Xsection/snake0.004.bdy.txt', skip_header=1)
foil -= foil.mean(axis=0)
foil = rotate_xc(foil, 0)
xx, yy = foil.T

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=.5)
ax.axhline(0, color='gray', lw=.5)
ax.fill(xx, yy, color='gray', alpha=.5)
ax.arrow(0, 0, 0, .3, head_width=.03, fc=bmap[1], ec=bmap[1], lw=1.5)
ax.arrow(0, 0, .3, 0, head_width=.03, fc=bmap[2], ec=bmap[2], lw=1.5)

ax.arrow(0, 0, .3, .15, head_width=.02, fc='black', ec='black', lw=1)
ax.arrow(0, 0, .15, -.3, head_width=.02, fc='black', ec='black', lw=1)
ax.arrow(0, 0, -.3, .15, head_width=.02, fc='black', ec='black', lw=1)
ax.arrow(0, 0, -.15, -.3, head_width=.02, fc='black', ec='black', lw=1)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
ax.set_xlim(-.6, .6)
ax.set_ylim(-.6, .6)
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(FIG.format("foil_single"), **FIGOPT)


# %% Opening angle of the airfoil

left_foil = foil[foil[:, 0] <= 0]
right_foil = foil[foil[:, 0] >= 0]

left_idx = left_foil[:, 1].argmin()
right_idx = right_foil[:, 1].argmin()
peak_idx = foil[:, 1].argmax()

fig, ax = plt.subplots()
#ax.plot(left_foil[:, 0], left_foil[:, 1])
#ax.plot(right_foil[:, 0], right_foil[:, 1])
ax.fill(xx, yy, color='gray', alpha=.5)
ax.plot(left_foil[left_idx, 0], left_foil[left_idx, 1], 'o', c=bmap[0])
ax.plot(right_foil[right_idx, 0], right_foil[right_idx, 1], 'o', c=bmap[1])
ax.plot(0, foil[peak_idx, 1], 'o', c=bmap[2])
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
ax.set_xlim(-.6, .6)
ax.set_ylim(-.6, .6)
sns.despine()
fig.set_tight_layout(True)

# angle of opening
dy_left = left_foil[left_idx, 1] - foil[peak_idx, 1]
dy_right = right_foil[right_idx, 1] - foil[peak_idx, 1]
ang_left = 90 - np.abs(np.rad2deg(np.arctan(dy_left / left_foil[left_idx, 0])))
ang_right = 90 - np.abs(np.rad2deg(np.arctan(dy_right / right_foil[right_idx, 0])))

#ang_right
#Out[166]: 51.300649327220107
#
#In [167]: ang_left
#Out[167]: 51.325950190913005
#
#In [168]: ang_left + ang_right
#Out[168]: 102.6265995181331

note = 'Total opening angle of the snake cross-section: {0:.2f} deg'
print(note.format(ang_left + ang_right))
# Total opening angle of the snake cross-section: 102.62 deg