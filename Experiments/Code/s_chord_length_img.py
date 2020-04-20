# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:16:25 2016

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

import seaborn as sns

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42}
sns.set('notebook', 'ticks', font_scale=1.5, rc=rc)
bmap = sns.color_palette()


# %%

import scipy.ndimage as ndi

from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.util import pad



def branch_point(skel):
    """Detect branch points in skeleton image.

    See: http://stackoverflow.com/a/19595646
    """

    from mahotas.morph import hitmiss

    xbranch0 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    xbranch1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    tbranch0 = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T
    tbranch3 = np.fliplr(tbranch2)
    tbranch4 = np.array([[1,0,1], [0,1,0], [1,0,0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)
    ybranch0 = np.array([[1,0,1], [0,1,0], [2,1,2]])
    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T
    ybranch3 = np.fliplr(ybranch2)
    ybranch4 = np.array([[0,1,2], [1,1,2], [2,2,1]])
    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)

    # ndi.binary_hit_or_miss
    xb0 = hitmiss(skel, xbranch0)
    xb1 = hitmiss(skel, xbranch1)
    tb0 = hitmiss(skel, tbranch0)
    tb1 = hitmiss(skel, tbranch1)
    tb2 = hitmiss(skel, tbranch2)
    tb3 = hitmiss(skel, tbranch3)
    tb4 = hitmiss(skel, tbranch4)
    tb5 = hitmiss(skel, tbranch5)
    tb6 = hitmiss(skel, tbranch6)
    tb7 = hitmiss(skel, tbranch7)
    yb0 = hitmiss(skel, ybranch0)
    yb1 = hitmiss(skel, ybranch1)
    yb2 = hitmiss(skel, ybranch2)
    yb3 = hitmiss(skel, ybranch3)
    yb4 = hitmiss(skel, ybranch4)
    yb5 = hitmiss(skel, ybranch5)
    yb6 = hitmiss(skel, ybranch6)
    yb7 = hitmiss(skel, ybranch7)

    xb = xb0 + xb1
    tb = tb0 + tb1 + tb2 + tb3 + tb4 + tb5 + tb6 + tb7
    yb = yb0 + yb1 + yb2 + yb3 + yb4 + yb5 + yb6 + yb7

    branches = xb + tb + yb

    return branches


def end_points(skel):
    """Calculate the end points of an image skeleton.

    See http://stackoverflow.com/q/16691924
    """

    from mahotas.morph import hitmiss

    endpoint1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
    endpoint2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    endpoint3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])
    endpoint4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
    endpoint5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
    endpoint6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
    endpoint7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
    endpoint8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])

    ep1 = hitmiss(skel, endpoint1)
    ep2 = hitmiss(skel, endpoint2)
    ep3 = hitmiss(skel, endpoint3)
    ep4 = hitmiss(skel, endpoint4)
    ep5 = hitmiss(skel, endpoint5)
    ep6 = hitmiss(skel, endpoint6)
    ep7 = hitmiss(skel, endpoint7)
    ep8 = hitmiss(skel, endpoint8)

    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8

    return ep


# %%

import pandas as pd

base_folder = '../Data/Snake silhouettes/'
rotated_folder = base_folder + '2_Selected/'

meta = pd.read_csv(base_folder + 'selected_chord_images.csv')

i = 5
snake = meta.ix[i]
fname = snake['File name'].split('.')[0]
img = plt.imread(rotated_folder + fname + '.tif')

print fname
print 'Snake ' + str(snake['Snake'])

# save the processed images
save_base = base_folder + '/Processed/' + fname + '_{0}.png'
save_args = dict(bbox_inches='tight', transparent=False)

fig, ax = plt.subplots()
ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
ax.set(title=fname)
sns.despine()
fig.set_tight_layout(True)


# %%

# find threshold between snake and background
threshold = threshold_otsu(img)

# snake as black
img_b = img.copy()
img_b[img_b < threshold] = 0
img_b[img_b >= threshold] = 1
img_b = img_b.astype(np.bool)

# snake as white
img_w = ~img_b

# crop out just the snake
slices = ndi.find_objects(img_w)
sl = slices[0]

# now crop the images
img = img[sl]
img_b = img_b[sl]
img_w = img_w[sl]

# pad a small boarder around the images
pixpad = 10
img_w = pad(img_w, ((pixpad, pixpad), (pixpad, pixpad)), 'constant',
             constant_values=0)
img_b = pad(img_b, ((pixpad, pixpad), (pixpad, pixpad)), 'constant',
             constant_values=1)


# %% Plot the snake black and white snake images

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
ax1.imshow(img_b, interpolation='nearest', cmap=plt.cm.gray)
ax2.imshow(img_w, interpolation='nearest', cmap=plt.cm.gray)
ax1.axis('off')
ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(save_base.format('0_cropped'), **save_args)


# %%

# skeleton the snake (this will have small unwanted branches)
skeleton = skeletonize(img_w)

# find branch points of skeleton
branches = branch_point(skeleton)
branch_y, branch_x = np.where(branches)

# find endpoints of skeleton
ends = end_points(skeleton)
end_y, end_x = np.where(ends)

# Euclidian distance from edge of snake (this is the width)
distance = ndi.distance_transform_edt(img_w)


# %% Plot distance map, skeleton, brach points, and end points

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
fig.set_facecolor('k')
ax1.imshow(distance, interpolation='nearest', cmap=plt.cm.inferno)
ax2.imshow(skeleton, interpolation='nearest', cmap=plt.cm.gray)
ax1.contour(img_w, [0.5], colors='w')
ax2.contour(img_w, [0.5], colors='w')

for bx, by in zip(branch_x, branch_y):
    ax2.plot(bx, by, '^', c=bmap[0])

for ex, ey in zip(end_x, end_y):
    ax2.plot(ex, ey, 'o', c=bmap[2])

ax1.axis('off')
ax2.axis('off')
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(save_base.format('1_EDT_skel'), facecolor='k', **save_args)


# %%
#
# Break-up skeleton at branch points, so we have multiple segments to
# iterate through and check the length of. We just want one large,
# continuous segment which is the backbone of the animal

# http://stackoverflow.com/a/30223225

skel = skeleton.astype(np.int)

mask = np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])

bp = ndi.filters.convolve(skel, mask).astype(np.int)

skbp = skel & bp

skel_disconnect = skel & ~skbp
#skel_disconnect = skel - branches

_structure = np.ones((3, 3), dtype=np.int)
label_im, num_label = ndi.label(skel_disconnect, structure=_structure)
slices = ndi.find_objects(label_im)


# %% Remove short skeleton segments

cutoff_size = 100

lens = []
dists = []
ii = []
skel_new = np.zeros(skel.shape)

for i in np.arange(1, num_label + 1):
    idx = np.where(label_im == i)
    cut = skel_disconnect[idx]
    lens.append(cut.size)

    dist = distance[idx]
    dists.append(dist)

    if cut.size > cutoff_size:
        skel_new[idx] = cut + skbp[idx]
        ii.append(i)

lens = np.array(lens)
ii = np.array(ii, dtype=np.int)


# clean-up holes between segments
skel_connected = (skbp | skel_new.astype(np.int)).astype(bool)

# now remove the small objects again
label_im, num_label = ndi.label(skel_connected, structure=np.ones((3, 3)))
slices = ndi.find_objects(label_im)

# this is the skeleton we will analyze
skel_closed = np.zeros(skel.shape, dtype=np.int)

for i in np.arange(1, num_label + 1):
    idx = np.where(label_im == i)
    cut = skel_connected[idx]

    if cut.size > cutoff_size:
        skel_closed[idx] = cut


# %%

fig, ax = plt.subplots(figsize=(9, 11))
#ax.imshow(distance * skel_new, interpolation='nearest', cmap=plt.cm.inferno)
#ax.imshow(distance * skel_connected, interpolation='nearest', cmap=plt.cm.inferno)
#ax.imshow(distance * skel_closed, interpolation='nearest', cmap=plt.cm.inferno)
#ax.imshow(skel, interpolation='nearest', cmap=plt.cm.gray)
#ax.imshow(skbp, interpolation='nearest', cmap=plt.cm.gray)
#ax.imshow(skel_disconnect, interpolation='nearest', cmap=plt.cm.gray)
#ax.imshow(label_im, interpolation='nearest', cmap=plt.cm.viridis)
#ax.imshow(skel_new, interpolation='nearest', cmap=plt.cm.gray)
#ax.imshow(skel_connected, interpolation='nearest', cmap=plt.cm.gray)
ax.imshow(skel_closed, interpolation='nearest', cmap=plt.cm.gray)

for bx, by in zip(branch_x, branch_y):
    ax.plot(bx, by, 'o', c=bmap[0])

for ex, ey in zip(end_x, end_y):
    ax.plot(ex, ey, 'o', c=bmap[2])

ax.contour(img_w, [0.5], colors='w')

ax.axis('image')
ax.axis('off')
fig.set_facecolor('k')
sns.despine()
fig.set_tight_layout(True)

fig.savefig(save_base.format('2_skel_closed'), facecolor='k', **save_args)


# %% Iterate through the skeleton, arranging points from snout to vent

# start of snake (since head at the top)
inds_y, inds_x = np.where(skel_closed)
start_y, start_x = inds_y[0], inds_x[0]
stop_y, stop_x = inds_y[-1], inds_x[-1]

start = np.r_[start_y, start_x]
stop = np.r_[stop_y, stop_x]

# indices array to check againts
img_shape = img_w.shape
img_ind = np.arange(img_w.size).reshape(img_shape)

all_inds = img_ind[inds_y, inds_x]
all_inds_old = all_inds.copy()
all_inds_sort = np.zeros(all_inds.shape, dtype=np.int)

path = np.zeros((len(inds_y), 2), dtype=np.int)
path[0] = start

for i in np.arange(1, len(inds_y)):
    py, px = path[i - 1]
    neighbors = img_ind[py-1:py+2, px-1:px+2].flatten()

    found_neighbor = False
    for neighbor in neighbors:
        if neighbor in all_inds:
            # location where we have the neighbor
            neighbor_idx = np.where(all_inds == neighbor)[0][0]
            found_neighbor = True

            path[i] = inds_y[neighbor_idx], inds_x[neighbor_idx]

            # remove the index from the list
            all_inds[neighbor_idx] = -1
            break

# location of tip of tail
tail_tip = path[-1]


# %% Select snout, start of spline, and vent

fig, ax = plt.subplots(figsize=(9, 11))
fig.set_facecolor('k')
ax.imshow(distance, interpolation='nearest', cmap=plt.cm.inferno)
ax.contour(img_w, [0.5], colors='gray', linewidth=1)
ax.plot(path[:, 1], path[:, 0], c='w', lw=1.5)
ax.axis('image')
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)

xlim = ax.get_xlim()
ylim = ax.get_ylim()


# store the digitized points
pts = np.array(plt.ginput(5))[:, ::-1]

# pts[0] is junk (zoom to head)
# pts[1] is the snout
# pts[2] is a good part of the distance map to start the path
# pts[3] is the junk (zoom to vent)
# pts[4] is the vent

snout = pts[1]
pts_start = pts[2]
pts_vent = pts[4]

# indices into path array
idx_start = np.sqrt(np.sum((path - pts_start)**2, axis=1)).argmin()
idx_vent = np.sqrt(np.sum((path - pts_vent)**2, axis=1)).argmin()

# plot path locations on the snake
ax.plot(path[idx_start, 1], path[idx_start, 0], 'o', c=bmap[0], ms=8, mew=0)
ax.plot(path[idx_vent, 1], path[idx_vent, 0], 'o', c=bmap[0], ms=8, mew=0)

# plot the snout and tail tip
ax.plot(snout[1], snout[0], 's', c=bmap[1], ms=8, mew=0)
ax.plot(tail_tip[1], tail_tip[0], 's', c=bmap[1], ms=8, mew=0)

ax.set(xlim=xlim, ylim=ylim)

plt.draw()

# fig.savefig(save_base.format('3_neck_vent'), facecolor='k', **save_args)


# %%

# truncate the path based on the idx_start index from above
path_all = path.copy()
path = path[idx_start:]

snout_pix = np.round(snout).astype(np.int)

# distance between the snout and the
snout_to_path = np.sqrt(np.sum((path[0] - snout)**2))

# arc length coordiante
arclen_path = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))  # .cumsum()
path_scoord = np.r_[snout_to_path, arclen_path].cumsum()


 # %%

fig, ax = plt.subplots(figsize=(9, 11))
ax.imshow(distance, interpolation='nearest', cmap=plt.cm.inferno)

ax.plot(path[:, 1], path[:, 0], '-o', c='w', ms=2)

ax.contour(img_w, [0.5], colors='gray', linewidth=1.5)

ax.axis('image')
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)

# fig.savefig(save_base.format('4_backbone'), facecolor='k', **save_args)


# %%

from scipy.interpolate import UnivariateSpline

SVL = snake['SVL (cm)']
VTL = snake['Tail (cm)']
pix2cm = SVL / path_scoord[idx_vent]

scoord_cm = pix2cm * path_scoord
SVL_dig = scoord_cm[idx_vent]
VTL_dig = scoord_cm[-1] - scoord_cm[idx_vent]

scoord_non = scoord_cm / SVL_dig

# multiply by 2 pecause we are at the centerline
distance_skel = 2 * distance[path[:, 0], path[:, 1]]

distance_cm = pix2cm * distance_skel
distance_non = distance_cm / SVL_dig  # TODO: maybe we want to use mass, not SVL

distance_fun = UnivariateSpline(path_scoord, distance_skel, k=3, s=None)
distance_smooth_cm = pix2cm * distance_fun(path_scoord)


# Plot the chord distribution

fig, ax = plt.subplots()
ax.plot(scoord_cm, distance_cm, c=bmap[0])
ax.plot(scoord_cm, distance_smooth_cm, c=bmap[2])
ax.axvline(pix2cm * path_scoord[idx_vent], color='gray', lw=1)
ax.set_xlabel('distance along body (cm)')
ax.set_ylabel('chord width (cm)')
ax.set(title=fname + '\nSVL = {0:.1f} cm'.format(snake['SVL (cm)']))
sns.despine()
fig.set_tight_layout(True)

save_name = save_base.format('5_chord_cm').replace('.png', '.pdf')
fig.savefig(save_name, facecolor='w', **save_args)


fig, ax = plt.subplots()
ax.axvline(1, color='gray', lw=1)
ax.plot(scoord_non, 1000 * distance_non)
ax.set_xlabel('distance along body (SVL)')
ax.set_ylabel(r'chord width (1000$\cdot$SVL)')
ax.set(title=fname)
sns.despine()
fig.set_tight_layout(True)

save_name = save_base.format('6_chord_non').replace('.png', '.pdf')
fig.savefig(save_name, facecolor='w', **save_args)


# %% Split into body and tail segment, interpolate, save data

import pandas as pd

idx_body = np.arange(0, idx_vent + 1)
idx_tail = np.arange(idx_vent, len(scoord_non))

s_body = scoord_non[idx_body]
#s_tail = scoord_non[idx_tail]
#s_tail = s_tail - s_tail[0]
s_tail = scoord_cm[idx_tail]
s_tail -= s_tail[0]
s_tail = s_tail / VTL_dig

d_body = distance_cm[idx_body] / SVL_dig
d_tail = distance_cm[idx_tail] / VTL_dig

s = np.linspace(0, 1, 101)
body = np.interp(s, s_body, d_body)
tail = np.interp(s, s_tail, d_tail)

columns = ['arc length (SVL)', 'body width (SVL)', 'tail width (VTL)']
df = pd.DataFrame(data=np.c_[s, body, tail], columns=columns)

data_name = base_folder + 'Processed/' + fname + '.csv'
# df.to_csv(data_name)