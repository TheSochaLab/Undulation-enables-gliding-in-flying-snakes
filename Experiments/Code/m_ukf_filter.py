# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:42:22 2015

@author: isaac
"""

from __future__ import division

import numpy as np

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


def _sub3(a1, a2, a3):
    """Construct a 9x9 matrix given the upper, middle, and lower submatrices.
    """
    M = np.zeros((9, 9))
    M[0:3, 0:3] = a1
    M[3:6, 3:6] = a2
    M[6:9, 6:9] = a3
    return M


def _hx(x):
    """Measurement function. Constructs a 3 x 9 measurement matrix that
    maps from physical to measurement space. These are the same for us.
    """
    H = np.zeros((3, 9))
    H[0, 0] = 1
    H[1, 3] = 1
    H[2, 6] = 1
    return np.dot(H, x)


def _fx(x, dt):
    """State transition matrix from x(k) to x(k + 1). This is simply
    x1 = x0 + dt * v0 + .5 * a0^2
    v1 = v1 + dt * a0
    a1 = a0
    """
    f = np.array([[1, dt, .5 * dt**2],
                  [0, 1, dt],
                  [0, 0, 1]])
    F = _sub3(f, f, f)
    return np.dot(F, x)


def ukf(pr, fs, meas_noise=3):
    """Use an unscented Kalman filter to smooth the data and predict
    positions px, py, pz when we have missing data.

    Parameters:
    pr = (ntime x nmark x 3) raw data array in mm
    fs = sampling rate
    meas_noise = measurement noise (from Qualisys), default=3

    Returns:
    out = dict that holds filtered position and calculated velocity
        and accleration. Keys are as follows:
        p, v, a: pos, vel, acc after RTS smoothing. These are the values
            to fill in the missing data gaps
        nans: (ntime x nmark) bool array that stores where we have bad values
    """

    ntime, nmark, ncoords = pr.shape

    g = 9810  # mm/s^2
    dim_x = 9  # tracked variables px, vx, ax, py, vy, ay, pz, vz, az
    dim_z = 3  # measured variables px, py, pz
    dt = 1 / fs

    # state uncertainty matrix (measurement noise)
    R = meas_noise**2 * np.eye(dim_z)

    # process uncertainty matrix (effect of unmodeled behavior)
    sigx, sigy, sigz = .5 * g, .5 * g, .5 * g
    qx = Q_discrete_white_noise(3, dt=dt, var=sigx**2)
    qy = Q_discrete_white_noise(3, dt=dt, var=sigy**2)
    qz = Q_discrete_white_noise(3, dt=dt, var=sigz**2)
    Q = _sub3(qx, qy, qz)

    # uncertainty covariance matrix
    p0 = np.diag([1, 500, 2 * g])**2
    P = _sub3(p0, p0, p0)

    # store the data in 3D arrays
    pf, vf, af = pr.copy(), pr.copy(), pr.copy()  # for after RTS smoothing
    pf0, vf0, af0 = pr.copy(), pr.copy(), pr.copy()  # for original pass
    nans = np.zeros((ntime, nmark)).astype(np.int)

    # indices for ntime x 9 arrays to get pos, vel, and acc
    pos_idx, vel_idx, acc_idx = [0, 3, 6], [1, 4, 7], [2, 5, 8]

    for j in np.arange(nmark):
        zs = pr[:, j].copy()

        # store the nan values
        idx = np.where(np.isnan(zs[:, 0]))[0]
        nans[idx, j] = 1

        # batch_filter needs a list, with a nan values as None
        zs = zs.tolist()
        for ii in idx:
            zs[ii] = None

        # initial conditions for the smoother
        x0 = np.zeros(9)
        x0[pos_idx] = zs[0]
        x0[vel_idx] = 0, 500, 1000  # guess v0, mm/s
        x0[acc_idx] = .1 * g, .1 * g, -.5 * g  # guess a0, mm/s^2

        # how to calculate the sigma points
        msp = MerweScaledSigmaPoints(n=dim_x, alpha=1e-3, beta=2, kappa=0)

        # setup the filter with our values
        kf = UKF(dim_x, dim_z, dt, _hx, _fx, msp)
        kf.x, kf.P, kf.R, kf.Q = x0, P, R, Q

        # filter 'forward'
        xs, covs = kf.batch_filter(zs)

        # apply RTS smoothing
        Ms, Ps, Ks = kf.rts_smoother(xs, covs, dt=dt)

        # get data out of the (ntime x 9) array
        pf0[:, j] = xs[:, pos_idx]
        vf0[:, j] = xs[:, vel_idx]
        af0[:, j] = xs[:, acc_idx]

        pf[:, j] = Ms[:, pos_idx]
        vf[:, j] = Ms[:, vel_idx]
        af[:, j] = Ms[:, acc_idx]

    # finally store everything in a dictionary
    out = {'p': pf, 'v': vf, 'a': af, 'pf0': pf0, 'vf0': vf0, 'af0': af0,
           'nans': nans, 'xs': xs, 'covs': covs, 'zs': zs, 'x0': x0}

    return out


def fill_gaps_ukf(pr, fs, meas_noise):
    """Fill-in missing data from the output of ukf. This only applies the
    filter to those arrays that are missing data.

    Parameters
    ----------
    pr : array, (ntime x nmark x 3)
        raw position data
    fs : float
        Sampling rate
    meas_noise : float
        Measurement noise in mm (output from Qualisys)

    Returns
    -------
    pfill : array, size (ntime x nmark)
        Array that holds the same (noisy) data as pr, except that
        nan locations have been filled with our predictions.
    nans : array, size (ntime x nmark)
        int array, 1's correspond to nan values
    pfill0 : array, size (ntime x nmark x 3)
        filled positions after the forward pass of the filter
        (excluding the RTS smoother).
    """

    ntime, nmark, ncoord = pr.shape

    pfill = pr.copy()
    pfill0 = pr.copy()
    nans = np.zeros((ntime, nmark)).astype(np.int)
    for j in np.arange(nmark):
        arr = pr[:, j]
        idx = np.where(np.isnan(arr[:, 0]))[0]
        nans[idx, j] = 1

        if len(idx) > 0:
            # convert to 3D array for ukf, since this function iterates
            # through all of the markers. now we have an ntime x 1 marker x
            # ncood array
            data = np.zeros((ntime, 1, ncoord))
            data[:, 0] = arr

            # perform the filtering
            out = ukf(data, fs, meas_noise)

            # take data from the smoothed array (output of ukf) and
            # copy over the naned values
            pfill[idx, j] = out['p'][idx, 0]
            pfill0[idx, j] = out['pf0'][idx, 0]

    return pfill, nans, pfill0