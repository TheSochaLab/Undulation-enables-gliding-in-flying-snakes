# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:46:20 2015

@author: isaac
"""

from __future__ import division

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, savgol_filter, filtfilt


def findiff(p, dt):
    """Second-order accurate finite difference velocites and accelerations.

    Parameters:
    p = 1D array (size ntime) to take finite difference of
    dt = time step between measurements

    Returns:
    v = velocity
    a = acceleration

    Finite difference code from:
    See: https://en.wikipedia.org/wiki/Finite_difference_coefficient

    We are using 2nd order accurate central, forward, and backward finite
    differences.
    """

    n = len(p)
    v, a = np.zeros_like(p), np.zeros_like(p)

    # at the center
    for i in np.arange(1, n - 1):
        v[i] = .5 * p[i + 1] - .5 * p[i - 1]
        a[i] = p[i + 1] - 2 * p[i] + p[i - 1]

    # left boundary (forward difference)
    v[0] = -1.5 * p[0] + 2 * p[1] - .5 * p[2]
    a[0] = 2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]

    # right boundary (backward differences)
    v[-1] = 1.5 * p[-1] - 2 * p[-2] + .5 * p[-3]
    a[-1] = 2 * p[-1] - 5 * p[-2] + 4 * p[-3] - p[-4]

    return v / dt, a / dt**2


def spl1d(pr, fs, kk=3, ss=.001):
    """Use scipy.interpolate.Univariate spline to fit the data and
    calculate velocity and acceleration from the spline

    Parameters:
    pr = (ntime x nmark x 3) raw data array
    fs = sampling rate
    kk = spline order (default is 3)
    ss = spline smoothing parameter (default is .001)

    Returns:
    out = dict that holds filtered position and calculated velocity
        and accleration
    """

    ntime, nmark, ncoord = pr.shape
    dt = 1 / fs
    times = np.arange(ntime) * dt

    # iterate through each marker and coordinate, smooth, and calculate
    # velocities and accelerations
    pf, vf, af = pr.copy(), pr.copy(), pr.copy()
    for j in np.arange(nmark):
        for k in np.arange(ncoord):
            d = pr[:, j, k]
            spl = UnivariateSpline(times, d, k=kk, s=ss)
            pf[:, j, k] = spl(times)
            vf[:, j, k] = spl.derivative(1)(times)
            af[:, j, k] = spl.derivative(2)(times)

    out = {'p': pf, 'v': vf, 'a': af}
    return out


def but(pr, fs, fc, order=2, padlen=None):
    """Smooth with two passes of a Butterworth filter and calculate
    velocity and acceleration with finite differences.

    Parameters:
    pr = (ntime x nmark x 3) raw data array
    fs = sampling rate
    fc = Butterworth filter cutoff frequency (pick this from a
        residual analysis)
    order = filter order (default is 2)
    padlen = padding to use in filtfilt (default is None, so use
        ntime - 1 length of padding)

    Returns:
    out = dict that holds filtered position and calculated velocity
        and accleration; also return Wn, the fraction of the Nyquist
        used to construct the filter coefficients
    """

    ntime, nmark, ncoord = pr.shape

    fny = fs / 2  # Nyquist frequency
    Wn = fc / fny  # non-dimensional fraction of the Nyquist
    dt = 1 / fs  # measurement interval

    # get the filter coefficients
    butb, buta = butter(order, Wn)

    # padlen for filtfilt
    if padlen is None:
        padlen = ntime - 1

    # iterate through each marker and coordinate, smooth, and calculate
    # velocities and accelerations
    pf, vf, af = pr.copy(), pr.copy(), pr.copy()
    for j in np.arange(nmark):
        for k in np.arange(ncoord):
            d = pr[:, j, k]
            p = filtfilt(butb, buta, d, padlen=padlen)
            v, a = findiff(p, dt)
            pf[:, j, k], vf[:, j, k], af[:, j, k] = p, v, a

    out = {'p': pf, 'v': vf, 'a': af, 'Wn': Wn}
    return out


def but_fcs(pr, fs, fcs, order=2, padlen=None):
    """Smooth with two passes of a Butterworth filter and calculate
    velocity and acceleration with finite differences. However, use
    difference cutoff frequencies for each marker and x, y, z
    time series.

    Parameters
    ----------
    pr = (ntime x nmark x ncoord) array
        raw data array
    fs = float
        sampling rate
    fcs = (nmark x ncoord) array
        Butterworth filter cutoff frequencies (pick this from the residual
        analysis)
    order = integer
        filter order (default is 2)
    padlen = integer or None
        padding to use in filtfilt (default is None, so use time - 1 length
        of padding)

    Returns
    -------
    out = dict
        Holds filtered position and calculated velocity
        and accleration; also return Wns, the fraction of the Nyquist
        used to construct the filter coefficients for each marker in
        the x, y, and z positions.
    """

    ntime, nmark, ncoord = pr.shape
    dt = 1 / fs  # measurement interval

    fny = fs / 2  # Nyquist frequency
    Wns = fcs / fny  # non-dimensional fraction of the Nyquist

    # padlen for filtfilt
    if padlen is None:
        padlen = ntime - 1

    # iterate through each marker and coordinate, smooth, and calculate
    # velocities and accelerations
    pf, vf, af = pr.copy(), pr.copy(), pr.copy()
    for j in np.arange(nmark):
        for k in np.arange(ncoord):
            # construct the filter
            butb, buta = butter(order, Wns[j, k])

            # select data and perform filtering
            d = pr[:, j, k]
            p = filtfilt(butb, buta, d, padlen=padlen)
            v, a = findiff(p, dt)
            pf[:, j, k], vf[:, j, k], af[:, j, k] = p, v, a

    out = {'p': pf, 'v': vf, 'a': af, 'Wns': Wns}
    return out


def svg(pr, fs, win, poly, use_fd_der=True):
    """Smooth with a Savitzky-Golay filter and calculate velocity and
    acceleration with finite differences or from the SG polynomials.

    Parameters:
    pr = (ntime x nmark x 3) raw data array
    fs = sampling rate
    win = window size to use around current point
    poly = order of polynomial to fit
    use_fd_der = whether to use finite differences or the polynomial to
        calculate velocities and accelerations

    Returns:
    out = dict that holds filtered position and calculated velocity
        and accleration

    Note:
    We can get velocities and accelerations from the smoothing polynomial
    directly, but we choose not to do that here
    """

    ntime, nmark, ncoord = pr.shape
    dt = 1 / fs

    pf, vf, af = pr.copy(), pr.copy(), pr.copy()
    for j in np.arange(nmark):
        for k in np.arange(ncoord):
            d = pr[:, j, k]
            p = savgol_filter(d, win, poly, deriv=0)
            if use_fd_der:
                v, a = findiff(p, dt)
            else:
                v = savgol_filter(d, win, poly, deriv=1) / dt
                a = savgol_filter(d, win, poly, deriv=2) / dt**2
            pf[:, j, k], vf[:, j, k], af[:, j, k] = p, v, a

    out = {'p': pf, 'v': vf, 'a': af}
    return out


def raw(pr, fs):
    """Use raw finite differences of the position data to
    calculated velocities and accelertions.

    Parameters:
    pr = (ntime x nmark x 3) raw data array
    fs = sampling rate

    Returns:
    out = dict that holds filtered position and calculated velocity
        and accleration
    """

    ntime, nmark, ncoord = pr.shape
    dt = 1 / fs

    vf, af = pr.copy(), pr.copy()
    for j in np.arange(nmark):
        for k in np.arange(ncoord):
            vf[:, j, k], af[:, j, k] = findiff(pr[:, j, k], dt)

    out = {'p': pr, 'v': vf, 'a': af}
    return out


def residual_butter(pr, fs, order=2, df=.5, fmin=1, fmax=35):
    """Calculate RMS residuals to determine a proper cutoff frequency
    for a Butterworth filtering.

    Parameters:
    pr = (ntime x nmark x 3) raw data array
    fs = sampling frequency
    order = order filter to use, default is 2
    df = difference between frequencies
    fmin = min freq to try, default is 1
    fmax = max freq to try, default is 35

    Returns:
    R = residual array of (nfreq x nmark x 3) that has the RMS residual
        for a given marker is in X, Y, and Z directions.
    fcs = array of cutoff frequencies tried
    """

    ntime, nmark, ncoord = pr.shape

    # fcs = np.linspace(fmin, fmax, nfreq)
    fcs = np.arange(fmin, fmax + .01, df)
    nfreq = len(fcs)
    R = np.zeros((nfreq, nmark, 3))

    for i, fc in enumerate(fcs):
        out = but(pr, fs, fc, order)
        R[i] = np.sqrt(np.mean((pr - out['p'])**2, axis=0))

    return R, fcs


def opt_cutoff(R, fcs, rsq_cutoff=.95):
    """Find an 'optimum' cutoff frequency based on the residuals.

    Parameters:
    R = (nfreq x nmark x 3) array of residuals
    fcs = cutoff frequences used
    rsq_cutoff = coefficient of determination minimum that determines the
        frequency to perform the noise fit

    Returns:
    inter = (nmark x ncoord) array of y-intercepts in mm
    fit_slope = (nmark x ncoord) array of slopes in mm/Hz
    fcopt = (nmark x ncoord) array of 'optimum' cutoff frequencies
    rsq = (nmark x ncoord) array of R^2 values
    flinreg = (2 x nmark x ncoord) array of the fmin, fmax values to
        construct the linear regression to get the cutoff frequency

    Notes:
    This function fits a linear line to the tail of the residual, finds
    the intercept (a residual) and the corresponding cutoff frequency
    this residual corresponds to. This is a 'rough' find, that is, it
    finds the nearest freqeuncy in the array of provided requencies and does
    not do root finding of an interpolation function, etc.

    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    """
    from scipy.stats import linregress
    from scipy.optimize import newton
    from scipy.interpolate import interp1d

    nfreq, nmark, ncoord = R.shape

    inter = np.zeros((nmark, ncoord))
    fit_slope = np.zeros((nmark, ncoord))
    fcopt = np.zeros((nmark, ncoord))
    rsq = np.zeros((nmark, ncoord))
    flinreg = np.zeros((2, nmark, ncoord))

    for j in np.arange(nmark):
        for k in np.arange(3):
            # res = R[idx, j, k]
            res = R[:, j, k]

            # iterative find the min freq to perform interpolation
            # step backwards through the array, until rsq drops to rsq_cutoff
            ixcut = np.inf  # if don't get into if, this will throw an error
            for mm in np.arange(nfreq - 2, 0, -1):
                slope, intercept, r_value, p_value, std_err = \
                    linregress(fcs[mm:], res[mm:])
                # print r_value**2
                if r_value**2 < rsq_cutoff:
                    ixcut = mm + 1
                    break

            # once we go under rsq_cutoff, use the previous value
            slope, intercept, r_value, p_value, std_err = \
                linregress(fcs[ixcut:], res[ixcut:])

            # optimize the cutoff frequency
            def tozero(freq_guess):
                return resid_interp(freq_guess) - intercept

            # interpolation function of the residuals
            resid_interp = interp1d(fcs, res)
            fopt_guess = fcs[np.argmin(np.abs(res - intercept))]
            fopt = newton(tozero, fopt_guess)

            # store the values
            inter[j, k] = intercept
            fit_slope[j, k] = slope
            fcopt[j, k] = fopt
            rsq[j, k] = r_value**2
            flinreg[:, j, k] = fcs[ixcut], fcs[-1]

    return inter, fit_slope, fcopt, rsq, flinreg


def fill_gaps_spl(p, times, kk=3, ss=1e-5):
    """Locate gaps in a 1D array. This function should probably not be used,
    since Kalman filtering will likely give better results.

    TODO: deal with the ends (don't extrapolate as the spline fails)
    """

    ntime, nmark, ncood = p.shape

    bounds = np.zeros((nmark, 2))
    pg = p.copy()

    for j in np.arange(nmark):
        x, y, z = p[:, j, :].T
        ixb = np.where(np.isnan(x))[0]
        ixg = np.where(~np.isnan(x))[0]

        imin, imax = 0, ntime
        if len(ixb) > 0:

            # fit a 1D spline to the good points
            spx = UnivariateSpline(times[ixg], x[ixg], k=kk, s=ss)
            spy = UnivariateSpline(times[ixg], y[ixg], k=kk, s=ss)
            spz = UnivariateSpline(times[ixg], z[ixg], k=kk, s=ss)

            # fill in the gap regions
            for ib in ixb:
                pg[ib, j, 0] = spx(times[ib])
                pg[ib, j, 1] = spy(times[ib])
                pg[ib, j, 2] = spz(times[ib])

            # index so we don't extrapolate
            imin, imax = ixb.min(), ixb.max()

        # save bounding region for later
        bounds[j] = imin, imax

    return pg, bounds