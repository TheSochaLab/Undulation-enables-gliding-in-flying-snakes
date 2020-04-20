# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:04:25 2016

@author: isaac
"""

from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import linregress


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


def para_fit(x, *params):
    h, k, p = params
    return (x - h)**2 / (4 * p) + k


def chord_dist(s, L):
    # using parabolic fit
    popt_chord = np.array([ 0.51220022,  0.02895101, -4.88957129])

    chord_para = para_fit(s / L, *popt_chord) * L

    return chord_para


def mass_dist(s, ds, L, mtot):
    """Apply parabolic mass distribution.
    """

    # parameters for parabolic fit for head and body
    popt_head = np.array([ 0.01661771,  1.36099325, -0.00012991])
    popt_body = np.array([ 0.52583598,  1.11461374, -0.10778636])

    # position along the body
    sneck = .037
    idx_head = np.where(s / L <= sneck)[0]
    idx_body = np.where(s / L > sneck)[0]
    shead = s[idx_head] / L
    sbody = s[idx_body] / L

    # non-dimensional density
    rho_non_head = para_fit(shead, *popt_head)
    rho_non_body = para_fit(sbody, *popt_body)
    rho_non = np.r_[rho_non_head, rho_non_body]  # combine into one array

    # average density
    rho_bar = mtot / L

    # density in physical units (mass/length)
    rho = rho_non * rho_bar

    # mass of each segment
    mass = rho * ds

    # ensure that sum(mass) is mtot
    mass = mass / mass.sum() * mtot

    return mass


def morph_from_svl(L):
    """Mass, chord, wing loading, average density from SVL.

    Parameters
    ----------
    L : float
        SVL in m

    Returns
    -------
    """

    # convert L to cm
    L = L * 100

    d = pd.read_csv('../Data/Morphometrics/Socha2005_morphometrics.csv')

    # chord vs. svl does better with this
    ix = d['Common name'] == 'flying snake'  # 17 snakes

    svl = d[u'SVL snake (cm)'][ix].values
    chord = d[u'Chord length (cm)'][ix].values
    mass = d[u'Mass (g)'][ix].values
    Ws = d[u'Wing loading (N/m^2)'][ix].values
    avg_den = mass / svl * 100  # mg/mm

    # linear fit of chord vs. SVL
    chord_vs_svl = linregress(svl, chord)

    # fit model of log variables vs. SVL
    B_mass_svl, P_mass_svl, R2_mass_svl = MLE_log(svl, mass)
    B_Ws_svl, P_Ws_svl, R2_Ws_svl = MLE_log(svl, Ws)
    B_avg_den_svl, P_avg_den_svl, R2_avg_den_svl = MLE_log(svl, avg_den)

    # apply the linear model
    chord_fit = chord_vs_svl.slope * L + chord_vs_svl.intercept

    # fit using the linear regression in log transformed units
    log_L = np.log(L)
    log_mass_fit = np.e ** (B_mass_svl[0] + B_mass_svl[1] * log_L)
    log_Ws_fit = np.e ** (B_Ws_svl[0] + B_Ws_svl[1] * log_L)
    log_avg_den_fit = np.e ** (B_avg_den_svl[0] + B_avg_den_svl[1] * log_L)

    # convert chord to m
    chord_fit = chord_fit / 100  # cm to m
    log_mass_fit = log_mass_fit / 1000  # g to kg

    return chord_fit, log_mass_fit, log_Ws_fit, log_avg_den_fit