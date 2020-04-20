# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:24:25 2015

@author: isaac
"""

from __future__ import division

import numpy as np
import pandas as pd


def load_qtm_tsv(fname, header=10):
    """Load in a tsv file that was exported from Qualisys Track Manager.

    Parameters:
    fname = total path to the file
    header = number of lines to skip (default is 10)

    Returns:
    df = DataFrame of the CSV file, columns should be Frame, Time,
        then the coordinate data.
    """

    # standard load in dataframe
    df = pd.read_csv(fname, sep='\t', header=header)

    # There may be an extra column tacked on at the end, so
    # get rid of it. Starts with "Unn".
    drop = []
    for idx, col in enumerate(df.columns):
        if col.startswith('Unn'):
            drop.append(idx)
    df = df.drop(df.columns[drop], axis=1)

    return df


def reconfig_raw_data(df):
    """Turn the DataFrame from load_qtm_tsv into a 3D array, normalize
    time, etc.

    Parameters:
    df = DataFrame from load_qtm_tsv

    Returns:
    pr: array
        raw coordinate data, a 3D array of (ntime x nmarks x (x, y, z))
    out: dict
        with pr, times, and frames
    """

    # remove Frame and Time columns; then have X, Y, Z coords for each marker
    nmark = np.int((df.shape[1] - 2) / 3)
    ntime = df.shape[0]

    frames = df['Frame'].values
    times = df['Time'].values
    # index = df.index.values

    times = times - times[0]  # make time start at zero
    marks = np.arange(nmark)  # marker index array

    # iterate through markers and select markers out, store these in a dict
    dd = {}
    b = '{0:02d} {1}'  # e.g. '01 X', '13 Z', etc.
    for i in marks + 1:  # one indexed files
        try:
            data = df[[b.format(i, 'X'), b.format(i, 'Y'), b.format(i, 'Z')]]
            dd[i] = data.values
        except:
            print('There is no column {0:d}'.format(i))

    # turn dict into a 3D array of (ntime x nmarks x (x, y, z))
    pr = np.zeros((ntime, nmark, 3))
    for i in np.arange(ntime):
        for j in np.arange(nmark):
            pr[i, j, :] = dd[j + 1][i]

    out = {'pr': pr, 'times': times, 'frames': frames}

    return pr, out


def save_filtered_data(filename, pf, vf, af, times):
    """Save filtered position, velocity, and acceleration data
    to a csv file.

    Parameters:
    filename = complete path and name of csv file
    pf = filtered position data (ntime x nmark x ncoord)
    vf = velocity data (ntime x nmark x ncoord)
    af = acceleration data (ntime x nmark x ncoord)
    times = time vector (ntime)

    Notes:
    This combines the time, position, velocity, and acceleration data
    into one array that is (ntime x nmark * ncoord * 3 + 1). The load
    function assumes this structure, but we write column neaders
    so another program can be used.
    """

    ntime, nmark, ncoord = pf.shape

    # convert 3D data arrays to 2D arrays
    pf_2d = pf.reshape((ntime, nmark * ncoord))
    vf_2d = vf.reshape((ntime, nmark * ncoord))
    af_2d = af.reshape((ntime, nmark * ncoord))

    # combine all of the data
    todf = np.c_[times, pf_2d, vf_2d, af_2d]

    # column names
    columns = ['Time (s)']
    quant = ['P', 'V', 'A']
    for k in np.arange(ncoord):
        for j in np.arange(nmark):
            columns.append(quant[k] + 'x_' + str(j))
            columns.append(quant[k] + 'y_' + str(j))
            columns.append(quant[k] + 'z_' + str(j))

    # make a DataFrame for ease of saving
    df = pd.DataFrame(todf, columns=columns)
    df.index.name = 'Index'

    # save the file
    df.to_csv(filename)


def load_filtered_data(filename):
    """Load filtered position, velocity, and acceleration data
    from a csv file saved with 'save_filtered_data'

    Parameters:
    filename = complete path and name of csv file

    Returns:
    pf = filtered position data (ntime x nmark x ncoord)
    vf = velocity data (ntime x nmark x ncoord)
    af = acceleration data (ntime x nmark x ncoord)
    times = time vector (ntime)

    Notes:
    We do not use the column headers here. Instead we deal directly
    with the shape of the saved 2D array.
    """

    # load the data in
    df = pd.read_csv(filename, index_col=0)

    # extract the time vector, then delete that column
    times = df['Time (s)'].values
    d = df.drop('Time (s)', axis=1).values

    # indices needed to slice large 2D array
    nmark = np.int(d.shape[1] / (3 * 3))
    idx = 3 * nmark

    # pull out subarrays and reshape to 3D data arrays (ntime x nmark x ncoord)
    pf, vf, af = d[:, :idx], d[:, idx:2 * idx], d[:, 2 * idx:3 * idx]
    pf = pf.reshape(-1, nmark, 3)
    vf = vf.reshape(-1, nmark, 3)
    af = af.reshape(-1, nmark, 3)

    return pf, vf, af, times


def save_residual_analysis(fname, R, fcs, inter, fcopt, rsq, flinreg):
    """Save the residual analysis as a binary numpy file.

    Parameters:
    R = (nfreq x nmark x ncoord) residual array
    fcs = frequencies residuals evaluated at
    inter = RMS noise level (mm)
    fcopt = optimal cutoff frquency
    rsq = coeff of determination of how well we estimate our noise residual
    flinreg = the frequency range used to determine the noise residual

    Returns:
    Noting
    """

    if not fname.endswith('.npz'):
        fname += '.npz'

    resid_dict = {'R': R, 'fcs': fcs, 'inter': inter, 'fcopt': fcopt,
                  'rsq': rsq, 'flinreg': flinreg}
    np.savez(fname, **resid_dict)


def load_residual_analysis(fname):
    """Load in the residual analysi from a binary numpy file.
    """

    if not fname.endswith('.npz'):
        fname += '.npz'

    d = np.load(fname)
    return d


def shift_to_com(pr):
    """Center each measurement location so the snake undulates in place.

    Parameters:
    pr : (ntime x nmark x ncoord) data array

    Returns:
    pc : (ntime x nmark x ncoord) shifted data array
    com : (ntime x ncoord) center of mass
    """

    # ntime, nmark, ncoord = pr.shape
    # pc = pr.copy()
    # for i in np.arange(ntime):
    #     pc[i] = pr[i] - pr[i].mean(axis=0)

    # faster, but harder to understand
    com = pr.mean(axis=1)
    pc = (pr.swapaxes(0, 1) - com).swapaxes(0, 1)

    return pc, com
