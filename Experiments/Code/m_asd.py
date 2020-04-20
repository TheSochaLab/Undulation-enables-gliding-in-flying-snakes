# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:47:07 2016

@author: isaac
"""

from __future__ import division

import numpy as np
from numba import jit
import m_smoothing as smoothing


def add_neck(pf, vf, dist_btn_markers):
    """Add a virtual marker to the data points, which constrains the spline and
    simulates head control/a neck.

    Parameters
    ----------
    pf : array, size (ntime, nmark, 3)
        Recorded marker locations (these are probably filtered)
    vf : array, size (ntime, nmark, 3)
        Recorded marker velocities
    dist_btn_markers : array, size (nmark)
        Arc lenth between the IR markers (measured from images)
    """

    ntime, nmark, _ = pf.shape

    # "extended" smooth points. 2nd entry for 'virtual' marker
    pfe = np.zeros((ntime, nmark + 1, 3))
    pfe[:, 0] = pf[:, 0]
    pfe[:, 2:] = pf[:, 1:]

    # new distance between markers (for spline calculations)
    # aka segment parameters
    neck_len = dist_btn_markers[0] / 2
    te = np.zeros(nmark)
    te[0] = neck_len
    te[1] = dist_btn_markers[0] - neck_len
    te[2:] = dist_btn_markers[1:]

    # add the virtual marker by finding rotation of pf[i, 0] into Yhat direction
    for i in np.arange(ntime):
        vx, vy = vf[i, 0, 0], vf[i, 0, 1]  # of head marker
        th_pfe = np.arctan2(vx, vy)

        # rotation matrix about Zhat
        Rth = np.array([[np.cos(th_pfe), -np.sin(th_pfe), 0],
                        [np.sin(th_pfe),  np.cos(th_pfe), 0],
                        [0, 0, 1]])


        # determine the y and z offset for the virtual marker
        p1 = pf[i, 0]  # head marker
        p2 = pf[i, 1]  # 2nd marker

        # average the z-coordinate b/n head and 2nd marker
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dl = np.sqrt(dx**2 + dy**2)
        yoff = dl / 2
        zoff = (p2[2] - p1[2]) / 2

        # add the virtual marker, calculate virtual point
        p1_rot = np.dot(Rth, p1)
        p1a_rot = p1_rot + np.array([0, -yoff, zoff])

        # rotate virtual marker back to other points
        pfe[i, 1] = np.dot(Rth.T, p1a_rot)

    tcoord_e = te.copy()

    return pfe, tcoord_e


def global_natural_spline(p, t, nspl):
    """Fit 3rd order global natural splines to the data.

    Parameters
    ----------
    p : array, size (nmark, 2)
        Measured data points
    t : array, size (nmark - 1)
        arc length distance between markers. Note that these don't have to
        be exact (and can be less than arc length between them),
        and can be modified to change the tangent angle at
        a given measurement point.
    nspl : int
        number of points to evaluate the spline at

    Returns
    -------
    r, dr, ddr, dddr : arrays, size (nspl, 3)
        x, y, z and the associated derivatives of the spline
    ts : array, size (nspl)
        cumulatve coordinate spine was **evaluated** at
    ss : array, size (nspl)
        cumulative **arc length** coordinate
    spl_ds : array, size (nspl)
        lengths of individual spline segments (in physical units)
    lengths_total : array, size (nmark - 1)
        the integrated length of the spline between the points
    idx_pts : array, size (nmark - 1)
        indices into the ts and ss for the measured points
    """

    # from scipy.integrate import cumtrapz

    # number of measured points
    n = p.shape[0]

    # construct C and D matries: Dm = Cp, solve for m (n x 3) matrix
    C = np.zeros((n, n))
    D = np.zeros((n, n))

    for j in np.arange(n - 2):
        D[j, j] = t[j + 1]
        D[j, j + 1] = 2 * (t[j + 1] + t[j])
        D[j, j + 2] = t[j]

        C[j, j] = 3 * (-t[j + 1] / t[j])
        C[j, j + 1] = 3 * (t[j + 1] / t[j] - t[j] / t[j + 1])
        C[j, j + 2] = 3 * (t[j] / t[j + 1])

    # end conditions for a global natural spline
    C[n - 2, 0] = -3 / t[0]  # x(0)
    C[n - 2, 1] = 3 / t[0]  # x(0)
    C[n - 1, n - 2] = -3 / t[-1]  # x_{n-1}(t_{n-1})
    C[n - 1, n - 1] = 3 / t[-1]  # x_{n-1}(t_{n-1})
    D[n - 2, 0] = 2  # x(0)
    D[n - 2, 1] = 1  # x(0)
    D[n - 1, n - 2] = 1  # x_{n-1}(t_{n-1})
    D[n - 1, n - 1] = 2  # x_{n-1}(t_{n-1})

    # solve for the tangent angles m
    m = np.linalg.solve(D, np.dot(C, p))

    # cubic spline coefficients
    a = p[:-1]
    b = m[:-1]
    c = (3 * (p[1:] - p[:-1]).T / t**2 - (2 * m[:-1] + m[1:]).T / t).T
    d = (2 * (p[:-1] - p[1:]).T / t**3 + (m[:-1] + m[1:]).T / t**2).T

    # number of spline points per segment, taking care so we have nspl total
    mm_per_spl_bit = t.sum() / nspl
    bits_per_seg_float = t / mm_per_spl_bit
    bits_per_seg = np.round(bits_per_seg_float).astype(np.int)

    nbits = bits_per_seg.sum()
    if nbits > nspl:
        bits_per_seg[-1] -= nbits - nspl
    elif nbits < nspl:
        bits_per_seg[-1] += nspl - nbits
    nspl_seg = bits_per_seg.copy()

    nspl_seg[:-1] += 1  # because of the inertior points we skip with
                        # jj > 0 below

    # indices in ts for the measured points
    # e.g. ts[idx_pts] == t.cumsum()
    idx_pts = (nspl_seg - 1).cumsum()

    # empty arrays to store the spline values
    r = np.zeros((nspl, 3))
    dr = np.zeros((nspl, 3))
    ddr = np.zeros((nspl, 3))
    dddr = np.zeros((nspl, 3))
    ts = np.zeros(nspl)

    # iterate through the (n - 1) segments and fit the spline
    cnt = 0
    for jj in np.arange(n - 1):
        ti = np.linspace(0, t[jj], nspl_seg[jj])
        if jj > 0:
            ti = ti[1:]

        for k in np.arange(len(ti)):
            ri = a[jj] + b[jj] * ti[k] + c[jj] * ti[k]**2 + d[jj] * ti[k]**3
            dri = b[jj] + 2 * c[jj] * ti[k] + 3 * d[jj] * ti[k]**2
            ddri = 2 * c[jj] + 6 * d[jj] * ti[k]
            dddri = 6 * d[jj]
            r[cnt] = ri
            dr[cnt] = dri
            ddr[cnt] = ddri
            dddr[cnt] = dddri
            ts[cnt] = t[:jj].sum() + ti[k]
            cnt += 1

    assert cnt == nspl

    # segment parameter lengths
    dts = np.gradient(ts, edge_order=2)

    # integrate arc length between the measured points
    ds = np.sqrt(np.sum(dr**2, axis=1))

    # length of each nspl segment
    seg_lens = ds * dts

    # arc length coordinate for the spline
    ss = seg_lens.cumsum()
    ss = ss - ss[0]  # because first element with be seg_lens[0] long

    # integrate arc length between points
    int_seg = np.r_[0, bits_per_seg.cumsum()]  # indices to integrate between
    lengths_total = np.zeros(n - 1)  # total length of each segment
    for jj in np.arange(n - 1):
        i0, i1 = int_seg[jj], int_seg[jj + 1]
        assert i1 <= nspl
        lens = seg_lens[i0:i1].cumsum()
        lengths_total[jj] = lens[-1]

    return r, dr, ddr, dddr, ts, ss, seg_lens, lengths_total, idx_pts


def splinize_snake(pfe, te, nspl, times, mass, dist_btn_markers, vent_idx, SVL, VTL,
                   density_df, chord_df):
    """Fit a spline to the recorded IR markers to model the backbone of the snake.
    Also overlay the mass and chord length distributions.

    Parameters
    ----------
    pfe : array, size (ntime, nmark + 1, 3)
        IR marker locations
    te : array, size (nmark + 1)
        Distance between markers
    nspl : int
        Number of points to evaluate spline at
    times : array, size (ntime)
        Measured time points (for making a 2D time point array)
    mass : float
        Measured mass of the snake
    marker_df : DataFrame
        Information about the markers for the snake
    density_df : DataFrame
        [s, rho] Normalized density (by average density) distribution
    chord_df : DataFrame
        [s, chord] Normalized chord length (by SVL) distribution

    Returns
    -------
    Dictionary with:
    out = dict(Ro_I=Ro_I, R_I=R_I, dRds_I=ddRds_I, ddRds_I=ddRds_I,
               dddRds_I=dddRds_I,
               spl_ds=spl_ds, mass_spl=mass_spl, chord_spl=chord_spl,
               vent_idx_spl=vent_idx_spl, times2D=times2D, t_coord=t_coord,
               s_coord=s_coord, spl_len_errors=spl_len_errors,
               idx_pts=idx_pts, SVL=SVL, VTL=VTL)
    """

    # density distribution
    s_rho = density_df['s'].values
    body_rho = density_df['rho'].values

    # chord length distribution
    s_chord  = chord_df['s'].values
    body_chord = chord_df['chord'].values

    ntime, nmark_e, _ = pfe.shape  # number of markers on 'extended' neck snake
    nmark = nmark_e - 1  # number of markers on acutal snake

    Ro_I = np.zeros((ntime, 3))
    times2D = np.zeros((ntime, nspl))
    t_coord = np.zeros((ntime, nspl))
    s_coord = np.zeros((ntime, nspl))
    vent_idx_spls = np.zeros(ntime, dtype=np.int)
    R_I = np.zeros((ntime, nspl, 3))  # spl
    dRds_I = np.zeros((ntime, nspl, 3))  # dspl
    ddRds_I = np.zeros((ntime, nspl, 3))  # ddspl
    dddRds_I = np.zeros((ntime, nspl, 3))
    spl_ds = np.zeros((ntime, nspl))  # length of each segment in mm
    mass_spl = np.zeros((ntime, nspl))  # in g
    chord_spl = np.zeros((ntime, nspl))
    #spl_len_totals = np.zeros(ntime)
    spl_len_errors = np.zeros((ntime, nmark - 1))  # -1 because difference b/n

    for i in np.arange(ntime):

        # fit spline (fpe is the arc length coordinate of the markers)
        out = global_natural_spline(pfe[i], te, nspl)
        r, dr, ddr, dddr, ts, ss, seg_lens, lengths_total_e, idx_pts = out

        # exclude the virtual marker for error calculations
        lengths_total = np.zeros(nmark - 1)
        lengths_total[0] = lengths_total_e[0] + lengths_total_e[1]
        lengths_total[1:] = lengths_total_e[2:]

        # arc length coordinate differences
        # (% along spline) of markers (no virtual marker)
        # %SVL of arc length coordinate
        spl_len_error = (dist_btn_markers - lengths_total) / SVL * 100

        # index into arc length coord where vent measurement is closest
        # based on segment parameters (maybe arc length would be better,
        # but it is making the tail too short)
        vent_idx_spl = idx_pts[vent_idx]

        # mass distribution
        mass_spl_i = np.interp(ts / SVL, s_rho, body_rho)
        mass_spl_i = mass * mass_spl_i / mass_spl_i.sum()

        # chord length distribution
        chord_spl[i] = SVL * np.interp(ts / SVL, s_chord, body_chord)

        # center of mass
        Ro_I[i] = np.sum((r.T * mass_spl_i).T, axis=0) / mass

        # store the spline and its derivative (for tangent angle calculations)
        R_I[i] = r
        dRds_I[i] = dr
        ddRds_I[i] = ddr
        dddRds_I[i] = dddr
        spl_ds[i] = seg_lens
        mass_spl[i] = mass_spl_i
        vent_idx_spls[i] = vent_idx_spl
        times2D[i] = times[i]
        t_coord[i] = ts
        s_coord[i] = ss
        spl_len_errors[i] = spl_len_error

    # the vent should be located at the same place for all splines
    assert(np.sum(vent_idx_spls == vent_idx_spls[0]) == ntime)

    # just use the one vent index into the spline
    # idx_pts should also be good then
    vent_idx_spl = vent_idx_spls[0]

    out = dict(Ro_I=Ro_I, R_I=R_I, dRds_I=dRds_I, ddRds_I=ddRds_I,
               dddRds_I=dddRds_I,
               spl_ds=spl_ds, mass_spl=mass_spl, chord_spl=chord_spl,
               vent_idx_spl=vent_idx_spl, times2D=times2D, t_coord=t_coord,
               s_coord=s_coord, spl_len_errors=spl_len_errors,
               idx_pts=idx_pts)

    return out


def body_vel_acc(pf_I, pfe_I, R_I, Ro_I, dt):
    """CoM shift the positions and calculate velocities and accelerations.

    Parameters
    ----------
    pf_I : array, size (ntime, nmark, 3)
        Marker positions, filtered
    pfe_I : array, size (ntime, nmark + 1, 3)
        Expended marker positions, filtered
    R_I : array, size (ntime, nspl, 3)
        Spline positions in inertial frame
    Ro_I : array, size (ntime, 3)
        CoM position in inertial frame
    dt : float
        time step

    Returns
    -------
    out = dict(R_Ic=R_Ic, pf_Ic=pf_Ic, pfe_Ic=pfe_Ic,
               vf_Ic=vf_Ic, af_Ic=af_Ic,
               dR_I=dR_I, ddR_I=ddR_I, dR_Ic=dR_Ic, ddR_Ic=ddR_Ic)
    """

    ntime, nspl, nmark = R_I.shape[0], R_I.shape[1], pf_I.shape[1]

    # CoM shift the positions
    R_Ic = R_I.copy()
    pf_Ic = pf_I.copy()
    pfe_Ic = pfe_I.copy()
    for i in np.arange(ntime):
        R_Ic[i] = R_I[i] - Ro_I[i]
        pf_Ic[i] = pf_I[i] - Ro_I[i]
        pfe_Ic[i] = pfe_Ic[i] - Ro_I[i]

    # marker velocies and accelerations relative to CoM
    vf_Ic = np.zeros_like(pf_I)
    af_Ic = np.zeros_like(pf_I)
    for j in np.arange(nmark):
        vv, aa = smoothing.findiff(pf_Ic[:, j], dt)
        vf_Ic[:, j] = vv
        af_Ic[:, j] = aa

    # spline velocity and accelerations relative to CoM
    dR_I = np.zeros((ntime, nspl, 3))
    ddR_I = np.zeros((ntime, nspl, 3))
    dR_Ic = np.zeros((ntime, nspl, 3))
    ddR_Ic = np.zeros((ntime, nspl, 3))
    for j in np.arange(nspl):
        vv, aa = smoothing.findiff(R_I[:, j], dt)
        dR_I[:, j] = vv
        ddR_I[:, j] = aa

        vv, aa = smoothing.findiff(R_Ic[:, j], dt)
        dR_Ic[:, j] = vv
        ddR_Ic[:, j] = aa

    out = dict(R_Ic=R_Ic, pf_Ic=pf_Ic, pfe_Ic=pfe_Ic,
               vf_Ic=vf_Ic, af_Ic=af_Ic,
               dR_I=dR_I, ddR_I=ddR_I, dR_Ic=dR_Ic, ddR_Ic=ddR_Ic)

    return out


def straighten_trajectory(filtered, com, splines, splines_ds):
    """Successively straighten the CoM trajectory.

    Parameters
    ----------
    Ro_I : array, size (ntime, 3)
        CoM position in inertial frame
    R_I : array, size (ntime, 3)
    """

    # unpack values
    pfe_I, pf_I, vf_I, af_I, pfe_Ic, pf_Ic, vf_Ic, af_Ic = filtered
    Ro_I, dRo_I, ddRo_I = com
    R_I, dR_I, ddR_I, R_Ic, dR_Ic, ddR_Ic = splines
    dRds_I, ddRds_I, dddRds_I = splines_ds

    nmark = pf_I.shape[1]
    ntime, nspl, _ = R_I.shape

    # straightened CoM position
    Ro0 = Ro_I[0]
    Ro_S = Ro_I - Ro0

    # iterate through the points a find the successive rotation angles and
    # straighten the CoM trajectory
    mus = np.zeros(ntime)
    Rmus = np.zeros((ntime, 3, 3))
    for i in np.arange(ntime):
        uu = Ro_S[i]
        mu = -np.arctan2(uu[0], uu[1])  # -tan^-1(px / py)
        Rmu = np.array([[np.cos(mu), np.sin(mu), 0],
                        [-np.sin(mu),  np.cos(mu), 0],
                        [0, 0, 1]])

        if i == 1:
            start = 0
        else:
            start = i

        mus[start] = mu
        Rmus[start] = Rmu

        # apply the rotation to each point along the spline
        for ii in np.arange(start, ntime):
            Ro_S[ii] = np.dot(Rmu, Ro_S[ii].T).T  # com

    # move back to start of inertial CoM position
    Ro_S = Ro_S + Ro0

    # use the yaw angle to rotate points and velocities
    yaw = np.arctan2(-dRo_I[:, 0], dRo_I[:, 1])  # yaw angle

    # rotation matrix about yaw axis
    C_I2S = np.zeros((ntime, 3, 3))

    # centered positions
    R_Sc = np.zeros_like(R_Ic)
    pf_Sc = np.zeros_like(pf_Ic)
    pfe_Sc = np.zeros_like(pfe_Ic)

    # velocities
    vf_S = np.zeros_like(vf_I)
    vf_Sc = np.zeros_like(vf_Ic)
    dRo_S = np.zeros_like(dRo_I)
    dR_S = np.zeros_like(dR_I)
    dR_Sc = np.zeros_like(dR_Ic)

    # accelerations
    af_S = np.zeros_like(af_I)
    af_Sc = np.zeros_like(af_Ic)
    ddRo_S = np.zeros_like(ddRo_I)
    ddR_S = np.zeros_like(ddR_I)
    ddR_Sc = np.zeros_like(ddR_Ic)

    # spline derivatives
    dRds_S = np.zeros_like(dRds_I)
    ddRds_S = np.zeros_like(ddRds_I)
    dddRds_S = np.zeros_like(dddRds_I)

    for i in np.arange(ntime):
        # rotation matrix
        C_I2S[i] = np.array([[np.cos(yaw[i]), np.sin(yaw[i]), 0],
                             [-np.sin(yaw[i]),  np.cos(yaw[i]), 0],
                             [0, 0, 1]])

        # centered positions
        R_Sc[i] = np.dot(C_I2S[i], R_Ic[i].T).T
        pf_Sc[i] = np.dot(C_I2S[i], pf_Ic[i].T).T
        pfe_Sc[i] = np.dot(C_I2S[i], pfe_Ic[i].T).T

        # velocities
        vf_S[i] = np.dot(C_I2S[i], vf_I[i].T).T
        vf_Sc[i] = np.dot(C_I2S[i], vf_Ic[i].T).T
        dRo_S[i] = np.dot(C_I2S[i], dRo_I[i].T).T
        dR_S[i] = np.dot(C_I2S[i], dR_I[i].T).T
        dR_Sc[i] = np.dot(C_I2S[i], dR_Ic[i].T).T

        # accelerations
        af_S[i] = np.dot(C_I2S[i], af_I[i].T).T
        af_Sc[i] = np.dot(C_I2S[i], af_Ic[i].T).T
        ddRo_S[i] = np.dot(C_I2S[i], ddRo_I[i].T).T
        ddR_S[i] = np.dot(C_I2S[i], ddR_I[i].T).T
        ddR_Sc[i] = np.dot(C_I2S[i], ddR_Ic[i].T).T

        # spline derivatives
        dRds_S[i] = np.dot(C_I2S[i], dRds_I[i].T).T
        ddRds_S[i] = np.dot(C_I2S[i], ddRds_I[i].T).T
        dddRds_S[i] = np.dot(C_I2S[i], dddRds_I[i].T).T

    # uncentered positions
    R_S = np.zeros_like(R_Sc)
    pf_S = np.zeros_like(pf_Sc)
    pfe_S = np.zeros_like(pfe_Sc)
    for i in np.arange(ntime):
        R_S[i] = Ro_S[i] + R_Sc[i]
        pf_S[i] = Ro_S[i] + pf_Sc[i]
        pfe_S[i] = Ro_S[i] + pfe_Sc[i]

    # glide angle based on straighted trajectory velcity down from horizontal
    # this ensures when rotated to this direction, only velocity in yhat
    gamma = -np.arctan2(dRo_S[:, 2], dRo_S[:, 1])  # glide angle

    out = dict(Ro_S=Ro_S, R_Sc=R_Sc, pf_Sc=pf_Sc, pfe_Sc=pfe_Sc,
               R_S=R_S, pf_S=pf_S, pfe_S=pfe_S,
               vf_S=vf_S, vf_Sc=vf_Sc, dRo_S=dRo_S, dR_S=dR_S, dR_Sc=dR_Sc,
               af_S=af_S, af_Sc=af_Sc, ddRo_S=ddRo_S, ddR_S=ddR_S, ddR_Sc=ddR_Sc,
               dRds_S=dRds_S, ddRds_S=ddRds_S, dddRds_S=dddRds_S,
               mus=mus, Rmus=Rmus,
               yaw=yaw, gamma=gamma, C_I2S=C_I2S)

    return out


# def OLD_straighten_trajectory(filtered, com, splines, splines_ds):
#     """Successively straighten the CoM trajectory.
#
#     Parameters
#     ----------
#
#     Returns
#     -------
#     out = dict(Ro_S=Ro_S, dRo_S=dRo_S, ddRo_S=ddRo_S, R_S=R_S, pfe_S=pfe_S,
#            pf_S=pf_S, vf_S=vf_S, af_S=af_S,
#            dRds_S=dRds_S, ddRds_S=ddRds_S, dddRds_S=dddRds_S,
#            dR_S=dR_S, ddR_S=ddR_S, dR_Sc=dR_Sc, ddR_Sc=ddR_Sc,
#            R_Sc=R_Sc, pf_Sc=pf_Sc, pfe_Sc=pfe_Sc, C_I2S=C_I2S)
#     """
#
#     # unpack values
#     pf_I, vf_I, af_I, pfe_I = filtered
#     Ro_I, dRo_I, ddRo_I = com
#     R_Ic, R_I, dR_I, ddR_I, dR_Ic, ddR_Ic = splines
#     dRds_I, ddRds_I, dddRds_I = splines_ds
#
#     nmark = pf_I.shape[1]
#     ntime, nspl, _ = R_I.shape
#
#     # shift to 'com'. Note, this is not strictly needed
#     Ro0 = Ro_I[0]
#     Ro_S = Ro_I - Ro0  # straightened CoM position
#     dRo_S = dRo_I.copy()
#     ddRo_S = ddRo_I.copy()
#     R_S = R_I - Ro0
#     pfe_S = pfe_I - Ro0
#
#     # apply to original markers
#     pf_S = pf_I - Ro0
#     vf_S = vf_I.copy()
#     af_S = af_I.copy()
#
#     # spline derivatives in straightened frame
#     dRds_S = dRds_I.copy()
#     ddRds_S = ddRds_I.copy()
#     dddRds_S = dddRds_I.copy()
#
#     # velocity and accelerations of the spline
#     dR_S = dR_I.copy()
#     ddR_S = ddR_I.copy()
#     dR_Sc = dR_Ic.copy()
#     ddR_Sc = ddR_Ic.copy()
#
#     # rotation angles to straighten trajectory
#     mus = np.zeros(ntime)
#     Rmus = np.zeros((ntime, 3, 3))
#     for i in np.arange(ntime):
#         Rmus[i] = np.eye(3)
#
#     # iterate through the points a find the successive roations
#     for i in np.arange(ntime):
#         uu = Ro_S[i]
#         mu = np.arctan2(uu[0], uu[1])  # tan^-1(px / py)
#         Rmu = np.array([[np.cos(mu), -np.sin(mu), 0],
#                         [np.sin(mu),  np.cos(mu), 0],
#                         [0, 0, 1]])  #NOTE: Rmu = R3(yaw).T
#
#         if i == 1:
#             start = 0
#         else:
#             start = i
#
#         mus[start] = mu
#         Rmus[start] = Rmu
#
#         # apply the rotation to each point along the spline
#         for ii in np.arange(start, ntime):
#             Ro_S[ii] = np.dot(Rmu, Ro_S[ii].T).T  # com
#             R_S[ii] = np.dot(Rmu, R_S[ii].T).T  # body
#             dRo_S[ii] = np.dot(Rmu, dRo_S[ii].T).T  # com velocity
#             ddRo_S[ii] = np.dot(Rmu, ddRo_S[ii].T).T  # com acceleration
#             pfe_S[ii] = np.dot(Rmu, pfe_S[ii].T).T
#             dRds_S[ii] = np.dot(Rmu, dRds_S[ii].T).T
#             ddRds_S[ii] = np.dot(Rmu, ddRds_S[ii].T).T
#             dddRds_S[ii] = np.dot(Rmu, dddRds_S[ii].T).T
#             pf_S[ii] = np.dot(Rmu, pf_S[ii].T).T
#             vf_S[ii] = np.dot(Rmu, vf_S[ii].T).T
#             af_S[ii] = np.dot(Rmu, af_S[ii].T).T
#             dR_S[ii] = np.dot(Rmu, dR_S[ii].T).T
#             ddR_S[ii] = np.dot(Rmu, ddR_S[ii].T).T
#             dR_Sc[ii] = np.dot(Rmu, dR_Sc[ii].T).T
#             ddR_Sc[ii] = np.dot(Rmu, ddR_Sc[ii].T).T
#
#     # com centered spline and points
#     R_Sc = np.zeros((ntime, nspl, 3))
#     pf_Sc = np.zeros((ntime, nmark, 3))
#     pfe_Sc = np.zeros((ntime, nmark + 1, 3))
#     for i in np.arange(ntime):
#         R_Sc[i] = R_S[i] - Ro_S[i]
#         pf_Sc[i] = pf_S[i] - Ro_S[i]
#         pfe_Sc[i] = pfe_S[i] - Ro_S[i]
#
#     # move the trajectory back to the inital com position, except com_X = 0
#     Ro0_move = Ro0  # np.r_[0, Ro0[1:]]
#     Ro_S = Ro_S + Ro0_move
#     R_S = R_S + Ro0_move
#     #pf_Sc = pf_Sc + Ro0_move
#     pfe_S = pfe_S + Ro0_move
#
#     # "chain together" rotation matrices
#     C_I2S = Rmus.copy()
#     for i in np.arange(1, ntime):
#         C_I2S[i] = np.dot(Rmus[i], C_I2S[i - 1])
#
#     # test the continuous rotation matrix
#     Ro_S_test = np.zeros(Ro_S.shape)
#     Ro_I_test = Ro_I - Ro0
#     R_Sc_test = np.zeros_like(R_S)
#     for i in np.arange(ntime):
#         Ro_S_test[i] = np.dot(C_I2S[i], Ro_I_test[i].T).T
#         R_Sc_test[i] = np.dot(C_I2S[i], R_Ic[i].T).T
#     Ro_S_test += Ro0_move
#     assert np.allclose(Ro_S, Ro_S_test)
#     assert np.allclose(R_Sc_test, R_Sc)
#
#     out = dict(Ro_S=Ro_S, dRo_S=dRo_S, ddRo_S=ddRo_S, R_S=R_S, pfe_S=pfe_S,
#            pf_S=pf_S, vf_S=vf_S, af_S=af_S,
#            dRds_S=dRds_S, ddRds_S=ddRds_S, dddRds_S=dddRds_S,
#            dR_S=dR_S, ddR_S=ddR_S, dR_Sc=dR_Sc, ddR_Sc=ddR_Sc,
#            R_Sc=R_Sc, pf_Sc=pf_Sc, pfe_Sc=pfe_Sc, C_I2S=C_I2S)
#
#     return out


def apply_body_cs(dRds_I, ddRds_I, dddRds_I, C_I2S):
    """Apply the body coordinate system (tangent, chord, backbone) unit vectors.

    Note: this method only uses the tangent vector and the Zhat direction to
    get bhat closes to Zhat

    Parameters
    ----------
    dRds_S, ddRds_S, ddRds_S, C_I2S

    Returns
    -------
    out = dict(Tdir_S=Tdir_S, Cdir_S=Cdir_S, Bdir_S=Bdir_S, Cb_S=Cb_S,
           a_angs=a_angs, b_angs=b_angs, kap_signed=kap_signed,
           kap_unsigned=kap_unsigned, tau=tau,
           Tdir_I=Tdir_I, Cdir_I=Cdir_I, Bdir_I=Bdir_I, Cb_I=Cb_I)
    """

    ntime, nspl, _ = dRds_I.shape

    Tdir_I = np.zeros((ntime, nspl, 3))
    Cdir_I = np.zeros((ntime, nspl, 3))
    Bdir_I = np.zeros((ntime, nspl, 3))

    Cb_I = np.zeros((ntime, nspl, 3, 3))

    a_angs = np.zeros((ntime, nspl))
    b_angs = np.zeros((ntime, nspl))

    kap_signed = np.zeros((ntime, nspl))
    kap_unsigned =np.zeros((ntime, nspl))
    tau = np.zeros((ntime, nspl))

    zhat = np.array([0, 0, 1])  # in the inertial frame

    for i in np.arange(ntime):

        # use spline derivatives in inertial frame
        dr = dRds_I[i]

        # TCB frame
        # https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
        Tdir_I[i] = (dr.T / np.linalg.norm(dr, axis=1)).T

        cdir_int = np.cross(zhat, Tdir_I[i])
        Cdir_I[i] = (cdir_int.T / np.linalg.norm(cdir_int, axis=1)).T
        Bdir_I[i] = np.cross(Tdir_I[i], Cdir_I[i])

        # store rotation matrix for the foil
        Cb_I[i, :, :, 0] = Tdir_I[i]
        Cb_I[i, :, :, 1] = Cdir_I[i]
        Cb_I[i, :, :, 2] = Bdir_I[i]

        # now iterate along the body (iterate to find the successive
        # bending and twisting angles)
        for j in np.arange(1, nspl):

            # bending and twisting angles of the snake/frame
            T0 = Tdir_I[i, j - 1]
            C0 = Cdir_I[i, j - 1]
            B0 = Bdir_I[i, j - 1]

            T1 = Tdir_I[i, j]
            C1 = Cdir_I[i, j]
            B1 = Bdir_I[i, j]

            T0_CT1 = T0 - np.dot(T0, B1) * B1
            T0_CT1 = T0_CT1 / np.linalg.norm(T0_CT1)
            alpha = np.arccos(np.dot(T1, T0_CT1))
            alpha_sign = np.sign(np.dot(C1, T0_CT1))
            alpha = alpha_sign * alpha

            C0_BC1 = C0 - np.dot(C0, T1) * T1
            C0_BC1 = C0_BC1 / np.linalg.norm(C0_BC1)
            beta = np.arccos(np.dot(C1, C0_BC1))
            beta_sign = np.sign(np.dot(B1, C0_BC1))
            beta = beta_sign * beta

            a_angs[i, j] = alpha
            b_angs[i, j] = beta

        # calculate signed and unsigned curvature and torsion
        dx, dy, dz = dRds_I[i].T
        ddx, ddy, ddz = ddRds_I[i].T
        dddx, dddy, dddz = dddRds_I[i].T
        for j in np.arange(nspl):
            k1 = ddz[j] * dy[j] - ddy[j] * dz[j]
            k2 = ddx[j] * dz[j] - ddz[j] * dx[j]
            k3 = ddy[j] * dx[j] - ddx[j] * dy[j]
            kn = (dx[j]**2 + dy[j]**2 + dz[j]**2)**1.5

            t1 = dddx[j] * k1
            t2 = dddy[j] * k2
            t3 = dddz[j] * k3
            tn = k1**2 + k2**2 + k3**2

            kap_signed[i, j] = (k1 + k2 + k3) / kn
            kap_unsigned[i, j] = np.sqrt(k1**2 + k2**2 + k3**2) / kn
            tau[i, j] = (t1 + t2 + t3) / tn


    # rotate the coordinate system to the interial frame
    Tdir_S = np.zeros_like(Tdir_I)
    Cdir_S = np.zeros_like(Cdir_I)
    Bdir_S = np.zeros_like(Bdir_I)
    Cb_S = np.zeros_like(Cb_I)
    for i in np.arange(ntime):
        Tdir_S[i] = np.dot(C_I2S[i], Tdir_I[i].T).T
        Cdir_S[i] = np.dot(C_I2S[i], Cdir_I[i].T).T
        Bdir_S[i] = np.dot(C_I2S[i], Bdir_I[i].T).T

        # torsion minimizing frame, but in inertial frame
        for j in np.arange(nspl):
            Cb_S[i, j] = np.dot(C_I2S[i], Cb_I[i, j])

    out = dict(Tdir_S=Tdir_S, Cdir_S=Cdir_S, Bdir_S=Bdir_S, Cb_S=Cb_S,
           a_angs=a_angs, b_angs=b_angs, kap_signed=kap_signed,
           kap_unsigned=kap_unsigned, tau=tau,
           Tdir_I=Tdir_I, Cdir_I=Cdir_I, Bdir_I=Bdir_I, Cb_I=Cb_I)

    return out


def apply_body_cs_bloomenthal(dRds_I, ddRds_I, dddRds_I, C_I2S):
    """Apply the body coordinate system (tangent, chord, backbone) unit vectors.

    Parameters
    ----------
    dRds_S, ddRds_S, ddRds_S, C_I2S

    Returns
    -------
    out = dict(Tdir_S=Tdir_S, Cdir_S=Cdir_S, Bdir_S=Bdir_S, Cb_S=Cb_S,
           a_angs=a_angs, b_angs=b_angs, kap_signed=kap_signed,
           kap_unsigned=kap_unsigned, tau=tau,
           Tdir_I=Tdir_I, Cdir_I=Cdir_I, Bdir_I=Bdir_I, Cb_I=Cb_I)
    """

    ntime, nspl, _ = dRds_I.shape

    Tdir_I = np.zeros((ntime, nspl, 3))
    Cdir_I = np.zeros((ntime, nspl, 3))
    Bdir_I = np.zeros((ntime, nspl, 3))

    Cb_I = np.zeros((ntime, nspl, 3, 3))

    a_angs = np.zeros((ntime, nspl))
    b_angs = np.zeros((ntime, nspl))

    kap_signed = np.zeros((ntime, nspl))
    kap_unsigned =np.zeros((ntime, nspl))
    tau = np.zeros((ntime, nspl))

    for i in np.arange(ntime):

        # use spline derivatives in inertial frame
        dr = dRds_I[i]

        # TNB frame
        # https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
        tdir = (dr.T / np.linalg.norm(dr, axis=1)).T
        tdir0 = tdir[0]  # this will be point back in -Yhat direction

        # xhat = np.r_[1, 0, 0]  # cdir0 should nominally be in xhat direciton
        # cdir0 = xhat - tdir0 * np.dot(tdir0, xhat)
        zhat = np.r_[0, 0, 1]
        tdir0_XY = tdir0 - zhat * np.dot(tdir0, zhat)  # = np.r_[tdir0[0], tdir0[1], 0]
        cdir0 = np.cross(zhat, tdir0_XY)
        cdir0 = cdir0 / np.linalg.norm(cdir0)
        bdir0 = np.cross(tdir0, cdir0)

        j = 0
        Tdir_I[i] = tdir
        Cdir_I[i, j] = cdir0
        Bdir_I[i, j] = bdir0

        # rotation matrix for foil shape defined in (y - z plane, with x = 0)
        Cr_foil = np.zeros((3, 3))
        Cr_foil[:, 0] = tdir0
        Cr_foil[:, 1] = cdir0
        Cr_foil[:, 2] = bdir0
        Cb_I[i, j] = Cr_foil

        # now iterate along the body, finding successive rotations
        # Bloomenthal (1990)
        for j in np.arange(1, nspl):
            T0 = Tdir_I[i, j - 1]  # tangent direction at head
            T1 = Tdir_I[i, j]
            T0 = T0 / np.linalg.norm(T0)
            T1 = T1 / np.linalg.norm(T1)
            A = np.cross(T0, T1)
            A = A / np.linalg.norm(A)  # why have to do this?

            a0 = np.cross(T0, A)
            a1 = np.cross(T1, A)

            m0 = np.array([T0, A, a0]).T
            m1 = np.array([T1, A, a1]).T
            Cr = np.dot(m0, m1.T)  # Cr = np.dot(m0.T, m1)

            # not 100% on why need to transpose (active vs. passive rotation?)
            # https://en.wikipedia.org/wiki/Active_and_passive_transformation
            Cr = Cr.T

            # store rotation matrix for the foil
            Cb_I[i, j] = np.dot(Cr, Cb_I[i, j - 1])

            C0 = Cdir_I[i, j - 1]
            B0 = Bdir_I[i, j - 1]
            C1 = np.dot(Cr, C0)
            B1 = np.dot(Cr, B0)
            Cdir_I[i, j] = C1
            Bdir_I[i, j] = B1

            # bending and twisting angles of the snake/frame
            T0_CT1 = T0 - np.dot(T0, B1) * B1
            T0_CT1 = T0_CT1 / np.linalg.norm(T0_CT1)
            alpha = np.arccos(np.dot(T1, T0_CT1))
            alpha_sign = np.sign(np.dot(C1, T0_CT1))
            alpha = alpha_sign * alpha

            C0_BC1 = C0 - np.dot(C0, T1) * T1
            C0_BC1 = C0_BC1 / np.linalg.norm(C0_BC1)
            beta = np.arccos(np.dot(C1, C0_BC1))
            beta_sign = np.sign(np.dot(B1, C0_BC1))
            beta = beta_sign * beta

            a_angs[i, j] = alpha
            b_angs[i, j] = beta

        # calculate signed and unsigned curvature and torsion
        dx, dy, dz = dRds_I[i].T
        ddx, ddy, ddz = ddRds_I[i].T
        dddx, dddy, dddz = dddRds_I[i].T
        for j in np.arange(nspl):
            k1 = ddz[j] * dy[j] - ddy[j] * dz[j]
            k2 = ddx[j] * dz[j] - ddz[j] * dx[j]
            k3 = ddy[j] * dx[j] - ddx[j] * dy[j]
            kn = (dx[j]**2 + dy[j]**2 + dz[j]**2)**1.5

            t1 = dddx[j] * k1
            t2 = dddy[j] * k2
            t3 = dddz[j] * k3
            tn = k1**2 + k2**2 + k3**2

            kap_signed[i, j] = (k1 + k2 + k3) / kn
            kap_unsigned[i, j] = np.sqrt(k1**2 + k2**2 + k3**2) / kn
            tau[i, j] = (t1 + t2 + t3) / tn


    # rotate the coordinate system to the interial frame
    Tdir_S = np.zeros_like(Tdir_I)
    Cdir_S = np.zeros_like(Cdir_I)
    Bdir_S = np.zeros_like(Bdir_I)
    Cb_S = np.zeros_like(Cb_I)
    for i in np.arange(ntime):
        Tdir_S[i] = np.dot(C_I2S[i], Tdir_I[i].T).T
        Cdir_S[i] = np.dot(C_I2S[i], Cdir_I[i].T).T
        Bdir_S[i] = np.dot(C_I2S[i], Bdir_I[i].T).T

        # torsion minimizing frame, but in inertial frame
        for j in np.arange(nspl):
            Cb_S[i, j] = np.dot(C_I2S[i], Cb_I[i, j])

    out = dict(Tdir_S=Tdir_S, Cdir_S=Cdir_S, Bdir_S=Bdir_S, Cb_S=Cb_S,
           a_angs=a_angs, b_angs=b_angs, kap_signed=kap_signed,
           kap_unsigned=kap_unsigned, tau=tau,
           Tdir_I=Tdir_I, Cdir_I=Cdir_I, Bdir_I=Bdir_I, Cb_I=Cb_I)

    return out


def apply_airfoil_shape(r, chord, Cb):
    """Apply the snake airfoil shape to the centered spline.

    Parameters
    ----------
    r : array, size (ntime, nspl, 3)
        Spline of the snake body (usually the CoM version)
    chord : array, size (ntime, nspl)
        Chord length in mm as a function of distance along the body
    Cb : array, size (ntime, nspl, 3, 3)
        Rotation matrices to rotate shape in a torsion-minimizing fashion

    Returns
    -------
    foils : array, size (ntime, nspl, nfoil, 3)
        X, Y, Z coordinates of the foil shape
    foil_color : array, size (ntime, nspl, nfoil)
        float array that the colormap uses (underside is yellow, top is green
        when using 'YlGn' colormap in Mayavi
    """

    ntime, nbody = r.shape[0], r.shape[1]

    # load in the airfoil shape
    rfoil = np.genfromtxt('../Data/Foil/snake0.004.bdy.txt', skip_header=1)
    rfoil = rfoil - rfoil.mean(axis=0)
    rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
    rfoil /= np.ptp(rfoil[:, 0])
    rfoil = rfoil[::5]
    _r0 = np.zeros(rfoil.shape[0])  # start with 0 in the Xhat direction
    rfoil = np.c_[_r0, rfoil[:, 0], rfoil[:, 1]]  # in YZ frame to start
    rfoil = np.c_[rfoil.T, rfoil[0]].T
    nfoil = rfoil.shape[0]

    # color the underside yellow
    idxbottom_L = np.argmin(rfoil[:, 1])
    idxbottom_R = np.argmax(rfoil[:, 1])
    idxbottom = np.arange(idxbottom_L + 1, idxbottom_R - 1)

    # airfoil shape
    # nbody + 2 to close the ends of the mesh for better plotting
    foils = np.zeros((ntime, nbody + 2, nfoil, 3))
    foil_color = .7 * np.ones((ntime, nbody + 2, nfoil))

    for i in np.arange(ntime):
        # interior mesh
        for j in np.arange(nbody):

            # index into foils and foil_colors, since have nbody + 2 points
            jj = j + 1
            # scale foil shape by chord length
            foil0 = rfoil * chord[i, j]

            # rotate into CB plane
            rotated_foil = np.dot(Cb[i, j], foil0.T).T

            # move to position along body
            foils[i, jj] = r[i, j] + rotated_foil

            # airfoil color for plotting
            foil_color[i, jj, idxbottom] = .3

        # end caps
        foils[i, 0] = foils[i, 1].mean(axis=0)
        foils[i, -1] = foils[i, -2].mean(axis=0)
        foil_color[i, 0] = foil_color[i, 1].mean()
        foil_color[i, -1] = foil_color[i, -2].mean()

    return foils, foil_color


def apply_airfoil_shape_cfd(r, chord, Cb):
    """Apply the snake airfoil shape to the centered spline,
    subsampled for CFD, hwere .

    Parameters
    ----------
    r : array, size (ntime, nspl, 3)
        Spline of the snake body (usually the CoM version)
    chord : array, size (ntime, nspl)
        Chord length in mm as a function of distance along the body
    Cb : array, size (ntime, nspl, 3, 3)
        Rotation matrices to rotate shape in a torsion-minimizing fashion

    Returns
    -------
    foils : array, size (ntime, nspl, nfoil, 3)
        X, Y, Z coordinates of the foil shape
    foil_color : array, size (ntime, nspl, nfoil)
        float array that the colormap uses (underside is yellow, top is green
        when using 'YlGn' colormap in Mayavi
    """

    ntime, nbody = r.shape[0], r.shape[1]

    # load in the airfoil shape
    rfoil = np.genfromtxt('../Data/Foil/snake0.004.bdy.txt', skip_header=1)
    rfoil = rfoil - rfoil.mean(axis=0)
    rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
    rfoil /= np.ptp(rfoil[:, 0])

    # downsample 585 to 118 points
    rfoil = rfoil[::5]
    rfoil = np.c_[rfoil.T, rfoil[0]].T

    # downsample 118 to 31 points
    rfoil_small = rfoil[::4]
    rfoil_small = np.c_[rfoil_small.T, rfoil_small[0]].T

    # combine rfoil and rfoil_small, emphaising the lips and top to 56 points
    rf = np.c_[rfoil[0:3].T,
               rfoil_small[1:7].T,
               rfoil[27:42].T,
               rfoil_small[11:20].T,
               rfoil[78:91].T,
               rfoil_small[23:29].T,
               rfoil[115:117].T].T
    rf = np.c_[rf.T, rf[0]].T
    rfoil = rf.copy()

    # make a 3D array to rotate
    _r0 = np.zeros(rfoil.shape[0])  # start with 0 in the Xhat direction
    rfoil = np.c_[_r0, rfoil[:, 0], rfoil[:, 1]]  # in YZ frame to start
    rfoil = np.c_[rfoil.T, rfoil[0]].T
    nfoil = rfoil.shape[0]

    # color the underside yellow
    idxbottom_L = np.argmin(rfoil[:, 1])
    idxbottom_R = np.argmax(rfoil[:, 1])
    idxbottom = np.arange(idxbottom_L + 1, idxbottom_R - 1)

    # airfoil shape
    # nbody + 2 to close the ends of the mesh for better plotting
    foils = np.zeros((ntime, nbody + 2, nfoil, 3))
    foil_color = .7 * np.ones((ntime, nbody + 2, nfoil))

    for i in np.arange(ntime):
        # interior mesh
        for j in np.arange(nbody):

            # index into foils and foil_colors, since have nbody + 2 points
            jj = j + 1
            # scale foil shape by chord length
            foil0 = rfoil * chord[i, j]

            # rotate into CB plane
            rotated_foil = np.dot(Cb[i, j], foil0.T).T

            # move to position along body
            foils[i, jj] = r[i, j] + rotated_foil

            # airfoil color for plotting
            foil_color[i, jj, idxbottom] = .3

        # end caps
        foils[i, 0] = foils[i, 1].mean(axis=0)
        foils[i, -1] = foils[i, -2].mean(axis=0)
        foil_color[i, 0] = foil_color[i, 1].mean()
        foil_color[i, -1] = foil_color[i, -2].mean()

    return foils, foil_color


def fit_comoving_frame_B(R_Ic, C_I2S, mass_spl, vent_idx_spl):
    """Fit a plane and body coordinate system to the snake.

    Parameters
    ----------
    R_Ic : array, size (ntime, nspl, 3)
        Position of the body in the inertial frame, CoM centered
    C_I2S : array, size (ntime, 3, 3)
        Rotation matrices to convert from inertial to straighted frame.
    mass_spl : array, size (ntime, nspl)
        Masses of the body
    vent_idx_spl : int
        Location of the vent along the spline

    Returns
    -------
    out = dict(C_I2B=C_I2B, Mw_I=Mw_I, Nhat_I=Nhat_I, Sfrac=Sfrac,
               planar_fit_error=planar_fit_error)
    """

    from scipy.linalg import svd

    ntime, nspl = mass_spl.shape

    # weights to apply to spline
    mass_body_weight = mass_spl[0, :vent_idx_spl + 1]  # head to vent
    weights = mass_body_weight / mass_body_weight.sum()

    # unit vectors
    Xhat = np.r_[1, 0, 0]
    Yhat = np.r_[0, 1, 0]
    Zhat = np.r_[0, 0, 1]

    # coordinates from SVD
    #V0 = np.zeros((ntime, 3))
    #V1 = np.zeros((ntime, 3))
    #V2 = np.zeros((ntime, 3))

    # planar normal vector, weighted spline
    Nhat_I = np.zeros((ntime, 3))
    Mw_I = np.zeros((ntime, len(weights), 3))

    # rotation matrices
    C_I2B = np.zeros((ntime, 3, 3))
    C1s = np.zeros((ntime, 3, 3))
    C2s = np.zeros((ntime, 3, 3))
    C3s = np.zeros((ntime, 3, 3))

    Sfrac = np.zeros((ntime, 3))  # measure of 'planeness'
    planar_fit_error = np.zeros(ntime)

    for i in np.arange(ntime):
        idx = vent_idx_spl + 1
        M = R_Ic[i, :idx]
        Mw = (M.T * weights).T  # weighted points

        U, S, V = svd(Mw)
        Svar = S**2

        # S is sorted singular values in descending order
        # rows of V are singluar values; we want row three
        n = V[2]

        # make sure the normal points up
        if np.dot(n, Zhat) < 0:
            n = -n

        # rotate normal vector into straightened frame
        na = np.dot(C_I2S[i], n)

        # calculate pitch angle
        pitch_i = np.arctan2(na[1], na[2])  # tan^-1(na_y, na_z)

        # C1(pitch)
        C1_i = np.array([[1, 0, 0],
                         [0, np.cos(pitch_i), np.sin(pitch_i)],
                         [0, -np.sin(pitch_i), np.cos(pitch_i)]])
        C1_i = C1_i.T  #TODO do this? yes (2016-10-19, but not sure why yet)
        # this is opposite convention Diebel, p. 5, p. 28 (2-1-3 rotation)
        # but we use this opposite rotation convention for yaw (C_I2S)

        # rotate normal about x-axis (Xhat?) into XZ plane
        nb = np.dot(C1_i, na)

        # calculate the roll angle
        roll_i = np.arctan2(nb[0], nb[2])  # tan^-1(nb_x, nb_z)

        # C2(roll)
        C2_i = np.array([[np.cos(roll_i), 0, -np.sin(roll_i)],
                         [0, 1, 0],
                         [np.sin(roll_i), 0, np.cos(roll_i)]])
        # C2_i = C2_i.T  #TODO do this? (2016-10-19, not, but not sure why yet)

        C1s[i] = C1_i
        C2s[i] = C2_i
        C3s[i] = C_I2S[i]

        Mw_I[i] = Mw
        Nhat_I[i] = n

        # C = C2(roll) * C1(pitch) * C3(yaw)
        C_I2B[i] = np.dot(C2s[i], np.dot(C1s[i], C3s[i]))
        #NOTE: np.dot(C_I2B[i], Nhat[i]) = Xhat

        Sfrac[i] = Svar / Svar.sum()
        # planar_fit_error[i] = np.sqrt(np.sum(np.dot(M, n)**2))
        planar_fit_error[i] = np.sqrt(np.sum(np.dot(Mw, n)**2))

        #V0[i] = V[0]
        #V1[i] = V[1]
        #V2[i] = V[2]

    out = dict(C_I2B=C_I2B, Mw_I=Mw_I, Nhat_I=Nhat_I, Sfrac=Sfrac,
               planar_fit_error=planar_fit_error)

    return out


def rotate_to_B(C_I2B, _Ro, _dRds, _R, _mark, _foils):
    """Rotate values from inertial to body frame.

    Parameters
    ----------
    C_I2B : array, size (ntime, 3, 3)
        Rotation matrix from intertial to body frames.
     _Ro, _dRds, _R, _mark, _foils

    Returns
    -------
    out = dict(dRo_B=dRo_B, ddRo_B=ddRo_B,
               dRds_B=dRds_B, ddRds_B=ddRds_B,
               R_B=R_B, dR_B=dR_B, ddR_B=ddR_B,
               pf_B=pf_B, vf_B=vf_B, af_B=af_B, pfe_B=pfe_B,
               dR_Bc=dR_Bc, ddR_Bc=ddR_Bc, vf_Bc=vf_Bc, af_Bc=af_Bc,
               foils_B=foils_B, Tdir_B=Tdir_B, Cdir_B=Cdir_B, Bdir_B=Bdir_B)
    """

    # unpack the arrays
    dRo_I, ddRo_I = _Ro
    dRds_I, ddRds_I = _dRds
    R_Ic, dR_Ic, ddR_Ic, dR_I, ddR_I = _R
    pf_Ic, vf_I, af_I, pfe_Ic, vf_Ic, af_Ic = _mark
    foils_Ic, Tdir_I, Cdir_I, Bdir_I = _foils

    ntime, nspl, _ = R_Ic.shape

    # CoM velocity and accerlation
    dRo_B = np.zeros_like(dRo_I)
    ddRo_B = np.zeros_like(ddRo_I)

    # spline derivatives
    dRds_B = np.zeros_like(dRds_I)
    ddRds_B = np.zeros_like(ddRds_I)

    # spline velocity and accerlation
    R_B = np.zeros_like(R_Ic)
    dR_B = np.zeros_like(dR_I)
    ddR_B = np.zeros_like(ddR_I)

    # markers
    pf_B = np.zeros_like(pf_Ic)
    vf_B = np.zeros_like(vf_I)
    af_B = np.zeros_like(af_I)
    pfe_B = np.zeros_like(pfe_Ic)

    # relative to CoM [...] velocities and accelerations
    dR_Bc = np.zeros_like(dR_Ic)
    ddR_Bc = np.zeros_like(ddR_Ic)
    vf_Bc = np.zeros_like(vf_Ic)
    af_Bc = np.zeros_like(af_Ic)

    # body coordinate system and foils for plotting
    foils_B = np.zeros_like(foils_Ic)
    Tdir_B = np.zeros_like(Tdir_I)
    Cdir_B = np.zeros_like(Cdir_I)
    Bdir_B = np.zeros_like(Bdir_I)

    for i in np.arange(ntime):
        dRo_B[i] = np.dot(C_I2B[i], dRo_I[i].T).T
        ddRo_B[i] = np.dot(C_I2B[i], ddRo_I[i].T).T

        dRds_B[i] = np.dot(C_I2B[i], dRds_I[i].T).T
        ddRds_B[i] = np.dot(C_I2B[i], ddRds_I[i].T).T

        R_B[i] = np.dot(C_I2B[i], R_Ic[i].T).T
        dR_B[i] = np.dot(C_I2B[i], dR_I[i].T).T
        ddR_B[i] = np.dot(C_I2B[i], ddR_I[i].T).T

        pf_B[i] = np.dot(C_I2B[i], pf_Ic[i].T).T
        vf_B[i] = np.dot(C_I2B[i], vf_I[i].T).T
        af_B[i] = np.dot(C_I2B[i], af_I[i].T).T
        pfe_B[i] = np.dot(C_I2B[i], pfe_Ic[i].T).T

        dR_Bc[i] = np.dot(C_I2B[i], dR_Ic[i].T).T
        ddR_Bc[i] = np.dot(C_I2B[i], ddR_Ic[i].T).T
        vf_Bc[i] = np.dot(C_I2B[i], vf_Ic[i].T).T
        af_Bc[i] = np.dot(C_I2B[i], af_Ic[i].T).T

        for j in np.arange(nspl):
            Tdir_B[i, j] = np.dot(C_I2B[i], Tdir_I[i, j])
            Cdir_B[i, j] = np.dot(C_I2B[i], Cdir_I[i, j])
            Bdir_B[i, j] = np.dot(C_I2B[i], Bdir_I[i, j])

        for j in np.arange(foils_Ic.shape[1]):
            foils_B[i, j] = np.dot(C_I2B[i], foils_Ic[i, j].T).T

    out = dict(dRo_B=dRo_B, ddRo_B=ddRo_B,
               dRds_B=dRds_B, ddRds_B=ddRds_B,
               R_B=R_B, dR_B=dR_B, ddR_B=ddR_B,
               pf_B=pf_B, vf_B=vf_B, af_B=af_B, pfe_B=pfe_B,
               dR_Bc=dR_Bc, ddR_Bc=ddR_Bc, vf_Bc=vf_Bc, af_Bc=af_Bc,
               foils_B=foils_B, Tdir_B=Tdir_B, Cdir_B=Cdir_B, Bdir_B=Bdir_B)

    return out


def apply_comoving_frame(C_I2B, C_I2S, _nmesh=21):
    """Apply the body coordinate system for plotting.

    Parameters
    ----------
    C_I2B : array, size (ntime, 3, 3)
        Rotation matrix from inertial to body frames
    C_I2S : array, size (ntime, 3, 3)
        Rotation matrix from inertial to straightened frames
    _nmesh : int, default=21
        Mesh for body planes

    Returns
    -------
    out = dict(Xp_B=Xp_B, Yp_B=Yp_B, Zp_B=Zp_B,
           Xp_I=Xp_I, Yp_I=Yp_I, Zp_I=Zp_I,
           Xp_S=Xp_S, Yp_S=Yp_S, Zp_S=Zp_S,
           YZ_B=YZ_B, XZ_B=XZ_B, XY_B=XY_B,
           YZ_I=YZ_I, XZ_I=XZ_I, XY_I=XY_I,
           YZ_S=YZ_S, XZ_S=XZ_S, XY_S=XY_S)

    Notes
    -----
    Yp_quiv = mlab.quiver3d(Yp_S[i, 0], Yp_S[i, 1], Yp_S[i, 2], scale_factor=75,
              color=bmap[1], mode='arrow', resolution=64)
    Xp_quiv = mlab.quiver3d(Xp_S[i, 0], Xp_S[i, 1], Xp_S[i, 2], scale_factor=75,
                  color=bmap[2], mode='arrow', resolution=64)
    Zp_quiv = mlab.quiver3d(Zp_S[i, 0], Zp_S[i, 1], Zp_S[i, 2], scale_factor=75,
                  color=bmap[0], mode='arrow', resolution=64)

    YZ_mesh = mlab.mesh(YZ_S[i, :, :, 0], YZ_S[i, :, :, 1], YZ_S[i, :, :, 2],
                        color=bmap[2], opacity=.25)
    XZ_mesh = mlab.mesh(XZ_S[i, :, :, 0], XZ_S[i, :, :, 1], XZ_S[i, :, :, 2],
                        color=bmap[1], opacity=.25)
    XY_mesh = mlab.mesh(XY_S[i, :, :, 0], XY_S[i, :, :, 1], XY_S[i, :, :, 2],
                        color=bmap[0], opacity=.25)
    """

    ntime = C_I2B.shape[0]

    # axes for plotting
    Xp_B = np.zeros((ntime, 3))
    Yp_B = np.zeros((ntime, 3))
    Zp_B = np.zeros((ntime, 3))
    Xp_I = np.zeros((ntime, 3))
    Yp_I = np.zeros((ntime, 3))
    Zp_I = np.zeros((ntime, 3))
    Xp_S = np.zeros((ntime, 3))
    Yp_S = np.zeros((ntime, 3))
    Zp_S = np.zeros((ntime, 3))

    # planes for plotting
    YZ_B = np.zeros((ntime, _nmesh, _nmesh, 3))
    XZ_B = np.zeros((ntime, _nmesh, _nmesh, 3))
    XY_B = np.zeros((ntime, _nmesh, _nmesh, 3))

    YZ_I = np.zeros((ntime, _nmesh, _nmesh, 3))
    XZ_I = np.zeros((ntime, _nmesh, _nmesh, 3))
    XY_I = np.zeros((ntime, _nmesh, _nmesh, 3))

    YZ_S = np.zeros((ntime, _nmesh, _nmesh, 3))
    XZ_S = np.zeros((ntime, _nmesh, _nmesh, 3))
    XY_S = np.zeros((ntime, _nmesh, _nmesh, 3))

    # extents of the mesh
    xx = np.linspace(-200, 200, _nmesh)
    yy = np.linspace(-200, 200, _nmesh)
    zz = np.linspace(-75, 75, _nmesh)
    YZ_y, YZ_z = np.meshgrid(yy, zz)
    XZ_x, XZ_z = np.meshgrid(xx, zz)
    XY_x, XY_y = np.meshgrid(xx, yy)

    for i in np.arange(ntime):
        Xp_B[i] = np.array([1, 0, 0])
        Yp_B[i] = np.array([0, 1, 0])
        Zp_B[i] = np.array([0, 0, 1])

        Xp_I[i] = np.dot(C_I2B[i].T, Xp_B[i])
        Yp_I[i] = np.dot(C_I2B[i].T, Yp_B[i])
        Zp_I[i] = np.dot(C_I2B[i].T, Zp_B[i])

        Xp_S[i] = np.dot(C_I2S[i], Xp_I[i])
        Yp_S[i] = np.dot(C_I2S[i], Yp_I[i])
        Zp_S[i] = np.dot(C_I2S[i], Zp_I[i])

        YZ_B[i, :, :, 0] = 0 * YZ_y
        YZ_B[i, :, :, 1] = YZ_y
        YZ_B[i, :, :, 2] = YZ_z

        XZ_B[i, :, :, 0] = XZ_x
        XZ_B[i, :, :, 1] = 0 * XZ_x
        XZ_B[i, :, :, 2] = XZ_z

        XY_B[i, :, :, 0] = XY_x
        XY_B[i, :, :, 1] = XY_y
        XY_B[i, :, :, 2] = 0 * XY_x

        for j in np.arange(_nmesh):
            # .T on rotation matrix b/c _B is in the body frame, so convert to _I
            YZ_I[i, j] = np.dot(C_I2B[i].T, YZ_B[i, j].T).T
            XZ_I[i, j] = np.dot(C_I2B[i].T, XZ_B[i, j].T).T
            XY_I[i, j] = np.dot(C_I2B[i].T, XY_B[i, j].T).T

            YZ_S[i, j] = np.dot(C_I2S[i], YZ_I[i, j].T).T
            XZ_S[i, j] = np.dot(C_I2S[i], XZ_I[i, j].T).T
            XY_S[i, j] = np.dot(C_I2S[i], XY_I[i, j].T).T

    out = dict(Xp_B=Xp_B, Yp_B=Yp_B, Zp_B=Zp_B,
               Xp_I=Xp_I, Yp_I=Yp_I, Zp_I=Zp_I,
               Xp_S=Xp_S, Yp_S=Yp_S, Zp_S=Zp_S,
               YZ_B=YZ_B, XZ_B=XZ_B, XY_B=XY_B,
               YZ_I=YZ_I, XZ_I=XZ_I, XY_I=XY_I,
               YZ_S=YZ_S, XZ_S=XZ_S, XY_S=XY_S)

    return out


def C2euler(C):
    """Euler angles from rotation matrix, using 3-1-2 convention.

    Parameters
    ----------
    C : proper rotation matrix which converts vectors from the inertial
        frame to the body frame

    Returns
    -------
    yaw, pitch, and roll angles in radians
    """

    # eqn 361 Diebel, but swap ROWS
    yaw = np.arctan2(-C[1, 0], C[1, 1])
    pitch = np.arcsin(C[1, 2])
    roll = np.arctan2(-C[0, 2], C[2, 2])

    return yaw, pitch, roll


def dang2omg(yaw, pitch, roll):
    """Body components of angular velocity from Euler angle rates. This
    is used to determine initial conditions of omega.

    Parameters
    ----------
    yaw : float
        yaw angle about z-axis in radians
    pitch : float
        pitch angle about y-axis in radians
    roll : float
        roll angle about x-axis in radians

    Returns
    -------
    euler_rates_matrix : array, size (3, 3)
        body-axis components of absolute angular velocity in terms of
        Euler angle rates. This needs to be dotted with Euler angle rates
        and C.T to get omega_0 in expressed in the inertial frame.
    """
    # p. 28 Diebel, eqn 365, but swap first and last COLUMNS
    c, s = np.cos, np.sin
    Kinv = np.array([[-c(pitch) * s(roll), c(roll), 0],
                     [s(pitch), 0, 1],
                     [c(pitch) * c(roll), s(roll), 0]])
    return Kinv


def euler_angles_and_omg(C_I2B, dt):
    """
    """

    ntime = C_I2B.shape[0]

    # calculate euler angles
    yaw = np.zeros(ntime)
    pitch = np.zeros(ntime)
    roll = np.zeros(ntime)
    for i in np.arange(ntime):
        yaw[i], pitch[i], roll[i] = C2euler(C_I2B[i])

    # yaw, pitch, and roll rates
    dyaw, ddyaw = smoothing.findiff(yaw, dt)
    dpitch, ddpitch= smoothing.findiff(pitch, dt)
    droll, ddroll = smoothing.findiff(roll, dt)

    # angular velocity and acceleration
    dang = np.c_[dyaw, dpitch, droll]
    omg_B = np.zeros_like(dang)
    omg_I = np.zeros_like(dang)
    for i in np.arange(ntime):
        Kinv = dang2omg(yaw[i], pitch[i], roll[i])
        omg_B[i] = np.dot(Kinv, dang[i])
        omg_I[i] = np.dot(C_I2B[i].T, omg_B[i])

    #TODO maybe these need to be smoothed?
    domg_B, ddomg_B = smoothing.findiff(omg_B, dt)
    domg_I, ddomg_I = smoothing.findiff(omg_I, dt)

    out = dict(yaw=yaw, pitch=pitch, roll=roll,
               dyaw=dyaw, dpitch=dpitch, droll=droll,
               ddyaw=ddyaw, ddpitch=ddpitch, ddroll=ddroll,
               omg_B=omg_B, domg_B=domg_B, ddomg_B=ddomg_B,
               omg_I=omg_I, domg_I=domg_I, ddomg_I=ddomg_I)

    return out


def dot(a, b):
    """Dot product for two (n x 3) arrays. This is a dot product
    for two vectors.

    Example from the aerodynamics function:
    a1 = np.array([np.dot(dR[i], Tv[i]) for i in np.arange(nbody)])
    a2 = np.diag(np.inner(dR, Tv))
    a3 = np.sum(dR * Tv, axis=1)
    np.allclose(a1, a2)
    np.allclose(a1, a3)
    """
    return np.sum(a * b, axis=1)


def aero_forces(Tv, Cv, Bv, dR, ds, c, rho, aero_interp, full_out=False):
    """Aerodynamic forces or each segment.

    Parameters
    ----------
    Tv : array, size (nbody, 3)
        tangent vector in interial frame
    Tv : array, size (nbody, 3)
        chord vector in inertial frame
    Tv : array, size (nbody, 3)
        backbone vector in inertial frame
    dR : array, size (nbody, 3)
        velocities of each mass, expressed in the inertial frame
    ds : float
        length of each piece of mass
    c : array, size (nbody)
        chord length in m of each piece of mass
    rho : float
        density of air in kg/m^3. A good value is rho = 1.165  # 30 C
    aero_interp : function
        function that returns force coeffients when passed
        aoa, U, c, and nu (optional)

    TODO UPDATE THE RETURN DICTIONARY
    Returns
    -------
    Fl : array, size (nbody)
        lift force in N
    Fd : array, size (nbody)
        drag force in N
    dRiBCs : array, size (nbody)
        velocity in the BC-plane of the snake airfoil, expressed in
        the inertial frame
    aoas : array, size (nbody)
        angles of attack in radians
    Res : array, size (nbody)
        Reynolds number
    """

    nbody = dR.shape[0]

    # # body coordinate system in intertial frame
    # Tv = rotate(C.T, tv)
    # Cv = rotate(C.T, cv)
    # Bv = rotate(C.T, bv)

    # we need consistent units -- meters
    mm2m = .001  # conversion from mm to m (length unit of c, ds, dRi)
    c = mm2m * c.copy()
    ds = mm2m * ds.copy()
    dR = mm2m * dR.copy()

    # velocity components parallel and perpendicular to arifoil
    dR_T = (dot(dR, Tv) * Tv.T).T  # dR_T = dot(dR, Tv) * Tv
    dR_BC = dR - dR_T  # velocity in B-C plan

    U_BC = np.linalg.norm(dR_BC, axis=1)  # reduced velocity in BC plane
    U_tot = np.linalg.norm(dR, axis=1)  # total velocity hitting mass (for Re calc)

    # angle of velocity in BC coordinate system
    cos_c = dot(dR_BC, Cv) / U_BC
    cos_b = dot(dR_BC, Bv) / U_BC

    # arccos is constrainted to [-1, 1] (due to numerical error)
    rad_c = np.arccos(np.clip(cos_c, -1, 1))
    rad_b = np.arccos(np.clip(cos_b, -1, 1))
    deg_c = np.rad2deg(rad_c)
    deg_b = np.rad2deg(rad_b)

    # unit vectors for drag and lift directions
    Dh = (-dR_BC.T / U_BC).T  # -dR_BC / U_BC
    Lh = np.cross(Tv, Dh)  # np.cross(Ti, Dh)
    aoa = np.zeros(nbody)

    # chat in -xhat, bhat = chat x that, bhat in +zhat
    Q1 = (deg_c < 90) & (deg_b >= 90)  # lower right
    Q2 = (deg_c < 90) & (deg_b < 90)  # upper right
    Q3 = (deg_c >= 90) & (deg_b < 90)  # upper left
    Q4 = (deg_c >= 90) & (deg_b >= 90)  # lower left

    # get sign and value of aoa and sign of Lh vector correct
    aoa = np.zeros(nbody)
    aoa[Q1] = rad_c[Q1]
    aoa[Q2] = -rad_c[Q2]
    aoa[Q3] = rad_c[Q3] - np.pi
    aoa[Q4] = np.pi - rad_c[Q4]
    Lh[Q1] = -Lh[Q1]
    Lh[Q2] = -Lh[Q2]

    # dynamic pressure
    dynP = .5 * rho * U_BC**2
    dA = ds * c  # area of each segment

    # now calculate the forces
    cl, cd, clcd, Re = aero_interp(aoa, U_tot, c)
    Fl = (dynP * dA * cl * Lh.T).T  # Fl = dynP * cl * Lh
    Fd = (dynP * dA * cd * Dh.T).T  # Fd = dynP * cd * Dh
    Fa = Fl + Fd  # total aerodynamic force

    if full_out:
        # sweep angle beta
        dR_B = (dot(dR, Bv) * Bv.T).T  # dR_B = np.dot(dR, Bv) * Bv
        dR_TC = dR - dR_B  # velocity in T-C plane
        U_TC = np.linalg.norm(dR_TC, axis=1)  # reduced velocity in TC plane
        cos_beta = dot(dR_TC, Tv) / U_TC
        beta = np.arccos(np.clip(cos_beta, -1, 1)) - np.pi / 2

        # fraction of dynP because of simple sweep theory assumption
        dynP_frac = U_BC**2 / U_tot**2

        # save aerodynamic variables in a dictionary
        out = dict(Fl=Fl, Fd=Fd, Fa=Fa, dR_T=dR_T, dR_BC=dR_BC, U_BC=U_BC,
                   U_tot=U_tot, Dh=Dh, Lh=Lh, aoa=aoa, dynP=dynP,
                   cl=cl, cd=cd, clcd=clcd, Re=Re,
                   dR_B=dR_B, dR_TC=dR_TC, U_TC=U_TC, beta=beta,
                   dynP_frac=dynP_frac)

        return out

    return Fa


def find_aero_forces_moments(R_Ic, TCB, dR_I, spl_ds, chord_spl, C_I2B, C_I2S):
    """Calculate aerodynamic forces, moments, and other quantities.

    Parameters
    ----------
    R_Ic : array, size (ntime, nspl, 3)
        Position of the snake in the inertial frame, relative to the CoM
    TCB : list, size (3)
        Tdir, Cdir, Bdir for the snake body coordinate system in the inertial frame
    dR_I : array, size (ntime, nspl, 3)
        Velocity in mm/s of each piece of the snake's body in the inertial frame
    spl_ds : array, size (ntime, nspl)
        The length of each spline segment
    chord_ds : array, szie (ntime, nspl)
        The width of each spline segment

    Returns
    -------
    out = dict(Fl_I=Fl_I, Fd_I=Fd_I, Fa_I=Fa_I,
               Ml_I=Ml_I, Md_I=Md_I, Ma_I=Ma_I,
               Fl_B=Fl_B, Fd_B=Fd_B, Fa_B=Fa_B,
               Ml_B=Ml_B, Md_B=Md_B, Ma_B=Ma_B,
               Fl_S=Fl_S, Fd_S=Fd_S, Fa_S=Fa_S,
               Ml_S=Ml_S, Md_S=Md_S, Ma_S=Ma_S,
               Re=Re, aoa=aoa, beta=beta, dynP_frac=dynP_frac,
               dR_BC_I=dR_BC_I, dR_TC_I=dR_TC_I,
               U_BC_I=U_BC_I, U_TC_I=U_TC_I,
               cl=cl, cd=cd, clcd=clcd)
    """

    Tdir_I, Cdir_I, Bdir_I = TCB
    ntime, nspl, _ = R_Ic.shape

    import m_aerodynamics as aerodynamics
    aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)

    rho = 1.17  # kg/m^3
    mm2m = .001

    Fl_I = np.zeros((ntime, nspl, 3))
    Fd_I = np.zeros((ntime, nspl, 3))
    Fa_I = np.zeros((ntime, nspl, 3))
    Ml_I = np.zeros((ntime, nspl, 3))
    Md_I = np.zeros((ntime, nspl, 3))
    Ma_I = np.zeros((ntime, nspl, 3))
    Fl_B = np.zeros((ntime, nspl, 3))
    Fd_B = np.zeros((ntime, nspl, 3))
    Fa_B = np.zeros((ntime, nspl, 3))
    Ml_B = np.zeros((ntime, nspl, 3))
    Md_B = np.zeros((ntime, nspl, 3))
    Ma_B = np.zeros((ntime, nspl, 3))
    Fl_S = np.zeros((ntime, nspl, 3))
    Fd_S = np.zeros((ntime, nspl, 3))
    Fa_S = np.zeros((ntime, nspl, 3))
    Ml_S = np.zeros((ntime, nspl, 3))
    Md_S = np.zeros((ntime, nspl, 3))
    Ma_S = np.zeros((ntime, nspl, 3))
    Re = np.zeros((ntime, nspl))
    aoa = np.zeros((ntime, nspl))
    beta = np.zeros((ntime, nspl))
    dynP = np.zeros((ntime, nspl))
    dynP_frac = np.zeros((ntime, nspl))
    dR_BC_I = np.zeros((ntime, nspl, 3))
    dR_TC_I = np.zeros((ntime, nspl, 3))
    U_BC_I = np.zeros((ntime, nspl))
    U_TC_I = np.zeros((ntime, nspl))
    cl = np.zeros((ntime, nspl))
    cd = np.zeros((ntime, nspl))
    clcd = np.zeros((ntime, nspl))

    for i in np.arange(ntime):
        # aerodynamic forces, angles
        out = aero_forces(Tdir_I[i], Cdir_I[i], Bdir_I[i], dR_I[i],
                          spl_ds[i], chord_spl[i], rho, aero_interp,
                          full_out=True)

        # store the values
        Fl_I[i] = out['Fl']
        Fd_I[i] = out['Fd']
        Fa_I[i] = out['Fa']
        Ml_I[i] = np.cross(R_Ic[i], Fl_I[i])  # Nmm
        Md_I[i] = np.cross(R_Ic[i], Fd_I[i])  # Nmm
        Ma_I[i] = np.cross(R_Ic[i], Fa_I[i])  # Nmm
        Re[i] = out['Re']
        aoa[i] = out['aoa']
        beta[i] = out['beta']
        dynP[i] = out['dynP']
        dynP_frac[i] = out['dynP_frac']
        dR_BC_I[i] = out['dR_BC']
        dR_TC_I[i] = out['dR_TC']
        U_BC_I[i] = out['U_BC']
        U_TC_I[i] = out['U_TC']
        cl[i] = out['cl']
        cd[i] = out['cd']
        clcd[i] = out['clcd']

        # in the body frame
        Fl_B[i] = np.dot(C_I2B[i], Fl_I[i].T).T
        Fd_B[i] = np.dot(C_I2B[i], Fd_I[i].T).T
        Fa_B[i] = np.dot(C_I2B[i], Fa_I[i].T).T
        Ml_B[i] = np.dot(C_I2B[i], Ml_I[i].T).T
        Md_B[i] = np.dot(C_I2B[i], Md_I[i].T).T
        Ma_B[i] = np.dot(C_I2B[i], Ma_I[i].T).T

        # in the straightened frame
        Fl_S[i] = np.dot(C_I2S[i], Fl_I[i].T).T
        Fd_S[i] = np.dot(C_I2S[i], Fd_I[i].T).T
        Fa_S[i] = np.dot(C_I2S[i], Fa_I[i].T).T
        Ml_S[i] = np.dot(C_I2S[i], Ml_I[i].T).T
        Md_S[i] = np.dot(C_I2S[i], Md_I[i].T).T
        Ma_S[i] = np.dot(C_I2S[i], Ma_I[i].T).T

    out = dict(Fl_I=Fl_I, Fd_I=Fd_I, Fa_I=Fa_I,
               Ml_I=Ml_I, Md_I=Md_I, Ma_I=Ma_I,
               Fl_B=Fl_B, Fd_B=Fd_B, Fa_B=Fa_B,
               Ml_B=Ml_B, Md_B=Md_B, Ma_B=Ma_B,
               Fl_S=Fl_S, Fd_S=Fd_S, Fa_S=Fa_S,
               Ml_S=Ml_S, Md_S=Md_S, Ma_S=Ma_S,
               Re=Re, aoa=aoa, beta=beta,
               dynP=dynP, dynP_frac=dynP_frac,
               dR_BC_I=dR_BC_I, dR_TC_I=dR_TC_I,
               U_BC_I=U_BC_I, U_TC_I=U_TC_I,
               cl=cl, cd=cd, clcd=clcd)

    return out


def Iten(r, m):
    """Full moment of inertia tensor.

    Parameters
    ----------
    r : array, size (nbody, 3)
        [x, y, z] coordinates of the point masses
    m : array, size (nbody)
        mass of each point

    Returns
    -------
    Ifull : array, size (3, 3)
        moment of inerta tensor
    """
    x, y, z = r.T
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y)
    Ixz = -np.sum(m * x * z)
    Iyz = -np.sum(m * y * z)
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])


def dIten_dt(r, dr, m):
    """Time-deriative of the full moment of inertia tensor.

    Parameters
    ----------
    r : array, size (nbody, 3)
        [x, y, z] coordinates of the point masses
    dr : array, size (nbody, 3)
        [dx, dy, dz] derivative of the coordinates of the point masses
    m : array, size (nbody)
        mass of each point

    Returns
    -------
    dIfull : array, size (3, 3)
        time derivative of the moment of inerta tensor
    """
    x, y, z = r.T
    dx, dy, dz = dr.T
    dIxx = np.sum(m * (2 * y * dy + 2 * z * dz))
    dIyy = np.sum(m * (2 * x * dx + 2 * z * dz))
    dIzz = np.sum(m * (2 * x * dx + 2 * y * dy))
    dIxy = -np.sum(m * (dx * y + x * dy))
    dIxz = -np.sum(m * (dx * z + x * dz))
    dIyz = -np.sum(m * (dy * z + y * dz))
    return np.array([[dIxx, dIxy, dIxz],
                     [dIxy, dIyy, dIyz],
                     [dIxz, dIyz, dIzz]])