# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:50:13 2015

@author: isaac
"""

from __future__ import division

import numpy as np

from scipy.integrate import cumtrapz, ode
from numba import jit
from math import sqrt

import time

np.seterr(divide='raise')


##### FASTER FUNCTIONS #####

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


@jit(nopython=True)
def nbnorm(a):
    """A faster vector norm of a length three vector.
    """
    return sqrt(a[0]**2 + a[1]**2 + a[2]**2)


@jit(nopython=True)
def nbcross(a, b):
    """A faster cross product. 'a' and 'b' are both length 3 arrays.
    """
    i = a[1] * b[2] - a[2] * b[1]
    j = -a[0] * b[2] + a[2] * b[0]
    k = a[0] * b[1] - a[1] * b[0]
    return i, j, k


def cross(a, b):
    """A faster cross product for two n x 3 arrays.

    a = np.random.rand(3 * 300).reshape(-1, 3)
    b = np.random.rand(3 * 300).reshape(-1, 3)

    %timeit np.cross(a, b)  # 10000 loops, best of 3: 30.9 µs per loop
    %timeit np.array(sim.nbcross(a.T, b.T)).T  # 100000 loops, best of 3: 6.04 µs per loop
    """

    return np.array(nbcross(a.T, b.T)).T


##### KINEMATICS FUNCTIONS #####


def aerialserp(s, t, m, n_neck, theta_dict, psi_dict, only_position=False):
    """Compound chrososerpenoid curve. Modified serpenoid curve for
    the planar wave and a serpenoid curve for the vertical wave.

    Parameters
    ----------
    s : array, size (nbody)
        arc length along body [0, L]
    t : float
        current time
    m : array, size (nbody)
        mass of each point mass
    n_neck : int
        index where the "neck" ends, so that the snake head points in the
        x direction always. This is based on Yamada (2007).
    theta_dict : dict
        Parameters for the planar wave. Inclues the spatial freqeuncy,
        temporal frequency, phase offset, amplitude function,
        amplitude function derivative, and the "circular shape parameter".
    psi_dict : dict
        Parameters for the vertical wave.
    only_position : bool (default=False)
        Only calculate positions, not derivatives wrt body, or curvature.

    Returns
    -------
    out : dict
        Constains arrays of position p, p', p'' (derivatives wrt s),
        theta (planar wave), psi (vertical wave), their derivatives (aka
        curvatures), and the signed curvature of the 3D curve.

    Note
    ----
    p is shifted to the center of mass
    """

    nbody = len(s)

    L = theta_dict['L']

    # unpack in-plane parameters
    nu_theta = theta_dict['nu_theta']
    f_theta = theta_dict['f_theta']
    phi_theta = theta_dict['phi_theta']
    amp_theta = theta_dict['amp_theta']
    damp_theta = theta_dict['damp_theta']
    d_theta = theta_dict['d_theta']

    # unpack out-of-plane parameters
    nu_psi = psi_dict['nu_psi']
    f_psi = psi_dict['f_psi']
    phi_psi = psi_dict['phi_psi']
    amp_psi = psi_dict['amp_psi']
    damp_psi = psi_dict['damp_psi']
    d_psi = psi_dict['d_psi']

    # in-plane angle
    int_theta = 2 * np.pi * nu_theta / L * s - 2 * np.pi * f_theta * t + phi_theta
    cos_theta = np.cos(int_theta)
    theta = amp_theta * np.sin(np.pi / 2 * cos_theta) + d_theta / L * s

    # out-of-plane angle
    int_psi = 2 * np.pi * nu_psi / L * s - 2 * np.pi * f_psi * t + phi_psi
    cos_psi = np.cos(int_psi)
    psi = amp_psi * cos_psi + d_psi / L * s

    # derivatives of in- and out-of-plane anlges
    if not only_position:
        dtheta = damp_theta * np.sin(np.pi / 2 * cos_theta) - \
            np.pi**2 * nu_theta / L * amp_theta * \
            np.sin(int_theta) * np.cos(np.pi / 2 * np.cos(int_theta)) + \
            d_theta / L

        dpsi = damp_psi * cos_psi - \
            2 * np.pi * nu_psi / L * amp_psi * np.sin(int_psi) + d_psi / L

    # add a neck to the snake
    if n_neck > 0:
        # tangent angle at end of 'neck'
        c_theta = theta[n_neck] / s[n_neck]
        theta_to_int = c_theta * np.ones(n_neck)

        c_psi = psi[n_neck] / s[n_neck]
        psi_to_int = c_psi * np.ones(n_neck)

        theta[:n_neck] = theta_to_int * s[:n_neck]
        psi[:n_neck] = psi_to_int * s[:n_neck]

        if not only_position:
            dtheta[:n_neck] = theta_to_int
            dpsi[:n_neck] = psi_to_int

    # derivatives wrt body
    dy = -np.cos(psi) * np.cos(theta)
    dx =  np.cos(psi) * np.sin(theta)
    dz = np.sin(psi)

    # integrate for backbone
    p = np.zeros((nbody, 3))
    p[:, 0] = cumtrapz(dx, s, initial=0)
    p[:, 1] = cumtrapz(dy, s, initial=0)
    p[:, 2] = cumtrapz(dz, s, initial=0)

    # shift the body to the center of mass
    com = np.sum((p.T * m).T, axis=0) / m.sum()
    p = p - com

    if only_position:
        out = dict(p=p)
    else:
        # derivatives along the body
        ddy =  dpsi * np.sin(psi) * np.cos(theta) + dtheta * np.cos(psi) * np.sin(theta)
        ddx = -dpsi * np.sin(psi) * np.sin(theta) + dtheta * np.cos(psi) * np.cos(theta)
        ddz = dpsi * np.cos(psi)
        dpds = np.c_[dx, dy, dz]
        ddpdds = np.c_[ddx, ddy, ddz]

        # total curvature of the body
        # kap = cross(dpds, ddpdds).sum(axis=1) / np.linalg.norm(dpds, axis=1)**3

        out = dict(p=p, dpds=dpds, ddpdds=ddpdds, theta=theta, psi=psi,
                   dthetads=dtheta, dpsids=dpsi)#, kap=kap)

    return out


def serp3d_tcb(dpds):
    """Body coordinate system for a 3D space curve. Based on the vertical
    Zhat direction.

    Parameters
    ----------
    dpds : array, size (nbody, 3)
        Derivative of the positions wrt s

    Returns
    -------
    tdir : array, size (nbody, 3)
        Tangent unit vector
    cdir : array, size (nbody, 3)
        Chord unit vector
    bdir : array, size (nbody, 3)
        Backbone unit vector
    Crs : array, size (nbody, 3, 3)
        Rotation matrices to transform a shape from the y-z plane
        to the c-b plane.

    Notes
    -----
    Cr[j] will take the airfoil shape, denfined in the x-z plane, and
    rotate it to the proper orientation in the c-b plane.
    """

    nbody = dpds.shape[0]

    tdir = (dpds.T / np.linalg.norm(dpds, axis=1)).T  # normalize to be safe

    Zhat = np.zeros((nbody, 3))
    Zhat[:, 2] = 1

    cdir_int = np.cross(Zhat, tdir)
    cdir = (cdir_int.T / np.linalg.norm(cdir_int, axis=1)).T
    bdir = np.cross(tdir, cdir)

    Crs = np.zeros((nbody, 3, 3))
    Crs[:, :, 0] = tdir  # column 0
    Crs[:, :, 1] = cdir  # column 1
    Crs[:, :, 2] = bdir  # column 2

#    |.   .   . |
#    |tv  cv  bv|
#    |.   .   . |

#    cdir = np.zeros_like(tdir)
#    bdir = np.zeros_like(tdir)
#    Crs = np.zeros((nbody, 3, 3))
#
#    zhat = np.array([0, 0, 1])
#
#    for j in np.arange(nbody):
#        # intermediate axis
#        cdir_int = np.cross(zhat, tdir[j])
#        cdir_int = cdir_int / np.linalg.norm(cdir_int)
#        bdir_int = np.cross(tdir[j], cdir_int)
#
#        bdir[j] = bdir_int
#        cdir[j] = cdir_int
#        Crs[j, :, 0] = tdir[j]
#        Crs[j, :, 1] = cdir[j]
#        Crs[j, :, 2] = bdir[j]

    return tdir, cdir, bdir, Crs


def serp3d_tcb_wang(dpds, p):
    """Body coordinate system for a 3D space curve.

    Wang (2008)
    https://github.com/bzamecnik/gpg/blob/master/rotation-minimizing-frame/rmf.py

    Parameters
    ----------
    dpds : array, size (nbody, 3)
        Derivative of the positions wrt s
    p : array, size (nbody, 3)
        Spline position in (x, y, z)

    Returns
    -------
    tdir : array, size (nbody, 3)
        Tangent unit vector
    cdir : array, size (nbody, 3)
        Chord unit vector
    bdir : array, size (nbody, 3)
        Backbone unit vector
    Crs : array, size (nbody, 3, 3)
        Rotation matrices to transform a shape from the y-z plane
        to the c-b plane.
    lateral_bend : array, size (nbody)
        Lateral rotation angle in rad (successive angle between cv vectors)
    back_bend : array, size (nbody)
        Backbone rotation angle  in rad (successive angle between bv vectors)

    Notes
    -----
    Cr[j] will take the airfoil shape, denfined in the x-z plane, and
    rotate it to the proper orientation in the c-b plane.
    """

    nbody = dpds.shape[0]

    tdir = (dpds.T / np.linalg.norm(dpds, axis=1)).T  # normalize to be safe
    cdir = np.zeros_like(tdir)
    bdir = np.zeros_like(tdir)
    Crs = np.zeros((nbody, 3, 3))

    # initialize the coordinate system, with chat in the +y direction
    xhat = np.r_[1, 0, 0]  # cdir0 should nominally be in xhat direciton
    # zhat = np.r_[0, 0, 1]
    cdir0 = xhat - tdir[0] * np.dot(tdir[0], xhat)
    cdir0 = cdir0 / nbnorm(cdir0)
    bdir0 = np.cross(tdir[0], cdir0)
    cdir[0] = cdir0
    bdir[0] = bdir0

    # rotation matrix for foil shape defined in (y - z plane, with x = 0)
    Cr_foil = np.zeros((3, 3))
    Cr_foil[:, 0] = tdir[0]
    Cr_foil[:, 1] = cdir[0]
    Cr_foil[:, 2] = bdir[0]
    Crs[0] = Cr_foil

    # Wang (2008) has Uo = (r0, s0, t0)
    # for us: Uo = (cdir, bdir, tdir)
    for j in np.arange(0, nbody - 1):
        v1 = p[j + 1] - p[j]
        c1 = np.dot(v1, v1)
        cdir_L = cdir[j] - (2 / c1) * np.dot(v1, cdir[j]) * v1
        tdir_L = tdir[j] - (2 / c1) * np.dot(v1, tdir[j]) * v1
        v2 = tdir[j + 1] - tdir_L
        c2 = np.dot(v2, v2)
        cdir[j + 1] = cdir_L - (2 / c2) * np.dot(v2, cdir_L) * v2
        bdir[j + 1] = np.cross(tdir[j + 1], cdir[j + 1])

        Crs[j + 1, :, 0] = tdir[j + 1]
        Crs[j + 1, :, 1] = cdir[j + 1]
        Crs[j + 1, :, 2] = bdir[j + 1]

    return tdir, cdir, bdir, Crs


#def _M(axis, angle):
#    """
#    http://stackoverflow.com/a/25709323
#    """
#
#    from scipy.linalg import expm3
#
#    return expm3(np.cross(np.eye(3), axis / np.linalg.norm(axis) * angle))
#
#
#def tcb_adjust(tdir, cdir, bdir, Crs):
#    """Adust the rotation about the That axis to prevent excess twist.
#
#    https://github.com/bzamecnik/gpg/blob/master/rotation-minimizing-frame/rmf.py
#    """
#
#    nbody = len(tdir)
#    zhat = np.array([0, 0, 1])
#    bdir_new = np.zeros_like(bdir)
#    cdir_new = np.zeros_like(cdir)
#    Crs_new = np.zeros_like(Crs)
#
#    bdir_new[0] = bdir[0]
#    cdir_new[0] = cdir[0]
#    Crs_new[0] = Crs[0]
#
#    for j in np.arange(1, nbody):
#        # intermediate axis
#        bdir_int = np.cross(tdir[j], np.cross(zhat, tdir[j]))
#        bdir_int = bdir_int / np.linalg.norm(bdir_int)
#
#        # angle between intermediate and bhat
#        angle = np.arccos(np.clip(np.dot(bdir[j], bdir_int), -1, 1))
#        M0 = _M(tdir[j], angle)
#
#        # v, axis, theta = [3,5,0], [4,4,1], 1.2
#        # M0 = _M(axis, angle)
#        # vrot = np.dot(M0, v)
#
#        bdir_new[j] = np.dot(M0, bdir[j])
#        cdir_new[j] = np.dot(M0, cdir[j])
#        Crs_new[j, :, 0] = tdir[j]
#        Crs_new[j, :, 1] = cdir_new[j]
#        Crs_new[j, :, 2] = bdir_new[j]
#
#    return cdir_new, bdir_new, Crs_new


def serp3d_tcb_bloomenthal(dpds, p):
    """Body coordinate system for a 3D space curve.

    This is the deprecated version from Bloomenthal (1990)

    Parameters
    ----------
    dpds : array, size (nbody, 3)
        Derivative of the positions wrt s
    p : array, size (nbody, 3)
        Spline position in (x, y, z)

    Returns
    -------
    tdir : array, size (nbody, 3)
        Tangent unit vector
    cdir : array, size (nbody, 3)
        Chord unit vector
    bdir : array, size (nbody, 3)
        Backbone unit vector
    Crs : array, size (nbody, 3, 3)
        Rotation matrices to transform a shape from the y-z plane
        to the c-b plane.
    lateral_bend : array, size (nbody)
        Lateral rotation angle in rad (successive angle between cv vectors)
    back_bend : array, size (nbody)
        Backbone rotation angle  in rad (successive angle between bv vectors)

    Notes
    -----
    Cr[j] will take the airfoil shape, denfined in the x-z plane, and
    rotate it to the proper orientation in the c-b plane.
    """

    nbody = dpds.shape[0]

    tdir = (dpds.T / np.linalg.norm(dpds, axis=1)).T  # normalize to be safe
    cdir = np.zeros_like(tdir)
    bdir = np.zeros_like(tdir)
#    back_bend = np.zeros(nbody)
#    lateral_bend = np.zeros(nbody)
    Crs = np.zeros((nbody, 3, 3))

    # initialize the coordinate system, with chat in the +y direction
    xhat = np.r_[1, 0, 0]  # cdir0 should nominally be in xhat direciton
    # zhat = np.r_[0, 0, 1]
    cdir0 = xhat - tdir[0] * np.dot(tdir[0], xhat)
    cdir0 = cdir0 / nbnorm(cdir0)
    bdir0 = np.cross(tdir[0], cdir0)
    cdir[0] = cdir0
    bdir[0] = bdir0

    # rotation matrix for foil shape defined in (y - z plane, with x = 0)
    Cr_foil = np.zeros((3, 3))
    Cr_foil[:, 0] = tdir[0]
    Cr_foil[:, 1] = cdir[0]
    Cr_foil[:, 2] = bdir[0]

    Crs[0] = Cr_foil

#    # bending angles
#    b0 = zhat
#    c0 = yhat
#    b1 = bdir0
#    c1 = cdir0
#    sin_b = np.cross(b0, b1)
#    cos_b = np.dot(b0, b1)
#    sin_l = np.cross(c0, c1)
#    cos_l = np.dot(c0, c1)
#
#    # so that the sign is correct
#    sign_b = np.sign(np.dot(sin_b, c1))
#    sign_l = np.sign(np.dot(sin_l, b1))
#
#    sin_b_mag = np.linalg.norm(sin_b)
#    sin_l_mag = np.linalg.norm(sin_l)
#
#    back_bend[0] = sign_b * np.arctan2(sin_b_mag, cos_b)
#    lateral_bend[0] = sign_l * np.arctan2(sin_l_mag, cos_l)

##    m0 = np.zeros((3, 3), dtype=np.float)
##    m1 = np.zeros((3, 3), dtype=np.float)

    for j in np.arange(1, nbody):
        t0 = tdir[j - 1]
        t1 = tdir[j]
##        A = cross(t0, t1)
##        nrm = nbnorm(A)
##        if nrm == 0:
##            nrm = 1
##        A = A / nrm
#        A = cross(t0, t1) / (nbnorm(t0) * nbnorm(t1))
#        a0 = cross(t0, A)
#        a1 = cross(t1, A)
##        m0 = np.array([[t0[0], A[0], a0[0]],
##                       [t0[1], A[1], a0[1]],
##                       [t0[2], A[2], a0[2]]])
##        m1 = np.array([[t1[0], A[0], a1[0]],
##                       [t1[1], A[1], a1[1]],
##                       [t1[2], A[2], a1[2]]])
##        m0[:, 0] = t0
##        m0[:, 1] = A
##        m0[:, 2] = a0
##        m1[:, 0] = t1
##        m1[:, 1] = A
##        m1[:, 2] = a1
#        m0 = np.array([t0, A, a0]).T
#        m1 = np.array([t1, A, a1]).T
#        Cr = np.dot(m0, m1.T)  # Cr = np.dot(m0.T, m1)
#
#        # not 100% on why need to transpose (active vs. passive rotation?)
#        # https://en.wikipedia.org/wiki/Active_and_passive_transformation
#        Cr = Cr.T

        # axis and angle of rotation
        A = cross(t0, t1)
        nrm = nbnorm(A)
        if nrm == 0:
            nrm = 1
        A = A / nrm

         # components of rotation matrix
        Ax, Ay, Az = A  # axis of rotation
        sqx, sqy, sqz = A**2
        cos = np.clip(np.dot(t0, t1), -1, 1)
        cos1 = 1 - cos
        xycos1 = Ax * Ay * cos1
        yzcos1 = Ay * Az * cos1  # check on Az
        zxcos1 = Ax * Az * cos1
        sin = np.sqrt(1 - cos**2)
        xsin, ysin, zsin =  A * sin

        # make the rotation matrix
        Cr = np.array([[sqx + (1 - sqx) * cos, xycos1 + zsin, zxcos1 - ysin],
                       [xycos1 - zsin, sqy + (1 - sqy) * cos, yzcos1 + xsin],
                       [zxcos1 + ysin, yzcos1 - xsin, sqz + (1 - sqz) * cos]])

        # not 100% on why need to transpose (active vs. passive rotation?)
        # https://en.wikipedia.org/wiki/Active_and_passive_transformation
        Cr = Cr.T

        # "chain together" the rotation matrices
        Crs[j] = np.dot(Cr, Crs[j - 1])

        c0 = cdir[j - 1]
        b0 = bdir[j - 1]
        c1 = np.dot(Cr, c0)
        b1 = np.dot(Cr, b0)
        cdir[j] = c1
        bdir[j] = b1

#        # back bend and lateral bend angles
#        # http://stackoverflow.com/a/10145056
#        sin_b = cross(b0, b1)
#        cos_b = np.dot(b0, b1)
#        sin_l = cross(c0, c1)
#        cos_l = np.dot(c0, c1)
#
#        # so that the sign is correct
#        sign_b = np.sign(np.dot(sin_b, c1))
#        sign_l = np.sign(np.dot(sin_l, b1))
#
#        sin_b_mag = np.linalg.norm(sin_b)
#        sin_l_mag = np.linalg.norm(sin_l)
#
#        back_bend[j] = sign_b * np.arctan2(sin_b_mag, cos_b)
#        lateral_bend[j] = sign_l * np.arctan2(sin_l_mag, cos_l)


#    return tdir, cdir, bdir, Crs, lateral_bend, back_bend
    return tdir, cdir, bdir, Crs


def serp3d_tcb_head_control(dpds, C):
    """Body coordinate system for a 3D space curve, but with head control.
    Contrain the chat direction to be horizontal with the ground (chat
    only has components in the Xhat and Yhat --- interial X, Y ---
    direction).

    Parameters
    ----------
    dpds : array, size (nbody, 3)
        Derivative of the positions wrt s
    C : array, size (3, 3)
        Rotation matrix to convert vector in inertial frame to body frame

    Returns
    -------
    tdir : array, size (nbody, 3)
        Tangent unit vector
    cdir : array, size (nbody, 3)
        Chord unit vector
    bdir : array, size (nbody, 3)
        Backbone unit vector
    Crs : array, size (nbody, 3, 3)
        Rotation matrices to transform a shape from the y-z plane
        to the c-b plane.

    Notes
    -----
    Cr[j] will take the airfoil shape, denfined in the x-z plane, and
    rotate it to the proper orientation in the c-b plane.
    """

    nbody = dpds.shape[0]

    tdir = (dpds.T / np.linalg.norm(dpds, axis=1)).T  # normalize to be safe
    cdir = np.zeros_like(tdir)
    bdir = np.zeros_like(tdir)
    Crs = np.zeros((nbody, 3, 3))

    # convert first tangent vector to inertial frame
    Tdir0 = rotate(C.T, tdir[0])

#    # need Tdir in XY plane
#    T_XY = np.array([Tdir0[0], Tdir0[1], 0])
#    T_XY = T_XY / np.linalg.norm(T_XY)  # convert to a unit vector
#
#    # construct the chord vector
#    C0 = np.r_[1, 0, 0]  # in the Xhat direction
#    C0 = C0 - T_XY * np.dot(T_XY, C0)
#    Cdir0 = C0 / nbnorm(C0)
#
#    # convert back to body frame
#    cdir0 = rotate(C, Cdir0)
#    bdir0 = np.cross(tdir[0], cdir0)

    # construct the chord vector
    C0 = np.r_[1, 0, 0]  # nominally in inertial Xhat direciton
    C0 = C0 - Tdir0 * np.dot(Tdir0, C0)
    Cdir0 = C0 / nbnorm(C0)

    # convert back to body frame
    cdir0 = rotate(C, Cdir0)
    bdir0 = np.cross(tdir[0], cdir0)

    # make sure that the coordinate system is orthogonal
    assert(np.allclose(0, np.dot(cdir0, tdir[0])))

    # store the coordinate system
    cdir[0] = cdir0
    bdir[0] = bdir0

    # rotation matrix for foil shape defined in (y - z plane, with x = 0)
    Cr_foil = np.zeros((3, 3))
    Cr_foil[:, 0] = tdir[0]
    Cr_foil[:, 1] = cdir[0]
    Cr_foil[:, 2] = bdir[0]

    Crs[0] = Cr_foil

    m0 = np.zeros((3, 3), dtype=np.float)
    m1 = np.zeros((3, 3), dtype=np.float)

    for j in np.arange(1, nbody):
        t0 = tdir[j - 1]
        t1 = tdir[j]
        A = cross(t0, t1) #TODO fix this business, like above. norm can be 0!
        A = A / nbnorm(A)
        a0 = cross(t0, A)
        a1 = cross(t1, A)

        m0 = np.array([t0, A, a0]).T
        m1 = np.array([t1, A, a1]).T
        Cr = np.dot(m0, m1.T)  # Cr = np.dot(m0.T, m1)

        # not 100% on why need to transpose (active vs. passive rotation?)
        # https://en.wikipedia.org/wiki/Active_and_passive_transformation
        Cr = Cr.T
        Crs[j] = np.dot(Cr, Crs[j - 1])

        c0 = cdir[j - 1]
        b0 = bdir[j - 1]
        c1 = np.dot(Cr, c0)
        b1 = np.dot(Cr, b0)
        cdir[j] = c1
        bdir[j] = b1

    return tdir, cdir, bdir, Crs


def aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict, dt=1e-5):
    """Calculate position and local velocities and acceleration of the
    aerial serpenoid curve.

    Parameters
    ----------
    s : array, size (nbody)
        arc length along body [0, L]
    t : float
        current time
    m : array, size (nbody)
        mass of each point mass
    n_neck : int
        index where the "neck" ends, so that the snake head points in the
        x direction always. This is based on Yamada (2007).
    theta_dict : dict
        Parameters for the planar wave. Inclues the spatial freqeuncy,
        temporal frequency, phase offset, amplitude function,
        amplitude function derivative, and the "circular shape parameter".
    psi_dict : dict
        Parameters for the vertical wave.
    dt : float, optional (default=1e-5)
        Time between two successive serpenoid curves to caluclate
        velocities and accelerations with finite differences.

    Returns
    -------
    out : dict
        Constains arrays of position p, p', p'' (derivatives wrt s),
        theta (planar wave), psi (vertical wave), their derivatives (aka
        curvatures), signed curvature of the 3D curve, dp (m/s), and ddp
        (m/s^2) velocity and acceleration (derivatives wrt time), tv
        tangent vector, cv chord vector, and bv backbone vector, and
        Crs rotation matrices for the orientation of the coordinate system,
        backbend angle (how the bv vector changes) and
    """

    # serpenoid curve at current time
    out = aerialserp(s, t, m, n_neck, theta_dict, psi_dict)

    # serpenoid curve nearby time points
    t_n2 = t - 2 * dt
    t_n1 = t - dt
    t_p1 = t + dt
    t_p2 = t + 2 * dt
    out_n2 = aerialserp(s, t_n2, m, n_neck, theta_dict, psi_dict, only_position=True)
    out_n1 = aerialserp(s, t_n1, m, n_neck, theta_dict, psi_dict, only_position=True)
    out_p1 = aerialserp(s, t_p1, m, n_neck, theta_dict, psi_dict, only_position=True)
    out_p2 = aerialserp(s, t_p2, m, n_neck, theta_dict, psi_dict, only_position=True)

    # 4th-order accurate finite differences
    p = out['p']
    p_n2 = out_n2['p']
    p_n1 = out_n1['p']
    p_p1 = out_p1['p']
    p_p2 = out_p2['p']

    # coefficients https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # d/ds:      1/12  -2/3    0   2/3  -1/12
    # d^2/ds^2: -1/12   4/3  -5/2  4/3  -1/12
    c112, c23, c43, c52 = 1 / 12., 2 / 3., 4 / 3., 5 / 2.

    # finite difference velocities and accelerations
    v = (c112 * p_n2 - c23 * p_n1 + c23 * p_p1 - c112 * p_p2) / dt
    a = (-c112 * p_n2 + c43 * p_n1 - c52 * p + c43 * p_p1 - c112 * p_p2) / dt**2

    out['dp'] = v
    out['ddp'] = a

    # construct the tcb frame and body rotation matrices
    # tv, cv, bv, Crs = serp3d_tcb(out['dpds'], out['p'])
    tv, cv, bv, Crs = serp3d_tcb(out['dpds'])

    out['tv'] = tv
    out['cv'] = cv
    out['bv'] = bv
    out['Crs'] = Crs

    return out


def func_ho_to_min(phi_theta, s, t, m, n_neck, theta_dict, psi_dict):
    """Return the angular momentum in the local frame. This function is
    called by newton, where we find a phi for the serpenoid curve
    such that the angular moment is zero in the local frame.

    Note
    ----
    Parameters after the first are the same inputs to aerialserpenoid

    Parameters
    ----------
    phi_theta : float
        Serpenoid curve phase shift
    s : array, size (nbody)
        arc length along body [0, L]
    t : float
        current time
    m : array, size (nbody)
        mass of each point mass
    n_neck : int
        index where the "neck" ends, so that the snake head points in the
        x direction always. This is based on Yamada (2007).
    theta_dict : dict
        Parameters for the planar wave. Inclues the spatial freqeuncy,
        temporal frequency, phase offset, amplitude function,
        amplitude function derivative, and the "circular shape parameter".
    psi_dict : dict
        Parameters for the vertical wave.

    Returns
    -------
    ||ho|| : norm of the angular momentum
        This should be as close to zero as possible.
    """

    theta_dict['phi_theta'] = phi_theta
    psi_dict['phi_psi'] = 2 * phi_theta - np.pi

    out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    p = out['p']
    dp = out['dp']
    ddp = out['ddp']

    # angular momentum and angular momentum rate
    ho = (m * np.cross(p, dp).T).T.sum(axis=0)
    dho = (m * np.cross(p, ddp).T).T.sum(axis=0)

    return np.linalg.norm(ho)


def apply_airfoil_shape(p, chord, Crs):
    """Apply the snake airfoil shape to the centered spline.

    Parameters
    ----------
    p : array, size (ntime, nbody, 3)
        Position of the backbone
    chord : array, size (nbody)
        Chord length in m as a function of distance along the body
    Crs : array, size (ntime, nbody, 3, 3)
        Rotation matrices to rotate shape in a torsion-minimizing fashion

    Returns
    -------
    foils : array, size (ntime, nbody + 2, nfoil, 3)
        X, Y, Z coordinates of the foil shape
    foil_color : array, size (ntime, nbody + 2, nfoil)
        float array that the colormap uses (underside is yellow, top is green
        when using 'YlGn' colormap in Mayavi

    Notes
    -----
    The size of foils is nbody + 2 because we add a mesh to close
    the ends of the snake.
    """

    # chord = .016 * np.ones(nbody)  # 16 mm constant

    ntime, nbody = p.shape[0], p.shape[1]

    # load in the airfoil shape
    rfoil = np.genfromtxt('../Data/Xsection/snake0.004.bdy.txt', skip_header=1)
    rfoil = rfoil - rfoil.mean(axis=0)
    rfoil[:, 1] -= rfoil[:, 1].max()  # center at top of airfoil
    #rfoil[:, 1] -= rfoil[:, 1].min()  # center at top of airfoil  #TODO
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
            foil0 = rfoil * chord[j]

            # rotate into CB plane
            rotated_foil = np.dot(Crs[i, j], foil0.T).T

            # move to position along body
            foils[i, jj] = p[i, j] + rotated_foil

            # airfoil color for plotting
            foil_color[i, jj, idxbottom] = .3

        # end caps
        foils[i, 0] = foils[i, 1].mean(axis=0)
        foils[i, -1] = foils[i, -2].mean(axis=0)
        foil_color[i, 0] = foil_color[i, 1].mean()
        foil_color[i, -1] = foil_color[i, -2].mean()

    return foils, foil_color


def serp_length(p):
    """Length of the serpenoid curve (should be the specified L).

    Parameters
    ----------
    p : array, size (nbody, 3)

    Returns
    -------
    ltot : float
        integrated arc-length along the body. Note that this uses
        numerical derivatives to calculate the "velocity" of the curve.
    """
    x, y, z = p.T
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)
    dz = np.gradient(z, edge_order=2)
    return np.sqrt(dx**2 + dy**2 + dz**2).sum()


def check_lengths(P, L):
    """Check integrated snake length to input length.

    Parameters
    ----------
    P : array, size (ntime, nbody, 3)
        predefined kinematics
    L : float
        length of the simulated snake

    Returns
    -------
    lengths : array, size (ntime)
        integrated length at each time point

    Notes
    -----
    This function simply calls 'serp_length' at each time point.
    """
    ntime, nbody, _ = P.shape
    lengths = np.zeros(ntime)
    for i in np.arange(ntime):
        lengths[i] = serp_length(P[i])

    print('Average error in lenths: {}'.format(lengths.mean() - L))
    return lengths


##### AERODYNAMICS FUNCTIONS #####

def aero_forces(tv, cv, bv, C, dR, ds, c, rho, aero_interp, full_out=False):
    """Aerodynamic forces or each segment.

    Parameters
    ----------
    tv : array, size (nbody, 3)
        tangent vector in body coordinates
    cv : array, size (nbody, 3)
        chord vector in body coordinates
    bv : array, size (nbody, 3)
        backbone vector in body coordinates
    C : array, size (3, 3)
        rotation matrix at the current time step
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

    # body coordinate system in intertial frame
    Tv = rotate(C.T, tv)
    Cv = rotate(C.T, cv)
    Bv = rotate(C.T, bv)

    # velocity components parallel and perpendicular to arifoil
    dR_T = (dot(dR, Tv) * Tv.T).T  # dR_T = dot(dR, Tv) * Tv
    dR_BC = dR - dR_T  # velocity in B-C plane
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
    Lh = cross(Tv, Dh)  # np.cross(Ti, Dh)
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


##### DYNAMICS FUNCTIONS #####

def Iten_2D(p, m):
    """Moment of inertia of serpenoid curve define in 2D.

    Parameters
    ----------
    p : array, size (nbody x 3)
        [x, y, z] positions of the masses in the local frame
    mi : array, size (nbody)
        mass of each element

    Returns
    -------
    I : array, size (3, 3)
        moment of inertia matrix
    """
    x, y, z = p.T

    Ixx = (m * y**2).sum()
    Iyy = (m * x**2).sum()
    Ixy = -(m * x * y).sum()
    Izz = Ixx + Iyy

    return np.array([[Ixx, Ixy, 0],
                     [Ixy, Iyy, 0],
                     [0, 0, Izz]])


def rotate(C, ri):
    """Rotate an N x 3 array.

    Parameters
    ----------
    C : array, size (3, 3)
        proper rotation matrix
    ri : array, size (nbody, 3)
        [x, y, z] coordintes of the point masses

    Returns
    -------
    ri_rot : array, size (nbody, 3)
        [xr, yr, zr] rotated coordinates
    """
    return np.dot(C, ri.T).T


def cpm(v):
    """Skew-symmetric cross-product matrix.

    Parameters
    -----------
    v : array, size (3)
        vector to make cross-product matrix from

    Returns
    -------
    cpm_v : array, size (3, 3)
        cross-product matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


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


def euler2C(yaw, pitch, roll):
    """Rotation matrix from Type-I (aircraft) Euler angles.

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
    C : array, size (3, 3)
        rotation matrix to convert vectors from inertial to body frame

    Notes
    -----
    ZYX (Type 1 aircraft Euler angles): Yaw Pitch Roll
    """
    c, s = np.cos, np.sin

    # C1(pitch)
    C1 = np.array([[1, 0, 0],
                   [0, c(pitch), s(pitch)],
                   [0, -s(pitch), c(pitch)]])

    # C2(roll)
    C2 = np.array([[c(roll), 0, -s(roll)],
                   [0, 1, 0],
                   [s(roll), 0, c(roll)]])

    # C3(yaw)
    C3 = np.array([[c(yaw), s(yaw), 0],
                   [-s(yaw), c(yaw), 0],
                   [0, 0, 1]])

    # C = C2(roll) * C1(pitch) * C3(yaw)
    return np.dot(C2, np.dot(C1, C3))  # 312 Greenwood? (213 Diebel)


def euler2kde(yaw, pitch, roll):
    """Kinematic differential equation from Euler angles. This is
    equation (3.16), p. 144 of Greenwood (2006).

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
    kde : array, size (3, 3)
        kinematic differential equation matrix, when dotted with the
        angular vleocity expressed in the body frame, gives the Euler
        angle rates.
    """

    # p. 28 Diebel, eqn 368, but swap first and last ROWS
    c, s = np.cos, np.sin

    K = np.array([[-s(roll), 0, c(roll)],
                  [c(roll) * c(pitch), 0, s(roll) * c(pitch)],
                  [s(roll) * s(pitch), c(pitch), -c(roll) * s(pitch)]])

    K = K / c(pitch)

    return K


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


def dkde_dt(ang, dang):
    """Derivative of the kinematic differential equations.

    Parameters
    ----------
    ang : array, size (3)
        yaw, pitch, and roll angles in radians
    dang : array, size (3)
        yaw, pitch, and roll angle rates in radians/sec

    Returns
    -------
    dkde : array, size (3, 3)
        derivative of the kinematic differential equaiton wrt time
    """

    #TODO: updated 2016-07-15; still needs to be checked!

    y, p, r = ang  # yaw, pitch, roll angles
    dy, dp, dr = dang  # yaw rate, pitch rate, roll rate

    sp, cp, tp = np.sin(p), np.cos(p), np.tan(p)
    sr, cr = np.sin(r), np.cos(r)

    row1 = [-dp * sr * tp - dr * cr, 0, dp * cr * tp - dr * sr]
    row2 = [-dr * sr * cp, 0, dr * cr * cp]
    row3 = [dr * cr * sp + dp * sr / cp, dp / cp, dr * sr * sp - dp * cr / cp]

    Kdot = np.array([row1, row2, row3]) / cp

    return Kdot

#    row1 = [0, dr * c(r) + dp * s(r) * t(p), -dr * s(r) + dp * c(r) * t(p)]
#    row2 = [0, -dr * s(r) * c(p), -dr * c(r) * c(p)]
#    row3 = [0, dr * c(r) * s(p) + dp * s(r) / c(p),
#            -dr * s(r) * s(p) + dp * c(r) / c(p)]


def ddang_ddt(ang, dang, omg, domg, C):
    """Euler angle accelerations.

    Parameters
    ----------
    ang : array, size (3)
        yaw, pitch, and roll angles in radians
    dang : array, size (3)
        yaw, pitch, and roll angle rates in radians/sec
    omg : array, size (3)
        angular velocity expressed in inertial frame
    domg : array, size (3)
        angular acceleration expressed in inertial frame
    C : array, size (3, 3)
        rotation matrix to convert from inertial to body frames

    Returns
    -------
    ddang : array, size (3, 3)
        yaw, pitch, and roll angle acclerations in radians/sec^2
    """

    yaw, pitch, roll = ang

    K = euler2kde(yaw, pitch, roll)
    dK = dkde_dt(ang, dang)

    omg_body = rotate(C, omg)
    omg_body_cpm = cpm(omg_body)

    term1 = np.dot(dK, np.dot(C, omg))
    term2 = -np.dot(K, np.dot(omg_body_cpm, np.dot(C, omg)))
    term3 = np.dot(K, np.dot(C, domg))

    return term1 + term2 + term3


##### FUNCTIONS TO SOLVE THE EQUATIONS OF MOTION #####


def dynamics_submatrices(r, dr, ddr, omg, m, F):
    """Construct components and matrix form of the dynamic equations.

    Parameters
    ----------
    ri : array, size (nbody, 3)
        local positions, expressed in inertial frame
    dri : array, size (nbody, 3)
        local velocities, expressed in inertial frame
    ddri : array, size (nbody, 3)
        local accelerations, expressed in inertial frame
    omg : array, size (3, 3)
        angular velocity expressed in inertial frame
    mi : array, size (nbody)
        mass of each point mass
    Fi : array, size (nbody)
        gravitational + aerodynamic forces on the body

    Returns
    -------
    M : array, size (6, 6)
        mass matrix
    N : array, size (6)
        forcing(?) vector
    N_newton : tuple, size (4)
        n1 - n4 components from linear momentum equation
    N_euler : tuple, size (4)
        n1 - n4 compoenets from angular momentum equation
    """

    # construct submatrices of mass matrix M
    m11 = m.sum() * np.eye(3)  # Ro, newton
    m12 = cpm((m * r.T).sum(axis=1)).T  # omega, newton
    m21 = cpm((m * r.T).sum(axis=1))  # Ro, euler
    m22 = np.zeros((3, 3))  # omega, euler
    for i in np.arange(len(r)):
        m22 += m[i] * np.dot(cpm(r[i]), cpm(r[i]).T)
    M = np.zeros((6, 6))
    M[:3, :3], M[:3, 3:] = m11, m12
    M[3:, :3], M[3:, 3:] = m21, m22

    # construct subvectors of N, newton
    n11 = np.sum(m * cross(omg, cross(omg, r)).T, axis=1)
    n12 = 2 * np.sum(m * cross(omg, dr).T, axis=1)
    n13 = np.sum(m * ddr.T, axis=1)
    n14 = -np.sum(F.T, axis=1)

    # construct subvectors of N, euler
    n21 = np.sum(m * cross(r, cross(omg, cross(omg, r))).T, axis=1)
    n22 = 2 * np.sum(m * cross(r, cross(omg, dr)).T, axis=1)
    n23 = np.sum(m * cross(r, ddr).T, axis=1)
    n24 = -np.sum(cross(r, F).T, axis=1)
    N = np.zeros(6)
    N[:3], N[3:] = n11 + n12 + n13 + n14, n21 + n22 + n23 + n24

    return M, N, (n11, n12, n13, n14), (n21, n22, n23, n24)


def dynamics(t, x, body_dict):
    """Function to integrate.

    Parameters
    ----------
    x : array, size (3 * 4)
        current state, [Ro, dRo, omg, ang]
    t : float
        current time
    args : tuple
        arguments to the function:
        (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)
        If aero_interp is None, then no aerodynamic forces, else it is
        a function that returns Fl and Fd. See its documentation for
        more details.

    Returns
    -------
    xdot : array, size (3 * 4)
        derivative of states, np.r_[dRo, qdot_n, qdot_e, dang]
    """

    # state vector
    Ro, dRo, omg, ang = np.split(x, 4)
    yaw, pitch, roll = ang

    # unpack additional arguments
    # s, m, n_neck, theta_dict, psi_dict = args[0]
    # ds, c, g, rho, aero_interp = args[1]
    s, m, n_neck = body_dict['s'], body_dict['m'], body_dict['n_neck']
    theta_dict, psi_dict = body_dict['theta_dict'], body_dict['psi_dict']
    rho, g = body_dict['rho'], body_dict['g']
    ds, c, aero_interp = body_dict['ds'], body_dict['c'], body_dict['aero_interp']

    # run the kinematics
    out = aerialserp_pos_vel_acc_tcb(s, t, m, n_neck, theta_dict, psi_dict)

    # extract kinematics variables
    p, dp, ddp = out['p'], out['dp'], out['ddp']
    tv, cv, bv = out['tv'], out['cv'], out['bv']

    # rotation matrix from inertial to body
    C = euler2C(yaw, pitch, roll)

    # control tv, cv, bv based on head orientation
    head_control = body_dict['head_control']
    if head_control:
        tv, cv, bv, _ = serp3d_tcb_head_control(out['dpds'], C)

    # positions, velocities, and accelerations
    r, dr, ddr = rotate(C.T, p), rotate(C.T, dp), rotate(C.T, ddp)
    # R = Ro + r
    dR = dRo + dr + np.cross(omg, r)

    # gravitational force in inertial frame
    F = np.zeros(r.shape)
    F[:, 2] = -m * g

    # aerodynamic forces
    if aero_interp is not None:
        Fa = aero_forces(tv, cv, bv, C, dR, ds, c, rho, aero_interp)
        F += Fa

    # form the dynamic equations
    M, N, _, _ = dynamics_submatrices(r, dr, ddr, omg, m, F)

    # solve for ddRo, domg
    qdot = np.linalg.solve(M, -N)

    # solve for change in Euler angles (kinematic differential equations)
    omg_body = rotate(C, omg)
    dang = np.dot(euler2kde(yaw, pitch, roll), omg_body)

    # combine our derivatives as the return parameter
    return np.r_[dRo, qdot, dang]


def integrate(soln0, args, dt, tend=None, print_time=False, angle_cond=True):
    """Integrate the equations of motion until the centroid drops
    below z = 0, or until the current time is greater than tend.

    Parameters
    ----------
    soln0 : array, size (12)
        initial conditions: Ro0, dRo0, omg0, ang0
    args : tuple
        arguments to the function:
        (s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp)
    dt : float
        time step for integration
    tend : float, optional
        maximum simulated time, default is None (integrate until z = 0)
    print_time : bool, optional
        whether to print timing information of the simulation, default is False
    angle_cond : bool, optional
        stop the simulation if pitch or roll angles exceed 85 deg,
        default is True

    Returns
    -------
    ts : array, size (ntime)
        time points of simulation, separated by dt
    Ro : array, size (ntime x 3)
        (x, y, z) position of local frame, expressed in inertial frame
    dRo : array, size (ntime x 3)
        (vx, vy, vz) velociies of local frame, expressed in inertial frame
    omg : array, size (ntime x 3)
        (wx, wy, wz) angular velocity vector, expressed in intertial frame
    ang : array, size (ntime x 3)
        (yaw, pitch, roll) angles, in radians
    """

    if print_time:
        import time
        now = time.time()

    if tend is None:
        tend = 1e6

    # setup the integrator
    # https://docs.scipy.org/doc/scipy-0.17.1/
    # reference/generated/scipy.integrate.ode.html
    solver = ode(dynamics)
    solver.set_integrator('dopri5')
    solver.set_initial_value(soln0, 0)  # x0, t0
    solver.set_f_params(args)

    # stop the simulation if we reach these angles
    ang_max = np.deg2rad(85)
    angle_cond = True

    # perform the integration
    soln, ts = [soln0], [0]
    while solver.y[2] > 0 and angle_cond and solver.t < tend:
        solver.integrate(solver.t + dt)
        soln.append(solver.y)
        ts.append(solver.t)

        # limit the angles when the solver stops
        yaw_cond = np.abs(solver.y[9]) < ang_max
        pitch_cond = np.abs(solver.y[10]) < ang_max
        roll_cond = np.abs(solver.y[11]) < ang_max
        angle_cond = yaw_cond and pitch_cond and roll_cond

    # we want these as arrays
    soln = np.array(soln)
    ts = np.array(ts)

    if print_time:
        base = 'Dynamics simulation time: {0:.3f} sec'
        print(base.format(time.time() - now))

    # unpack the values
    # 0  1    2    3     4     5     6   7   8   9    10     11
    Rox, Roy, Roz, dRox, dRoy, dRoz, wx, wy, wz, yaw, pitch, roll = soln.T
    Ro = np.c_[Rox, Roy, Roz]
    dRo = np.c_[dRox, dRoy, dRoz]
    omg = np.c_[wx, wy, wz]
    ang = np.c_[yaw, pitch, roll]

    return ts, Ro, dRo, omg, ang


##### RUNNING SIMULATIONS IN PARALLEL #####


def parallel_simulation(arguments):
    """Called by multiprocessing.Pool.map for a parallel run.
    """

    import aerodynamics

    savename, soln0, dt, args, params = arguments

    # can't pickle a function, so do the aerodynamics here
    aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)
    args_list = list(args)
    args_list.append(aero_interp)
    args = tuple(args_list)

    out = integrate(soln0, args, dt, print_time=True)

    # extract values
    ts, Ro, dRo, omg, ang = out
    yaw, pitch, roll = ang.T
    ntime = len(ts)

    save_derived_quantities(savename, ntime, out, args, params, print_time=True)


##### SAVE OUTPUT FROM THE SIMULATION #####


def save_derived_quantities(savename, ntime, out, args, params, print_time=False):

    if print_time:
        now = time.time()

    # output from simulation run
    ts, Ro, dRo, omg, ang = out
    yaw, pitch, roll = ang.T

    # arguments that define body shape, etc.
    s, A, k, w, phi, n_neck, ds, c, mi, g, rho, aero_interp = args

    dt, L, nbody, mass_total, rho_body, neck_length, Stot, \
        wing_loading, wave_length_m, freq_undulation_hz = params

    # allocate space
    S, T = np.meshgrid(s, ts)
    Sn, Tn = np.meshgrid(s / L, ts / np.sqrt(L / g))  # normalized

    # gliding angle (angle between (x-y) plane and CoM velcoity vector
    nhat = np.array([0, 0, -1])  # angle from horizontal down is positive
    glide_angle_mag = np.dot(nhat, dRo.T) / np.linalg.norm(dRo, axis=1)
    glide_angle_r = np.arcsin(glide_angle_mag)
    glide_angle_d = np.rad2deg(glide_angle_r)

    # heading angle (angle between (x-z) plane and CoM vleocity vector
    nhat = np.array([0, 1, 0])
    heading_angle_mag = np.dot(nhat, dRo.T) / np.linalg.norm(dRo, axis=1)
    heading_angle_r = np.arcsin(heading_angle_mag)
    heading_angle_d = np.rad2deg(heading_angle_r)

    ntnb3 = (ntime, nbody, 3)

    C = np.zeros((ntime, 3, 3))
    Norig, Borig = np.zeros((ntime, 3, 3)), np.zeros((ntime, 3, 3))
    ho, Ho = np.zeros((ntime, 3)), np.zeros((ntime, 3))

    p, dp, ddp = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)
    tang, kap = np.zeros((ntime, nbody)), np.zeros((ntime, nbody))
    # tcb, TCB = np.zeros((ntime, nbody, 3, 3)), np.zeros((ntime, nbody, 3, 3))
    tv, cv, bv = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)
    Tv, Cv, Bv = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)

    r, dr, ddr = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)
    R, dR, ddR = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)
    ddRo, domg, dang = np.zeros((ntime, 3)), np.zeros((ntime, 3)), \
        np.zeros((ntime, 3))
    adv_ratio = np.zeros((ntime, nbody))

    Mm = np.zeros((ntime, 6, 6))
    Nm = np.zeros((ntime, 6))
    Nnew = np.zeros((ntime, 4, 3))
    Neul = np.zeros((ntime, 4, 3))

    Fg, Fl, Fd = np.zeros(ntnb3), np.zeros(ntnb3), np.zeros(ntnb3)
    Faero, Ftot = np.zeros(ntnb3), np.zeros(ntnb3)
    aoa_r, aoa_d = np.zeros((ntime, nbody)), np.zeros((ntime, nbody))
    beta_r, beta_d = np.zeros((ntime, nbody)), np.zeros((ntime, nbody))
    Re, dR_BC = np.zeros((ntime, nbody)), np.zeros(ntnb3)
    Maero, cop = np.zeros((ntime, 3)), np.zeros((ntime, 3))
    Maero_z0, cop_z0 = np.zeros((ntime, 3)), np.zeros((ntime, 3))

    for i in np.arange(ntime):
        t = ts[i]

        # kinematics in local frame
        out = serp_pos_vel_acc_tcb(s, t, A, k, w, phi, n_neck)
        pi, dpi, ddpi, tangi, kapi, tv_i, cv_i, bv_i = out

        # store kinematics in loacal frame
        p[i], dp[i], ddp[i] = pi, dpi, ddpi
        tang[i], kap[i] = tangi, kapi
        tv[i], cv[i], bv[i] = tv_i, cv_i, bv_i

        # accelerations from dynamics
        soln_i = np.r_[Ro[i], dRo[i], omg[i], ang[i]]
        dynamics_out = dynamics(t, soln_i, args)
        _, ddRoi, domgi, dangi = np.split(dynamics_out, 4)

        # store accelerations from dynamics
        ddRo[i], domg[i], dang[i] = ddRoi, domgi, dangi

        # rotation matrix, velocities, acceleration in inertial frame
        yawi, pitchi, rolli = ang[i]
        Ci = euler2C(yawi, pitchi, rolli)
        ri, dri, ddri = rotate(Ci.T, pi), rotate(Ci.T, dpi), rotate(Ci.T, ddpi)
        Ri = Ro[i] + ri
        dRi = dRo[i] + dri + np.cross(omg[i], ri)
        ddRi = ddRoi + np.cross(domgi, ri) + \
               np.cross(omg[i], np.cross(omg[i], ri)) + \
               2 * np.cross(omg[i], dri) + ddri
        Tv[i] = rotate(Ci.T, tv_i)
        Cv[i] = rotate(Ci.T, cv_i)
        Bv[i] = rotate(Ci.T, bv_i)
        # TCB[i] = rotate(Ci.T, tcb_i)

        # store rotation matrix, velocities, acceleration in inertial frame
        C[i] = Ci
        r[i], dr[i], ddr[i] = ri, dri, ddri
        R[i], dR[i], ddR[i] = Ri, dRi, ddRi

        # calc + store angular momentum, expressed in local and inertial frames
        ho[i] = np.cross(pi, (mi * dpi.T).T).sum(axis=0)
        Ho[i] = np.cross(ri, (mi * dRi.T).T).sum(axis=0)

        # store the orientation of the coordinate systems
        Norig[i] = np.eye(3)
        Borig[i] = rotate(Ci, Norig[i])

        # advance ratio
        for j in np.arange(nbody):
            dR_hat = dR[i, j] / np.linalg.norm(dR[i, j])
            Upar = np.dot(dr[i, j], dR_hat) * dR_hat
            Uper = dr[i, j] - Upar
            adv_ratio[i, j] = np.linalg.norm(dR[i, j]) / np.linalg.norm(Uper)

        # forces on the body
        Fgi = np.zeros((nbody, 3))
        Fgi[:, 2] = -mi * g  # gravitational force in inertial frame
        if aero_interp is not None:
            aeroi = aero_forces(tv_i, cv_i, bv_i, Ci, dRi, ds, c, rho, aero_interp)
            Fli, Fdi, dRiBC, aoai, Rei = aeroi

            # sweep angle beta
            betai = np.zeros(nbody)
            for j in np.arange(nbody):
                dRi = dR[i, j]
                Ti, Bi = Tv[i, j], Bv[i, j]
                # Ci = Cv[i, j]

                dRiB = np.dot(dRi, Bi) * Bi
                dRiTC = dRi - dRiB  # velocity in T-C plane
                Ui = np.linalg.norm(dRiTC)  # reduced velocity in plane

                bTi = np.dot(dRiTC, Ti) / Ui
                beta_di = np.rad2deg(np.arccos(bTi)) - 90
                betai[j] = beta_di

        else:
            Fli, Fdi = np.zeros((nbody, 3)), np.zeros((nbody, 3))
            dRiBC = np.zeros((nbody, 3))
            aoai, Rei = np.zeros(nbody), np.zeros(nbody)
            betai = np.zeros(nbody)
        Faero_i = Fli + Fdi
        Ftot_i = Fgi + Faero_i

        # store forces on body
        Fg[i], Fl[i], Fd[i] = Fgi, Fli, Fdi
        Faero[i], Ftot[i] = Faero_i, Ftot_i
        aoa_r[i], aoa_d[i], Re[i] = aoai, np.rad2deg(aoai), Rei
        beta_r[i], beta_d[i] = np.deg2rad(betai), betai
        dR_BC[i] = dRiBC

        # terms in the moment equation
        submat_i = dynamics_submatrices(ri, dri, ddri, omg[i], mi, Ftot_i)
        Mi, Ni, newtoni, euleri = submat_i
        n11, n12, n13, n14 = newtoni
        n21, n22, n23, n24 = euleri
        Mm[i] = Mi
        Nm[i] = Ni
        Nnew[i] = n11, n12, n13, n14  # note: n12 = -sum(Ftot_i)
        Neul[i] = n21, n22, n23, n24

        # CoP closest to origin
        Maero_i = np.cross(ri, Faero_i).sum(axis=0)
        Faero_tot = Faero_i.sum(axis=0)
        funit = Faero_tot / np.linalg.norm(Faero_tot)
        M_pure_couple = np.dot(Maero_i, funit) * funit
        Ms = Maero_i - M_pure_couple
        Fx, Fy, Fz = Faero_tot
        Mx, My, Mz = Ms
        m = np.r_[Mx, My, Mz, 0]
        f = np.array([[0, Fz, -Fy],
                      [-Fz, 0, Fx],
                      [Fy, -Fx, 0],
                      [Fx, Fy, Fz]])
        cop_i = np.linalg.lstsq(f, m)[0]

        # store center of pressure information
        Maero[i], cop[i] = Maero_i, cop_i

        # CoP in the x-y plane of the undulation
        Maero_i = np.cross(ri, Faero_i).sum(axis=0)
        Faero_tot = Faero_i.sum(axis=0)

        # put everything into body frame
        Maero_i = rotate(Ci, Maero_i)
        Faero_tot = rotate(Ci, Faero_tot)

        funit = Faero_tot / np.linalg.norm(Faero_tot)
        M_pure_couple = np.dot(Maero_i, funit) * funit
        Ms = Maero_i - M_pure_couple
        Fx, Fy, Fz = Faero_tot
        Mx, My, Mz = Ms
        m = np.r_[Mx, My, Mz, 0]
        f = np.array([[0, Fz, -Fy],
                      [-Fz, 0, Fx],
                      [Fy, -Fx, 0],
                      [0, 0, 1]])
        cop_i = np.linalg.lstsq(f, m)[0]

        # store center of pressure information
        Maero_z0[i], cop_z0[i] = Maero_i, cop_i

    # save the data
    np.savez(savename,
         L=L,
         ds=ds,
         s=s,
         nbody=nbody,
         ntime=ntime,
         dt=dt,
         mass_total=mass_total,
         rho_body=rho_body,
         mi=mi,
         c=c,
         neck_length=neck_length,
         n_neck=n_neck,
         Stot=Stot,
         wing_loading=wing_loading,
         g=g,
         rho=rho,
         A=A,
         wave_length_m=wave_length_m,
         freq_undulation_hz=freq_undulation_hz,
         w=w,
         k=k,
         phi=phi,
         ts=ts,
         Ro=Ro,
         dRo=dRo,
         omg=omg,
         ang=ang,
         yaw=yaw,
         pitch=pitch,
         roll=roll,
         S=S,
         T=T,
         C=C,
         Norig=Norig,
         Borig=Borig,
         ho=ho,
         Ho=Ho,
         p=p,
         dp=dp,
         ddp=ddp,
         tang=tang,
         kap=kap,
         tv=tv,
         cv=cv,
         bv=bv,
         Tv=Tv,
         Cv=Cv,
         Bv=Bv,
         r=r,
         dr=dr,
         ddr=ddr,
         R=R,
         dR=dR,
         ddR=ddR,
         ddRo=ddRo,
         domg=domg,
         dang=dang,
         Mm=Mm,
         Nm=Nm,
         Nnew=Nnew,
         Neul=Neul,
         Fg=Fg,
         Fl=Fl,
         Fd=Fd,
         Faero=Faero,
         Ftot=Ftot,
         aoa_r=aoa_r,
         aoa_d=aoa_d,
         beta_r=beta_r,
         beta_d=beta_d,
         Re=Re,
         dR_BC=dR_BC,
         Maero=Maero,
         cop=cop,
         Maero_z0=Maero_z0,
         cop_z0=cop_z0,
         Sn=Sn,
         Tn=Tn,
         glide_angle_r=glide_angle_r,
         glide_angle_d=glide_angle_d,
         heading_angle_r=heading_angle_r,
         heading_angle_d=heading_angle_d,
         adv_ratio=adv_ratio)

    if print_time:
        base = 'Save output time: {0:.3f} sec'
        print(base.format(time.time() - now))
