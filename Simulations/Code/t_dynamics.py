# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:54:51 2015

@author: isaac
"""

def test_mean_mom_dir(eig_vecs):
    evm = eig_vecs.mean(axis=0)
    evmx = eig_vecs[:, :, 0].mean(axis=0)
    evmy = eig_vecs[:, :, 1].mean(axis=0)
    evmz = eig_vecs[:, :, 2].mean(axis=0)
    evmxyz = np.c_[evmx, evmy, evmz]
    magevm = np.linalg.norm(evm, axis=0)
    magevmxyz = np.linalg.norm(evmxyz, axis=0)
    assert np.allclose(evmxyz, evm)
    evm /= magevm
    evmxyz /= magevmxyz
    assert np.allclose(evmxyz, evm)


def test_euler2C():
    Cyaw = euler2C(*np.deg2rad([50, 0, 0]))
    print Cyaw
    print
    Cpitch = euler2C(*np.deg2rad([0, 50, 0]))
    print Cpitch
    print
    Croll = euler2C(*np.deg2rad([0, 0, 50]))
    print Croll


def test_nbcross():
    aa = np.array([1., 2., 3.])
    bb = np.array([-3., 2., -8.])

    assert np.allclose(np.cross(aa, bb), nbcross(aa, bb))

    now = time.time()
    np.cross(aa, bb)
    print('np.cross : {0:.8f} sec'.format(time.time() - now))

    now = time.time()
    nbcross(aa, bb)
    print('nbcross  : {0:.8f} sec'.format(time.time() - now))


def tt_old(Ri, mi, ri, dri, ddri, omg, Fi):

    # construct submatrices of mass matrix M
    m11 = mi.sum() * np.eye(3)  # Ro, newton
    m12 = cpm((mi * ri.T).sum(axis=1)).T  # omega, newton
    m21 = cpm((mi * Ri.T).sum(axis=1))  # Ro, euler
    m22 = np.zeros((3, 3))  # omega, euler
    for i in np.arange(len(ri)):
        m22 += mi[i] * np.dot(cpm(Ri[i]), cpm(ri[i]).T)
    M = np.zeros((6, 6))
    M[:3, :3], M[:3, 3:] = m11, m12
    M[3:, :3], M[3:, 3:] = m21, m22

    # construct subvectors of N, newton
    n11 = np.sum(mi * np.cross(omg, np.cross(omg, ri)).T, axis=1)
    n12 = 2 * np.sum(mi * np.cross(omg, dri).T, axis=1)
    n13 = np.sum(mi * ddri.T, axis=1)
    n14 = -np.sum(Fi.T, axis=1)

    # construct subvectors of N, euler
    n21 = np.sum(mi * np.cross(Ri, np.cross(omg, np.cross(omg, ri))).T, axis=1)
    n22 = 2 * np.sum(mi * np.cross(Ri, np.cross(omg, dri)).T, axis=1)
    n23 = np.sum(mi * np.cross(Ri, ddri).T, axis=1)
    n24 = -np.sum(np.cross(Ri, Fi).T, axis=1)
    N = np.zeros(6)
    N[:3], N[3:] = n11 + n12 + n13 + n14, n21 + n22 + n23 + n24

    # solve for ddRo, domg
    qdot = np.linalg.solve(M, -N)

    return M, N, qdot


def tt(mi, ri, dri, ddri, omgi, Fi):

    # construct submatrices of mass matrix M
    m11 = mi.sum() * np.eye(3)  # Ro, newton
    m12 = cpm((mi * ri.T).sum(axis=1)).T  # omega, newton
    m21 = cpm((mi * ri.T).sum(axis=1))  # Ro, euler
    m22 = np.zeros((3, 3))  # omega, euler
    for i in np.arange(len(ri)):
        m22 += mi[i] * np.dot(cpm(ri[i]), cpm(ri[i]).T)
    M = np.zeros((6, 6))
    M[:3, :3], M[:3, 3:] = m11, m12
    M[3:, :3], M[3:, 3:] = m21, m22

    # construct subvectors of N, newton
    n11 = np.sum(mi * np.cross(omgi, np.cross(omgi, ri)).T, axis=1)
    n12 = 2 * np.sum(mi * np.cross(omgi, dri).T, axis=1)
    n13 = np.sum(mi * ddri.T, axis=1)
    n14 = -np.sum(Fi.T, axis=1)

    # construct subvectors of N, euler
    n21 = np.sum(mi * np.cross(ri, np.cross(omgi, np.cross(omgi, ri))).T, axis=1)
    n22 = 2 * np.sum(mi * np.cross(ri, np.cross(omgi, dri)).T, axis=1)
    n23 = np.sum(mi * np.cross(ri, ddri).T, axis=1)
    n24 = -np.sum(np.cross(ri, Fi).T, axis=1)
    N = np.zeros(6)
    N[:3], N[3:] = n11 + n12 + n13 + n14, n21 + n22 + n23 + n24

    # solve for ddRo, domg
    qdot = np.linalg.solve(M, -N)

    return M, N, qdot



def test_hot_dyn_loop(mi, ri, dri, ddri, omgi, Fi):

    np.random.seed(10)

    nbody, ncood = 101, 3
    ri = np.random.normal(size=(nbody, ncoord))
    dri = np.random.normal(size=(nbody, ncoord))
    ddri = np.random.normal(size=(nbody, ncoord))
    Fi = np.random.normal(0, 2, (nbody, ncoord))
    omgi = np.random.normal(0, .5, 3)

    # constrcut submatrices of mass matrix M
    m11, m12 = np.zeros((3, 3)), np.zeros((3, 3))
    m21, m22 = np.zeros((3, 3)), np.zeros((3, 3))
    for i in np.arange(nbody):
        m11 = m11 + mi[i] * np.eye(3)
        m12 = m12 + mi[i] * cpm(ri[i]).T
        m21 = m21 + mi[i] * cpm(ri[i])
        m22 = m22 + mi[i] * np.dot(cpm(ri[i]), cpm(ri[i]).T)
    M = np.zeros((6, 6))
    M[:3, :3], M[:3, 3:] = m11, m12
    M[3:, :3], M[3:, 3:] = m21, m22

    # construct subvectors of N
    n11, n12, n13, n14 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    n21, n22, n23, n24 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    for i in np.arange(nbody):
        # newton
        n11 += np.cross(omgi, np.cross(omgi, mi[i] * ri[i]))
        n12 += np.cross(omgi, mi[i] * dri[i])
        n13 += mi[i] * ddri[i]
        n14 += -Fi[i]

        # euler
        n21 += np.cross(ri[i], np.cross(omgi, np.cross(omgi, mi[i] * ri[i])))
        n22 += np.cross(ri[i], np.cross(omgi, mi[i] * dri[i]))
        n23 += np.cross(ri[i], mi[i] * ddri[i])
        n24 += -np.cross(ri[i], Fi[i])
    n12 *= 2
    n22 *= 2
    N = np.zeros(6)
    N[:3], N[3:] = n11 + n12 + n13 + n14, n21 + n22 + n23 + n24

    # solve for ddRo, domg
    qdot = np.linalg.solve(M, -N)

    return M, N, qdot


Ms, Ns, qdots = test_hot_dyn_loop(mi, ri, dri, ddri, omgi, Fi)

Mo, No, qdot0 = tt_old(Ri, mi, ri, dri, ddri, omgi, Fi)
M, N, qdot = tt( mi, ri, dri, ddri, omgi, Fi)

assert(np.allclose(Ms, M))
assert(np.allclose(Ns, N))
assert(np.allclose(qdots, qdot))
