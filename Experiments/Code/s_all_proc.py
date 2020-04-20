# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:03:35 2016

%reset -f
%clear
%pylab
%load_ext autoreload
%autoreload 2

@author: isaac
"""

from __future__ import division

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

import time, sys

import m_data_utils as data_utils
import m_smoothing as smoothing
import m_ukf_filter as ukf_filter
import m_asd as asd
import m_plotting as plotting

import seaborn as sns

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()


# %%

# load in data files used for all snake trials
master = pd.read_csv('../Data/snake_master.csv')
chord_df = pd.read_csv('../Data/snake_width.csv', index_col=0)
density_df = pd.read_csv('../Data/snake_density.csv', index_col=0)
density_df.columns = ['s', 'rho']
chord_df.columns = ['s', 'chord']

ntrials = len(master)

fs = 179.
dt = 1 / fs
nspl = 200

# file name templates
fn_coordinates = '../Data/Raw Qualisys output/{trial}_{snake}.tsv'
fn_marker = '../Data/Snake markers/Processed/Marker-summary_{0}00_{1}.csv'
fn_save = '../Data/Processed Qualisys output/{0}.npz'

plot_intermediates = False


# %%

# set to skip particular trials
skip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
skip = []
loop_times = []

# iterate through all trials, processing marker points to full snake body
for trial_idx in np.arange(ntrials):
    # timing information
    now = time.time()

    # extract information about the trial from the master file
    snake_id = master['Snake'].ix[trial_idx]
    trial_id = master['Trial ID'].ix[trial_idx]
    file_id = '{0}_{1}'.format(trial_id, snake_id)
    height = master['Height (m)'].ix[trial_idx]
    mass = master['Mass (g)'].ix[trial_idx]
    weight = mass / 1000 * 9.81  # N
    SVL_avg = master['SVL (mm)'].ix[trial_idx]
    VTL_avg = master['VTL (mm)'].ix[trial_idx]
    frame_of_jump = master['Frame jump'].ix[trial_idx]
    is_gold = bool(master['Gold'].ix[trial_idx])

    _anal_str = 'Analyzing trial {0} of {1}: {2}'
    print(_anal_str.format(trial_idx, ntrials - 1, file_id))
    sys.stdout.flush()

    if trial_idx in skip:
        continue

    # marker information
    _day = str(trial_id)[0]
    marker_df = pd.read_csv(fn_marker.format(_day, snake_id))
    dist_btn_markers = marker_df['Dist to next, mm'].dropna().values
    vent_idx = np.where(marker_df['Marker type'] == 'vent')[0][0]
    SVL = marker_df['svl (mm)'].values[0]
    VTL = marker_df['tail (mm)'].values[0]

    # load in the data
    fname = fn_coordinates.format(trial=trial_id, snake=snake_id)
    df = data_utils.load_qtm_tsv(fname)
    pr, out = data_utils.reconfig_raw_data(df)
    times, frames = out['times'], out['frames']

    # remove the 2nd marker information, since that did not record well and
    # was exculded fromt the Qualisys output
    if snake_id == 86:
        vent_idx = vent_idx - 1
        dtmp = np.zeros(len(dist_btn_markers) - 1)
        dtmp[0] = dist_btn_markers[0] + dist_btn_markers[1]
        dtmp[1:] = dist_btn_markers[2:]
        dist_btn_markers = dtmp

    # start time based on when jump begins, not when we export the data
    assert frames[0] >= frame_of_jump
    time_offset = dt * (frames[0] - frame_of_jump)
    times = times + time_offset

    # find first and last indices of a complete snake
    # these are the frames we will analyze
    no_nans = np.where(np.isnan(pr[:, :, 0]).sum(axis=1) == 0)[0]
    idx_keep_0 = no_nans[0]
    idx_keep_1 = no_nans[-1]

    # select out the data
    #TODO check the + 1 on idx_keep_1
    pr_with_nans = pr.copy()
    pgap = pr[idx_keep_0:idx_keep_1 + 1]  # pr with nan values
    times = times[idx_keep_0:idx_keep_1 + 1]
    frames = frames[idx_keep_0:idx_keep_1 + 1]
    ntime = len(times)

    # fill in missing values with an unscented Kalman filter
    pfill, nans_fill, pfill0 = ukf_filter.fill_gaps_ukf(pgap, fs, meas_noise=5)

    # find the optimal Butterworth filter cutoff frequencies
    R, fcs = smoothing.residual_butter(pfill, fs, df=.5, fmin=1, fmax=35)
    inter, slope, fcopt, rsq, flinreg = smoothing.opt_cutoff(R, fcs, rsq_cutoff=.95)

    # plot the residual values
    if plot_intermediates:
        fig, ax = plotting.plot_residuals(pfill, R, fcs, inter, fcopt)

    # perform Butterworth filtering
    but_unique = smoothing.but_fcs(pfill, fs, fcopt)
    bup, buv, bua = but_unique['p'], but_unique['v'], but_unique['a']

    # extract/rename smooth position, velocity, acceleration valueds
    pf_I, vf_I, af_I = bup.copy(), buv.copy(), bua.copy()

    # add a neck to the snake's body to constrain the spline
    pfe_I, tcoord_e = asd.add_neck(pf_I, vf_I, dist_btn_markers)  # , neck_len=30)

    # fit a spline to the snake
    out = asd.splinize_snake(pfe_I, tcoord_e, nspl, times, mass,
                             dist_btn_markers, vent_idx, SVL, VTL,
                             density_df, chord_df)
    Ro_I, R_I, dRds_I = out['Ro_I'], out['R_I'], out['dRds_I']
    ddRds_I, dddRds_I = out['ddRds_I'], out['dddRds_I']
    spl_ds, mass_spl  = out['spl_ds'], out['mass_spl']
    chord_spl, vent_idx_spl = out['chord_spl'], out['vent_idx_spl']
    times2D, t_coord = out['times2D'], out['t_coord']
    s_coord, spl_len_errors = out['s_coord'], out['spl_len_errors']
    idx_pts = out['idx_pts']

    # error metric for the splines
    err_mean = spl_len_errors[:, :vent_idx].sum(axis=1).mean()
    err_std = spl_len_errors[:, :vent_idx].sum(axis=1).std()

    # plot spline length errors
    if plot_intermediates:
        pass

    # raw CoM velocities
    Ro_I_raw = Ro_I.copy()
    dRo_I_raw, ddRo_I_raw = smoothing.findiff(Ro_I_raw, dt)

    # filter the CoM position to calculate velocities
    Ro_resid_array = np.zeros((ntime, 1, 3))  # turn 2d array into 3d array
    Ro_resid_array[:, 0] = Ro_I_raw
    resid_Ro, fcs_Ro = smoothing.residual_butter(Ro_resid_array, fs, df=.5,
                                                 fmin=1, fmax=35)
    inter_Ro, slope_Ro, fcopt_Ro, rsq_Ro, flinreg_Ro = smoothing.opt_cutoff(resid_Ro, fcs_Ro, rsq_cutoff=.95)

    # CoM velocity and acceleration
    Ro_unique = smoothing.but_fcs(Ro_resid_array, fs, fcopt_Ro)
    Ro_I = Ro_unique['p'].reshape(-1, 3)
    dRo_I = Ro_unique['v'].reshape(-1, 3)
    ddRo_I = Ro_unique['a'].reshape(-1, 3)

    # CoM shift, calculate velocity and accelerations of spline and markers
    out = asd.body_vel_acc(pf_I, pfe_I, R_I, Ro_I, dt)
    R_Ic, pf_Ic, pfe_Ic = out['R_Ic'], out['pf_Ic'], out['pfe_Ic']
    vf_Ic, af_Ic = out['vf_Ic'], out['af_Ic']
    dR_I, ddR_I = out['dR_I'], out['ddR_I']
    dR_Ic, ddR_Ic = out['dR_Ic'], out['ddR_Ic']

    # straighten the trajectory
    _filtered = [pfe_I, pf_I, vf_I, af_I, pfe_Ic, pf_Ic, vf_Ic, af_Ic]
    _com = [Ro_I, dRo_I, ddRo_I]
    _splines = [R_I, dR_I, ddR_I, R_Ic, dR_Ic, ddR_Ic]
    _splines_ds = [dRds_I, ddRds_I, dddRds_I]
    out = asd.straighten_trajectory(_filtered, _com, _splines, _splines_ds)
    Ro_S, R_Sc = out['Ro_S'], out['R_Sc'],
    pf_Sc, pfe_Sc = out['pf_Sc'], out['pfe_Sc']
    R_S, pf_S, pfe_S = out['R_S'], out['pf_S'], out['pfe_S']
    vf_S, vf_Sc, dRo_S = out['vf_S'], out['vf_Sc'], out['dRo_S']
    dR_S, dR_Sc = out['dR_S'], out['dR_Sc']
    af_S, af_Sc, ddRo_S = out['af_S'], out['af_Sc'], out['ddRo_S']
    ddR_S, ddR_Sc = out['ddR_S'], out['ddR_Sc']
    dRds_S, ddRds_S, dddRds_S = out['dRds_S'], out['ddRds_S'], out['dddRds_S']
    mus, Rmus = out['mus'], out['Rmus']
    yaw, gamma, C_I2S = out['yaw'], out['gamma'], out['C_I2S']

    # apply the body coordinate system (TBC) and calculate curvatures
    out = asd.apply_body_cs(dRds_I, ddRds_I, dddRds_I, C_I2S)
    Tdir_S, Cdir_S, Bdir_S = out['Tdir_S'], out['Cdir_S'], out['Bdir_S']
    Cb_S, a_angs, b_angs = out['Cb_S'], out['a_angs'], out['b_angs']
    kap_signed, kap_unsigned = out['kap_signed'], out['kap_unsigned']
    Tdir_I, Cdir_I, Bdir_I = out['Tdir_I'], out['Cdir_I'], out['Bdir_I']
    Cb_I, tau = out['Cb_I'], out['tau']

    # airfoils shape for plotting and movies in straightened frame
    foils_Sc, foil_color = asd.apply_airfoil_shape_cfd(R_Sc, chord_spl, Cb_S)
    foils_Ic, foil_color = asd.apply_airfoil_shape_cfd(R_Ic, chord_spl, Cb_I)
    foils_S, foils_I = np.zeros_like(foils_Sc), np.zeros_like(foils_Ic)
    for i in np.arange(ntime):
        foils_S[i] = foils_Sc[i] + Ro_S[i]
        foils_I[i] = foils_Ic[i] + Ro_I[i]

    # fit a plane to the snake's body
    out = asd.fit_comoving_frame_B(R_Ic, C_I2S, mass_spl, vent_idx_spl)
    C_I2B, Mw_I, Nhat_I = out['C_I2B'], out['Mw_I'], out['Nhat_I']
    Sfrac, planar_fit_error = out['Sfrac'], out['planar_fit_error']

    # rotate everything to body frame
    _Ro = [dRo_I, ddRo_I]
    _dRds = [dRds_I, ddRds_I]
    _R = [R_Ic, dR_Ic, ddR_Ic, dR_I, ddR_I]
    _mark = [pf_Ic, vf_I, af_I, pfe_Ic, vf_Ic, af_Ic]
    _foils = [foils_Ic, Tdir_I, Cdir_I, Bdir_I]
    out = asd.rotate_to_B(C_I2B, _Ro, _dRds, _R, _mark, _foils)
    dRo_B, ddRo_B = out['dRo_B'], out['ddRo_B']
    dRds_B, ddRds_B = out['dRds_B'], out['ddRds_B']
    R_B, dR_B, ddR_B = out['R_B'], out['dR_B'], out['ddR_B']
    pf_B, vf_B, af_B = out['pf_B'], out['vf_B'], out['af_B']
    pfe_B, foils_B = out['pfe_B'], out['foils_B']
    Tdir_B, Cdir_B, Bdir_B = out['Tdir_B'], out['Cdir_B'], out['Bdir_B']
    dR_Bc, ddR_Bc = out['dR_Bc'], out['ddR_Bc']
    vf_Bc, af_Bc = out['vf_Bc'], out['af_Bc']

    # apply the body coordinate system (for plotting)
    out = asd.apply_comoving_frame(C_I2B, C_I2S, _nmesh=21)
    Xp_B, Yp_B, Zp_B = out['Xp_B'], out['Yp_B'], out['Zp_B'],
    Xp_I, Yp_I, Zp_I = out['Xp_I'], out['Yp_I'], out['Zp_I']
    Xp_S, Yp_S, Zp_S = out['Xp_S'], out['Yp_S'], out['Zp_S']
    YZ_B, XZ_B, XY_B = out['YZ_B'], out['XZ_B'], out['XY_B']
    YZ_I, XZ_I, XY_I = out['YZ_I'], out['XZ_I'], out['XY_I']
    YZ_S, XZ_S, XY_S = out['YZ_S'], out['XZ_S'], out['XY_S']

    # yaw, pitch, roll angles; angular velcity and acceleration
    out = asd.euler_angles_and_omg(C_I2B, dt)
    yaw, pitch, roll = out['yaw'], out['pitch'], out['roll']
    dyaw, dpitch, droll = out['dyaw'], out['dpitch'], out['droll']
    ddyaw, ddpitch, ddroll = out['ddyaw'], out['ddpitch'], out['ddroll']
    omg_B, domg_B, ddomg_B = out['omg_B'], out['domg_B'], out['ddomg_B']
    omg_I, domg_I, ddomg_I = out['omg_I'], out['domg_I'], out['ddomg_I']

    # aerodynamic forces
    _TCB = [Tdir_I, Cdir_I, Bdir_I]
    out = asd.find_aero_forces_moments(R_Ic, _TCB, dR_I, spl_ds, chord_spl,
                                       C_I2B, C_I2S)
    Fl_I, Fd_I, Fa_I = out['Fl_I'], out['Fd_I'], out['Fa_I']
    Ml_I, Md_I, Ma_I = out['Ml_I'], out['Md_I'], out['Ma_I']
    Fl_B, Fd_B, Fa_B = out['Fl_B'], out['Fd_B'], out['Fa_B']
    Ml_B, Md_B, Ma_B = out['Ml_B'], out['Md_B'], out['Ma_B']
    Fl_S, Fd_S, Fa_S = out['Fl_S'], out['Fd_S'], out['Fa_S']
    Ml_S, Md_S, Ma_S = out['Ml_S'], out['Md_S'], out['Ma_S']
    Re, aoa, beta = out['Re'], out['aoa'], out['beta']
    dynP, dynP_frac = out['dynP'], out['dynP_frac']
    dR_BC_I, dR_TC_I = out['dR_BC_I'], out['dR_TC_I']
    U_BC_I, U_TC_I = out['U_BC_I'], out['U_TC_I']
    cl, cd, clcd = out['cl'], out['cd'], out['clcd']
    U_tot_I = out['U_tot']

    # center of pressure nearest CoM


    # something to do the CoM wiggles


    # components of the variable-geometry rigid body equations of motion

    # save as an npz
    np.savez(fn_save.format(file_id),
            # extract information about the trial from the master file
            fs = fs,
            dt = dt,
            nspl = nspl,
            snake_id = snake_id,
            trial_id = trial_id,
            file_id = file_id,
            height = height,
            mass = mass,
            weight = weight,
            SVL_avg = SVL_avg,
            VTL_avg = VTL_avg,
            frame_of_jump = frame_of_jump,
            is_gold = is_gold,
            # marker information
            dist_btn_markers = dist_btn_markers,
            vent_idx = vent_idx,
            # load in the data
            pr = pr,
            time_offset = time_offset,
            # find first and last indices of a complete snake
            no_nans = no_nans,
            idx_keep_0 = idx_keep_0,
            idx_keep_1 = idx_keep_1,
            # select out the data
            pr_with_nans = pr_with_nans,
            pgap = pgap,
            times = times,
            frames = frames,
            ntime = ntime,
            # fill in missing values with an unscented Kalman filter
            pfill = pfill,
            nans_fill = nans_fill,
            pfill0 = pfill0,
            # find the optimal Butterworth filter cutoff frequencies
            R = R,
            fcs = fcs,
            inter = inter,
            fcopt = fcopt,
            rsq = rsq,
            flinreg = flinreg,
            # perform Butterworth filtering
            bup = bup,
            buv = buv,
            bua = bua,
            # extract/rename smooth position, velocity, acceleration valueds
            pf_I = pf_I,
            vf_I = vf_I,
            af_I = af_I,
            # add a neck to the snake's body to constrain the spline
            pfe_I = pfe_I,
            tcoord_e = tcoord_e,
            # fit a spline to the snake
            R_I = R_I,
            dRds_I = dRds_I,
            ddRds_I = ddRds_I,
            dddRds_I = dddRds_I,
            spl_ds = spl_ds,
            mass_spl = mass_spl,
            chord_spl = chord_spl,
            vent_idx_spl = vent_idx_spl,
            times2D = times2D,
            t_coord = t_coord,
            s_coord = s_coord,
            spl_len_error = spl_len_errors,
            idx_pts = idx_pts,
            SVL = SVL,
            VTL = VTL,
            # error metric for the splines
            err_mean = err_mean,
            err_std = err_std,
            # raw CoM velocities
            Ro_I_raw = Ro_I_raw,
            dRo_I_raw = dRo_I_raw,
            ddRo_I_raw = ddRo_I_raw,
            # filter the CoM position to calculate velocities
            resid_Ro = resid_Ro,
            fcs_Ro = fcs_Ro,
            inter_Ro = inter_Ro,
            fcopt_Ro = fcopt_Ro,
            rsq_Ro = rsq_Ro,
            flinreg_Ro = flinreg_Ro,
            # CoM velocity and acceleration
            Ro_I = Ro_I,
            dRo_I = dRo_I,
            ddRo_I = ddRo_I,
            # CoM shift, calculate velocity and accelerations of spline and markers
            R_Ic = R_Ic,
            pf_Ic = pf_Ic,
            pfe_Ic = pfe_Ic,
            vf_Ic = vf_Ic,
            af_Ic = af_Ic,
            dR_I = dR_I,
            ddR_I = ddR_I,
            dR_Ic = dR_Ic,
            ddR_Ic = ddR_Ic,
            # straighten the trajectory
            Ro_S = Ro_S,
            R_Sc = R_Sc,
            pf_Sc = pf_Sc,
            pfe_Sc = pfe_Sc,
            R_S = R_S,
            pf_S = pf_S,
            pfe_S = pfe_S,
            vf_S = vf_S,
            vf_Sc = vf_Sc,
            dRo_S = dRo_S,
            dR_S = dR_S,
            dR_Sc = dR_Sc,
            af_S = af_S,
            af_Sc = af_Sc,
            ddRo_S = ddRo_S,
            ddR_S = ddR_S,
            ddR_Sc = ddR_Sc,
            dRds_S = dRds_S,
            ddRds_S = ddRds_S,
            dddRds_S = dddRds_S,
            mus = mus,
            Rmus = Rmus,
            #yaw = yaw,
            gamma = gamma,
            C_I2S = C_I2S,
            # apply the body coordinate system (TBC) and calculate curvatures
            Tdir_S = Tdir_S,
            Cdir_S = Cdir_S,
            Bdir_S = Bdir_S,
            Cb_S = Cb_S,
            a_angs = a_angs,
            b_angs = b_angs,
            kap_signed = kap_signed,
            kap_unsigned = kap_unsigned,
            Tdir_I = Tdir_I,
            Cdir_I = Cdir_I,
            Bdir_I = Bdir_I,
            Cb_I = Cb_I,
            tau = tau,
            # airfoils shape for plotting and movies in straightened frame
            foils_Sc = foils_Sc,
            foils_Ic = foils_Ic,
            foil_color = foil_color,
            foils_S = foils_S,
            foils_I = foils_I,
            # fit a plane to the snake's body
            C_I2B = C_I2B,
            Mw_I = Mw_I,
            Nhat_I  = Nhat_I ,
            Sfrac = Sfrac,
            planar_fit_error = planar_fit_error,
            # rotate everything to body frame
            dRo_B = dRo_B,
            ddRo_B = ddRo_B,
            dRds_B = dRds_B,
            ddRds_B = ddRds_B,
            R_B = R_B,
            dR_B = dR_B,
            ddR_B = ddR_B,
            pf_B = pf_B,
            vf_B = vf_B,
            af_B = af_B,
            pfe_B = pfe_B,
            foils_B = foils_B,
            Tdir_B = Tdir_B,
            Cdir_B = Cdir_B,
            Bdir_B = Bdir_B,
            dR_Bc = dR_Bc,
            ddR_Bc = ddR_Bc,
            vf_Bc = vf_Bc,
            af_Bc = af_Bc,
            # apply the body coordinate system (for plotting)
            Xp_B = Xp_B, Yp_B = Yp_B, Zp_B = Zp_B,
            Xp_I = Xp_I, Yp_I = Yp_I, Zp_I = Zp_I,
            Xp_S = Xp_S, Yp_S = Yp_S, Zp_S = Zp_S,
            YZ_B = YZ_B, XZ_B = XZ_B, XY_B = XY_B,
            YZ_I = YZ_I, XZ_I = XZ_I, XY_I = XY_I,
            YZ_S = YZ_S, XZ_S = XZ_S, XY_S = XY_S,
            # yaw, pitch, roll angles; angular velcity and acceleration
            yaw = yaw,
            pitch = pitch,
            roll = roll,
            dyaw = dyaw,
            dpitch = dpitch,
            droll = droll,
            ddyaw = ddyaw,
            ddpitch = ddpitch,
            ddroll = ddroll,
            omg_B = omg_B,
            domg_B = domg_B,
            ddomg_B = ddomg_B,
            omg_I = omg_I,
            domg_I = domg_I,
            ddomg_I = ddomg_I,
            # aerodynamic forces
            Fl_I = Fl_I,
            Fd_I = Fd_I,
            Fa_I = Fa_I,
            Ml_I = Ml_I,
            Md_I = Md_I,
            Ma_I = Ma_I,
            Fl_B = Fl_B,
            Fd_B = Fd_B,
            Fa_B = Fa_B,
            Ml_B = Ml_B,
            Md_B = Md_B,
            Ma_B  = Ma_B,
            Fl_S = Fl_S,
            Fd_S = Fd_S,
            Fa_S = Fa_S,
            Ml_S = Ml_S,
            Md_S = Md_S,
            Ma_S = Ma_S,
            Re = Re,
            aoa = aoa,
            beta = beta,
            dynP = dynP,
            dynP_frac = dynP_frac,
            dR_BC_I = dR_BC_I,
            dR_TC_I  = dR_TC_I ,
            U_BC_I = U_BC_I,
            U_TC_I = U_TC_I,
            cl = cl,
            cd = cd,
            clcd = clcd,
            U_tot_I = U_tot_I)

    # timing information
    tend = time.time() - now
    print('Elapsed time: {0:.3f} sec\n'.format(tend))
    loop_times.append(tend)

loop_times = np.array(loop_times)