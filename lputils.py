#!/usr/bin/python

import pdb
import numpy as np
import utils

from settings import *

import matplotlib.pyplot as plt

import cPickle as pickle
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter



def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

def get_score(pr, pth, tr, tth):

    px, py = pol2cart(pr, pth)
    tx, ty = pol2cart(tr, tth)

    false_pos = 0
    false_neg = 0
    true_pos = len(px)

    L1 = len(px)
    L2 = len(tx)
    d = np.zeros((L1,L2))
    for i in range(L1):
        d[i,:] = np.sqrt((px[i] - tx)**2 + (py[i] - ty)**2)

    while d.shape[0] and d.shape[1]: 
        i,j = np.unravel_index(d.argmin(), d.shape)
        if d[i,j] > DIST_THRESH:
            break

        d = np.delete(d, (i), axis=0)
        d = np.delete(d, (j), axis=1)

    false_pos = d.shape[0]
    false_neg = d.shape[1]
    true_pos = len(px) - false_pos

    return false_pos, false_neg, true_pos


def apply_thresholds(prob, thresholds, r, th, raw_r=None, raw_th=None, ped_pos=None):
    '''
    Takes probability data y and applies thresholds to determine
    pedestrian or not. Returns r and theta as 2d arrays of lists.

    Inputs: 
      prob: probability of a pedestrian for each sliding window. Dimensiosn are
         [N, M] whrere n is the number of frames and m is the number
         of sliding windows 

    thresholds: a list of L thresholds to apply to y for pedestrian detection

      r:  The range at the center of each sliding window, dimensions [N, M] 

      th: The theta at the center of each sliding window, dimensions [N, M]

    Outputs:

    r_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing range values of pedestrian detections. 

    th_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing theta values of pedestrian detections. 

    '''

    N_frames = prob.shape[0]
    N_threshes = len(thresholds)

    prob = np.tile(prob, (N_threshes, 1, 1))
    prob = np.moveaxis(prob, 0, 2)
    idx = prob > thresholds  

    r_list = np.empty((N_frames, N_threshes), dtype=object)
    th_list = np.empty((N_frames, N_threshes), dtype=object)
    for i in range(N_frames):
        utils.print_progress_bar(i, N_frames, 
            prefix = 'Progress:', suffix = 'Complete', bar_length = 50)

        xc, yc = pol2cart(r[i,:], th[i,:])

        # Plot looks better if background covers all data
        if DO_THRESHOLD_PLOT: 
            Yb = np.arange(XLIMS[0], XLIMS[1], SCORE_RES)
            Xb = np.arange(YLIMS[0], YLIMS[1], SCORE_RES)
        else:
            Xb = np.arange(xc.min()-SCORE_RES, xc.max()+SCORE_RES, SCORE_RES)
            Yb = np.arange(yc.min()-SCORE_RES, yc.max()+SCORE_RES, SCORE_RES)
        Xm, Ym = np.meshgrid(Xb, Yb)

        for j in range(N_threshes):
            k = idx[i,:, j]

            score = voting(xc[k], yc[k], Xb, Yb)
            xout, yout = non_max_suppression(score.flatten(), Xm.flatten(), Ym.flatten())
            r_list[i, j], th_list[i, j] = cart2pol(xout, yout)

            if DO_THRESHOLD_PLOT and raw_r is not None:
                rawx, rawy = pol2cart(raw_r[i,:], raw_th)

                truth_pos = np.array(ped_pos[i])
                truth_x = np.array([])
                truth_y = np.array([])
                if truth_pos.ndim == 2:
                    truth_x = truth_pos[:,0]
                    truth_y = truth_pos[:,1]

                # Skip negative examples if desired
                if len(truth_x) < 3 or len(xout) < 2:
                    continue

                fig = plt.figure()
                score[score < MIN_SCORE] = np.nan # blank out background 0's
                plt.pcolormesh(Ym.T, Xm.T, score.T, cmap='Reds') # magma looks nice
                plt.plot(rawy, rawx, linestyle='', marker='.', markeredgecolor='black', markersize=2)
                plt.plot(yout, xout, linestyle='', marker='o', 
                        markeredgecolor='red', markersize=15, fillstyle='none',
                        markeredgewidth=0.5)
                plt.plot(truth_y, truth_x,  linestyle='', marker='s', 
                    markeredgecolor='g', markersize=15, fillstyle='none',
                    markeredgewidth=0.5)
                # plt.plot(yc, xc,  linestyle='', marker='.', 
                        # markeredgecolor='yellow', markersize=2)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.gca().set_xlim((rawy.min(), rawy.max()))
                plt.gca().set_ylim((rawx.min(), rawx.max()))
                plt.gca().invert_xaxis()
                plt.title("Test Frame: {}".format(i))

                plt.show()


    return r_list, th_list

def voting(x, y, Xb, Yb):

    # Make data.
    ix = np.digitize(x, Xb) - 1
    iy = np.digitize(y, Yb) - 1

    score = np.zeros((len(Yb), len(Xb)))
    for i in range(len(ix)):
        score[iy[i], ix[i]] += 1

    # Constant mode pads edges with cval 
    # Truncate stops filter at this many standard deviations
    max1 = np.max(score)
    if max1 == 0:
        return score

    score = gaussian_filter(score, FILTER_SIGMA, 
        mode='constant', cval=0.0, truncate=FILTER_TRUNCATE)

    # Scale back up so 1 vote is still 1 vote
    score = score * max1/score.max()

    return score

def non_max_suppression(score, x, y):

    idx = score > MIN_SCORE
    score = score[idx]
    x = x[idx]
    y = y[idx]

    isort = np.argsort(-score) # descending sort
    x = x[isort]
    y = y[isort]

    xout = []
    yout = []
    while len(x) > 0: 
        xout.append(x[0])
        yout.append(y[0])
        x = np.delete(x, 0)
        y = np.delete(y, 0)
        if len(x) == 0:
            break
        d = ((xout[-1] - x)**2 + (yout[-1] - y)**2)**.5
        rm_ind = np.where(d < NMS_DIST_THRESH)
        x = np.delete(x, rm_ind)
        y = np.delete(y, rm_ind)

    xout = np.array(xout)
    yout = np.array(yout)

    return xout, yout

def snap_to_closest(xp, yp, xd, yd):
    # expand dimensions so we can get an automatic broadcasting to 2 dimensions
    xd = np.expand_dims(xd, 0)
    yd = np.expand_dims(yd, 0)
    xp = np.expand_dims(xp, 1)
    yp = np.expand_dims(yp, 1)

    # Calculate distance
    dis = (xp - xd)**2 + (yp - yd)**2

    # Find closest
    idx = np.argmin(dis, axis=1)

    # Set xp/yp to closest
    xp = xd[0,idx]
    yp = yd[0,idx]

    return xp, yp

def pos2cart(pos):
        x = []
        y = []
        for i in range(len(pos)):
            x.append(pos[i][0])
            y.append(pos[i][1])

        return np.array(x), np.array(y)

def pos2pol(pos):
        x, y = pos2cart(pos)
        r, theta = cart2pol(x, y)

        return r, theta

def ped_to_onehot(ped_pos, lidar_angle):
    N_frames = len(ped_pos)

    # Use nearest neighbor interpolation to find angle for pedestrians
    interp_func = interp1d(lidar_angle, range(len(lidar_angle)), kind='nearest')
    ped_onehot = np.zeros((N_frames, len(lidar_angle)), dtype=bool)
    for i in range(N_frames):
        angles = np.array([np.arctan2(y / x) for x,y in ped_pos[i]])
        idx = interp_func(angles).astype(int)
        ped_onehot[i, idx] = 1

    return ped_onehot

def pol2cart(r, theta):
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return x, y

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def load_laser_data(pickle_file):
    print('parsing laser data: {}'.format(pickle_file))

    lidar_time = []
    lidar_range = []
    
    count = 0
    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
                count += 1
                lidar_time.append(object[0]) 
                lidar_range.append( list(object[1]))

                # if count > 1000:
                    # break
            except EOFError:
                break

    lidar_time = np.array(lidar_time)
    lidar_range = np.array(lidar_range)

    return lidar_time, lidar_range


def load_pedestrian_data(pickle_file):
    print('parsing pedestrian data: {}'.format(pickle_file))

    ped_time = []
    ped_pos = []

    prev_time = None
    
    count = 0
    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
                count += 1

                cur_time = object[0]
                if cur_time == prev_time: 
                    ped_pos[-1].append([object[1], object[2]])
                else:
                    ped_time.append(object[0])
                    if object[1] is not None:
                        ped_pos.append([[object[1], object[2]]])
                    else:
                        ped_pos.append([])

                prev_time = cur_time

                # if count > 1000:
                    # break
            except EOFError:
                break

    ped_time = np.array(ped_time)

    return ped_time, ped_pos