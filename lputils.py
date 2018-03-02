#!/usr/bin/python

import cPickle as pickle
import numpy as np
from scipy.interpolate import interp1d

# Threshold for acceptable ped detection distance
DIST_THRESH = 1

def pol2cart(r, theta):
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return x, y

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

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

def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

def get_score(pr, pth, tr, tth):
    px, py = pol2cart(pr, pth)
    tx, ty = pol2cart(tr, tth)
    # false_pos = 0
    # false_neg = 0
    # true_pos = len(px)

    L1 = len(px)
    L2 = len(tx)
    d = np.zeros((L1, L2))
    for i in range(L1):
        d[i, :] = np.sqrt((px[i] - tx)**2 + (py[i] - ty)**2)

    while d.shape[0] and d.shape[1]: 
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] > DIST_THRESH:
            break
        d = np.delete(d, (i), axis=0)
        d = np.delete(d, (j), axis=1)

    false_pos = d.shape[0]
    false_neg = d.shape[1]
    true_pos = len(px) - false_pos
    return false_pos, false_neg, true_pos

def apply_thresholds(y, thresholds, r, th):
    '''
    Takes probability data y and applies thresholds to determine
    pedestrian or not. Returns r and theta as 2d arrays of lists.

    Inputs
    ------
    Y: probability of a pedestrian for each sliding window. Dimensions are
        [N, M] whrere n is the number of frames and m is the number
        of sliding windows 

    thresholds: a list of L thresholds to apply to y for pedestrian detection

    r: The range at the center of each sliding window, dimensions [N, M] 

    th: The theta at the center of each sliding window, dimensions [N, M]

    Outputs
    -------
    r_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing range values of pedestrian detections. 

    th_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing theta values of pedestrian detections. 
    '''
    N_frames = y.shape[0]
    N_threshes = len(thresholds)

    y = np.tile(y, (N_threshes, 1, 1))
    y = np.moveaxis(y, 0, 2)
    idx = y > thresholds

    r_list = np.empty((N_frames, N_threshes), dtype=object)
    th_list = np.empty((N_frames, N_threshes), dtype=object)
    for i in range(N_frames):
        for j in range(N_threshes):
            k = idx[i, :, j]
            r_list[i, j] = r[i, k]
            th_list[i, j] = th[i, k]

    return r_list, th_list

def ped_to_onehot(ped_pos, lidar_angle):
    N_frames = len(ped_pos)

    # Use nearest neighbor interpolation to find angle for pedestrians
    interp_func = interp1d(lidar_angle, range(len(lidar_angle)), kind='nearest')
    ped_onehot = np.zeros((N_frames, len(lidar_angle)), dtype=bool)
    for i in range(N_frames):
        angles = np.array([np.arctan(y / x) for x,y in ped_pos[i]])
        idx = interp_func(angles).astype(int)
        ped_onehot[i, idx] = 1

    return ped_onehot

def load_laser_data(pickle_file):
    print('parsing laser data: {}'.format(pickle_file))
    lidar_time = []
    lidar_range = []

    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
                lidar_time.append(object[0]) 
                lidar_range.append(list(object[1]))
            except EOFError:
                break

    return np.array(lidar_time), np.array(lidar_range)

def load_pedestrian_data(pickle_file):
    print('parsing pedestrian data: {}'.format(pickle_file))
    ped_time = []
    ped_pos = []

    prev_time = None   
    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
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
            except EOFError:
                break

    return np.array(ped_time), ped_pos
