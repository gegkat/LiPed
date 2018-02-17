#!/usr/bin/python

import cPickle as pickle
# import pdb
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
# import time

class LiPed(object):
    def __init__(self, init=False, laser_file='', pedestrian_file='', data_dir='data'):
        if init:
            lidar_time, lidar_range = load_laser_data(laser_file)
            ped_time, ped_pos = load_pedestrian_data(pedestrian_file)
            # print(lidar_time.shape, lidar_range.shape, ped_time.shape, len(ped_x), len(ped_y))

            interp_func = interp1d(lidar_time, range(len(lidar_time)), 
                kind='nearest', fill_value='extrapolate')
            idx = interp_func(ped_time).astype(int)

            # Keep only lidar scans that match pedestrian detections
            lidar_time = lidar_time[idx]
            lidar_range = lidar_range[idx,:]
            np.save(data_dir + '/lidar_time', lidar_time)
            np.save(data_dir + '/lidar_range', lidar_range)
            np.save(data_dir + '/ped_time', ped_time)
            np.save(data_dir + '/ped_pos', ped_pos)
        else:
            lidar_time = np.load(data_dir + '/lidar_time.npy')
            lidar_range = np.load(data_dir + '/lidar_range.npy')
            ped_time = np.load(data_dir + '/ped_time.npy')
            ped_pos = np.load(data_dir + '/ped_pos.npy')

        self.lidar_angle = np.arange(-1.69296944141, 1.6929693222, 0.00872664619237)
        self.in_view = np.logical_and(self.lidar_angle > -0.5, self.lidar_angle < 0.5)

        self.lidar_time = lidar_time
        self.lidar_range = lidar_range
        self.ped_time = ped_time
        self.ped_pos = ped_pos

        # Generic neural network
        self.nn = self._build_nn()

    # Abstract method, to be implemented in subclasses
    def _build_nn(self):
        return None

    # Abstract method, to be implemented in subclasses
    def segment_data(self):
        pass

    # Abstract method, to be implemented in subclasses
    def train(self):
        pass

    def plot(self, frame):
        lx = np.cos(self.lidar_angle) * self.lidar_range[frame, :]
        ly = np.sin(self.lidar_angle) * self.lidar_range[frame, :]
        plt.plot(lx[self.in_view], ly[self.in_view], '.b')
        plt.plot(lx[~self.in_view], ly[~self.in_view], '.r')
        for i in range(len(self.ped_x[frame])):
            plt.plot(self.ped_x[frame][i], self.ped_y[frame][i], 'x')
        plt.show()

    # def movie(self, frames):
    #     for frame in frames:
    #         plt.clf()
    #         self.plot(frame)
    #         time.sleep(0.05)


def load_laser_data(pickle_file):
    print('parsing laser data')

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

                if count > 1000:
                    break
            except EOFError:
                break

    lidar_time = np.array(lidar_time)
    lidar_range = np.array(lidar_range)

    return lidar_time, lidar_range


def load_pedestrian_data(pickle_file):
    print('parsing pedestrian data')

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

                if count > 1000:
                    break
            except EOFError:
                break

    ped_time = np.array(ped_time)

    return ped_time, ped_pos
