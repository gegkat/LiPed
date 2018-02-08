#!/usr/bin/python

import sys
import cPickle as pickle
import pdb
import scipy.io
import numpy as np

def load_laser_data(pickle_file):

        lidar_time = []
        lidar_range = []
        
        count = 0
        with (open(pickle_file, "rb")) as openfile:
            while True:
                try:
                    object = pickle.load(openfile)
                    count += 1
                    # print(count)
                    # pdb.set_trace()
                    # object = [object[0]] + list(object[1])
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

        ped_time = []
        ped_x = []
        ped_y = []
        
        count = 0
        with (open(pickle_file, "rb")) as openfile:
            while True:
                try:
                    object = pickle.load(openfile)
                    count += 1
                    # print(count)
                    # pdb.set_trace()
                    # object = [object[0]] + list(object[1])
                    ped_time.append(object[0])
                    ped_x.append(object[1])
                    ped_y.append(object[2])

                    if count > 1000:
                        break
                except EOFError:
                    break

        ped_time = np.array(ped_time)
        ped_x = np.array(ped_x)
        ped_y = np.array(ped_y)

        return ped_time, ped_x, ped_y

if __name__ == "__main__":
    lidar_time, lidar_range = load_laser_data(sys.argv[1])
    ped_time, ped_x, ped_y = load_pedestrian_data(sys.argv[2])
    # with open('lidar.p', 'wb') as f:
    #     pickle.dump([lidar_time, lidar_range], f)