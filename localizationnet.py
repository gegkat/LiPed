#!/usr/bin/python

import pdb
import time
import numpy as np

from liped import LiPed
from lputils import *
from settings import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler


def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

class LocNet(LiPed):
    def __init__(self, *args, **kwargs):
        super(LocNet, self).__init__(*args, **kwargs)

    def _build_nn(self):
        width = self.X_train.shape[1]
        model = Sequential()
        model.add(Conv1D(N_LOCNET_NODES,SEGL, strides=1, activation='relu', 
            input_shape=(None, 1)))
        for _ in range(N_LOCNET_LAYERS - 1):
            model.add(Dropout(DROPOUT_RATE))
            model.add(Conv1D(N_LOCNET_NODES,1, strides=1, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Conv1D(2,1, strides=1))
        model.compile(loss='mse',
                      optimizer='adam', 
                      metrics=['mae'])
        return model

    def segment_data(self, rd, ped_pos):


        # In this function the suffix d is for data from lidar
        # while the suffix p is for pedestrian from YOLO
        # Examples: xd, yd, rd, thd, xp, yp, rp, thp

        N_frames = len(ped_pos)
        thd = self.lidar_angle[self.in_view]
        interp_func = interp1d(thd, range(len(thd)), kind='nearest')
        X = []
        Y = []
        start = time.time()

        # Convert lidar data to cartesian
        xd, yd = pol2cart(rd, thd)

        for i in range(N_frames):
            curr_ped_pos = np.array(ped_pos[i])

            # Skip if no pedestrians in this frame
            if curr_ped_pos.ndim < 2:
                continue

            # pull out x and y
            xp = curr_ped_pos[:, 0]
            yp = curr_ped_pos[:, 1]

            if TRAIN_SNAP_TO_CLOSEST:
                xp, yp = snap_to_closest(xp, yp, xd[i,:], yd[i,:])

            # convert pedestrian coordinate to polar
            rp, thp = cart2pol(xp, yp)

            # Find closest match for pedestrian angle to lidar angle
            idx = interp_func(thp).astype(int)

            # Find index of beginning of segment with
            # idx at the beginning of the WINDOW 
            q1 = idx - (SEGL - PADDING) + 1

            # Iterate over pedestrian detections
            for j in range(len(q1)):

                # Iterate over all positions of pedestrian in WINDOW
                for k in range(WINDOW):
                    q2 = q1[j] + k

                    if q2 >= 0:
                        X.append(rd[i, q2:q2+SEGL])
                        if LOCNET_TYPE == 'cartesian':
                            # x/y relative to first point in the segment
                            Y.append([xp[j] - xd[i,q2], yp[j] - yd[i,q2]])
                        elif LOCNET_TYPE == 'polar':
                            # th relative to first point in the segment
                            Y.append([rp[j], thp[j] - thd[q2]])

        X = np.array(X)
        Y = np.array(Y)

        if LOCNET_TYPE == 'polar':
            Y[:,0] = (Y[:,0] - R_BIAS)/R_SCALE
            Y[:,1] = (Y[:,1] - TH_BIAS)/TH_SCALE

        end = time.time()
        print('Segmented data in {:.2f} seconds'.format(end-start))

        return X, Y

    def train(self, epochs=5):
        print("Segmenting training data")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.ped_pos_train)
        # Change dimensions for convolutional
        self.X_train = np.expand_dims(self.X_train, 2)
        self.Y_train = np.expand_dims(self.Y_train, 1)

        super(LocNet, self).train(epochs=epochs, regression=True)
     



