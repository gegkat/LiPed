#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pdb

SEGL = 15 # segmentation size
SEG_STRIDE = 5 # segementation stride
PRED_THRESH = 0.9 # threshold for labeling a prediction as pedestrian

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)
        self.segment_data()
        # Should probably do more preprocessing here

    def _build_nn(self):
        model = Sequential()
        model.add(Dense(300, activation='relu', input_shape=(SEGL,)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # TODO: save and load weights
        return model

    def segment_data(self):
        # TODO: Set up test data for evaluation
        X_pos = []
        X_neg = []
        for i in range(self.N_frames):
            data = self.lidar_range[i]
            persons = self.ped_pos[i]
            angles = np.array([np.arctan(y / x) for x,y in persons])
            for j in range(len(data) // SEGL):
                idx = j * SEGL
                for k in range(len(data[idx : idx + SEGL]) // SEG_STRIDE):
                    curr_x = data[idx + k*SEG_STRIDE : idx + k*SEG_STRIDE + SEGL]
                    angle = -1.69296944141 + (idx + k * SEG_STRIDE) * 0.00872664619237 # Can we index into self.lidar_angles?
                    if np.any(np.logical_and(angles >= angle, angles < angle + SEGL*0.00872664619237 )):
                        X_pos.append(curr_x)
                    else:
                        X_neg.append(curr_x)

        X_pos = np.array(X_pos)
        X_neg = np.array(X_neg)

        # shuffle data along first dimension
        np.random.shuffle(X_pos)
        np.random.shuffle(X_neg)

        # Select number of positive examples to keep 
        L = X_pos.shape[0]
        # print("{} positive examples found".format(L))

        # Keep only the first L data points from pos/neg examples so
        # we have an equal proportion of pos/neg samples
        X_pos = X_pos[:L, :]
        X_neg = X_neg[:L, :]

        # Stack data
        X = np.vstack((X_pos, X_neg))

        # Construct labels
        Y = np.ones(2*L)
        Y[L:] = 0

        # Train/test split with shuffle
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, 
            shuffle=True, random_state=42)

    def predict(self, frame):
        data = self.lidar_range[frame,:]
        X = []
        angles = []
        ranges = []
        for j in range(len(data) // SEGL):
            idx = j * SEGL
            for k in range(len(data[idx : idx + SEGL]) // SEG_STRIDE):
                # Pull out current segment
                curr_x = data[idx + k*SEG_STRIDE : idx + k*SEG_STRIDE + SEGL]
                X.append(curr_x)

                # Figure out where to center pedestrian in this segment
                # The strategy is to find the minimum reasonable range
                cx = np.copy(curr_x)

                # remove 0 ranges
                cx[cx < 0.2] = 10000 

                # remove ranges too far from the median
                med = np.median(cx[cx < 10000])
                cx[cx - med < -0.6] = 10001

                ranges.append(cx.min())
                angles.append(self.lidar_angle[idx + k*SEG_STRIDE + np.argmin(cx)])


                # A different strategy is to use the entire segment
                # ranges.append(curr_x)
                # angles.append(self.lidar_angle[idx + k*SEG_STRIDE : idx + k*SEG_STRIDE + SEGL])

                # Another strategy is to use the center of the segment
                # angles.append(self.lidar_angle[idx + k*SEG_STRIDE : idx + k*SEG_STRIDE + SEGL].mean())
                # ranges.append(curr_x[SEGL // 2])
                

        angles = np.array(angles)
        ranges = np.array(ranges)

        X = np.array(X)
        y_pred = self.nn.predict(X)

        idx = y_pred[:,0] > PRED_THRESH
        angles = angles[idx]
        ranges = ranges[idx]
        x = np.cos(angles) * ranges
        y = np.sin(angles) * ranges

        return x, y




