#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pdb

SEGL = 15 # segmentation size
SEG_STRIDE = 5 # segementation stride
PRED_THRESH = 0.9 # threshold for labeling a prediction as pedestrian

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)
        # Split last 20% of data as test set
        split = int(self.lidar_range.shape[0] * 0.8)
        self.X_test, self.Y_test = self.segment_data(self.lidar_range[split:], self.ped_pos[split:])

        # Train/validation split with shuffle and oversampling
        X, Y = self.segment_data(self.lidar_range[:split], self.ped_pos[:split])
        X_train, self.X_val, Y_train, self.Y_val = train_test_split(X, Y, test_size=0.1, 
            shuffle=True, random_state=42)
        sm = SMOTE(ratio='minority', random_state=42)
        self.X_train, self.Y_train = sm.fit_sample(X_train, Y_train)

    def _build_nn(self):
        model = Sequential()
        model.add(Dense(300, activation='relu', input_shape=(SEGL,)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def segment_data(self, data, ped_pos):
        X = []
        Y = []
        for i in range(data.shape[0]):
            datum = data[i]
            persons = ped_pos[i]
            angles = np.array([np.arctan(y / x) for x,y in persons])
            for j in range(len(data) // SEGL):
                idx = j * SEGL
                for k in range(len(data[idx : idx + SEGL]) // SEG_STRIDE):
                    X.append(datum[idx + k*SEG_STRIDE : idx + k*SEG_STRIDE + SEGL])
                    angle = -1.69296944141 + (idx + k * SEG_STRIDE) * 0.00872664619237 # Can we index into self.lidar_angles?
                    if np.any(np.logical_and(angles >= angle, angles < angle + SEGL*0.00872664619237)):
                        Y.append(1)
                    else:
                        Y.append(0)

        return np.array(X), np.array(Y)

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




