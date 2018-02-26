#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pdb

SEGL = 15 # segmentation size
SEG_STRIDE = 5 # segementation stride
PRED_THRESH = 0.9 # threshold for labeling a prediction as pedestrian

def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)


        print("Segmenting test datadata")
        self.X_test, self.Y_test = self.segment_data(self.X_test, self.Y_test)

        print("Segmenting training datadata")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.Y_train)

        print("Over sampling")
        sm = SMOTE(ratio='minority', random_state=42)
        self.X_train, self.Y_train = sm.fit_sample(self.X_train, self.Y_train)

    def _build_nn(self):
        model = Sequential()
        model.add(Dense(300, activation='relu', input_shape=(SEGL,)))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def segment_data(self, data, ped_onehot):
        N_frames = data.shape[0]
        width = data.shape[1]
        segments_per_frame = get_segments_per_frame(width, SEGL, SEG_STRIDE)

        X = np.zeros((N_frames, segments_per_frame,SEGL))
        Y = np.zeros((N_frames, segments_per_frame), dtype=int)
        count = 0
        for j in range(segments_per_frame):
            idx1 = j * SEG_STRIDE
            idx2 = idx1 + SEGL
            X[:, j, :] = data[:, idx1:idx2]
            Y[:,j] = np.any(ped_onehot[:, idx1:idx2], axis=1)

        X = X.reshape((-1, SEGL))
        Y = Y.flatten()

        return X, Y


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




