#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)
        self.segment_data()
        # Should probably do more preprocessing here

    def _build_nn(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(15,)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # TODO: save and load weights
        return model

    def segment_data(self):
        # TODO: Set up test data for evaluation
        self.X_train = []
        self.Y_train = []
        for i in range(self.lidar_range.shape[0]):
            data = self.lidar_range[i]
            persons = self.ped_pos[i]
            angles = np.array([np.arctan(y / x) for x,y in persons])
            for j in range(len(data) // 15):
                idx = j * 15
                for k in range(len(data[idx : idx + 15]) // 5):
                    self.X_train.append(data[idx + k*5 : idx + k*5 + 15])
                    angle = -1.69296944141 + (idx + k * 5) * 0.00872664619237
                    if np.any(np.logical_and(angles >= angle, angles < angle + 0.13089969288)):
                        self.Y_train.append(1)
                    else:
                        self.Y_train.append(0)
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)


    def train(self):
        self.nn.fit(self.X_train, self.Y_train, 
                    batch_size=32, epochs=1, verbose=1)

    # TODO: write evaluation function
