#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pdb

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)
        self.segment_data()
        # Should probably do more preprocessing here

    def _build_nn(self):
        model = Sequential()
        model.add(Dense(300, activation='relu', input_shape=(15,)))
        model.add(Dense(300, activation='relu'))
        # model.add(Dense(300, activation='relu'))
        # model.add(Dense(300, activation='relu'))
        # model.add(Dense(300, activation='relu'))
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
        for i in range(self.lidar_range.shape[0]):
            data = self.lidar_range[i]
            persons = self.ped_pos[i]
            angles = np.array([np.arctan(y / x) for x,y in persons])
            for j in range(len(data) // 15):
                idx = j * 15
                for k in range(len(data[idx : idx + 15]) // 5):
                    curr_x = data[idx + k*5 : idx + k*5 + 15]
                    angle = -1.69296944141 + (idx + k * 5) * 0.00872664619237
                    if np.any(np.logical_and(angles >= angle, angles < angle + 0.13089969288)):
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
        # L = 100
        print("{} positive examples used".format(L))

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



    def train(self):
        self.nn.fit(self.X_train, self.Y_train, 
                    batch_size=128, epochs=20, verbose=1, 
                    shuffle=True, validation_split=0.2)

    # TODO: write evaluation function
