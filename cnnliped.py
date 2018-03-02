#!/usr/bin/python

from liped import LiPed
from liped import apply_thresholds
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
import pdb
import time

SEGL = 10 # segmentation size
SEG_STRIDE = 1 # segementation stride

def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

class CNNLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(CNNLiPed, self).__init__(*args, **kwargs)

    def _build_nn(self):
        width = self.X_train.shape[1]
        model = Sequential()
        model.add(Conv1D(300,SEGL, strides=1, activation='relu', 
            input_shape=(None, 1)))
        model.add(Conv1D(300,1, strides=1, activation='relu'))
        model.add(Conv1D(1,1, strides=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def segment_data(self, data, ped_onehot):
        N_frames = data.shape[0]
        width = data.shape[1]
        segments_per_frame = get_segments_per_frame(width, SEGL, SEG_STRIDE)

        X = np.zeros((N_frames, segments_per_frame, SEGL))
        Y = np.zeros((N_frames, segments_per_frame), dtype=int)
        count = 0
        for j in range(segments_per_frame):
            idx1 = j * SEG_STRIDE
            idx2 = idx1 + SEGL
            X[:, j, :] = data[:, idx1:idx2]
            Y[:,j] = ped_onehot[:, idx1 + SEGL//2]

        X = X.reshape((-1, SEGL))
        Y = Y.flatten()

        return X, Y

    def train(self, epochs=5):
        print("Segmenting training data")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.Y_train)

        print("Over sampling")
        sampler = SMOTE(ratio='majority', random_state=42)

        # Under sampling option
        # print("Under sampling")
        # sampler = ClusterCentroids(random_state=42)
        # sampler = RandomUnderSampler(random_state=42)
        self.X_train, self.Y_train = sampler.fit_sample(self.X_train, self.Y_train)

        # Change dimensions for convolutional
        self.X_train = np.expand_dims(self.X_train, 2)
        self.Y_train = np.expand_dims(self.Y_train, 1)
        self.Y_train = np.expand_dims(self.Y_train, 2)

        super(CNNLiPed, self).train(epochs=epochs)

    def predict_prob(self, r, th):
        '''
        Runs range data r through the network producing probabilities of pedestrian
        presence on a sliding window. Sliding window is performed fully convolutionally. 
        Also returns the range and angle at the center of each sliding window which is
        why theta needs to be input. 

        Note that no thresholding is applied here. 

        Inputs: 
          r:  range data to feed into network. Must have
              dimension [n, m] where n is number of frames 
              and m is width of lidar scan 

          th: angle corresponding to r. Must have dimension
              [m] where m is width of lidar scan

          Since the network is fully convolutional, it will perform a dense 
          sliding window across any lidar scan width m. The window size is 
          given by SEGL and must be adhered to only in training. In prediciton
          any data width can be used and the prediction output will grow accordingly. 

        Outputs:
          Y: probability of a pedestrian for each sliding window. Dimensiosn are
             [n, m] whrere n is the number of frames and m is the number
             of sliding windows that fit in the data

        pred_r: The range at the center of each sliding window, dimensions [n, m] 

        pred_th: The theta at the center of each sliding window, dimensions [n, m]

        '''

        padding = SEGL//2
        pred_th = th[padding:-(padding-1)]
        pred_th = np.tile(pred_th, (r.shape[0], 1))
        pred_r = r[:,padding:-(padding-1)]

        X = np.expand_dims(r, 2)
        Y = self.nn.predict(X)
        Y = Y[:,:,0]
        
        return Y, pred_r, pred_th

    def predict(self, r, th):
        '''
            Intended for a single frame of data 
        '''

        r = np.expand_dims(r, 0)
        pred_probability, pred_r, pred_th = self.predict_prob(r, th)

        pred_r, pred_th = apply_thresholds(pred_probability, 
            [self.pred_thresh], pred_r, pred_th)

        return pred_r[0, 0], pred_th[0, 0]       



