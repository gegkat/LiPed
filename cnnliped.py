#!/usr/bin/python

import pdb
import time
import numpy as np

from liped import LiPed
from lputils import apply_thresholds, get_segments_per_frame
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from keras.models import load_model
import pdb
import time

WINDOW = 6
PADDING = 7 # Amount on each side segment that is only for padding
SEGL = WINDOW + 2*PADDING # total segment length 
SEG_STRIDE = 1 # segementation stride
R_BIAS = 0
R_SCALE = 10
TH_BIAS = 0.0087266461923 * PADDING
TH_SCALE = 0.00872664619237 * WINDOW

class CNNLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        self.locmodel = load_model('locnet_2018-03-03_23-17-05/model.h5')
        super(CNNLiPed, self).__init__(*args, **kwargs)

    def _build_nn(self):
        width = self.X_train.shape[1]
        model = Sequential()
        model.add(Conv1D(50,SEGL, strides=1, activation='relu', 
            input_shape=(None, 1)))
        model.add(Conv1D(50,1, strides=1, activation='relu'))
        model.add(Conv1D(50,1, strides=1, activation='relu'))
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
            Y[:,j] = np.any(ped_onehot[:, (idx1+PADDING):(idx1+PADDING + WINDOW) ], 
                axis=1)

        X = X.reshape((-1, SEGL))
        Y = Y.flatten()

        return X, Y

    def train(self, epochs=5):
        print("Segmenting training data")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.Y_train)

        print("Over sampling")
        sampler = SMOTE(ratio='minority', random_state=42)

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

        start = time.time()
    

        X = np.expand_dims(r, 2)
        Y = self.nn.predict(X)
        Y = Y[:,:,0]
        stop = time.time()
        # print("Processed {} frames in {} seconds. {} frames per sec".format(
        #     Y.shape[0], stop - start, (Y.shape[0])/float(stop-start)))

        start = time.time()
        use_locnet = True
        if use_locnet:
            Y2 = self.locmodel.predict(X)
            pred_r = Y2[:,:,0]*R_SCALE + R_BIAS
            pred_th = Y2[:,:,1]*TH_SCALE + TH_BIAS
            pred_th += th[:pred_th.shape[1]]
        else:
            padding = SEGL//2
            pred_th = th[padding:-(padding-1)]
            pred_th = np.tile(pred_th, (r.shape[0], 1))
            pred_r = r[:,padding:-(padding-1)]
        stop = time.time()
        # print("Processed {} frames in {} seconds. {} frames per sec".format(
        #     Y.shape[0], stop - start, (Y.shape[0])/float(stop-start)))
        
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



