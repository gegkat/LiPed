#!/usr/bin/python

from liped import LiPed
from liped import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
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

def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

class LocNet(LiPed):
    def __init__(self, *args, **kwargs):
        super(LocNet, self).__init__(*args, **kwargs)

    def _build_nn(self):
        width = self.X_train.shape[1]
        model = Sequential()
        model.add(Conv1D(500,SEGL, strides=1, activation='relu', 
            input_shape=(None, 1)))
        model.add(Conv1D(500,1, strides=1, activation='relu'))
        model.add(Conv1D(500,1, strides=1, activation='relu'))
        model.add(Conv1D(500,1, strides=1, activation='relu'))
        # model.add(Conv1D(500,1, strides=1, activation='relu'))
        # model.add(Conv1D(2,1, strides=1))
        model.add(Conv1D(2,1, strides=1))
        model.compile(loss='mse',
                      optimizer='adam', 
                      metrics=['mae'])
        return model

    def segment_data(self, data, ped_pos):

        N_frames = len(ped_pos)
        angles = self.lidar_angle[self.in_view]
        interp_func = interp1d(angles, range(len(angles)), kind='nearest')
        X = []
        Y = []
        start = time.time()
        for i in range(N_frames):
            pols = np.array([cart2pol(x, y) for x,y in ped_pos[i]])


            for j in range(len(pols)):
                r = pols[j,0]
                th = pols[j,1]
                idx = interp_func(th).astype(int)
                q1 = idx - (SEGL - PADDING) + 1

                for k in range(WINDOW):
                    q2 = q1 + k
                    if q2 >= 0:
                        X.append(data[i, q2:q2+SEGL])
                        Y.append([r, th - angles[q2]])
                        # Y.append([th - angles[q2]])

        X = np.array(X)
        Y = np.array(Y)
        Y[:,0] = (Y[:,0] - R_BIAS)/R_SCALE
        Y[:,1] = (Y[:,1] - TH_BIAS)/TH_SCALE
        # Y[:,0] = (Y[:,0] - TH_BIAS)/TH_SCALE
        stop = time.time()
        print(stop - start)

        return X, Y

    def train(self, epochs=5):
        print("Segmenting training data")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.ped_pos_train)
        # Change dimensions for convolutional
        self.X_train = np.expand_dims(self.X_train, 2)
        self.Y_train = np.expand_dims(self.Y_train, 1)

        super(LocNet, self).train(epochs=epochs, regression=True)

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
        padding = SEGL//2
        pred_th = th[padding:-(padding-1)]
        pred_th = np.tile(pred_th, (r.shape[0], 1))
        pred_r = r[:,padding:-(padding-1)]

        X = np.expand_dims(r, 2)
        Y = self.nn.predict(X)
        Y = Y[:,:,0]
        stop = time.time()
        # print("Processed {} frames in {} seconds. {} frames per sec".format(
            # Y.shape[0], stop - start, (Y.shape[0])/float(stop-start)))
        
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



