#!/usr/bin/python

import pdb
import time
import numpy as np

from liped import LiPed
from lputils import *
from settings import *

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler



class CNNLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(CNNLiPed, self).__init__(*args, **kwargs)

    def _build_nn(self):
        width = self.X_train.shape[1]
        model = Sequential()
        model.add(Conv1D(200,SEGL, strides=1, activation='relu', 
            input_shape=(None, 1)))
        model.add(Conv1D(200,1, strides=1, activation='relu'))
        model.add(Conv1D(200,1, strides=1, activation='relu'))
        model.add(Conv1D(1,1, strides=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def load_localization_model(self, model_file):
        self.locmodel = load_model(model_file)

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

    def predict_prob(self, rd, thd):
        '''
        Runs range data r through the network producing probabilities of pedestrian
        presence on a sliding window. Sliding window is performed fully convolutionally. 
        Also returns the range and angle at the center of each sliding window which is
        why theta needs to be input. 

        Note that no thresholding is applied here. 

        Inputs: 
          rd:  range data to feed into network. Must have
              dimension [n, m] where n is number of frames 
              and m is width of lidar scan 

          thd: angle corresponding to r. Must have dimension
              [m] where m is width of lidar scan

          Since the network is fully convolutional, it will perform a dense 
          sliding window across any lidar scan width m. The window size is 
          given by SEGL and must be adhered to only in training. In prediciton
          any data width can be used and the prediction output will grow accordingly. 

        Outputs:
           Y: probability of a pedestrian for each sliding window. Dimensiosn are
              [n, m] whrere n is the number of frames and m is the number
              of sliding windows that fit in the data

          rp: The range at the center of each sliding window, dimensions [n, m] 

         thp: The theta at the center of each sliding window, dimensions [n, m]

        '''

        X = np.expand_dims(rd, 2)
        Y = self.nn.predict(X)
        Y = Y[:,:,0]


        if USE_LOCNET:
            Y2 = self.locmodel.predict(X)

            # In this function the suffix d is for data from lidar
            # while the suffix p is for prediction
            # Examples: xd, yd, rd, thd, xp, yp, rp, thp

            # Convert lidar data to x,y
            xd, yd = pol2cart(rd, thd)

            if LOCNET_TYPE == 'cartesian':
                xp = Y2[:,:,0]
                yp = Y2[:,:,1]

                # Adjust prediction to be absolute
                # rather than relative to window
                xp += xd[:,:xp.shape[1]]
                yp += yd[:,:yp.shape[1]]

                rp, thp = cart2pol(xp, yp)

            elif LOCNET_TYPE == 'polar':
                rp =  Y2[:,:,0]*R_SCALE + R_BIAS
                thp = Y2[:,:,1]*TH_SCALE + TH_BIAS

                # Adjust prediction to be absolute
                # rather than relative to window
                thp += thd[:thp.shape[1]]

                xp, yp = pol2cart(rp, thp)
            else:
                error("Did not recognize locnet type: {}".format(LOCNET_TYPE))

            # Snap to closest lidar point
            if PRED_SNAP_TO_CLOSEST:
                for i in range(xp.shape[0]):
                    xp[i,:], yp[i,:] = snap_to_closest(xp[i,:], yp[i,:], xd[i,:], yd[i,:])

                rp, thp = cart2pol(xp, yp)

        else:
            # If no localization net, return data at center of each segment
            # which is just the same as the raw data with some padding taken
            # off at the beginning and the end. 
            padding = SEGL//2
            thp = thd[padding:-(padding-1)]
            thp = np.tile(thp, (rd.shape[0], 1))
            rp = rd[:,padding:-(padding-1)]

        return Y, rp, thp

    def predict(self, r, th):
        '''
            Intended for a single frame of data 
        '''

        r = np.expand_dims(r, 0)
        pred_probability, pred_r, pred_th = self.predict_prob(r, th)

        pred_r, pred_th = apply_thresholds(pred_probability, 
            [self.pred_thresh], pred_r, pred_th)

        return pred_r[0, 0], pred_th[0, 0]       



