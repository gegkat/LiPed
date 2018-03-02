#!/usr/bin/python

from liped import LiPed
from liped import apply_thresholds
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pdb

SEGL = 10 # segmentation size
SEG_STRIDE = 10 # segementation stride

def get_segments_per_frame(length, seg_length, stride):
    return (length - seg_length) // stride + 1

class SimpleLiPed(LiPed):
    def __init__(self, *args, **kwargs):
        super(SimpleLiPed, self).__init__(*args, **kwargs)

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

        X = np.zeros((N_frames, segments_per_frame, SEGL))
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

    def train(self, epochs=5):
        print("Segmenting training data")
        self.X_train, self.Y_train = self.segment_data(self.X_train, self.Y_train)

        print("Over sampling")
        sm = SMOTE(ratio='minority', random_state=42)
        self.X_train, self.Y_train = sm.fit_sample(self.X_train, self.Y_train)

        super(SimpleLiPed, self).train(epochs=epochs)

    def predict_prob(self, r, th):

        N_frames = r.shape[0]
        width = r.shape[1]

        # Segment range data with sliding window
        idx1 = np.arange(0, width - SEGL + 1, SEG_STRIDE)
        idx2 = np.arange(SEGL)
        idx = np.expand_dims(idx1, 1)+ idx2

        # Initialize
        segments_per_frame = idx.shape[0]
        y_pred = np.zeros((N_frames, segments_per_frame))
        pred_th = np.zeros((N_frames, segments_per_frame))
        pred_r = np.zeros((N_frames, segments_per_frame))

        for i in range(N_frames):
            X = r[i, idx]
            y_pred[i,:] = self.nn.predict(X)[:,0]

            # Determine most likely pedestrian location for each segment
            for j in range(segments_per_frame):
                idx1 = j * SEG_STRIDE
                idx2 = idx1 + SEGL

                # Figure out where to center pedestrian in this segment
                # The strategy is to find the minimum reasonable range
                cx = np.copy(X[j,:])

                # remove 0 ranges
                cx[cx < 0.2] = 10000 

                # remove ranges too far from the median
                med = np.median(cx[cx < 10000])
                cx[cx - med < -0.6] = 10001

                pred_r[i, j] = cx.min()
                pred_th[i, j] = th[idx1 + cx.argmin()]

                # A different strategy is to use the entire segment
                # pred_r[j] = cx
                # pred_th[j] = th[idx1 : idx2]

                # Another strategy is to use the center of the segment
                # pred_th[j] = th[idx1 : idx2].mean()
                # pred_r[j] = cx[SEGL // 2]
            
        return y_pred, pred_r, pred_th

    def predict(self, r, th):
        '''
            Intended for a single frame of data 
        '''

        r = np.expand_dims(r, 0)
        pred_probability, pred_r, pred_th = self.predict_prob(r, th)

        pred_r, pred_th = apply_thresholds(pred_probability, 
            [self.pred_thresh], pred_r, pred_th)

        return pred_r[0, 0], pred_th[0, 0]   



