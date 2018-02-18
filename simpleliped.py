#!/usr/bin/python

from liped import LiPed
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import utils
import os
import pickle
import json
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

    def load(self, model_file):
        self.nn = load_model(model_file)

    def train(self, epochs=5):
        start_time = time.time()

        history = self.nn.fit(self.X_train, self.Y_train, 
                    batch_size=128, epochs=epochs, verbose=1, 
                    shuffle=True, validation_split=0.2)

        end_time = time.time()
        print('Trained model in {:.2f} seconds'.format(end_time-start_time))

        timestamp = utils.get_timestamp()
        udir = utils.mkdir_unique(timestamp)

        # Save the model
        print("Saving model weights and configuration file.")
        self.nn.save(os.path.join(udir,'model.h5'))

        with open(os.path.join(udir, 'history.csv'), 'w') as f:
            for i in range(0, len(history.history['loss'])):
                f.write("{}, {}\n".format(history.history['loss'][i], history.history['val_loss'][i]))

        with open(os.path.join(udir, 'train_history_dict.pickle'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # with open(os.path.join(udir, 'config.log'), 'w') as f:
        #     f.write(json.dumps(vars(args)))
        #     self.nn.summary(print_fn=lambda x: f.write(x + '\n'))

        with open(os.path.join(udir, 'model.json'), 'w') as f:
            f.write(json.dumps(self.nn.to_json()))

        plt.figure()
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.xlabel('# Epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(udir, 'loss_vs_epoch.png'), dpi=400)

        plt.figure()
        plt.plot(history.history['acc'], label='training')
        plt.plot(history.history['val_acc'], label='validation')
        plt.legend()
        plt.xlabel('# Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(udir, 'accuracy_vs_epoch.png'), dpi=400)

        # TODO: write evaluation function

    def predict(self):
        for i in range(0, self.lidar_range.shape[0], 20):
            plt.axis([0, 20, -10, 10])
            self.predict_frame(i)
            plt.title(i)
            print(i)
            plt.pause(0.01)
            plt.cla()


    def predict_frame(self, frame):
        data = self.lidar_range[frame,:]
        X = []
        angles = []
        ranges = []
        for j in range(len(data) // 15):
            idx = j * 15
            for k in range(len(data[idx : idx + 15]) // 5):
                curr_x = data[idx + k*5 : idx + k*5 + 15]
                X.append(curr_x)
                # angle1 = -1.69296944141 + (idx + k * 5 + 15/2) * 0.00872664619237
                # angle2 = self.lidar_angle[idx + k*5 : idx + k*5 + 15].mean()
                ranges.append(curr_x.min())
                angles.append(self.lidar_angle[idx + k*5 : idx + k*5 + 15].mean())
                

        angles = np.array(angles)
        ranges = np.array(ranges)

        X = np.array(X)
        y_pred = self.nn.predict(X)

        # plt.subplot(2,1,1)
        # plt.plot(angles, y_pred)

        # plt.subplot(2,1,2)
        self.plot(frame)
        idx = y_pred[:,0] > 0.9
        angles = angles[idx]
        ranges = ranges[idx]
        x = np.cos(angles) * ranges
        y = np.sin(angles) * ranges
        plt.plot(x, y, 'g.')

