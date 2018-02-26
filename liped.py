#!/usr/bin/python

import cPickle as pickle
import pdb
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import matplotlib.animation as animation
import time
import utils
import os
import pickle
import json
import shutil

# fixed axis limits for animation
XLIMS = (-10, 10) 
YLIMS = (-2, 12)

class LiPed(object):
    def __init__(self, init=False, laser_file='', pedestrian_file='', data_dir='data'):

        self.lidar_angle = np.arange(-1.69296944141, 1.6929693222, 0.00872664619237)
        self.in_view = np.logical_and(self.lidar_angle > -0.5, self.lidar_angle < 0.5)

        if init:
            lidar_time, lidar_range = load_laser_data(laser_file)
            ped_time, ped_pos = load_pedestrian_data(pedestrian_file)
            # print(lidar_time.shape, lidar_range.shape, ped_time.shape, len(ped_x), len(ped_y))

            interp_func = interp1d(lidar_time, range(len(lidar_time)), 
                kind='nearest', fill_value='extrapolate')
            idx = interp_func(ped_time).astype(int)

            ped_onehot = ped_to_onehot(ped_pos, self.lidar_angle)

            # Keep only lidar scans that match pedestrian detections
            lidar_time = lidar_time[idx]
            lidar_range = lidar_range[idx,:]

            # Clear data dir
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                print("Deleting dir: {}".format(data_dir))
                shutil.rmtree(data_dir)

            print("Creating dir: {}".format(data_dir))
            os.mkdir(data_dir)

            np.save(data_dir + '/lidar_time', lidar_time)
            np.save(data_dir + '/lidar_range', lidar_range)
            np.save(data_dir + '/ped_time', ped_time)
            np.save(data_dir + '/ped_pos', ped_pos)
            np.save(data_dir + '/ped_onehot', ped_onehot)
        else:
            lidar_time = np.load(data_dir + '/lidar_time.npy')
            lidar_range = np.load(data_dir + '/lidar_range.npy')
            ped_time = np.load(data_dir + '/ped_time.npy')
            ped_pos = np.load(data_dir + '/ped_pos.npy')
            ped_onehot = np.load(data_dir + '/ped_onehot.npy')

        self.lidar_time = lidar_time
        self.lidar_range = lidar_range
        self.ped_time = ped_time
        self.ped_pos = ped_pos
        self.ped_onehot = ped_onehot

        self.N_frames = self.lidar_range.shape[0]
        print("Data contains {} frames".format(self.N_frames))

        self.train_test_val_split()

        # Generic neural network
        self.nn = self._build_nn()

    def train_test_val_split(self):

        # Restrict to data in_view
        X = self.lidar_range[:, self.in_view]
        Y = self.ped_onehot[:, self.in_view]

        # Train/test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, 
            shuffle=True, random_state=42)


    # Abstract method, to be implemented in subclasses
    def _build_nn(self):
        return None

    # Abstract method, to be implemented in subclasses
    def segment_data(self, data, ped_pos):
        pass

    # Abstract method, to be implemented in subclasses
    def predict(self):
        pass

    def load_model(self, model_file):
        self.nn = load_model(model_file)
        self.udir = os.path.dirname(model_file)

    # Abstract method, to be implemented in subclasses
    def train(self, epochs=5):
        start_time = time.time()

        history = self.nn.fit(self.X_train, self.Y_train, 
                    batch_size=128, epochs=epochs, verbose=1, 
                    shuffle=True, validation_split=0.2)

        end_time = time.time()
        print('Trained model in {:.2f} seconds'.format(end_time-start_time))

        timestamp = utils.get_timestamp()
        udir = utils.mkdir_unique(timestamp)
        self.udir = udir

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


    def plot_init(self):
        self.fig = plt.figure()
        lx = self.lidar_range * np.cos(self.lidar_angle)
        ly = self.lidar_range * np.sin(self.lidar_angle)
        self.ax = plt.axes(xlim=(ly.min(), ly.max()), ylim=(lx.min(), lx.max()))

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(XLIMS)
        self.ax.set_ylim(YLIMS)

        plt.plot([0, lx.max()*np.tan(0.5)], [0, lx.max()], 'k-', linewidth=1)
        plt.plot([0, -lx.max()*np.tan(0.5)], [0, lx.max()], 'k-', linewidth=1)

        self.lidar_in_view_plot = Line2D([], [], linestyle='', marker='.', 
            markeredgecolor='gray', markersize=2)
        self.ax.add_line(self.lidar_in_view_plot)

        self.lidar_not_in_view_plot = Line2D([], [], linestyle='', marker='.', 
            markeredgecolor='gray', markersize=2)
        self.ax.add_line(self.lidar_not_in_view_plot)

        self.ped_truth_plot = Line2D([], [], linestyle='', marker='s', 
            markeredgecolor='g', markersize=5, fillstyle='none', 
            markeredgewidth=0.5)
        self.ax.add_line(self.ped_truth_plot)

        self.ped_pred_plot = Line2D([], [], linestyle='', marker='o', 
            markeredgecolor='r', markersize=5, fillstyle='none',
            markeredgewidth=0.5)
        self.ax.add_line(self.ped_pred_plot)

        self.ax.invert_xaxis()

    def plot(self, frame):

        lx, ly = pol2cart(self.lidar_range[frame, :], self.lidar_angle)

        self.lidar_in_view_plot.set_data(ly[self.in_view], lx[self.in_view])
        self.lidar_not_in_view_plot.set_data(ly[~self.in_view], lx[~self.in_view])

        ped_x = []
        ped_y = []
        for i in range(len(self.ped_pos[frame])):
            ped_x.append(self.ped_pos[frame][i][0])
            ped_y.append(self.ped_pos[frame][i][1])

        self.ped_truth_plot.set_data(ped_y, ped_x)

        x, y = self.predict(frame)
        # x = lx[self.ped_onehot[frame,:]]
        # y = ly[self.ped_onehot[frame,:]]

        self.ped_pred_plot.set_data(y, x)

        plt.title("Frame: {}, ped detected: {}, ped truth: {}".format(
            frame, len(x), len(ped_x)))

        # @TODO: Add polar plot for y_pred

        # Return list of artists
        return [self.lidar_in_view_plot, 
                self.lidar_not_in_view_plot,
                self.ped_truth_plot, 
                self.ped_pred_plot]

    def plot_frame_idx(self, idx):
        frame = self.frames_to_animate[idx]
        artists = self.plot(frame)
        utils.print_progress_bar(idx + 1, len(self.frames_to_animate), 
            prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
        return artists

    def sample_frames(self, sections=10, width=40):

        frames = []
        jump = self.lidar_range.shape[0] // sections
        array = np.arange(width)
        for i in range(sections):
            frames += np.ndarray.tolist(array + i*jump)
        return frames

    def animate(self, frames=None, show_plot=False, dpi=500):

        # all frames
        if frames is None:
            frames = np.arange(0, self.lidar_range.shape[0], 1)

        self.frames_to_animate = frames
        print('Animating {} frames'.format(len(frames)))

        self.plot_init()

        # Call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, func=self.plot_frame_idx, 
                                       frames=range(len(self.frames_to_animate)),
                                       interval=1, # animate as fast as possible
                                       blit=True)

        if show_plot:
            # @TODO: This doesn't seem to work but I don't know why
            plt.show()

        anim.save(os.path.join(self.udir,'animation.mp4'), 
                  fps=12, bitrate=-1, dpi=dpi) 


def ped_to_onehot(ped_pos, lidar_angle):
    N_frames = len(ped_pos)

    # Use nearest neighbor interpolation to find angle for pedestrians
    interp_func = interp1d(lidar_angle, range(len(lidar_angle)), kind='nearest')
    ped_onehot = np.zeros((N_frames, len(lidar_angle)), dtype=bool)
    for i in range(N_frames):
        angles = np.array([np.arctan(y / x) for x,y in ped_pos[i]])
        idx = interp_func(angles).astype(int)
        ped_onehot[i, idx] = 1

    return ped_onehot

def pol2cart(r, theta):
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return x, y

def load_laser_data(pickle_file):
    print('parsing laser data: {}'.format(pickle_file))

    lidar_time = []
    lidar_range = []
    
    count = 0
    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
                count += 1
                lidar_time.append(object[0]) 
                lidar_range.append( list(object[1]))

                # if count > 1000:
                    # break
            except EOFError:
                break

    lidar_time = np.array(lidar_time)
    lidar_range = np.array(lidar_range)

    return lidar_time, lidar_range


def load_pedestrian_data(pickle_file):
    print('parsing pedestrian data: {}'.format(pickle_file))

    ped_time = []
    ped_pos = []

    prev_time = None
    
    count = 0
    with (open(pickle_file, 'rb')) as openfile:
        while True:
            try:
                object = pickle.load(openfile)
                count += 1

                cur_time = object[0]
                if cur_time == prev_time: 
                    ped_pos[-1].append([object[1], object[2]])
                else:
                    ped_time.append(object[0])
                    if object[1] is not None:
                        ped_pos.append([[object[1], object[2]]])
                    else:
                        ped_pos.append([])

                prev_time = cur_time

                # if count > 1000:
                    # break
            except EOFError:
                break

    ped_time = np.array(ped_time)

    return ped_time, ped_pos
