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
import keras.callbacks

# fixed axis limits for animation
XLIMS = (-10, 10) 
YLIMS = (-2, 12)
DIST_THRESH = 1 # Threshold for acceptable ped detection distance

class LiPed(object):
    def __init__(self, init=False, laser_file='', pedestrian_file='', data_dir='data', regression=False):

        self.pred_thresh = 0.5 # threshold for labeling a prediction as pedestrian

        self.lidar_angle = np.arange(-1.69296944141, 1.6929693222, 0.00872664619237)
        self.in_view = np.logical_and(self.lidar_angle > -0.5, self.lidar_angle < 0.5)
        if regression:
            self.in_view[:] = True

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

        self.train_test_split()

        # Generic neural network
        self.nn = self._build_nn()
        print(self.nn.summary())


    def train_test_split(self):

        # Restrict to data in_view
        X = self.lidar_range[:, self.in_view]
        Y = self.ped_onehot[:, self.in_view]

        # Train/test split
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
        self.ped_pos_train, self.ped_pos_test = train_test_split(
            X, Y, self.ped_pos, test_size=0.2, 
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

    def precision_recall(self):

        precisions = []
        recalls = []
        F1s = []
        # threshes = np.linspace(0, .98, 10)
        threshes = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, .97, .975, 0.98, .985, 0.99, .995, 0.999]
        # threshes = [0.5] #[0.98]
        precisions, recalls, F1s = self.evaluate(threshes)
        i_max = np.argmax(F1s)
        self.pred_thresh = threshes[i_max]

        plt.figure()
        plt.plot(threshes, precisions)
        plt.plot(threshes, recalls)
        plt.plot(threshes, F1s)
        plt.legend(['precision', 'recall', 'F1 Score'])
        plt.title('Max F1: {:.3f} at {}'.format(F1s[i_max], threshes[i_max]))
        plt.xlabel('threshold')
        self.savefig('F1_vs_thresh.png')

        plt.figure()
        plt.plot(precisions, recalls)
        plt.xlabel('precision')
        plt.ylabel('recall')
        self.savefig('precision_recall.png')

        data_dict = {"threshes": threshes, 
                     "precisions": precisions,
                     "recalls": recalls,
                     "F1s": F1s}
        self.savedict(data_dict, 'precision_recall.p')


    def savedict(self, data_dict, fname):
        with open(os.path.join(self.udir, fname), 'wb') as f:
            pickle.dump(data_dict, f)

    def savefig(self, fname):
        plt.savefig(os.path.join(self.udir, fname), dpi=400)

    # Abstract method, to be implemented in subclasses
    def evaluate(self, thresholds):
        do_plot = False
        N_frames = self.X_test.shape[0]
        N_threshes = len(thresholds)
        print("Evaluating {} frames".format(N_frames))
        false_pos = np.zeros((N_threshes))
        false_neg = np.zeros((N_threshes))
        true_pos =  np.zeros((N_threshes))

        # Run entire test set through network
        pred_probability, pred_r, pred_th = self.predict_prob(
            self.X_test, self.lidar_angle[self.in_view])

        # Apply multiple thresholds and get list of r/theta for detected
        # pedestrians at each frame and each threshold
        pred_r, pred_th = apply_thresholds(pred_probability, 
            thresholds, pred_r, pred_th)

        print('counting scores')
        t1 = time.time()
        for i in range(N_frames):
            utils.print_progress_bar(i, N_frames, 
            prefix = 'Progress:', suffix = 'Complete', bar_length = 50)

            truth_r, truth_th = pos2pol(self.ped_pos_test[i])

            for j in range(N_threshes):

                fp, fn, tp = get_score(
                    pred_r[i,j], pred_th[i,j], truth_r, truth_th)

                if do_plot: 
                    lx, ly = pol2cart(self.X_test[i,:], self.lidar_angle[self.in_view])
                    plt.plot(ly, lx, '.', linestyle='', marker='.', 
                    markeredgecolor='gray', markersize=2)
                    pred_x, pred_y = pol2cart(pred_r[i,j], pred_th[i,j])
                    plt.plot(pred_y, pred_x, linestyle='', marker='o', 
                    markeredgecolor='r', markersize=5, fillstyle='none',
                    markeredgewidth=0.5)
                    truth_x, truth_y = pol2cart(truth_r, truth_th)
                    plt.plot(truth_y, truth_x,  linestyle='', marker='s', 
                    markeredgecolor='g', markersize=5, fillstyle='none',
                    markeredgewidth=0.5)
                    plt.title("fp: {}, fn: {}, tp: {}".format(fp, fn, tp))
                    plt.gca().set_xlim(XLIMS)
                    plt.gca().set_ylim(YLIMS)
                    plt.gca().invert_xaxis()
                    plt.show()

                false_pos[j] += fp
                false_neg[j] += fn
                true_pos[j] += tp

        t2 = time.time()
        print("\nComplete in {} seconds, {} samples per sec".format(
            t2 - t1, N_frames/float(t2-t1)))

        eps = np.finfo(float).eps
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        F1_score = 2 * (precision * recall) / (precision + recall + eps)

        # print("False pos: {}".format(false_pos))
        # print("False neg: {}".format(false_neg))
        # print("True pos: {}".format(true_pos))
        # print("Precision: {}".format(precision))
        # print("Recall: {}".format(recall))
        print("F1 Score: {}".format(F1_score))

        # with open(os.path.join(self.udir, 'evaluate.csv'), 'w') as f:
        #     f.write("N frames, {}\n".format(N_frames))
        #     f.write("False pos, {}\n".format(false_pos))
        #     f.write("False neg, {}\n".format(false_neg))
        #     f.write("True pos, {}\n".format(true_pos))
        #     f.write("Precision, {}\n".format(precision))
        #     f.write("Recall, {}\n".format(recall))
        #     f.write("F1 Score, {}\n".format(F1_score))

        return precision, recall, F1_score


    def load_model(self, model_file):
        self.nn = load_model(model_file)
        self.udir = os.path.dirname(model_file)

    # Abstract method, to be implemented in subclasses
    def train(self, regression=False, epochs=5):
        start_time = time.time()

        # Get recall/precision/f1 metrics to set as callback function
        if regression:
            callbacks = []
        else:
            callbacks = [Metrics()]

        self.X_train, self.X_val, self.Y_train, self.Y_val, = train_test_split(
            self.X_train, self.Y_train, test_size=0.2, 
            shuffle=True, random_state=42)
        history = self.nn.fit(self.X_train, self.Y_train, 
                    batch_size=128, epochs=epochs, verbose=1, 
                    shuffle=True, validation_data=(self.X_val, self.Y_val), 
                    callbacks=callbacks)

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

        if regression:
            plt.figure()
            plt.plot(history.history['mean_absolute_error'], label='training')
            plt.plot(history.history['val_mean_absolute_error'], label='validation')
            plt.legend()
            plt.xlabel('# Epochs')
            plt.ylabel('Mean absolute error')
            plt.savefig(os.path.join(udir, 'error_vs_epoch.png'), dpi=400)
        else:
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

        ped_x, ped_y = pos2cart(self.ped_pos[frame])

        self.ped_truth_plot.set_data(ped_y, ped_x)

        r, th = self.predict(self.lidar_range[frame,:], self.lidar_angle)
        x, y = pol2cart(r, th)

        # Use this for perfect predictions or test ped_onehot
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


def get_score(pr, pth, tr, tth):

    px, py = pol2cart(pr, pth)
    tx, ty = pol2cart(tr, tth)

    false_pos = 0
    false_neg = 0
    true_pos = len(px)

    L1 = len(px)
    L2 = len(tx)
    d = np.zeros((L1,L2))
    for i in range(L1):
        d[i,:] = np.sqrt((px[i] - tx)**2 + (py[i] - ty)**2)

    while d.shape[0] and d.shape[1]: 
        i,j = np.unravel_index(d.argmin(), d.shape)
        if d[i,j] > DIST_THRESH:
            break

        d = np.delete(d, (i), axis=0)
        d = np.delete(d, (j), axis=1)

    false_pos = d.shape[0]
    false_neg = d.shape[1]
    true_pos = len(px) - false_pos

    # print(len(px), len(tx), false_pos, false_neg, true_pos)

    # if true_pos > 0:
        # pdb.set_trace()

    return false_pos, false_neg, true_pos


def apply_thresholds(y, thresholds, r, th):
    '''
    Takes probability data y and applies thresholds to determine
    pedestrian or not. Returns r and theta as 2d arrays of lists.

    Inputs: 
      Y: probability of a pedestrian for each sliding window. Dimensiosn are
         [N, M] whrere n is the number of frames and m is the number
         of sliding windows 

    thresholds: a list of L thresholds to apply to y for pedestrian detection

      r:  The range at the center of each sliding window, dimensions [N, M] 

      th: The theta at the center of each sliding window, dimensions [N, M]

    Outputs:

    r_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing range values of pedestrian detections. 

    th_list: 2d array of lists with dimension [N, L]. Each element is a list
    containing theta values of pedestrian detections. 

    '''

    N_frames = y.shape[0]
    N_threshes = len(thresholds)

    y = np.tile(y, (N_threshes, 1, 1))
    y = np.moveaxis(y, 0, 2)
    idx = y > thresholds

    r_list = np.empty((N_frames, N_threshes), dtype=object)
    th_list = np.empty((N_frames, N_threshes), dtype=object)
    for i in range(N_frames):
        for j in range(N_threshes):
            k = idx[i,:, j]
            r_list[i, j], th_list[i, j] = max_suppression(y[i, k, j], r[i,k], th[i, k])
            # r_list[i, j] = r[i,k]
            # th_list[i, j] = th[i,k]

    return r_list, th_list

def max_suppression(y, r, th):
    isort = np.argsort(y)
    r = r[isort]
    th = th[isort]
    x, y = pol2cart(r, th)

    xout = []
    yout = []
    while len(x) > 0: 
        xout.append(x[0])
        yout.append(y[0])
        x = np.delete(x, 0)
        y = np.delete(y, 0)
        if len(x) == 0:
            break
        d = ((xout[-1] - x)**2 + (yout[-1] - y)**2)**.5
        rm_ind = np.where(d < DIST_THRESH)
        # pdb.set_trace()
        x = np.delete(x, rm_ind)
        y = np.delete(y, rm_ind)

    xout = np.array(xout)
    yout = np.array(yout)
    if False:
        plt.figure()
        plt.plot(xout, yout, linestyle='', marker='o', 
                        markeredgecolor='r', markersize=5, fillstyle='none',
                        markeredgewidth=0.5)
        x, y = pol2cart(r, th)
        plt.plot(x, y, '.', linestyle='', marker='.', 
                        markeredgecolor='black', markersize=2)
        plt.show()
    rout, thout = cart2pol(xout, yout)

    return rout, thout

def pos2cart(pos):
        x = []
        y = []
        for i in range(len(pos)):
            x.append(pos[i][0])
            y.append(pos[i][1])

        return np.array(x), np.array(y)

def pos2pol(pos):
        x, y = pos2cart(pos)
        r, theta = cart2pol(x, y)

        return r, theta


def ped_to_onehot(ped_pos, lidar_angle):
    N_frames = len(ped_pos)

    # Use nearest neighbor interpolation to find angle for pedestrians
    interp_func = interp1d(lidar_angle, range(len(lidar_angle)), kind='nearest')
    ped_onehot = np.zeros((N_frames, len(lidar_angle)), dtype=bool)
    for i in range(N_frames):
        angles = np.array([np.arctan2(y / x) for x,y in ped_pos[i]])
        idx = interp_func(angles).astype(int)
        ped_onehot[i, idx] = 1

    return ped_onehot

def pol2cart(r, theta):
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return x, y

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

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

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0])) > 0.5
        y_true = self.validation_data[1]

        eps = np.finfo(float).eps
        true_positives = np.sum(y_true * y_pred)
        true_negatives = np.sum((1-y_true) * (1-y_pred))
        predicted_positives = np.sum(y_pred)
        predicted_negatives = np.sum(1-y_pred)
        possible_positives = np.sum(y_true)
        possible_negatives = np.sum(1-y_true)
        precision = true_positives / (predicted_positives + eps)
        recall = true_positives / (possible_positives + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (true_positives + true_negatives) / float(possible_positives + possible_negatives)
        print("Validation: precision {:.4}, recall {:.4}, F1 {:.4}, acc: {:.4}".format(
            precision, recall, f1_score, accuracy))
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy

        return

