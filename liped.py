#!/usr/bin/python

import pdb
import os, time 
import utils
import pickle, json
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import keras.callbacks

from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from lputils import *
from settings import *


class LiPed(object):
    def __init__(self, init=False, laser_file='', pedestrian_file='', data_dir='data',
                 regression=False):
    #################################################################
    ### INITIALIZATION AND UTILITIES
    #################################################################

        # Threshold for labeling a prediction as pedestrian
        # Gets overwritten by precision_reacall function to threshold
        # that gives max F1 score
        self.pred_thresh = 0.5 


        self.lidar_angle = np.arange(LIDAR_MIN, LIDAR_MAX, LIDAR_STEP)

        #### @TODO: TEMPORARY HACK to change dimension from 389 to 387 ####
        self.lidar_angle = self.lidar_angle[1:-1]
        self.in_view = np.logical_and(self.lidar_angle > YOLO_FOV_MIN, self.lidar_angle < YOLO_FOV_MAX)

        if regression:
            self.in_view[:] = True

        if init:
            lidar_time, lidar_range = load_laser_data(laser_file)
            ped_time, ped_pos = load_pedestrian_data(pedestrian_file)

            interp_func = interp1d(lidar_time, range(len(lidar_time)), 
                kind='nearest', fill_value='extrapolate')
            idx = interp_func(ped_time).astype(int)


            # Keep only lidar scans that match pedestrian detections
            lidar_time = lidar_time[idx]
            lidar_range = lidar_range[idx, :]

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
        else:
            lidar_time = np.load(data_dir + '/lidar_time.npy')
            lidar_range = np.load(data_dir + '/lidar_range.npy')
            ped_time = np.load(data_dir + '/ped_time.npy')
            ped_pos = np.load(data_dir + '/ped_pos.npy')


        if PLOT_REFINEMENT: 
            before = np.copy(lidar_range)

        # Clamping data
        np.clip(lidar_range, lidar_range.min(), MAX_R, out=lidar_range)
        # Filter out partial points
        filter_partial_points(lidar_range, self.lidar_angle)

        if PLOT_REFINEMENT:
            x, y = pol2cart(before, self.lidar_angle)
            x2, y2 = pol2cart(lidar_range, self.lidar_angle)
            for i in range(0, x.shape[0], 20):
                plt.plot(y[i,:], x[i,:], '.')
                idx = y[i,:] != y2[i,:]
                plt.plot(y[i,idx], x[i,idx], 'o', fillstyle='none')
                plt.plot(y2[i,:], x2[i,:], 'x')
                plt.gca().set_aspect('equal', adjustable='box-forced')
                plt.show()

        # Remove YOLO pedestrian detections beyond max R
        for i in range(ped_pos.shape[0]):
            ped_pos[i] = [[x,y] for x, y in ped_pos[i] if not (x**2 + y**2)**0.5 > MAX_R]

        self.lidar_time = lidar_time
        self.lidar_range = lidar_range
        self.ped_time = ped_time
        self.ped_pos = ped_pos

        print('ped to one hot')
        self.ped_onehot = ped_to_onehot(self.ped_pos, self.lidar_angle)

        self.N_frames = self.lidar_range.shape[0]
        print("Data contains {} frames".format(self.N_frames))

        # Restrict to data in_view
        X = self.lidar_range[:, self.in_view]
        Y = self.ped_onehot[:, self.in_view]

        # Train/test split
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
        self.ped_pos_train, self.ped_pos_test = train_test_split(
            X, Y, self.ped_pos, test_size=TEST_SIZE, 
            shuffle=TRAIN_TEST_SHUFFLE, random_state=42)

        # Generic neural network
        self.nn = self._build_nn()

    def load_model(self, model_file):
        self.nn = keras.models.load_model(model_file)
        self.udir = os.path.dirname(model_file)

    def savedict(self, data_dict, fname):
        with open(os.path.join(self.udir, fname), 'wb') as f:
            pickle.dump(data_dict, f)

    def savefig(self, fname):
        plt.savefig(os.path.join(self.udir, fname), dpi=400)

    #################################################################
    ### ABSTRACT METHODS
    #################################################################

    def _build_nn(self):
        return None

    def segment_data(self, data, ped_pos):
        pass

    def predict(self):
        pass

    #################################################################
    ### NEURAL NETWORK METHODS
    #################################################################

    def average_precision(self, precisions):
        # using Eq. 1 and 2 (Sec. 4.2) in
        # http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
        p = 0
        for i in range(len(precisions)):
            p += max(precisions[i:])
        return p / len(precisions)

    def precision_recall(self):
        precisions = []
        recalls = []
        F1s = []

        precisions, recalls, F1s = self.evaluate(THRESHOLDS)
        i_max = np.argmax(F1s)
        self.pred_thresh = THRESHOLDS[i_max]

        # Only single class, so AP and not mAP
        # AP dependent on thresholds chosen
        # Standard COCO IoU thresholds for mAP[.5:.95]
        # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        ap = self.average_precision(precisions)
        print('Average precision: {:.3f}'.format(ap))

        plt.figure()
        plt.plot(THRESHOLDS, precisions)
        plt.plot(THRESHOLDS, recalls)
        plt.plot(THRESHOLDS, F1s)
        plt.gca().set_ylim((0,1))
        plt.legend(['precision', 'recall', 'F1 Score'])
        plt.title('Max F1: {:.3f} at {}'.format(F1s[i_max], THRESHOLDS[i_max]))
        plt.xlabel('threshold')
        self.savefig('F1_vs_thresh.png')

        plt.figure()
        plt.plot(precisions, recalls)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.gca().set_xlim((0,1))
        plt.gca().set_ylim((0,1))
        self.savefig('precision_recall.png')

        data_dict = {"threshes": THRESHOLDS, 
                     "precisions": precisions,
                     "recalls": recalls,
                     "F1s": F1s}
        self.savedict(data_dict, 'precision_recall.p')

    def evaluate(self, thresholds):

        N_frames = self.X_test.shape[0]
        N_threshes = len(thresholds)

        print("Evaluating {} frames".format(N_frames))
        t1 = time.time()
        false_pos = np.zeros((N_threshes))
        false_neg = np.zeros((N_threshes))
        true_pos =  np.zeros((N_threshes))

        # Run entire test set through network
        pred_probability, pred_r, pred_th = self.predict_prob(
            self.X_test, self.lidar_angle[self.in_view])

        t2 = time.time()
        print("\nComplete in {:.1f} seconds, {:.1f} samples per sec".format(
            t2 - t1, N_frames/float(t2-t1)))

        # Apply multiple thresholds and get list of r/theta for detected
        # pedestrians at each frame and each threshold
        print("Applying thresholds")
        t1 = time.time()
        pred_r, pred_th = apply_thresholds(pred_probability, 
            thresholds, pred_r, pred_th, 
            self.X_test, self.lidar_angle[self.in_view], 
            self.ped_pos_test)

        t2 = time.time()
        print("\nComplete in {:.1f} seconds, {:.1f} samples per sec".format(
            t2 - t1, N_frames/float(t2-t1)))

        print('Counting scores')
        t1 = time.time()
        for i in range(N_frames):
            utils.print_progress_bar(i, N_frames, 
            prefix = 'Progress:', suffix = 'Complete', bar_length = 50)

            truth_r, truth_th = pos2pol(self.ped_pos_test[i])

            for j in range(N_threshes):

                fp, fn, tp = get_score(
                    pred_r[i,j], pred_th[i,j], truth_r, truth_th)

                if DO_EVALUATION_PLOT: 
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
        print("\nComplete in {:.1f} seconds, {:.1f} samples per sec".format(
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

    # Abstract method, to be implemented in subclasses
    def train(self, regression=False, epochs=5):
        start_time = time.time()

        # Get recall/precision/f1 metrics to set as callback function
        if regression:
            callbacks = []
        else:
            callbacks = [Metrics()]

        # Train/validation split
        print(self.nn.summary())
        self.X_train, self.X_val, self.Y_train, self.Y_val, = train_test_split(
            self.X_train, self.Y_train, test_size=CROSS_VAL_SIZE, 
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

        shelve_settings(udir)

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

    #################################################################
    ### PLOTTING/ANIMATION METHODS
    #################################################################

    def plot_init(self):
        self.fig = plt.figure()
        lx = self.lidar_range * np.cos(self.lidar_angle)
        ly = self.lidar_range * np.sin(self.lidar_angle)
        self.ax = plt.axes(xlim=(ly.min(), ly.max()), ylim=(lx.min(), lx.max()))
        self.ax.set_aspect('equal', adjustable='box-forced')

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

    def sample_frames(self):

        sections = N_ANIMATION_SECTIONS
        width = ANIMATION_SECTION_WIDTH

        if USE_SUBSET_FOR_ANIMATION:
            clips = [ [0,     623], 
                      [2553,  2898],
                      [3123,  3575],
                      [4619,  5088],
                      [10097, 11005],
                      [11773, 12781],
                      [13637, 14421],
                      [18145, 18881],
                      [19464, 19760],
                      [19901 ,20423]]

            allframes = []
            for clip in clips:
                allframes += range(clip[0], clip[1])



        else:
            # Use all frames
            allframes = range(self.lidar_range.shape[0])

        # If section is None, return allframes
        if sections is None:
            return allframes

        frames = []
        jump = len(allframes) // sections
        array = np.arange(width)
        allframes = np.array(allframes)
        for i in range(sections):
            start_frame = np.random.randint(i*jump, (i+1)*jump - width)
            frames += np.ndarray.tolist(allframes[start_frame + array])

        return frames

    def animate(self, show_plot=False):
        # all frames


        self.frames_to_animate = self.sample_frames() 
        print('Animating {} frames'.format(len(self.frames_to_animate)))

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
                  fps=FPS, bitrate=-1, dpi=DPI) 


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
