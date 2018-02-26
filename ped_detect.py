#!/usr/bin/python

import sys, argparse
from enum import Enum
from simpleliped import SimpleLiPed
import pdb

class LiPedType(Enum):
    simple = SimpleLiPed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--do_animation', type=bool, default=False,
        help='Do prediction')
    parser.add_argument('--show_plot', type=bool, default=False,
        help='Show plot, default is to just save movie file without showing plot')
    parser.add_argument('-t', '--type', default='simple',
        help='LiPed type to train')
    parser.add_argument('-i', '--init', nargs=2,
        help='initial load of raw data, else will load preprocessed data')
    parser.add_argument('-e', '--epochs', type=int, default=20,
        help='epochs to train')
    parser.add_argument('-d', '--data_dir', default='data',
        help='initial load of raw data, else will load preprocessed data') 
    parser.add_argument('-l', '--load_model', nargs=1, 
        help='load a trained model')
    parser.add_argument('--dpi', type=int, default=100, 
        help='dpi for animation. Use 100 for speed, 300 for quality')
    args = parser.parse_args()

    lptype = LiPedType[args.type].value
    if args.init:
        print("Processing pickle files...")
        lp = lptype(True, args.init[0], args.init[1], args.data_dir)
    else:
        print("Loading npy files from {}".format(args.data_dir))
        lp = lptype(data_dir=args.data_dir)

    # train neural network
    if args.load_model:
        print("Loading model from {}".format(args.load_model[0]))
        lp.load_model(args.load_model[0])
    else:
        print("Traning model for {} epochs".format(args.epochs))
        lp.train(epochs=args.epochs)

    # lp.evaluate()
    lp.precision_recall()

    if args.do_animation:
        # frames = range(0, lp.N_frames, 1) # use all frames
        # frames = range(0, 100, 5) # specify a specific range of frames
        frames = lp.sample_frames(sections=4, width=30) # process evenly spaced sections of fixed width
        print("Running prediction animation for {} frames at {} dpi".format(len(frames), args.dpi))
        lp.animate(frames=frames, show_plot=args.show_plot, dpi=args.dpi)
