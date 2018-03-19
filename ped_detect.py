#!/usr/bin/python

import pdb
import sys, argparse

from enum import Enum
from simpleliped import SimpleLiPed
from cnnliped import CNNLiPed
from localizationnet import LocNet
import os


class LiPedType(Enum):
    simple = SimpleLiPed
    cnn = CNNLiPed
    locnet = LocNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--do_animation', type=bool, default=True,
        help='Do prediction')
    parser.add_argument('--show_plot', type=bool, default=False,
        help='Show plot, default is to just save movie file without showing plot')
    parser.add_argument('-t', '--type', default='cnn',
        help='LiPed type to train')
    parser.add_argument('-i', '--init', nargs=2,
        help='initial load of raw data, else will load preprocessed data')
    parser.add_argument('-e', '--epochs', type=int, default=3,
        help='epochs to train')
    parser.add_argument('-d', '--data_dir', default='data',
        help='initial load of raw data, else will load preprocessed data') 
    parser.add_argument('-l', '--load_model', nargs=1, 
        help='load a trained model')
    parser.add_argument('--loc_model', nargs=1, 
        help='load a trained localizaiton model')
    args = parser.parse_args()

    lptype = LiPedType[args.type].value
    regression = False
    if args.type == 'locnet':
        regression = True

    if args.init:
        print("Processing pickle files...")
        lp = lptype(True, args.init[0], args.init[1], args.data_dir, regression=regression)
    else:
        print("Loading npy files from {}".format(args.data_dir))
        lp = lptype(data_dir=args.data_dir, regression=regression)

    # Load trianed model
    if args.load_model:
        print("Loading model from {}".format(args.load_model[0]))
        lp.load_model(args.load_model[0])

    # Train network
    else:
        print("Traning model for {} epochs".format(args.epochs))
        lp.train(epochs=args.epochs)

    if not regression:
        if args.loc_model:
            lp.load_localization_model(args.loc_model[0])
        else:
            lptype2 = LiPedType['locnet'].value
            lp2 = lptype2(data_dir=args.data_dir, regression=True)
            lp2.train(epochs=10)
            lp.load_localization_model(os.path.join(lp2.udir, 'model.h5'))

        lp.precision_recall()

        if args.do_animation:
            lp.animate(show_plot=args.show_plot)
