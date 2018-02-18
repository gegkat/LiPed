#!/usr/bin/python

import sys, argparse
from enum import Enum
from simpleliped import SimpleLiPed
import pdb

class LiPedType(Enum):
    simple = SimpleLiPed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', type=int,
        help='plot data at given frame')
    parser.add_argument('-t', '--type', default='simple',
        help='LiPed type to train')
    parser.add_argument('-i', '--init', nargs=2,
        help='initial load of raw data, else will load preprocessed data')
    parser.add_argument('-e', '--epochs', type=int, default=5,
        help='epochs to train')
    parser.add_argument('-d', '--data_dir', default='data',
        help='initial load of raw data, else will load preprocessed data') 
    parser.add_argument('-l', '--load_model', nargs=1, 
        help='load a trained model')
    args = parser.parse_args()

    lptype = LiPedType[args.type].value
    if args.init:
        lp = lptype(True, args.init[0], args.init[1], args.data_dir)
    else:
        lp = lptype(data_dir=args.data_dir)

    if args.plot:
        lp.plot(args.plot)
        sys.exit(0)
    # lp.movie(range(20))

    # train neural network
    if args.load_model:
        lp.load(args.load_model[0])
    else:
        lp.train(epochs=args.epochs)

    lp.predict()
