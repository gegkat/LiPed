#!/usr/bin/python

import sys, argparse
from enum import Enum
from simpleliped import SimpleLiPed

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
    args = parser.parse_args()

    lptype = LiPedType[args.type].value
    if args.init:
        lp = lptype(True, args.init[0], args.init[1])
    else:
        lp = lptype()

    if args.plot:
        lp.plot(args.plot)
        sys.exit(0)
    # lp.movie(range(20))

    # train neural network
    lp.train()

