#!/usr/bin/python

import sys, argparse
from enum import Enum
from simpleliped import SimpleLiPed

class LiPedType(Enum):
    simple = SimpleLiPed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('laserfile', help='people detection data file')
    parser.add_argument('personfile', help='laseer scan data file')
    parser.add_argument('-p', '--plot', type=int,
        help='plot data at given frame')
    parser.add_argument('-t', '--type', default='simple',
        help='LiPed type to train')
    args = parser.parse_args()

    lptype = LiPedType[args.type].value

    lp = lptype(args.laserfile, args.personfile)
    if args.plot:
        lp.plot(args.plot)
        sys.exit(0)
    # lp.movie(range(20))

    # train neural network
    lp.train()

