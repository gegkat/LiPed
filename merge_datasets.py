#!/usr/bin/python

import cPickle as pickle
import sys

def merge_datasets(lf0, pf0, lf1, pf1, combined_lf, combined_pf):
    with (open(combined_lf, 'ab')) as clf:
        with (open(lf0, 'rb')) as lf:
            while True:
                try:
                    o = pickle.load(lf)
                    pickle.dump(o, clf)
                except EOFError:
                    break
        with (open(lf1, 'rb')) as lf:
            while True:
                try:
                    o = pickle.load(lf)
                    pickle.dump(o, clf)
                except EOFError:
                    break

    with (open(combined_pf, 'ab')) as cpf:
        with (open(pf0, 'rb')) as pf:
            while True:
                try:
                    o = pickle.load(pf)
                    pickle.dump(o, cpf)
                except EOFError:
                    break
        with (open(pf1, 'rb')) as pf:
            while True:
                try:
                    o = pickle.load(pf)
                    pickle.dump(o, cpf)
                except EOFError:
                    break


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('merge_datasets needs 6 arguments')
    merge_datasets(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
