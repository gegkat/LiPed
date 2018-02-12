#!/usr/bin/python

from liped import LiPed

class SimpleLiPed(LiPed):
    def __init__(self, *args):
        super(SimpleLiPed, self).__init__(*args)

    def _build_nn(self):
        return None

    def segment_data(self):
        pass

    def train(self):
        pass
