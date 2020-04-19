# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import pdb

from antispoofing.sfsnet.utils import load_images


class HybridImage(object):

    def __init__(self, data, input_fname, output_fname,
                 file_type="png",
                 n_channel=1,
                 ):

        self.data = data
        self.input_fname = input_fname
        self.output_fname = output_fname

        self.n_channel = n_channel
        self.file_type = file_type

    def save_new_img(self, new_img):

        print("-- saving image in {0}".format(self.output_fname))
        sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(self.output_fname))
        except OSError:
            pass

        cv2.imwrite(self.output_fname, new_img)

    def save_ndarray(self, new_arr):

        print("-- saving image in {0}".format(self.output_fname))
        sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(self.output_fname))
        except OSError:
            pass

        np.save(self.output_fname, new_arr)

    def run(self):

        if os.path.exists(self.output_fname):
            return True

        img = load_images(self.input_fname, n_channel=self.n_channel)
        img = np.rollaxis(img, 0, 4)
        n_rows, n_cols, n_channel, n_imgs = img.shape
        img = np.reshape(img, (n_rows, n_cols, -1))

        if self.n_channel == 1:
            self.save_new_img(img)
        else:
            self.save_ndarray(img)

        return True
