# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import pdb

from antispoofing.sfsnet.features.descriptors import RawImage
from antispoofing.sfsnet.utils import load_images


class Extraction(object):

    def __init__(self, data, input_fname, output_path,
                 descriptor='',
                 file_type="png",
                 n_channel=1,
                 hybrid_image=False,
                 ):

        self.data = data
        self.input_fname = input_fname
        self.output_path = output_path
        self.descriptor = descriptor
        self.file_type = file_type
        self.n_channel = n_channel
        self.hybrid_image = hybrid_image

        self.debug = False

        self.flatten = True

    def extract_features(self):

        if 'RawImage' in self.descriptor:

            if self.n_channel > 3:
                feats = np.load(self.input_fname[0])
                self.save_features(feats)

            else:
                imgs = load_images(self.input_fname, n_channel=self.n_channel)

                feature_descriptor = RawImage()
                feats = feature_descriptor.extraction(imgs)
                self.save_imgs(feats[0])

        elif 'RawVideo' in self.descriptor:
            imgs = self.data.get_imgs([self.input_fname])[0]
            feature_descriptor = RawImage()

            feature_vector = []
            for idx in range(len(imgs)):
                feats = feature_descriptor.extraction(imgs[idx, :, :, :])
                feature_vector.append(feats)
            feature_vector = np.array(feature_vector)
            self.save_features_1(feature_vector)

        else:
            pass

    def save_features_1(self, feats):

        print("-- saving {0} features extracted from {1}".format(self.descriptor, self.output_path))
        sys.stdout.flush()

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        for i, feat in enumerate(feats):
            output_fname = os.path.join(self.output_path, '{:03}.png'.format(i))
            cv2.imwrite(output_fname, feat)

    def save_features(self, feats):

        print("-- saving {0} features extracted from {1}".format(self.descriptor, self.output_path))
        sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(self.output_path))
        except OSError:
            pass

        np.save(self.output_path, feats)

    def save_imgs(self, feats):

        print("-- saving {0} features extracted from {1}".format(self.descriptor, self.output_path))
        sys.stdout.flush()

        try:
            os.makedirs(os.path.dirname(self.output_path))
        except OSError:
            pass

        cv2.imwrite(self.output_path, feats)

    def run(self):

        if os.path.exists(self.output_path):
            return True

        self.extract_features()

        return True
