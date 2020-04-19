# -*- coding: utf-8 -*-

from antispoofing.sfsnet.utils import *


class RawImage(object):

    def __init__(self):
        pass

    def extraction(self, img):
        return img


class BSIF(object):

    def __init__(self):
        pass

    def extraction(self, img):
        return np.reshape(img, (1, -1))
