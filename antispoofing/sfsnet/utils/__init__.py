# -*- coding: utf-8 -*-

# -- common functions and constants
from antispoofing.sfsnet.utils.constants import N_JOBS
from antispoofing.sfsnet.utils.constants import CONST
from antispoofing.sfsnet.utils.constants import PROJECT_PATH
from antispoofing.sfsnet.utils.constants import UTILS_PATH
from antispoofing.sfsnet.utils.constants import SEED
from antispoofing.sfsnet.utils.constants import SCALE_FACTOR
from antispoofing.sfsnet.utils.constants import MIN_NEIGHBORS
from antispoofing.sfsnet.utils.constants import MIN_SIZE
from antispoofing.sfsnet.utils.constants import HAARCASCADE_FACE_PATH
from antispoofing.sfsnet.utils.constants import SHAPE_PREDICTOR_PATH
from antispoofing.sfsnet.utils.constants import MMOD_HUMAN_FACE_DETECTOR_PATH
from antispoofing.sfsnet.utils.constants import fusion_type_dict
from antispoofing.sfsnet.utils.constants import color_space_dict
from antispoofing.sfsnet.utils.constants import map_type_dict
from antispoofing.sfsnet.utils.constants import N_MAPS
from antispoofing.sfsnet.utils.constants import MAX_FRAME_NUMBERS
from antispoofing.sfsnet.utils.constants import BOX_SHAPE_DICT
from antispoofing.sfsnet.utils.constants import IMG_OP_DICT
from antispoofing.sfsnet.utils.constants import IMG_CHANNEL
from antispoofing.sfsnet.utils.misc import modification_date
from antispoofing.sfsnet.utils.misc import get_time
from antispoofing.sfsnet.utils.misc import total_time_elapsed
from antispoofing.sfsnet.utils.misc import RunInParallel
from antispoofing.sfsnet.utils.misc import progressbar
from antispoofing.sfsnet.utils.misc import save_object
from antispoofing.sfsnet.utils.misc import load_object
from antispoofing.sfsnet.utils.misc import load_img
from antispoofing.sfsnet.utils.misc import load_images
from antispoofing.sfsnet.utils.misc import load_videos
from antispoofing.sfsnet.utils.misc import crop_img
from antispoofing.sfsnet.utils.misc import sampling_frames
from antispoofing.sfsnet.utils.misc import safe_create_dir
from antispoofing.sfsnet.utils.misc import mosaic
from antispoofing.sfsnet.utils.misc import read_csv_file
from antispoofing.sfsnet.utils.misc import get_interesting_samples
from antispoofing.sfsnet.utils.misc import create_mosaic
from antispoofing.sfsnet.utils.misc import prune_data
from antispoofing.sfsnet.utils.misc import retrieve_fnames
from antispoofing.sfsnet.utils.misc import retrieve_fnames_py3
from antispoofing.sfsnet.utils.misc import classification_results_summary
from antispoofing.sfsnet.utils.misc import convert_numpy_dict_items_to_list
from antispoofing.sfsnet.utils.misc import unique_everseen

# -- common imports
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mlab

import os
import sys
import time
import numpy as np
import random as rn
import tensorflow as tf

# # -- get reproducible results
# os.environ['PYTHONHASHSEED'] = '{0}'.format(SEED)
# np.random.rand(SEED)
# rn.seed(SEED)
# tf.set_random_seed(SEED)

import keras
from keras import activations
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import LocallyConnected2D
from keras.layers import advanced_activations
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Layer
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

