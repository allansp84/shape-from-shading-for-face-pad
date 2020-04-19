# -*- coding: utf-8 -*-

import os
from multiprocessing import cpu_count


# -- available color spaces
color_space_dict = {1:'grayscale', 3:'rgb'}

# -- type of fusion for video-based decision
fusion_type_dict = {'image_based': 0, 'max': 1, 'min': 2, 'median': 3, 'mean': 4, 'skew': 5, 'test': 6, 'q1': 7}


SEED = 42

N_JOBS = cpu_count()//2 if cpu_count() > 1 else 1

PROJECT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
UTILS_PATH = os.path.dirname(__file__)

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 7
MIN_SIZE = (40, 40)

current_path = os.path.abspath(os.path.dirname(__file__))
# HAARCASCADE_FACE_PATH = os.path.join(current_path, 'haarcascades/haarcascade_frontalface_default.xml')
HAARCASCADE_FACE_PATH = os.path.join(current_path, 'lbpcascades/lbpcascade_frontalface.xml')
HAARCASCADE_EYES_PATH = os.path.join(current_path, 'haarcascades/haarcascade_eye.xml')
SHAPE_PREDICTOR_PATH = os.path.join(current_path, 'shape_predictor_68_face_landmarks.dat')
MMOD_HUMAN_FACE_DETECTOR_PATH = os.path.join(current_path, 'mmod_human_face_detector.dat')


# -- available type of maps
map_type_dict = {0: 'albedo',
                 1: 'depth',
                 2: 'reflectance',
                 3: 'origin_image',
                 4: 'hybrid',
                 }

N_MAPS = 6
MAX_FRAME_NUMBERS = 999
BOX_SHAPE_DICT = {0: (200, 200), 1: (200, 200), 2: (200, 200)}
IMG_OP_DICT = {'crop': 0, 'resize': 1, 'None': 2}
IMG_CHANNEL = 3


class CONST:

    def __init__(self):
        pass

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
