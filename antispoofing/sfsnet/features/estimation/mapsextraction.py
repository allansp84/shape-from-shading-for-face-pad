# -*- coding: utf-8 -*-

import cv2
from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.features.estimation.tsai import Tsai


class MapsExtraction(object):

    def __init__(self, data, dataset_path, output_paths, input_fnames,
                 sfs_algo=Tsai,
                 analyze=False,
                 n_channel=1,
                 frame_numbers=1,
                 mp=1,
                 local_estimation=False,
                 light_direction="constant",
                 box_shape=BOX_SHAPE_DICT[0],
                 operation=IMG_OP_DICT['resize'],
                 only_face_detection=False,
                 padding=1,
                 ):

        self.__pad = 1

        self.data = data
        self.dataset_path = dataset_path
        self.output_paths = output_paths
        self.input_fnames = input_fnames
        self.sfs_algo = sfs_algo
        self.analyze = analyze
        self.n_channel = n_channel
        self.frame_numbers = frame_numbers
        self.mp = mp
        self.local_estimation = local_estimation
        self.light_direction = light_direction
        self.box_shape = box_shape
        self.operation = operation
        self.only_face_detection = only_face_detection
        self.padding = padding

    @property
    def dataset_path(self):
        return self.__dataset_path

    @dataset_path.setter
    def dataset_path(self, path):
        try:
            self.__dataset_path = os.path.abspath(path)
        except AttributeError:
            sys.exit(1)

    def save_maps(self, output_path, feats):

        try:
            os.makedirs(output_path)
        except OSError:
            pass

        n_frames, n_rows, n_cols, n_channels = feats.shape

        assert self.frame_numbers == n_frames, "missing some frames"

        for idx in range(self.frame_numbers):
            fname = "{0}/{1:03}.png".format(output_path, idx)
            cv2.imwrite(fname, feats[idx, :, :, ::-1])

        return True

    def export_result_dict(self, output_path, feats):

        print("-- saving map(s)")
        sys.stdout.flush()

        try:
            os.makedirs(output_path)
        except OSError:
            pass

        n_frames, n_rows, n_cols, n_channels = feats.shape

        # assert self.frame_numbers == n_frames, "missing some frames"
        n_frames_missing = self.frame_numbers - n_frames
        if n_frames_missing:
            last_frame = feats[-1, :, :, :][np.newaxis, :, :]
            for idx in range(n_frames_missing):
                feats = np.concatenate((feats, last_frame), axis=0)

        output_fnames = []
        for idx in range(self.frame_numbers):
            fname = "{0}/{1:03}.png".format(output_path, idx)
            output_fnames += [fname]

        return output_fnames, feats

    def extract_maps(self, input_fname, output_path, frames):

        print("-- extracting {0} maps".format(self.mp))
        sys.stdout.flush()

        n_frames, n_rows, n_cols, n_channels = frames.shape

        feat_maps = np.zeros((4, n_frames, n_rows - 2 * self.__pad, n_cols - 2 * self.__pad, n_channels,), dtype=np.float32)

        for idx in range(n_frames):

            for ch in range(n_channels):

                sfs = self.sfs_algo(frames[idx, :, :, ch], local_estimation=self.local_estimation,
                                    light_direction=self.light_direction)

                sfs.compute_sfs()

                feat_maps[0, idx, :, :, ch] = sfs.albedo_map[self.__pad:-self.__pad, self.__pad:-self.__pad]
                feat_maps[1, idx, :, :, ch] = sfs.depth_map[self.__pad:-self.__pad, self.__pad:-self.__pad]
                feat_maps[2, idx, :, :, ch] = sfs.reflectance_map[self.__pad:-self.__pad, self.__pad:-self.__pad]
                feat_maps[3, idx, :, :, ch] = sfs.image[self.__pad:-self.__pad, self.__pad:-self.__pad]

        return feat_maps

    def analyze_one_sample(self):
        pass

    def run(self):

        extension = os.path.splitext(self.input_fnames)[1]
        img_extensions = ('.jpg', '.png')

        if extension in img_extensions:
            frames = self.data.load_images([self.input_fnames])
        else:
            frames = self.data.load_videos([self.input_fnames])[0]

        # -- pre-processing
        frames = self.data.pre_processing([self.input_fnames], [frames], padding=self.padding)[0]

        if 'hybrid' in map_type_dict[self.mp]:

            exists = []

            l_keys = list(map_type_dict.keys())
            for key in l_keys[:self.mp]:
                exists += [os.path.exists(self.output_paths.replace('hybrid', map_type_dict[key]))]

            if not exists.count(False):
                print("-- done {0}".format(os.path.relpath(self.input_fnames, self.dataset_path)))
                sys.stdout.flush()
                return True

            print("-- processing video {0}".format(os.path.relpath(self.input_fnames, self.dataset_path)))
            sys.stdout.flush()

            feat_maps = self.extract_maps(self.input_fnames, self.output_paths, frames)

            if not self.only_face_detection:
                for key in l_keys[:self.mp]:
                    output_path = self.output_paths.replace('hybrid', map_type_dict[key])
                    self.save_maps(output_path, feat_maps[key])

        else:

            if os.path.exists(self.output_paths):
                print("-- done {0}".format(os.path.relpath(self.input_fnames, self.dataset_path)))
                sys.stdout.flush()
                return True

            print("-- processing video {0}".format(os.path.relpath(self.input_fnames, self.dataset_path)))
            sys.stdout.flush()

            feat_maps = self.extract_maps(self.input_fnames, self.output_paths, frames)

            if not self.only_face_detection:
                self.save_maps(self.output_paths, feat_maps[self.mp])

        return True
