# -*- coding: utf-8 -*-

import cv2
import dlib

from abc import ABCMeta
from abc import abstractmethod
from antispoofing.sfsnet.utils import *

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

from scipy.misc import imresize


class Dataset(metaclass=ABCMeta):

    def __init__(self, dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                 n_channel, frame_offset, total_n_frames, file_type_face_annotation='.face', delimiter_face_annotation=" "):

        self.dataset_path = os.path.abspath(dataset_path)
        self.output_path = os.path.abspath(output_path)
        self.face_locations_path = face_locations_path

        self.file_types = file_types

        self.operation = operation
        self.max_axis = max_axis
        self.n_channel = n_channel
        self.frame_offset = frame_offset
        self.total_n_frames = total_n_frames
        self.file_type_face_annotation = file_type_face_annotation
        self.delimiter_face_annotation = delimiter_face_annotation

        # -- classes
        self.POS_LABEL = 1
        self.NEG_LABEL = 0

        self.__meta_info = None
        self.meta_was_built = False

    def prune_set(self, labels, indexes):
        """ This method is responsible for pruning an input subset in order to get a balanced dataset in terms of their classes.

        Args:
            labels (numpy.ndarray):
            indexes (numpy.ndarray):

        Returns:

        """

        # -- prune samples if necessary to have equal sized splits
        neg_idxs = [idx for idx in indexes if labels == self.NEG_LABEL]
        pos_idxs = [idx for idx in indexes if labels == self.POS_LABEL]
        n_samples = min(len(neg_idxs), len(pos_idxs))

        rstate = np.random.RandomState(42)
        rand_idxs_neg = rstate.permutation(neg_idxs)
        rand_idxs_pos = rstate.permutation(pos_idxs)

        neg_idxs = rand_idxs_neg[:n_samples]
        pos_idxs = rand_idxs_pos[:n_samples]
        indexes = np.concatenate((pos_idxs, neg_idxs))

        return indexes

    @staticmethod
    def __crop_img(img, cx, cy, max_axis, padding=0):
        """ This method is responsible for cropping tha input image.

        Args:
            img (numpy.ndarray):
            cx (float):
            cy (float):
            max_axis (int):
            padding (int):

        Returns:
            numpy.ndarray: Cropped image.

        """

        new_height = max_axis
        new_width = max_axis

        n_rows, n_cols, n_channels = img.shape

        cy -= new_height // 2
        cx -= new_width // 2

        if (cy + new_height) > n_rows:
            shift = (cy + new_height) - n_rows
            cy -= shift

        if (cx + new_width) > n_cols:
            shift = (cx + new_width) - n_cols
            cx -= shift

        cy = int(max(0., cy))
        cx = int(max(0., cx))

        cx = padding if cx == 0 else cx
        cy = padding if cy == 0 else cy

        cropped_img = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

        return cropped_img

    # @staticmethod
    def __resize_img(self, img, cx, cy, max_axis, padding=0):
        """ This method is responsible for resizing tha input image.

        Args:
            img (numpy.ndarray):
            cx (float):
            cy (float):
            max_axis (int):
            padding (int):

        Returns:
            numpy.ndarray: Cropped image.

        """

        new_height = max_axis
        new_width = max_axis

        n_rows, n_cols, n_channels = img.shape

        cy -= new_height // 2
        cx -= new_width // 2

        if (cy + new_height) > n_rows:
            shift = (cy + new_height) - n_rows
            cy -= shift

        if (cx + new_width) > n_cols:
            shift = (cx + new_width) - n_cols
            cx -= shift

        cy = int(max(0., cy))
        cx = int(max(0., cx))

        cx = padding if cx == 0 else cx
        cy = padding if cy == 0 else cy

        roi = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

        resized_img = cv2.resize(roi, (self.max_axis+2*padding, self.max_axis+2*padding), interpolation=cv2.INTER_LANCZOS4)

        if len(resized_img.shape) == 2:
            resized_img = resized_img[:, :, np.newaxis]

        return resized_img

    def __detecting_faces_via_dlib(self, imgs_):

        face_boxes = []
        face_cascade = dlib.get_frontal_face_detector()
        # face_cascade = dlib.cnn_face_detection_model_v1(MMOD_HUMAN_FACE_DETECTOR_PATH)

        imgs = imgs_.copy()
        n_frames, n_rows, n_cols, n_channels = imgs.shape

        if n_channels == 3:
            gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2HSV)[:, :, 2]
        else:
            gray = imgs[0]

        # gray_eq = cv2.equalizeHist(gray)

        face_box = face_cascade(gray, 1)

        if len(face_box) == 0:
            x, y, w, h = n_cols / 2, n_rows / 2, self.max_axis, self.max_axis
            x_prev, y_prev, w_prev, h_prev = x, y, w, h

        else:
            x = face_box[0].left()
            y = face_box[0].top()
            w = face_box[0].right() - face_box[0].left()
            h = face_box[0].bottom() - face_box[0].top()
            x_prev, y_prev, w_prev, h_prev = x, y, w, h

        face_boxes.append([0, x, y, w, h])

        for i in range(1, n_frames):

            if n_channels == 3:
                gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2HSV)[:, :, 2]
            else:
                gray = imgs[i]

            gray_eq = cv2.equalizeHist(gray)

            face_box = face_cascade(gray_eq, 1)

            n_boxes = len(face_box)

            if n_boxes == 0:
                x, y, w, h = x_prev, y_prev, w_prev, h_prev
            else:
                face_idx = 0
                face_found = False
                while face_idx < n_boxes and not face_found:

                    x = face_box[face_idx].left()
                    y = face_box[face_idx].top()
                    w = face_box[face_idx].right() - face_box[face_idx].left()
                    h = face_box[face_idx].bottom() - face_box[face_idx].top()

                    if abs(x - x_prev) < 60 and abs(y - y_prev) < 60:
                        x_prev, y_prev, w_prev, h_prev = x, y, w, h
                        face_found = True
                    else:
                        x, y, w, h = x_prev, y_prev, w_prev, h_prev
                    face_idx += 1

            face_boxes.append([i, x, y, w, h])

        return np.array(face_boxes, np.int)

    def __detecting_faces(self, imgs_):

        face_boxes = []
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_FACE_PATH)

        imgs = imgs_.copy()
        n_frames, n_rows, n_cols, n_channels = imgs.shape

        if n_channels == 3:
            gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2HSV)[:, :, 2]
        else:
            gray = imgs[0]

        gray_eq = cv2.equalizeHist(gray.astype(np.uint8))

        face_box = face_cascade.detectMultiScale(gray_eq, SCALE_FACTOR, MIN_NEIGHBORS,
                                                 minSize=MIN_SIZE,
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face_box) == 0:
            x, y, w, h = n_cols / 2, n_rows / 2, self.max_axis, self.max_axis
            x_prev, y_prev, w_prev, h_prev = x, y, w, h

        else:
            x, y, w, h = face_box[0]
            x_prev, y_prev, w_prev, h_prev = x, y, w, h

        face_boxes.append([0, x, y, w, h])

        for i in range(1, n_frames):

            if n_channels == 3:
                gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2HSV)[:, :, 2]
            else:
                gray = imgs[i]

            gray_eq = cv2.equalizeHist(gray.astype(np.uint8))

            face_box = face_cascade.detectMultiScale(gray_eq, SCALE_FACTOR, MIN_NEIGHBORS,
                                                     minSize=MIN_SIZE,
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

            n_boxes = len(face_box)

            if n_boxes == 0:
                x, y, w, h = x_prev, y_prev, w_prev, h_prev
            else:
                face_idx = 0
                face_found = False
                while face_idx < n_boxes and not face_found:
                    x, y, w, h = face_box[face_idx]
                    if abs(x - x_prev) < 60 and abs(y - y_prev) < 60:
                        x_prev, y_prev, w_prev, h_prev = x, y, w, h
                        face_found = True
                    else:
                        x, y, w, h = x_prev, y_prev, w_prev, h_prev
                    face_idx += 1

            face_boxes.append([i, x, y, w, h])

        return np.array(face_boxes, np.int)

    def __normalized_faces(self, frames, boxes, padding=0):

        norm_faces = []
        for img, box in zip(frames, boxes):

            # if max(box[3], box[4]) == 0:
            #     box = self.__detecting_faces(img[np.newaxis]).flatten()

            nf, x, y, w, h = box
            cx = x + w/2
            cy = y + h/2

            if 'crop' in self.operation:
                img = self.__crop_img(img, cx, cy, self.max_axis)

            elif 'resize' in self.operation:

                # # landmarks
                # predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
                # face_aligner = FaceAligner(predictor, desiredFaceWidth=self.max_axis)
                #
                # img = face_aligner.align(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), box[1:])

                maxvalue = self.max_axis if not max(w, h) else max(w, h)

                img = self.__resize_img(img, cx, cy, maxvalue, padding=padding)

            else:
                pass

            norm_faces += [img]

        return np.array(norm_faces)

    def alignment(self, frames):

        # -- initialize dlib's face detector (HOG-based) and then create the facial landmark predictor and the face aligner
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        fa = FaceAligner(predictor, desiredFaceWidth=self.max_axis)

        boxes = []

        for frame in frames:

            if self.n_channel == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 2]
            else:
                gray = frame

            rects = detector(gray, 2)

            # -- loop over the face detections
            for rect in rects:
                # -- extract the ROI of the *original* face, then align the face using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)

    def get_faces_locations(self, fname):
        rel_fname = os.path.relpath(fname, self.dataset_path)
        rel_fname = os.path.splitext(rel_fname)[0] + self.file_type_face_annotation
        fname_face_location = os.path.join(os.path.abspath(self.face_locations_path), rel_fname)
        boxes = np.loadtxt(fname_face_location, dtype=np.int, delimiter=self.delimiter_face_annotation)
        if boxes.ndim == 1:
            boxes = np.reshape(boxes, (1, -1))

        return boxes

    # @profile
    def load_videos(self, fnames, padding=0):

        videos = []

        for i, fname in enumerate(fnames):
            print('-- {0} {1}'.format(i, fname), flush=True)

            # -- load the frames of a video
            frames = load_videos(fname, n_channel=self.n_channel)

            videos += [frames]

        return videos

    def load_images(self, fnames):

        all_images = []

        for i, fname in enumerate(fnames):
            print('-- {0} {1}'.format(i, fname), flush=True)

            # -- load the frames of a video
            img = load_img(fname, n_channel=self.n_channel)
            all_images += [img]

        all_images = np.array(all_images)

        return all_images

    def pre_processing(self, fnames, videos, padding=0):

        all_videos = []
        for fname, frames in zip(fnames, videos):

            if self.face_locations_path:
                # -- load the face locations from a file
                boxes = self.get_faces_locations(fname)
            else:
                # -- detect the face locations of a video
                boxes = self.__detecting_faces(frames)
                # boxes = self.__detecting_faces_via_dlib(frames)

                # -- saving the face locations
                rel_fname = os.path.relpath(fname, self.dataset_path)
                rel_fname = os.path.splitext(rel_fname)[0] + self.file_type_face_annotation
                fname_face_location = os.path.join(self.output_path, self.face_locations_path, rel_fname)
                safe_create_dir(os.path.dirname(fname_face_location))
                np.savetxt(fname_face_location, boxes, fmt='%d', delimiter=self.delimiter_face_annotation)

            # -- check if all faces were located
            try:
                assert frames.shape[0] == boxes.shape[0]
            except AssertionError:
                raise Exception('ERROR: some faces cannot be located in {} (n_faces:{} - n_boxes:{})'.format(fname, frames.shape[0],
                                                                                                             boxes.shape[0]))

            # -- ignore black frames
            ignored_frames_idxs = []
            for k, frame in enumerate(frames):
                if np.mean(frame) < 10:
                    ignored_frames_idxs += [k]

            if len(ignored_frames_idxs):
                valid_frames = np.setdiff1d(np.arange(len(frames)), ignored_frames_idxs)
                frames = frames[valid_frames]
                boxes = boxes[valid_frames]

            try:
                assert frames.shape[0]
            except AssertionError:
                raise Exception('ERROR: There is no valid frames')

            # -- sampling of the input video
            n_frames = frames.shape[0]
            frames = sampling_frames(frames, n_frames, self.total_n_frames)
            boxes = sampling_frames(boxes, n_frames, self.total_n_frames)

            norm_faces = self.__normalized_faces(frames, boxes, padding=padding)

            if norm_faces.shape[0] < self.total_n_frames:
                n_frames = norm_faces.shape[0]
                n_frames_missing = self.total_n_frames - n_frames

                last_frame = norm_faces[-1]
                last_frame = last_frame[np.newaxis, :, :, :]
                for _ in range(n_frames_missing):
                    norm_faces = np.concatenate((norm_faces, last_frame), axis=0)

            try:
                assert norm_faces.shape[0] == self.total_n_frames
                print('-- shape:', norm_faces.shape, flush=True)
            except AssertionError:
                raise Exception("ERROR: Some frames are missing: %s: %d" % (fname, n_frames))

            all_videos += [norm_faces]

        return all_videos

    # -- TODO: Deprecated
    def get_imgs(self, fnames):

        videos = []

        for i, fname in enumerate(fnames):
            print('-- {0} {1}'.format(i, fname), flush=True)

            # -- load the frames of a video
            frames = load_img(fname, n_channel=self.n_channel)

            if self.face_locations_path:
                # -- load the face locations from a file
                boxes = self.get_faces_locations(fname)
            else:
                # -- detect the face locations of a video

                boxes = self.__detecting_faces(frames)
                # boxes = self.__detecting_faces_via_dlib(frames)

                # -- saving the face locations
                rel_fname = os.path.relpath(fname, self.dataset_path)
                rel_fname = os.path.splitext(rel_fname)[0] + self.file_type_face_annotation
                fname_face_location = os.path.join(self.output_path, self.face_locations_path, rel_fname)
                safe_create_dir(os.path.dirname(fname_face_location))
                np.savetxt(fname_face_location, boxes, fmt='%d', delimiter=" ")

            # -- check if all faces were located
            try:
                assert frames.shape[0] == boxes.shape[0]
            except AssertionError:
                raise Exception('ERROR: some faces cannot be located in {} (n_faces:{} - n_boxes:{})'.format(fname,
                                                                                                             frames.shape[0],
                                                                                                             boxes.shape[0]))

            # -- ignore black frames
            ignored_frames_idxs = []
            for k, frame in enumerate(frames):
                if not np.any(frame):
                    ignored_frames_idxs += [k]

            if len(ignored_frames_idxs):
                valid_frames = np.setdiff1d(np.arange(len(frames)), ignored_frames_idxs)
                frames = frames[valid_frames]
                boxes = boxes[valid_frames]

            # -- sampling of the input video
            n_frames = frames.shape[0]
            frames = sampling_frames(frames, n_frames, self.total_n_frames)
            boxes = sampling_frames(boxes, n_frames, self.total_n_frames)

            norm_faces = self.__normalized_faces(frames, boxes)

            if norm_faces.shape[0] < self.total_n_frames:
                n_frames = norm_faces.shape[0]
                n_frames_missing = self.total_n_frames - n_frames

                last_frame = norm_faces[-1]
                last_frame = last_frame[np.newaxis, :, :, :]
                for _ in range(n_frames_missing):
                    norm_faces = np.concatenate((norm_faces, last_frame), axis=0)

            try:
                assert norm_faces.shape[0] == self.total_n_frames
                print('-- shape:', norm_faces.shape, flush=True)
            except AssertionError:
                raise Exception("ERROR: Some frames are missing: %s: %d" % (fname, n_frames))

            videos += [norm_faces]

        return videos

    def info(self, meta_info):

        try:
            print('-*- Dataset Info -*-')
            print('-- all_labels:', meta_info['all_labels'].shape)
            print('-- train_idxs:', meta_info['train_idxs'].shape)
            print('   - pos:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.POS_LABEL)[0].shape)
            print('   - neg:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.NEG_LABEL)[0].shape)

            test_idxs = meta_info['test_idxs']
            for subset in test_idxs:
                print('-- %s:' % subset, test_idxs[subset].shape)
                print('   - pos:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.POS_LABEL)[0].shape)
                print('   - neg:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.NEG_LABEL)[0].shape)

            print('')
            sys.stdout.flush()
        except IndexError:
            pass

    @staticmethod
    def list_dirs(roo_tpath, file_types):
        """ This method returns the name of the subfolders that contain, at least, one file whose type is into the list file_types.

        Args:
            roo_tpath (str):
            file_types (Tuple[str]):

        Returns:
            list: Subfolders that contains at least one file of interest.

        """

        folders = []

        for root, dirs, files in os.walk(roo_tpath, followlinks=True):
            for f in files:
                if os.path.splitext(f)[1] in file_types:
                    folders += [os.path.relpath(root, roo_tpath)]
                    break

        return folders

    def meta_info_feats(self, output_path, file_types):
        """ Metadata of the feature vectors.

        Args:
            output_path:
            file_types:

        Returns:
            dict: A dictionary contaning the metadata.
        """
        return self.build_meta(output_path, file_types)

    @property
    def meta_info(self):
        """ Metadata of the images.

        Returns:
            dict: A dictionary contaning the metadata.
        """

        if not self.meta_was_built:
            self.__meta_info = self.build_meta(self.dataset_path, self.file_types)
            self.meta_was_built = True

        return self.__meta_info

    @abstractmethod
    def build_meta(self, in_path, file_types):
        pass

    @abstractmethod
    def protocol_eval(self):
        pass

    # def meta_info_face_locations(self, file_types, frame_numbers=1):
    #     if self.face_locations_path is None:
    #         return None
    #     else:
    #         return self._build_meta(self.face_locations_path, file_types, frame_numbers)
    #
    # def meta_info_images(self, output_path, file_types, frame_numbers):
    #     return self._build_meta(output_path, file_types, frame_numbers)
