# -*- coding: utf-8 -*-

import sys
import numpy as np
import itertools as it
from multiprocessing import Pool, Value, Lock
from antispoofing.sfsnet.utils.constants import *
import datetime
import pickle
from PIL import Image
import cv2
from skimage import measure
from operator import itemgetter
from glob import glob
import pdb
import csv


counter = Value('i', 0)
counter_lock = Lock()


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_time():
    return datetime.datetime.now()


def total_time_elapsed(start, finish):
    elapsed = finish - start

    total_seconds = int(elapsed.total_seconds())
    total_minutes = int(total_seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(round(total_seconds % 60))

    return "{0:02d}+{1:02d}:{2:02d}:{3:02d} ({4})".format(elapsed.days, hours, minutes, seconds, elapsed)


def progressbar(name, current_task, total_task, bar_len=20, new_line=False):
        percent = float(current_task) / total_task

        progress = ""
        for i in range(bar_len):
            if i < int(bar_len * percent):
                progress += "="
            else:
                progress += " "

        print("\r{0}{1}: [{2}] {3}/{4} ({5:.1f}%).{6:30}".format(CONST.OK_GREEN, name,
                                                                 progress, current_task,
                                                                 total_task, percent * 100,
                                                                 CONST.END),
              end="\n")

        if new_line:
            print()

        sys.stdout.flush()


def start_process():
    pass


def launch_tasks(arg):

    global counter
    global counter_lock

    index, n_tasks, task = arg

    result = task.run()

    with counter_lock:
        # elapsed = datetime.datetime.now() - start_time
        # time_rate = ((counter.value-1) * previous_time_rate + elapsed.total_seconds())/float(counter.value)
        counter.value += 1
        progressbar('-- RunInParallel', counter.value, n_tasks)
        # print('\r{0}-- RunInParallel: {1} task(s) done.{2:30}'.format(CONST.OK_GREEN, counter.value, CONST.END), end="")
        # sys.stdout.flush()

    return result


class RunInParallel(object):
    def __init__(self, tasks, n_proc=N_JOBS):

        # -- private attributes
        self.__pool = Pool(initializer=start_process, processes=n_proc)
        self.__tasks = []

        # -- public attributes
        self.tasks = tasks

    @property
    def tasks(self):
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks_list):
        self.__tasks = []
        for i, task in enumerate(tasks_list):
            self.__tasks.append((i, len(tasks_list), task))

    def run(self):
        global counter
        counter.value = 0

        pool_outs = self.__pool.map_async(launch_tasks, self.tasks)
        self.__pool.close()
        self.__pool.join()

        try:
            work_done = [out for out in pool_outs.get() if out]
            assert (len(work_done)) == len(self.tasks)

            print('\n{0}-- finish.{1:30}'.format(CONST.OK_GREEN, CONST.END))
            sys.stdout.flush()

        except AssertionError:
            print('\n{0}ERROR: some objects could not be processed!{1:30}\n'.format(CONST.ERROR, CONST.END))
            sys.exit(1)


def save_object(obj, fname):

    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass

    fo = open(fname, 'wb')
    pickle.dump(obj, fo)
    fo.close()


def load_object(fname):
    fo = open(fname, 'rb')
    obj = pickle.load(fo)
    fo.close()

    return obj


def preprocessing(img):

    gimg = cv2.medianBlur(img.mean(axis=2).astype(np.uint8), 31)
    _, bimg = cv2.threshold(gimg, 10, 255, cv2.THRESH_BINARY_INV)

    size = 3
    kernel = np.ones((size, size), np.uint8)
    bimg = cv2.morphologyEx(bimg[size:-size, size:-size], cv2.MORPH_OPEN, kernel, iterations=3)
    bimg = np.pad(bimg, size, 'constant', constant_values=0)

    return bimg


def find_bounding_box(contour):
    min_x, max_x = contour[:, 0].min(), contour[:, 0].max()
    min_y, max_y = contour[:, 1].min(), contour[:, 1].max()
    width = max_x - min_x
    height = max_y - min_y
    return np.array([min_x, min_y, width, height])


def get_center_of_iris_image(img):

    bimage = preprocessing(img)

    labels = measure.label(bimage)

    label_number = 0

    results = []
    while True:
        temp = np.uint8(labels == label_number) * 255
        if not cv2.countNonZero(temp):
            break
        results.append(temp)
        label_number += 1
    results = np.array(results)

    db = []
    for res in results:
        images, contours, hierarchy = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):

            if contour.shape[0] > 2:

                # contour = cv2.convexHull(contour)

                # -- compute the circularity
                area = cv2.contourArea(contour)

                # -- finding the bounding box
                bbox = find_bounding_box(np.squeeze(contour))

                # -- compute the circularity
                circumference = cv2.arcLength(contour, True)
                circularity = circumference ** 2 / (4 * np.pi * area)

                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                n_approx = len(approx)

                aux_img = np.zeros(bimage.shape, dtype=np.uint8) + 255
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, np.int)
                cv2.drawContours(aux_img, [box], 0, (0,0,0), cv2.FILLED)
                min_rect_area = np.count_nonzero(255 - aux_img)

                aux_img = np.zeros(bimage.shape, dtype=np.uint8) + 255
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(aux_img, center, radius, (0, 0, 0), cv2.FILLED)
                min_enclosing_circle_area = np.count_nonzero(255 - aux_img)

                if not min_enclosing_circle_area:
                    min_enclosing_circle_area = 1.

                compactness = min_rect_area/min_enclosing_circle_area

                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center_of_mass = [cx, cy]

                db.append([area, circularity, center_of_mass, bbox, n_approx, compactness])

    db = sorted(db, key=itemgetter(0), reverse=True)

    center_of_mass = []
    idx, found = 0, 0
    while idx < len(db) and not found:
        area, circularity, center_of_mass, bbox, n_approx, compactness = db[idx]

        x, y, width, height = bbox

        # print("area, circularity, center_of_mass, n_approx, compactness")
        # print(area, circularity, center_of_mass, n_approx, compactness)
        # sys.stdout.flush()

        if circularity > 0.9 and circularity < 1.2:
            if area > 30*30 and area < 104*104:
                if img[y:y+height, x:x+width, :].mean() < 50:
                    found = 1
        idx += 1

    return center_of_mass


def crop_img(img, cx, cy, max_axis, padding=0):
    new_height = max_axis
    new_width = max_axis

    cy -= new_height//2
    cx -= new_width//2

    if (cy + new_height) > img.shape[0]:
        shift = (cy + new_height) - img.shape[0]
        cy -= shift

    if (cx + new_width) > img.shape[1]:
        shift = (cx + new_width) - img.shape[1]
        cx -= shift

    cy = max(0, cy)
    cx = max(0, cx)

    cx = padding if cx == 0 else cx
    cy = padding if cy == 0 else cy

    cropped_img = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

    return cropped_img


def __resize_img(img, max_axis):
    ratio = max_axis / np.max(img.shape)
    n_rows, n_cols = img.shape[:2]
    new_n_rows = int(n_rows * ratio)
    new_n_cols = int(n_cols * ratio)

    new_shape = (new_n_rows, new_n_cols, img.shape[2])

    return np.resize(img, new_shape)


def __try_get_iris_region(img, max_axis):

    center_of_mass = get_center_of_iris_image(img)
    # center_of_mass = intensity_profile(img)

    if len(center_of_mass) == 0:
        center_of_mass = [img.shape[1]//2, img.shape[0]//2]

    cx, cy = center_of_mass
    img = crop_img(img, cx, cy, max_axis)

    return img


def intensity_profile(img):
    """Intensity profile available in https://github.com/sbanerj1/IrisSeg

    Reference:
    S. Banerjee and D. Mery. Iris Segmentation using Geodesic Active Contours and GrabCut.
    In Workshop on 2D & 3D Geometric Properties from Incomplete Data at PSIVT (PSIVT Workshops), 2015.
    """

    h, w, d = img.shape
    h3 = h // 3
    w3 = w // 3

    lft = 1 * w3
    rt = 2 * w3
    up = 1 * h3
    down = 2 * h3

    hor_l = [0] * (int(down - up) // 5 + 1)
    ver_l = [0] * (int(rt - lft) // 5 + 1)
    temp_l = []
    hor_list = []
    ver_list = []
    min_val = 100
    ellipse_size = 0
    min_x = 0
    min_y = 0
    maxf = 0
    maxs = 0
    eoc = 0

    i = lft
    j = up
    while i <= rt:
        j = up
        while j <= down:
            if int(img[j][i][0]) < min_val:
                min_val = int(img[j][i][0])
            j += 1
        i += 1

    m = 0
    n = up
    k = 0
    max_blah = 0
    while n <= down:
        m = lft
        while m <= rt:
            temp = int(img[n][m][0])
            if temp < (min_val + 10):
                hor_l[k] += 1
                temp_l.append([m, n])
            else:
                pass
            m += 1
        if hor_l[k] > max_blah:
            max_blah = hor_l[k]
            hor_list = temp_l
        temp_l = []
        n += 5
        k += 1

    max_t = max_blah

    m = 0
    n = lft
    k = 0
    max_blah = 0
    temp_l = []
    while n <= rt:
        m = up
        while m <= down:
            temp = int(img[m][n][0])
            if temp < (min_val + 10):
                ver_l[k] += 1
                temp_l.append([n, m])
            else:
                pass
            m += 1
        if ver_l[k] > max_blah:
            max_blah = ver_l[k]
            ver_list = temp_l
        temp_l = []
        n += 5
        k += 1

    if max_blah > max_t:
        max_t = max_blah

    cx = 0
    cy = 0
    hlst = []
    vlst = []
    sumh = 0
    sumv = 0

    i = lft

    while i <= rt:
        j = up
        while j <= down:
            if int(img[j][i][0]) < (min_val + 10):
                hlst.append(i)
                sumh += i
                vlst.append(j)
                sumv += j
            j += 1
        i += 1

    cx = int(sumh // len(hlst))
    cy = int(sumv // len(vlst))

    return [cx, cy]


def __grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)


def safe_create_dir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def mosaic(n, imgs):
    """
    Make a grid from images.
    n    -- number of grid columns
    imgs -- images (must have same size and format)
    :param imgs:
    :param w:
    """
    imgs = iter(imgs)
    img0 = imgs.__next__()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = __grouper(n, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def read_csv_file(fname, sequenceid_col=1, delimiter=',', remove_header=True):

    csv_hash = {}
    csv_data = []

    with open(fname) as f:
        data = list(csv.reader(f, delimiter=delimiter))
        # -- removing header
        if remove_header:
            data = data[1:]
        for r_idx, row in enumerate(data):
            csv_data += [row]
            csv_hash[os.path.splitext(row[sequenceid_col])[0]] = r_idx
    csv_data = np.array(csv_data)

    return csv_data, csv_hash


def get_interesting_samples(ground_truth, scores, threshold, n=1, label_neg=0, label_pos=1):
    """
    Return the n most confusing positive and negative sample indexes. Positive samples have
    scores >= threshold and are labeled label_pos in ground_truth. Negative samples are labeled label_neg.
    @param ground_truth:
    @param scores:
    @param threshold:
    @param n:
    @param label_neg:
    @param label_pos:
    """
    pos_hit = []
    neg_miss = []
    neg_hit = []
    pos_miss = []

    for idx, (gt, score) in enumerate(zip(ground_truth, scores)):
        if score >= threshold:
            if gt == label_pos:
                # -- positive hit
                pos_hit += [idx]
            else:
                # -- negative miss
                neg_miss += [idx]
        else:
            if gt == label_neg:
                # -- negative hit
                neg_hit += [idx]
            else:
                # -- positive miss
                pos_miss += [idx]

    # -- interesting samples
    scores_aux = np.empty(scores.shape, dtype=scores.dtype)

    scores_aux[:] = np.inf
    scores_aux[pos_hit] = scores[pos_hit]
    idx = min(n, len(pos_hit))
    int_pos_hit = scores_aux.argsort()[:idx]

    scores_aux[:] = np.inf
    scores_aux[neg_miss] = scores[neg_miss]
    idx = min(n, len(neg_miss))
    int_neg_miss = scores_aux.argsort()[:idx]

    scores_aux[:] = -np.inf
    scores_aux[neg_hit] = scores[neg_hit]
    idx = min(n, len(neg_hit))
    if idx == 0:
        idx = -len(scores_aux)
    int_neg_hit = scores_aux.argsort()[-idx:]

    scores_aux[:] = -np.inf
    scores_aux[pos_miss] = scores[pos_miss]
    idx = min(n, len(pos_miss))
    if idx == 0:
        idx = -len(scores_aux)
    int_pos_miss = scores_aux.argsort()[-idx:]

    r_dict = {'true_positive': int_pos_hit,
              'false_negative': int_neg_miss,
              'true_negative': int_neg_hit,
              'false_positive': int_pos_miss,
              }

    return r_dict


def create_mosaic(all_data, resize=False, max_axis=64, n_col=50, quality=50, output_fname='mosaic.jpg'):
    """ Create a mosaic.

    Args:
        all_data:
        resize:
        max_axis:
        n_col:
        quality:
        output_fname:

    Returns:

    """
    print('-- creating mosaic ...')
    sys.stdout.flush()

    all_data = np.array(all_data)
    new_shape = (-1,) + all_data.shape[2:]
    all_data = np.reshape(all_data, new_shape)

    alldata = []
    for idx in range(len(all_data)):
        try:
            img = all_data[idx]
            img = np.squeeze(img)
            if resize:
                ratio = max_axis / np.max(img.shape)
                new_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
                alldata += [cv2.resize(img, new_shape)]
            else:
                alldata += [img]
        except:
            pdb.set_trace()
    mosaic_img = mosaic(n_col, alldata)

    print('-- saving mosaic', output_fname)
    sys.stdout.flush()

    cv2.imwrite(output_fname, mosaic_img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def sampling_frames(frames, n_frames, total_n_frames):

    if total_n_frames < 0:
        total_n_frames = len(frames)

    try:
        assert total_n_frames < n_frames
        k = int(np.floor(n_frames / float(total_n_frames)))
        idxs = range(0, n_frames, k)
        frames = frames[idxs[:total_n_frames]]
    except AssertionError:
        pass

    return frames


def load_videos(fname, n_channel=1):

    dt = np.uint8

    # -- load the frames of a input video
    cap = cv2.VideoCapture()
    cap.open(fname)

    frames = []
    has_frame, frame = cap.read()
    while has_frame:
        # -- convert BGR to RGB
        frames.append(frame[:, :, ::-1])
        has_frame, frame = cap.read()
    frames = np.array(frames, dtype=dt)

    cap.release()

    # -- pre-processing of the loaded frames
    n_frames, n_rows, n_cols, n_channels = frames.shape

    if n_channel == 1:
        # -- convert the RGB to gray-scale images
        gray_frames = np.zeros((n_frames, n_rows, n_cols), dtype=frames.dtype)
        for f in range(n_frames):
            gray_frames[f] = cv2.cvtColor(frames[f], cv2.COLOR_RGB2GRAY)

        frames = gray_frames[:, :, :, np.newaxis]
    else:
        # for f in range(n_frames):
        #     # -- perform a color correction using a color constancy algorithm
        #     frames[f] = cv2.cvtColor(frames[f], cv2.COLOR_RGB2HSV)
        pass

    return frames


def load_images(fnames, n_channel=None, new_shape=None):

    dt = np.float32
    imgs = []

    for i, fname in enumerate(fnames):
        try:
            if n_channel is None:
                img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
            elif n_channel == 1:
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(fname, cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            if new_shape is not None:
                img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LANCZOS4)

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]

            img = np.array(img, dtype=dt)

            imgs += [img]

        except Exception:
            raise (Exception, 'Can not read the image {}'.format(fname))

    imgs = np.array(imgs, dtype=dt)

    return imgs


def load_img(fname, n_channel=None):

    dt = np.float32

    try:

        if n_channel is None:
            img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)

        elif n_channel == 1:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        else:
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        img = np.array(img, dtype=dt)

    except Exception:
        raise (Exception, 'Can not read the image {}'.format(fname))

    return img


def prune_data(data, labels, frames_per_video, label_neg=0, label_pos=1):
    """ This method is responsible for pruning an input subset in order to get a balanced dataset in terms of their classes.

    Args:
        labels (numpy.ndarray):
        data (numpy.ndarray):
        n_frames_ber_video (int):
        label_neg:
        label_pos:

    Returns:

    """

    # -- get the indexes for positive and negative samples
    neg_idxs = np.where(labels == label_neg)[0]
    pos_idxs = np.where(labels == label_pos)[0]

    # n_samples = min(len(neg_idxs), len(pos_idxs))
    rand_state = np.random.RandomState(42)

    # rand_idxs_neg = rand_state.permutation(neg_idxs)
    # rand_idxs_pos = rand_state.permutation(pos_idxs)
    #
    # neg_idxs = rand_idxs_neg[:n_samples]
    # pos_idxs = rand_idxs_pos[:n_samples]

    rate = min(len(neg_idxs), len(pos_idxs))/max(len(neg_idxs), len(pos_idxs))

    # -- reshape to get the frames from a person per line
    neg_idxs = np.reshape(neg_idxs, (-1, frames_per_video))

    # -- compute the number of frames to be held
    n_frames = int(frames_per_video*rate)

    # --  prune the negative data and its respective labels
    neg_idxs = neg_idxs[:, :n_frames].flatten()

    all_idxs = np.concatenate((neg_idxs, pos_idxs))
    # all_idxs = rand_state.permutation(all_idxs)

    return data[all_idxs], labels[all_idxs]


def retrieve_fnames(input_path, file_type):

    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if file_type in [os.path.splitext(f)[1], ".*"]:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(input_path, dir_name, '*' + file_type)))
        fnames += dir_fnames

    return fnames


def retrieve_fnames_py3(input_path, file_type):
    fnames = []
    for filename in glob(os.path.join(input_path, '**/*' + file_type), recursive=True):
        fnames += [filename]
    return fnames


def classification_results_summary(report):
        """ This method is responsible for printing a summary of the classification results.

        Args:
            report (dict): A dictionary containing the measures (e.g., Acc, APCER)  computed for each test set.

        """

        print('-- Classification Results', flush=True)

        headers = ['Testing set', 'Threshold (value)', 'AUC', 'ACC', 'BACC', 'APCER (FAR)', 'BPCER (FRR)', 'EER', 'HTER', 'BPCER@FAR_1', 'BPCER@FAR_5', 'BPCER@FAR_10']
        header_line = "| {:<60s} | {:<17s} | {:<8s} | {:<8s} | {:<12s} | {:<12s} | {:<12s} | {:<8s} | {:<8s} | {:<8s} | {:<8s} | {:<8s} |\n".format(*headers)
        sep_line = '-' * (len(header_line)-1) + '\n'

        final_report = sep_line
        final_report += header_line

        for k1 in sorted(report):
            final_report += sep_line
            for k2 in sorted(report[k1]):
                values = [k1,
                          "{} ({:.2f})".format(k2, report[k1][k2]['threshold']),
                          report[k1][k2]['auc'],
                          report[k1][k2]['acc'],
                          report[k1][k2]['bacc'],
                          report[k1][k2]['apcer'],
                          report[k1][k2]['bpcer'],
                          report[k1][k2]['eer'],
                          report[k1][k2]['hter'],
                          report[k1][k2]['bpcer_at_one'],
                          report[k1][k2]['bpcer_at_five'],
                          report[k1][k2]['bpcer_at_ten'],
                          ]

                line = "| {:<60s} | {:<17s} | {:<8.4f} | {:<8.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} |\n".format(*values)
                final_report += line
        final_report += sep_line

        print(final_report, flush=True)


def convert_numpy_dict_items_to_list(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = convert_numpy_dict_items_to_list(v)
        else:
            new_dict = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                new_dict[k] = v
            return new_dict


def unique_everseen(iterable, key=None):
    # -- list unique elements, preserving order. Remember all elements ever seen
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in it.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


# def list_of_results(f_results, output_path, test_set, measure):
#
#     f_measure  = [f for f in f_results if((test_set in f) and (measure in f))]
#
#     configs, values = [], []
#     for f_m in f_measure:
#         configs += [os.path.dirname(os.path.relpath(f_m, output_path))]
#         values += [float(open(f_m,'r').readline())]
#
#     configs_orin = configs
#     configs = replace_from_list(configs, '/', ',')
#
#     configs = replace_from_list(configs, test_set, '')
#     configs = replace_from_list(configs, 'classifiers,', '')
#
#     configs = replace_from_list(configs, '300,', '')
#
#     configs = replace_from_list(configs, 'realization_1', 'R1')
#     configs = replace_from_list(configs, 'realization_2', 'R2')
#     configs = replace_from_list(configs, 'realization_3', 'R3')
#
#     configs = replace_from_list(configs, 'centerframe', 'C')
#     configs = replace_from_list(configs, 'wholeframe', 'W')
#
#     configs = replace_from_list(configs, 'dftenergymag',   'ME')
#     configs = replace_from_list(configs, 'dftentropymag',  'MS')
#     configs = replace_from_list(configs, 'dftenergyphase', 'PE')
#     configs = replace_from_list(configs, 'dftentropyphase','PS')
#
#     configs = replace_from_list(configs, 'kmeans', 'K')
#     configs = replace_from_list(configs, 'random', 'R')
#
#     configs = replace_from_list(configs, 'class_based', 'D')
#     configs = replace_from_list(configs, 'unified', 'S')
#
#     configs = replace_from_list(configs, 'svm', 'SVM')
#     configs = replace_from_list(configs, 'pls', 'PLS')
#
#     configs = replace_from_list(configs, 'energy_phase', 'PE')
#     configs = replace_from_list(configs, 'entropy_phase', 'PH')
#     configs = replace_from_list(configs, 'energy_mag', 'ME')
#     configs = replace_from_list(configs, 'entropy_mag', 'MH')
#     configs = replace_from_list(configs, 'mutualinfo_phase', 'PMI')
#     configs = replace_from_list(configs, 'mutualinfo_mag', 'MMI')
#     configs = replace_from_list(configs, 'correlation_phase', 'PC')
#     configs = replace_from_list(configs, 'correlation_mag', 'MC')
#
#     reverse = False if 'hter' in measure else True
#
#     results = sorted(zip(configs, values), key=operator.itemgetter(1), reverse=reverse)
#
#     fname = "{0}/{1}.{2}.csv".format(output_path, test_set, measure)
#     f_csv = open(fname, 'w')
#     f_csv.write("N,LGF,M,CS,SDD,DS,CP,C,%s\n" % str(measure).upper())
#     for r in results:
#         f_csv.write("%s%s\n" % (r[0], r[1]))
#     f_csv.close()
#
#     print fname, results[:4]
#
#     return sorted(zip(configs_orin, values), key=operator.itemgetter(1), reverse=reverse)
