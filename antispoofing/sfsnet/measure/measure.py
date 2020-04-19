# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from antispoofing.sfsnet.utils import *
from scipy import stats


def split_score_distributions(ground_truth, predicted_scores, label_neg, label_pos):
    """ Split the scores in negative and positive scores based on the ground-truth.

    Args:
        ground_truth (numpy.ndarray): Ground-truth.
        predicted_scores (numpy.ndarray): The predicted scores.
        label_neg (int): Label for the negative class.
        label_pos (int):  Label for the positive class.

    Returns:

    """

    # -- get the score distributions of positive and negative classes
    neg = [score for label, score in zip(ground_truth, predicted_scores) if label == label_neg]
    pos = [score for label, score in zip(ground_truth, predicted_scores) if label == label_pos]

    return np.array(neg), np.array(pos)


def predicted_labels_threshold(scores, threshold, label_neg, label_pos):
    """ This function compute the predicted labels for the testing set based on the their scores.

    Args:
        scores (numpy.ndarray): A numpy array containing the predicted scores.
        threshold (float): Threshold value.
        label_neg (int): Label for the negative class.
        label_pos (int):  Label for the positive class.

    Returns:
        numpy.ndarray: Return the predicted labels.

    """

    scores = np.array(scores)

    predicted_labels = np.array([label_pos if score >= threshold else label_neg for score in scores])

    return predicted_labels


def acc_threshold(ground_truth, scores, threshold, label_neg, label_pos):
    """ Compute the Accuracy upon a given threshold.

    Args:
        ground_truth (numpy.ndarray): The ground-truth.
        scores (numpy.ndarray): The predicted scores.
        threshold (float): Threshold value.
        label_neg (int): Label for the negative class.
        label_pos (int):  Label for the positive class.

    Returns:
        float: Return the balanced accuracy, whose value can varies between 0.0 and 1.0, which means the worst and the
        perfect classification results, respectively.

    """

    scores = np.array(scores)

    n_scores = len(scores)
    predicted_labels = np.array([label_pos if score >= threshold else label_neg for score in scores])
    acc = (predicted_labels == ground_truth).sum() / float(n_scores)

    return acc


def balanced_accuracy(ground_truth, predicted_labels):
    """ Compute the Balanced accuracy

    Args:
        ground_truth (numpy.ndarray): The ground-truth.
        predicted_labels (numpy.ndarray): The predicted labels.

    Returns:
        float: Return the balanced accuracy, whose value can varies between 0.0 and 1.0, which means the worst and the
        perfect classification results, respectively.

    """
    categories = np.unique(ground_truth)

    bal_acc = []
    for cat in categories:
        idx = np.where(ground_truth == cat)[0]
        y_true_cat, y_pred_cat = ground_truth[idx], predicted_labels[idx]
        tp = [1 for k in range(len(y_pred_cat)) if y_true_cat[k] == y_pred_cat[k]]
        tp = np.sum(tp)
        bal_acc += [tp / float(len(y_pred_cat))]

    bal_acc = np.array(bal_acc)

    return np.mean(bal_acc)


def bacc_threshold(ground_truth, predicted_scores, threshold, label_neg, label_pos):
    """ Compute the Balanced accuracy upon a given threshold.

    Args:
        ground_truth (numpy.ndarray): The ground-truth.
        predicted_scores (numpy.ndarray): The predicted scores.
        threshold (float): Threshold value.
        label_neg (int): Label for the negative class.
        label_pos (int):  Label for the positive class.

    Returns:
        float: Return the balanced accuracy considering a given threshold. The balanced accuracy can varies between 0.0 and 1.0, which
        means the worst and the perfect classification results, respectively.

    """

    y_pred = predicted_labels_threshold(predicted_scores, threshold, label_neg=label_neg, label_pos=label_pos)

    categories = np.unique(ground_truth)

    bal_acc = []
    for cat in categories:
        idx = np.where(ground_truth == cat)[0]
        y_true_cat, y_pred_cat = ground_truth[idx], y_pred[idx]
        tp = [1 for k in range(len(y_pred_cat)) if y_true_cat[k] == y_pred_cat[k]]
        tp = np.sum(tp)
        bal_acc += [tp / float(len(y_pred_cat))]

    bal_acc = np.array(bal_acc)

    return np.mean(bal_acc)


def min_eer_threshold(neg_scores, pos_scores, n_points=1000):
    """ Compute the EER threshold, that is, the threshold in that the FAR = FRR

    Args:
        neg_scores (numpy.ndarray): The scores for the negative class.
        pos_scores (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve. Default is 1000.

    Returns:

    Note:
        We suppose that negative scores is on the left side of the positive scores
        Deprecated condition: We suppose that negative scores is lower than 0 and positive scores os upper than 0
    """

    threshold = 0.0
    delta_min = 1e5

    if neg_scores.size == 0:
        return 0.

    if pos_scores.size == 0:
        return 0.

    lower_bound = np.min(neg_scores)
    upper_bound = np.max(pos_scores)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    thr = lower_bound

    for pt in range(n_points):

        far, frr = farfrr(neg_scores, pos_scores, thr)

        if abs(far - frr) < delta_min:
            delta_min = abs(far - frr)
            threshold = thr

        thr += steps

    return threshold


def eer_threshold(neg_scores, pos_scores, n_points=1000):
    """ Compute the Minimum HTER threshold

    Args:
        neg_scores (numpy.ndarray): The scores for the negative class.
        pos_scores (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve. Default is 1000.

    Returns:

    Note:
        We suppose that negative scores is on the left side of the positive scores
        Deprecated condition: We suppose that negative scores is lower than 0 and positive scores os upper than 0
    """

    if neg_scores.size == 0:
        return 0.

    if pos_scores.size == 0:
        return 0.

    lower_bound = np.min(neg_scores)
    upper_bound = np.max(pos_scores)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    current_threshold = lower_bound

    predicates = []
    thresholds = []
    for pt in range(n_points):

        far, frr = farfrr(neg_scores, pos_scores, current_threshold)
        current_predicate = abs(far - frr)

        predicates += [current_predicate]
        thresholds += [current_threshold]

        current_threshold += steps

    predicates = np.array(predicates)
    thresholds = np.array(thresholds)

    n_predicates = len(predicates)

    first_idx = np.argmin(predicates)
    min_predicate = predicates[first_idx]
    last_idx = first_idx

    keep_in_loop = True
    while keep_in_loop and (last_idx < n_predicates):
        if min_predicate == predicates[last_idx]:
            last_idx += 1
        else:
            keep_in_loop = False

    threshold = np.median(thresholds[first_idx:last_idx+1])

    return threshold


def min_hter_threshold(neg_scores, pos_scores, n_points=1000):
    """ Compute the Minimum HTER threshold

    Args:
        neg_scores (numpy.ndarray): The scores for the negative class.
        pos_scores (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve. Default is 1000.

    Returns:

    Note:
        We suppose that negative scores is on the left side of the positive scores
        Deprecated condition: We suppose that negative scores is lower than 0 and positive scores os upper than 0
    """

    if neg_scores.size == 0:
        return 0.

    if pos_scores.size == 0:
        return 0.

    lower_bound = np.min(neg_scores)
    upper_bound = np.max(pos_scores)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    current_threshold = lower_bound

    predicates = []
    thresholds = []
    for pt in range(n_points):

        far, frr = farfrr(neg_scores, pos_scores, current_threshold)
        current_predicate = (far+frr)/2.

        predicates += [current_predicate]
        thresholds += [current_threshold]

        current_threshold += steps

    predicates = np.array(predicates)
    thresholds = np.array(thresholds)

    n_predicates = len(predicates)

    first_idx = np.argmin(predicates)
    min_predicate = predicates[first_idx]
    last_idx = first_idx

    keep_in_loop = True
    while keep_in_loop and (last_idx < n_predicates):
        if min_predicate == predicates[last_idx]:
            last_idx += 1
        else:
            keep_in_loop = False

    threshold = np.median(thresholds[first_idx:last_idx+1])

    return threshold


def farfrr_curve(negatives, positives, n_points=1000):
    """ Compute the far and frr curve

    Args:
        negatives (numpy.ndarray): The scores for the negative class.
        positives (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve. Default is 1000.

    Returns:

    Note:
        We suppose that negative scores is on the left side of the positive scores
        Deprecated condition: We suppose that negative scores is lower than 0 and positive scores os upper than 0

    """

    if negatives.size == 0:
        return 0.

    if positives.size == 0:
        return 0.

    lower_bound = np.min(negatives)
    upper_bound = np.max(positives)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    thr = lower_bound

    fars, frrs, thresholds = [], [], []
    for pt in range(n_points):
        far, frr = farfrr(negatives, positives, thr)
        fars += [far]
        frrs += [frr]
        thresholds += [thr]

        thr += steps

    fars = np.array(fars)
    frrs = np.array(frrs)
    thresholds = np.array(thresholds)

    return fars, frrs, thresholds


def far_threshold(negatives, positives, far_at):
    """ Compute the threshold in that FAR=far_value

    Args:
        negatives (numpy.ndarray): The scores for the negative class.
        positives (numpy.ndarray): The scores for the positive class.
        far_at (float): A value for the false acceptance rate.

    Returns:

    """

    n_points = 1000
    threshold = 0.0

    if negatives.size == 0:
        return 0.

    if positives.size == 0:
        return 0.

    lower_bound = np.min(negatives)
    upper_bound = np.max(positives)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    thr = lower_bound

    found, pt = 0, 0
    while pt < n_points and not found:
        far, frr = farfrr(negatives, positives, thr)
        if far <= far_at:
            threshold = thr
            found = 1

        thr += steps

    return threshold


def farfrr(negatives, positives, threshold):
    """ Compute the False Acceptance and the False Rejection Rates considering a given threshold value.

    Args:
        negatives (numpy.ndarray): The scores for the negative class.
        positives (numpy.ndarray): The scores for the positive class.
        threshold (float): A threshold value.

    Returns:

    """

    if negatives.size != 0:
        far = (np.array(negatives) >= threshold).mean()
    else:
        far = 1.

    if positives.size != 0:
        frr = (np.array(positives) < threshold).mean()
    else:
        frr = 1.

    return far, frr


def calc_hter(neg_devel, pos_devel, neg_test, pos_test):
    """ Compute the Half Total Error Rate.

    Args:
        neg_devel (numpy.ndarray):
        pos_devel (numpy.ndarray):
        neg_test (numpy.ndarray):
        pos_test (numpy.ndarray):

    Returns:

    """

    # -- calculate threshold upon eer point
    threshold = eer_threshold(neg_devel, pos_devel)

    # -- calculate far and frr
    far, frr = farfrr(neg_test, pos_test, threshold)

    far *= 100.
    frr *= 100.

    hter = ((far + frr) / 2.)

    return threshold, far, frr, hter


def ppndf_over_array(cum_prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    n_rows, n_cols = cum_prob.shape

    norm_dev = np.zeros((n_rows, n_cols))
    for irow in range(n_rows):
        for icol in range(n_cols):

            prob = cum_prob[irow, icol]
            if prob >= 1.0:
                prob = 1-eps
            elif prob <= 0.0:
                prob = eps

            q = prob - 0.5
            if abs(prob-0.5) <= split:
                r = q * q
                pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
                pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

            else:
                if q > 0.0:
                    r = 1.0-prob
                else:
                    r = prob

                r = np.sqrt((-1.0) * np.log(r))
                pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
                pf /= ((d_2 * r + d_1) * r + 1.0)

                if q < 0:
                    pf *= -1.0

            norm_dev[irow, icol] = pf

    return norm_dev


def ppndf(prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    if prob >= 1.0:
        prob = 1-eps
    elif prob <= 0.0:
        prob = eps

    q = prob - 0.5
    if abs(prob-0.5) <= split:
        r = q * q
        pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
        pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

    else:
        if q > 0.0:
            r = 1.0-prob
        else:
            r = prob

        r = np.sqrt((-1.0) * np.log(r))
        pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
        pf /= ((d_2 * r + d_1) * r + 1.0)

        if q < 0:
            pf *= -1.0

    return pf


def compute_det(negatives, positives, n_points):
    """

    Args:
        negatives (numpy.ndarray): The scores for the negative class.
        positives (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve.

    Returns:

    Note:
        We suppose that negative scores is on the left side of the positive scores
        Deprecated condition: We suppose that negative scores is lower than 0 and positive scores os upper than 0

    """

    lower_bound = np.min(negatives)
    upper_bound = np.max(positives)

    steps = float((upper_bound - lower_bound) / (n_points - 1.))

    threshold = lower_bound
    curve = []
    for pt in range(n_points):

        far, frr = farfrr(negatives, positives, threshold)

        curve.append([far, frr])
        threshold += steps

    curve = np.array(curve)

    return ppndf_over_array(curve.T)


def det(negatives, positives, n_points, axis_font_size='x-small', **kwargs):
    """

    Args:
        negatives (numpy.ndarray): The scores for the negative class.
        positives (numpy.ndarray): The scores for the positive class.
        n_points (int): Number of points considered to build the curve.
        axis_font_size (str): A string specifying the axis font size.
        **kwargs: Optional arguments for the plot function of the matplotlib.pyplot package.

    """

    # these are some constants required in this method
    desired_ticks = [
        '0.00001', '0.00002', '0.00005',
        '0.0001', '0.0002', '0.0005',
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.4', '0.6', '0.8', '0.9',
        '0.95', '0.98', '0.99',
        '0.995', '0.998', '0.999',
        '0.9995', '0.9998', '0.9999',
        '0.99995', '0.99998', '0.99999',
    ]

    desired_labels = [
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.5',
        '1', '2', '5',
        '10', '20', '40', '60', '80', '90',
        '95', '98', '99',
        '99.5', '99.8', '99.9',
        '99.95', '99.98', '99.99',
        '99.995', '99.998', '99.999',
    ]

    curve = compute_det(negatives, positives, n_points)

    output_plot = plt.plot(curve[0, :], curve[1, :], **kwargs)

    # -- now the trick: we must plot the tick marks by hand using the PPNDF method
    p_ticks = [ppndf(float(v)) for v in desired_ticks]

    # -- and finally we set our own tick marks
    ax = plt.gca()
    ax.set_xticks(p_ticks)
    ax.set_xticklabels(desired_labels, size=axis_font_size)
    ax.set_yticks(p_ticks)
    ax.set_yticklabels(desired_labels, size=axis_font_size)

    return output_plot


def det_axis(v, **kwargs):
    """

    Args:
        v:
        **kwargs:

    Returns:

    """

    tv = list(v)
    tv = [ppndf(float(k)/100) for k in tv]
    ret = plt.axis(tv, **kwargs)

    return ret


def perform_frame_fusion(gt, predicted_scores, frame_fusion_type, frame_numbers):

    if frame_fusion_type not in fusion_type_dict.keys():
        raise Exception('error: fusion type not found!')

    r_scores = predicted_scores.flatten()
    pred_labels = np.zeros(r_scores.shape, dtype=np.int32)
    pred_labels[np.where(r_scores > 0.5)[0]] = 1.

    if fusion_type_dict[frame_fusion_type] == 0:
        # -- image-based classification

        r_gt = gt.flatten()
        r_scores = predicted_scores.flatten()
        fuse_idxs = np.arange(r_scores.size).reshape(r_scores.size, -1)

    else:
        # -- video-based classification

        # -- get one label by video
        r_gt = np.reshape(gt, (-1, frame_numbers))
        r_gt = r_gt.mean(axis=1).astype(np.int)

        # -- get one score by video
        fuse_idxs = np.arange(predicted_scores.size).reshape(-1, frame_numbers)
        r_scores = np.reshape(predicted_scores, (-1, frame_numbers))

        if fusion_type_dict[frame_fusion_type] == 1:
            # -- max
            r_scores = r_scores.max(axis=1)

        elif fusion_type_dict[frame_fusion_type] == 2:
            # -- min
            r_scores = r_scores.min(axis=1)

        elif fusion_type_dict[frame_fusion_type] == 3:
            # -- median
            r_scores = np.median(r_scores, axis=1)

        elif fusion_type_dict[frame_fusion_type] == 4:
            # -- mean
            r_scores = r_scores.mean(axis=1)

        elif fusion_type_dict[frame_fusion_type] == 5:
            # -- gmean
            r_scores = stats.mstats.skew(r_scores, axis=1)

        elif fusion_type_dict[frame_fusion_type] == 6:
            # -- hmean
            # pdb.set_trace()
            # params = [stats.exponweib.fit(score, floc=0, f0=0.1) for score in r_scores]
            # params = np.array(params)
            # r_scores = params[:, 3]
            # print r_scores
            # pdb.set_trace()
            # r_scores = r_scores.flatten()
            # signs = np.sign(r_scores)
            # r_scores = np.abs(r_scores)
            # neg_idxs = np.where(r_scores>1.85)
            # r_scores[neg_idxs] *= -1

            # r_scores = np.reshape(r_scores, (-1, frame_numbers))
            # r_scores = r_scores.mean(axis=1)

            # scores = np.sort(r_scores, axis=1)[:, ::-1]
            # scores = scores[:,:5]
            # r_scores = scores.sum(axis=1)

            # # r_scores = stats.signaltonoise(r_scores, axis=1)
            # data_min = np.reshape(r_scores.min(axis=1), (-1,1))
            # data_max = np.reshape(r_scores.max(axis=1), (-1,1))
            # X_std = (r_scores-data_min)/(data_max-data_min)
            # X_scaled = X_std * (0.99 - 0.01) + 0.01
            # r_scores = np.prod(X_scaled, axis=1)

            # # # np.correlate(r_scores[0, :], r_scores[0, :], mode='full')
            # # aa = r_scores[0, :]-r_scores[0, :].mean()
            # # np.correlate(aa, aa, mode='same')/np.sum(aa**2)
            raise Exception('Not Implemented yet!')

        elif fusion_type_dict[frame_fusion_type] == 7:

            pos_rate = round(frame_numbers * 0.95)
            r_pred_labels = np.reshape(pred_labels, (-1, frame_numbers))
            r_pred_labels = r_pred_labels.sum(axis=1).astype(np.int)
            r_pred_labels[r_pred_labels < pos_rate] = 0
            r_pred_labels[r_pred_labels >= pos_rate] = 1

            pos_idxs = np.where(r_pred_labels == 1)[0]
            neg_idxs = np.where(r_pred_labels == 0)[0]

            f_scores = np.zeros(len(r_scores), dtype=np.float32)
            f_scores[pos_idxs] = r_scores[pos_idxs].max(axis=1)
            f_scores[neg_idxs] = r_scores[neg_idxs].min(axis=1)
            r_scores = f_scores

        else:
            print('error: fusion type not found!')
            sys.exit(1)

    return r_gt, r_scores, fuse_idxs
