# -*- coding: utf-8 -*-

import cv2
import json
import shutil

from abc import ABCMeta
from abc import abstractmethod
from sklearn import metrics
from matplotlib import ticker

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.measure import *
from tflearn.objectives import roc_auc_score

from scipy.stats import norm


# # -- get reproducible results
# os.environ['PYTHONHASHSEED'] = '{0}'.format(0)
# np.random.rand(0)
# rn.seed(0)
# tf.set_random_seed(0)


def metric_roc_auc_score(y_true, y_pred):
    return roc_auc_score(y_pred, y_true)


def loss_hter(y_true, y_pred):

    with tf.name_scope("loss_hter"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.cast(tf.round(pos), tf.float32)+0.00001
        neg = tf.cast(tf.round(neg), tf.float32)+0.00001

        false_positive_rate = tf.reduce_sum(neg) / tf.cast(tf.size(neg), tf.float32)
        false_negative_rate = tf.reduce_sum((1. - pos)) / tf.cast(tf.size(pos), tf.float32)

        false_positive_rate = tf.minimum(false_positive_rate, 1.)
        false_negative_rate = tf.minimum(false_negative_rate, 1.)

        total_error = tf.reduce_sum(false_positive_rate + false_negative_rate)
        half_total_error = tf.divide(total_error, 2.)

        return half_total_error


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true)*alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1-p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss


def metric_hter(y_true, y_pred):

    with tf.name_scope("metric_hter"):
        # y_true = y_true[:, 0]
        # y_pred = y_pred[:, 0]

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.cast(tf.round(pos), tf.float32)+0.00001
        neg = tf.cast(tf.round(neg), tf.float32)+0.00001

        false_positive_rate = tf.reduce_sum(neg) / tf.cast(tf.size(neg), tf.float32)
        false_negative_rate = tf.reduce_sum((1. - pos)) / tf.cast(tf.size(pos), tf.float32)

        false_positive_rate = tf.minimum(false_positive_rate, 1.)
        false_negative_rate = tf.minimum(false_negative_rate, 1.)

        total_error = tf.reduce_sum(false_positive_rate + false_negative_rate)
        half_total_error = tf.divide(total_error, 2.)

        return half_total_error


def metric_bal_accuracy(y_true, y_pred):

    with tf.name_scope("metric_bal_accuracy"):
        # y_true = y_true[:, 0]
        # y_pred = y_pred[:, 0]

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.cast(tf.round(pos), tf.float32)
        neg = tf.cast(tf.round(neg), tf.float32)

        false_positive_rate = tf.reduce_sum(neg) / (tf.cast(tf.size(neg), tf.float32)+0.00001)
        false_negative_rate = tf.reduce_sum((1. - pos)) / (tf.cast(tf.size(pos), tf.float32)+0.00001)

        false_positive_rate = tf.minimum(false_positive_rate, 1.)
        false_negative_rate = tf.minimum(false_negative_rate, 1.)

        total_error = tf.reduce_sum(false_positive_rate + false_negative_rate)
        half_total_error = tf.divide(total_error, 2.)

        bal_accuracy = 1. - half_total_error

        return bal_accuracy


class BaseClassifier(metaclass=ABCMeta):

    def __init__(self, output_path, dataset, dataset_b=None, dataset_name="", dataset_b_name="", input_shape=200, n_channel=3,
                 frames_per_video=10, n_frames_for_testing=10, frame_fusion_type='max', load_n_frames=-1,
                 force_train=False, fold=0, seed=0, testing_best_weights=False,
                 force_testing=False, load_weights='', linearize_hybrid_imgs=False,
                 feature_visualization=False,
                 debug=False):

        self.verbose = True
        self.output_path = os.path.abspath(output_path)
        self.dataset = dataset
        self.dataset_b = dataset_b
        self.dataset_name = dataset_name
        self.dataset_b_name = dataset_b_name
        self.input_shape = (input_shape, input_shape, n_channel)
        self.n_channel = n_channel
        self.frames_per_video = frames_per_video
        self.n_frames_for_testing = n_frames_for_testing
        self.frame_fusion_type = frame_fusion_type
        self.load_n_frames = load_n_frames
        self.fold = fold
        self.force_train = force_train
        self.testing_best_weights = testing_best_weights
        self.force_testing = force_testing
        self.load_weights = load_weights
        self.linearize_hybrid_imgs = linearize_hybrid_imgs
        self.feature_visualization = feature_visualization
        self.debug = debug

    def linearizing_hybrid_imgs(self, all_data, all_labels):

        all_data = all_data[:, :, :, :, np.newaxis]
        all_data = np.reshape(all_data, all_data.shape[:3] + (3, -1))
        all_data = np.rollaxis(all_data, 3, 1)
        all_data = np.reshape(all_data, (-1,) + all_data.shape[2:])

        all_labels = np.reshape(all_labels, (-1, 1))
        all_labels = np.concatenate((all_labels, all_labels, all_labels), axis=1)
        all_labels = all_labels.flatten()

        return all_data, all_labels

    def load_data(self, all_fnames, all_labels):
        # -- TODO: Reshape all_fnames too

        if self.load_n_frames > 0:
            all_fnames = np.reshape(all_fnames, (-1, self.frames_per_video))
            all_labels = np.reshape(all_labels, (-1, self.frames_per_video))

            all_fnames = all_fnames[:, :self.load_n_frames].flatten()
            all_labels = all_labels[:, :self.load_n_frames].flatten()

        if self.debug: pdb.set_trace()

        file_type = os.path.splitext(all_fnames[0])[1]

        if '.npy' in file_type:

            img_shape = np.load(all_fnames[0]).shape

            if img_shape[0] != self.input_shape[0] or img_shape[1] != self.input_shape[1]:
                print('-- WarningImageSize: changing image size from {} to {}'.format(img_shape, self.input_shape), end='', flush=True)

                new_shape = (len(all_fnames),) + self.input_shape
                all_data = np.zeros(new_shape, dtype=np.float32)
                for i, fname in enumerate(all_fnames):
                    hybrid_img = np.load(fname)
                    for ch in range(hybrid_img.shape[2]):
                        all_data[i, :, :, ch] = cv2.resize(hybrid_img[:, :, ch], self.input_shape[:2], interpolation=cv2.INTER_LANCZOS4)

            else:
                new_shape = (len(all_fnames),) + np.load(all_fnames[0]).shape
                all_data = np.zeros(new_shape, dtype=np.float32)

                n_fnames = len(all_fnames)
                for i, fname in enumerate(all_fnames):
                    all_data[i] = np.load(fname)
                    progressbar('-- loading dataset', i, n_fnames)

        else:
            img_shape = cv2.imread(all_fnames[0], cv2.IMREAD_ANYCOLOR).shape

            if img_shape[0] != self.input_shape[0] or img_shape[1] != self.input_shape[1]:
                print('-- WarningImageSize: changing image size from {} to {}'.format(img_shape, self.input_shape), end='', flush=True)
                all_data = load_images(all_fnames, new_shape=self.input_shape[:2])
            else:
                all_data = load_images(all_fnames)

        if self.linearize_hybrid_imgs:
            all_data, all_labels = self.linearizing_hybrid_imgs(all_data, all_labels)

        print('-- data loaded in memory:', all_data.shape, flush=True)

        return all_data, all_labels

    def interesting_samples(self, all_fnames, test_sets, class_report, predictions, threshold_type='DEV_EER', prefix=''):
        """ This method persists a dictionary containing interesting samples for later visual assessments, which contains the filenames of
        the samples that were incorrectly classified.

        Args:
            all_fnames (numpy.ndarray): A list containing the filename for each image in the dataset.
            test_sets (dict): A dictionary containing the data and the labels for all testing sets that compose the dataset.
            class_report (dict): A dictionary containing several evaluation measures for each testing set.
            predictions (dict): A dictionary containing the predicted scores and labels for each testing set.
            threshold_type (str): Defines what threshold will be considered for deciding the false acceptance and false rejections.
            prefix (str):
        """

        output_path = os.path.join(self.output_path)
        safe_create_dir(output_path)

        int_samples = {}
        predictions_test = predictions.copy()

        for key in sorted(predictions_test):

            pfx, train_key, devel_key, test_key = key.split('-')

            if test_key:
                gt = predictions_test[key]['gt']
                scores = predictions_test[key]['predicted_scores']
                test_idxs = test_sets[test_key]

                int_samples_idxs = get_interesting_samples(gt, scores, class_report[key][threshold_type]['threshold'], n=5, label_neg=0, label_pos=1)

                int_samples[key] = {}

                for key_samples in int_samples_idxs.keys():
                    int_samples[key][key_samples] = {'input_fnames': []}
                    for idx in int_samples_idxs[key_samples]:
                        int_samples[key][key_samples]['input_fnames'] += [all_fnames[test_idxs[idx]]]

        json_fname = os.path.join(output_path, '%s-interesting_samples.json'%prefix)
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(int_samples, indent=4))

        print('-- saving the interesting samples', flush=True)
        # -- saving int_samples
        for key in sorted(int_samples):
            for key_samples in sorted(int_samples[key]):
                for fname in int_samples[key][key_samples]['input_fnames']:
                    rel_fname = os.path.relpath(fname, os.getcwd())
                    output_fname = os.path.join('%s-interesting_samples'%prefix, key, key_samples, rel_fname)
                    safe_create_dir(os.path.dirname(output_fname))
                    shutil.copy2(fname, output_fname)

                    if '.png' in os.path.splitext(fname)[1]:
                        img = cv2.imread(fname, cv2.IMREAD_COLOR)
                        b, g, r = cv2.split(img)
                        cv2.imwrite('%s-R.png' % os.path.splitext(output_fname)[0], r)
                        cv2.imwrite('%s-G.png' % os.path.splitext(output_fname)[0], g)
                        cv2.imwrite('%s-B.png' % os.path.splitext(output_fname)[0], b)
                    else:
                        hybrid_img = np.load(fname)
                        for ch in range(hybrid_img.shape[2]):
                            cv2.imwrite('%s-%d.png' % (os.path.splitext(output_fname)[0], ch), hybrid_img[:, :, ch].astype(np.uint8))

    def save_performance_results(self, class_report):
        """ Save the performance results in a .json file.

        Args:
            class_report (dict): A dictionary containing the evaluation results for each testing set.

        """

        print('-- saving the performance results in {0}\n'.format(self.output_path))
        sys.stdout.flush()

        for k in class_report:
            output_dir = os.path.join(self.output_path, k)
            safe_create_dir(output_dir)
            json_fname = os.path.join(output_dir, 'results.%s.json' % self.frame_fusion_type)
            with open(json_fname, mode='w') as f:
                print("--saving results.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(class_report[k], indent=4))

    @staticmethod
    def plot_score_distributions(thresholds, neg_scores, pos_scores, set_name, output_path):
        """ Plot the score distribution for a binary classification problem.

        Args:
            thresholds (list): A list of tuples containing the types and the values of the thresholds applied in this work.
            neg_scores (numpy.ndarray): The scores for the negative class.
            pos_scores (numpy.ndarray): The scores for the positive class.
            set_name (str): Name of the set used for computing the scores
            output_path (str):

        """

        safe_create_dir(output_path)

        plt.clf()
        plt.figure(figsize=(12, 10), dpi=300)

        plt.title("Score distributions (%s set)" % set_name)
        n, bins, patches = plt.hist(neg_scores, bins=25, normed=True, facecolor='green', alpha=0.5, histtype='bar', label='Negative class')
        na, binsa, patchesa = plt.hist(pos_scores, bins=25, normed=True, facecolor='red', alpha=0.5, histtype='bar', label='Positive class')

        # -- add a line showing the expected distribution
        y = norm.pdf(bins, np.mean(neg_scores), np.std(neg_scores))
        _ = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = norm.pdf(binsa, np.mean(pos_scores), np.std(pos_scores))
        _ = plt.plot(binsa, y, 'k--', linewidth=1.5)

        for thr_type, thr_value in thresholds:
            plt.axvline(x=thr_value, linewidth=2, color='blue')
            plt.text(thr_value, -5, str(thr_type).upper(), rotation=0)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel('Scores', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        plt.legend()

        filename = os.path.join(output_path, '%s.score.distribution.png' % set_name)
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def plot_crossover_error_rate(neg_scores, pos_scores, filename, n_points=1000):
        """ TODO: Not ready yet.

        Args:
            neg_scores (numpy.ndarray):
            pos_scores (numpy.ndarray):
            filename (str):
            n_points (int):
        """

        fars, frrs, thrs = farfrr_curve(neg_scores, pos_scores, n_points=n_points)
        x_range = np.arange(0, len(thrs), 1)

        # -- create the general figure
        fig1 = plt.figure(figsize=(12, 8), dpi=300)

        # -- plot the FAR curve
        ax1 = fig1.add_subplot(111)
        ax1.plot(fars[x_range], 'b-')
        plt.ylabel("(BPCER) FAR")

        # -- plot the FRR curve
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(frrs[x_range], 'r-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("(APCER) FRR")

        plt.xticks()
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')

        plt.show()
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def plot_det_curve(negatives, positives, set_name, output_path, color=0):
        # -- plotting DET Curve

        safe_create_dir(output_path)

        n_points = 1000
        color_graph = (0, 0, 0)

        plt.clf()
        plt.figure(figsize=(12, 10), dpi=300)

        # if color < 16:
        #     list_chars = list("{0:#b}".format(10)[2:])
        #     color_graph = tuple(list_chars)

        det(negatives, positives, n_points, color=color_graph, linestyle='-',  # -- label='test'
            )
        det_axis([1, 40, 1, 40])

        plt.xlabel('APCER (%)')
        plt.ylabel('BPCER (%)')

        title = "DET Curve"
        plt.title(title)

        plt.grid(True)

        filename = os.path.join(output_path, 'det_plot.%s.pdf' % set_name)
        plt.savefig(filename)
        plt.close('all')

    def make_det_plots(self, scores):

        title_font = {'size': '22', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}  # -- bottom vertical alignment for more space
        axis_font = {'size': '18'}
        font_size_axis = 16
        font_size_legend = 18

        title = "DET Curve"

        n_points = 1000
        axis_range = [1, 40, 1, 40]
        fig_size = (8, 6)
        dpi = 300

        if 'casia' in self.dataset_name:

            # -- plot attacks by type
            fig = plt.figure(figsize=fig_size, dpi=dpi)
            plt.clf()

            try:
                det(scores['test_grand']['negatives'], scores['test_grand']['positives'], n_points, color='black',
                    marker='o', linestyle='-', linewidth=2, label='Overall test set')
            except KeyError:
                print('WarningDETCurvePlots: Overall test scores not found!', flush=True)

            try:
                det(scores['test_scenario_4']['negatives'], scores['test_scenario_4']['positives'], n_points, color='firebrick',
                    marker='s', linestyle='-', linewidth=2, label='Warped attack')
            except KeyError:
                print('WarningDETCurvePlots: Warped attack scores not found!', flush=True)

            try:
                det(scores['test_scenario_5']['negatives'], scores['test_scenario_5']['positives'], n_points, color='royalblue',
                    marker='d', linestyle='-', linewidth=2, label='Cut attack')
            except KeyError:
                print('WarningDETCurvePlots: Cut attack scores not found!', flush=True)

            try:
                det(scores['test_scenario_6']['negatives'], scores['test_scenario_6']['positives'], n_points, color='seagreen',
                    marker='*', linestyle='-', linewidth=2, label='Video attack')
            except KeyError:
                print('WarningDETCurvePlots: Video attack scores not found!', flush=True)

            det_axis(axis_range)

            plt.xlabel('APCER (%)', **axis_font)
            plt.ylabel('BPCER (%)', **axis_font)

            plt.xticks(size=font_size_axis)
            plt.yticks(size=font_size_axis)

            plt.legend(title='', fontsize=font_size_legend)
            plt.title(title, **title_font)
            plt.grid(True)

            filename = "{}/det_plot_casia_type_attacks.pdf".format(scores['output_path'])
            print('-- saving DET Curves:', filename, flush=True)
            fig.savefig(filename, bbox_inches='tight')
            plt.close('all')

        elif 'replay' in self.dataset_name:

            # -- plot attacks by type
            fig = plt.figure(figsize=fig_size, dpi=dpi)
            plt.clf()

            try:
                det(scores['test_grand']['negatives'], scores['test_grand']['positives'], n_points, color='black',
                    marker='o', linestyle='-', linewidth=2, label='Overall test set')
            except KeyError:
                print('WarningDETCurvePlots: Overall test scores not found!', flush=True)

            try:
                det(scores['test_attack_highdef']['negatives'], scores['test_attack_highdef']['positives'], n_points, color='firebrick',
                    marker='s', linestyle='-', linewidth=2, label='High-definition attack')
            except KeyError:
                print('WarningDETCurvePlots: High-definition attack scores not found!', flush=True)

            try:
                det(scores['test_attack_mobile']['negatives'], scores['test_attack_mobile']['positives'], n_points, color='royalblue',
                    marker='d', linestyle='-', linewidth=2, label='Mobile attack')
            except KeyError:
                print('WarningDETCurvePlots: Mobile attack scores not found!', flush=True)

            try:
                det(scores['test_attack_print']['negatives'], scores['test_attack_print']['positives'], n_points, color='seagreen',
                    marker='*', linestyle='-', linewidth=2, label='Print attack')
            except KeyError:
                print('WarningDETCurvePlots: Print attack scores not found!', flush=True)

            det_axis(axis_range)

            plt.xlabel('APCER (%)', **axis_font)
            plt.ylabel('BPCER (%)', **axis_font)

            plt.xticks(size=font_size_axis)
            plt.yticks(size=font_size_axis)

            plt.legend(title='', fontsize=font_size_legend)
            plt.title(title, **title_font)
            plt.grid(True)

            filename = "{}/det_plot_replayattack_type_attacks.pdf".format(scores['output_path'])
            print('-- saving DET curves:', filename, flush=True)
            fig.savefig(filename, bbox_inches='tight')
            plt.close('all')

        return True

    def performance_evaluation(self, predictions, train_scenarios_idxs, devel_scenarios_idxs, test_scenarios_idxs, prefix='intra'):
        """ Compute the performance of the fitted model for each test set.

        Args:
            predictions (dict): A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data. For Exemple: {'test': {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}}
            train_scenarios_idxs:
            devel_scenarios_idxs:
            test_scenarios_idxs:
            prefix (str):

        Returns:
            dict: A dictionary containing the performance results for each testing set. For example: {'test': {'acc': acc, 'apcer': apcer, 'bpcer': bpcer}}

        """

        report = {}

        for train_key, devel_key in zip(sorted(train_scenarios_idxs), sorted(devel_scenarios_idxs)):

            print('-- performance_evaluation:', prefix, train_key, devel_key, flush=True)

            # -- compute the thresholds using the training set
            gt_devel = predictions['{}-{}-{}-'.format(prefix, train_key, devel_key)]['gt']
            pred_scores_devel = predictions['{}-{}-{}-'.format(prefix, train_key, devel_key)]['predicted_scores']
            attack_scores_devel, genuine_scores_devel = split_score_distributions(gt_devel, pred_scores_devel, label_neg=0, label_pos=1)

            min_eer_thr = min_eer_threshold(attack_scores_devel, genuine_scores_devel)
            eer_thr = eer_threshold(attack_scores_devel, genuine_scores_devel)
            min_hter_thr = min_hter_threshold(attack_scores_devel, genuine_scores_devel)

            thresholds = [('DEV_MIN_EER', min_eer_thr),
                          ('DEV_EER', eer_thr),
                          ('DEV_MIN_HTER', min_hter_thr),
                          ]

            print('-- attack_scores_devel', attack_scores_devel, flush=True)
            print('-- genuine_scores_devel', genuine_scores_devel, flush=True)
            print('-- eer_thr', eer_thr, flush=True)

            # -- plotting the score distribution for the training set
            output_path = os.path.join(self.output_path, '{}-{}-{}-'.format(prefix, train_key, devel_key))
            self.plot_score_distributions(thresholds, attack_scores_devel, genuine_scores_devel, 'devel', output_path)

            det_curve_scores = {'output_path': output_path}

            # -- compute the evaluation metrics for the test sets
            for test_key in sorted(test_scenarios_idxs):

                key = "{}-{}-{}-{}".format(prefix, train_key, devel_key, test_key)

                report[key] = {}

                if self.load_n_frames > 0:
                    if 'intra' in prefix:
                        frames_per_video = self.frames_per_video
                    else:
                        frames_per_video = self.n_frames_for_testing
                else:
                    frames_per_video = self.frames_per_video

                ground_truth, pred_scores, fuse_idxs = perform_frame_fusion(predictions[key]['gt'],
                                                                            predictions[key]['predicted_scores'],
                                                                            self.frame_fusion_type,
                                                                            frames_per_video,
                                                                            )

                neg_scores, pos_scores = split_score_distributions(ground_truth, pred_scores, label_neg=0, label_pos=1)

                det_curve_scores[test_key] = {'negatives': neg_scores, 'positives': pos_scores}

                # -- compute the Area Under ROC curve
                try:
                    roc_auc = metrics.roc_auc_score(ground_truth, pred_scores)
                except ValueError:
                    roc_auc = 0.

                for thr_type, thr_value in thresholds:

                    # -- compute the FAR and FRR
                    # -- FAR (APCER) is the rate of Presentation Attack images classified as Bona fide images
                    # -- FRR (BPCER) is the rate of Bona fide images classified as Presentation Attacks images
                    # -- Note: Bona fide images represent the Positive Class (label 1) and the
                    # --       Presentation attack images represent the Negative Class (0)
                    apcer, bpcer = farfrr(neg_scores, pos_scores, thr_value)

                    eer = -1.0
                    hter = (apcer + bpcer)/2.

                    if 'TEST' in thr_type:
                        eer = (apcer + bpcer) / 2.
                        hter = -1.0

                    # -- compute the ACC and Balanced ACC
                    acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                    bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                    far_thr_at_one = far_threshold(neg_scores, pos_scores, 0.01)
                    apcer_at_one, bpcer_at_one = farfrr(neg_scores, pos_scores, far_thr_at_one)

                    far_thr_at_five = far_threshold(neg_scores, pos_scores, 0.05)
                    apcer_at_five, bpcer_at_five = farfrr(neg_scores, pos_scores, far_thr_at_five)

                    far_thr_at_ten = far_threshold(neg_scores, pos_scores, 0.10)
                    apcer_at_ten, bpcer_at_ten = farfrr(neg_scores, pos_scores, far_thr_at_ten)

                    # -- save the results in a dictionary
                    report[key][thr_type] = {'auc': roc_auc,
                                             'acc': acc,
                                             'bacc': bacc,
                                             'threshold': thr_value,
                                             'apcer': apcer,
                                             'bpcer': bpcer,
                                             'bpcer_at_one': bpcer_at_one,
                                             'bpcer_at_five': bpcer_at_five,
                                             'bpcer_at_ten': bpcer_at_ten,
                                             'eer': eer,
                                             'hter': hter,
                                            }

                # -- plotting the score distribution and the Crossover Error Rate (CER) graph
                self.plot_score_distributions(thresholds, neg_scores, pos_scores, key, output_path)

            if 'intra' in prefix:
                self.make_det_plots(det_curve_scores)

        return report

    @staticmethod
    def _save_raw_scores(predictions, fname):
        """ Save the scores obtained for the trained model.

        Args:
            predictions (dict): A dictionary containing the obtained scores.
            fname (numpy.ndarray): List of filenames of all images in the dataset.
        """
        save_object(predictions, fname)

    def run_evaluation_train_scenarios(self, dataset_a, dataset_b, prefix='intra'):

        predictions = {}

        train_scenarios_idxs = self.dataset['train_scenarios_idxs']
        devel_scenarios_idxs = self.dataset['devel_scenarios_idxs']

        # -- loading the training data and its labels
        all_fnames_a = dataset_a['all_fnames']
        all_labels_a = dataset_a['all_labels']

        # -- loading the testing data and its labels
        if 'intra' in prefix:
            all_fnames = dataset_a['all_fnames']
            all_labels = dataset_a['all_labels']
            test_scenarios_idxs = dataset_a['test_scenarios_idxs']
        else:
            all_fnames = dataset_b['all_fnames']
            all_labels = dataset_b['all_labels']
            test_scenarios_idxs = dataset_b['test_scenarios_idxs']

        # -- compute the predicted scores and labels for the testing sets
        f_predictions = os.path.join(self.output_path, '%s.predictions.pkl' % prefix)
        if self.force_testing or not os.path.exists(f_predictions):

            for train_key, devel_key in zip(sorted(train_scenarios_idxs), sorted(devel_scenarios_idxs)):

                print('-- run_evaluation_train_scenarios:', prefix, train_key, devel_key, flush=True)

                # -- compute the predicted scores and labels for the devel set
                x_train, y_train = self.load_data(all_fnames_a[train_scenarios_idxs[train_key]], all_labels_a[train_scenarios_idxs[train_key]],
                                                  # frames_per_video=self.frames_per_video,
                                                  )

                pred_train = self.testing(x_train, y_train,
                                          prefix=train_key,
                                          test_fnames=all_fnames_a[train_scenarios_idxs[train_key]],
                                          test_labels=all_labels_a[train_scenarios_idxs[train_key]],
                                          visualization=False,
                                          )

                del x_train, y_train

                predictions["{}-{}-{}-".format(prefix, train_key, train_key)] = {'gt': pred_train['gt'],
                                                                                 'predicted_labels': pred_train['predicted_labels'],
                                                                                 'predicted_scores': pred_train['predicted_scores'],
                                                                                 }

                # -- compute the predicted scores and labels for the devel set
                x_devel, y_devel = self.load_data(all_fnames_a[devel_scenarios_idxs[devel_key]], all_labels_a[devel_scenarios_idxs[devel_key]],
                                                  # frames_per_video=self.frames_per_video,
                                                  )

                pred_devel = self.testing(x_devel, y_devel,
                                          prefix=train_key,
                                          test_fnames=all_fnames_a[devel_scenarios_idxs[devel_key]],
                                          test_labels=all_labels_a[devel_scenarios_idxs[devel_key]],
                                          visualization=False,
                                          )

                del x_devel, y_devel

                predictions["{}-{}-{}-".format(prefix, train_key, devel_key)] = {'gt': pred_devel['gt'],
                                                                                 'predicted_labels': pred_devel['predicted_labels'],
                                                                                 'predicted_scores': pred_devel['predicted_scores'],
                                                                                 }

                for test_key in sorted(test_scenarios_idxs):

                    print('-- testing', test_key, flush=True)

                    # if 'intra' in prefix:
                    #     frames_per_video = self.frames_per_video
                    # else:
                    #     frames_per_video = self.n_frames_for_testing

                    x_test, y_test = self.load_data(all_fnames[test_scenarios_idxs[test_key]], all_labels[test_scenarios_idxs[test_key]],
                                                    # frames_per_video=self.frames_per_video,
                                                    )

                    results = self.testing(x_test, y_test,
                                           prefix=train_key,
                                           test_fnames=all_fnames[test_scenarios_idxs[test_key]],
                                           test_labels=all_labels[test_scenarios_idxs[test_key]],
                                           visualization=self.feature_visualization,
                                           )

                    del x_test, y_test

                    pred_scores = results['predicted_scores']
                    pred_labels = results['predicted_labels']
                    gt = results['gt']

                    predictions["{}-{}-{}-{}".format(prefix, train_key, devel_key, test_key)] = {'gt': gt,
                                                                                                 'predicted_labels': pred_labels,
                                                                                                 'predicted_scores': pred_scores,
                                                                                                 }

            # -- saving the raw scores
            self._save_raw_scores(predictions, f_predictions)

        # -- load the predictions if we don't have it
        if not predictions:
            print('-- loading the predictions saved on disk', flush=True)
            predictions = load_object(f_predictions)

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(predictions, train_scenarios_idxs, devel_scenarios_idxs, test_scenarios_idxs, prefix=prefix)

        # -- saving the performance results
        self.save_performance_results(class_report)

        # # -- saving the interesting samples for further analysis
        # self.interesting_samples(all_fnames, test_scenarios_idxs, class_report, predictions, threshold_type='DEV_EER', prefix=prefix)

    @staticmethod
    def saving_training_history(history, output_path, metric='acc'):
        """ Saving the plot containg the training history.

        Args:
            history (dict): A dictionary containing the values of accuracy and losses obtainied in each epoch of the learning stage.
            output_path (str):

        """

        # -- save the results obtained during the training process
        json_fname = os.path.join(output_path, 'training.history.{}.json'.format(metric))
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(history, indent=4))

        output_history = os.path.join(output_path, 'training.history.{}.png'.format(metric))
        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        title_font = {'size': '18', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
        axis_font = {'size': '14'}
        font_size_axis = 12
        title = "Training History"

        plt.clf()
        plt.plot(range(1, len(history['{}'.format(metric)]) + 1), history['{}'.format(metric)], color=(0, 0, 0), marker='o', linestyle='-', linewidth=2, label='train')
        plt.plot(range(1, len(history['val_{}'.format(metric)]) + 1), history['val_{}'.format(metric)], color=(0, 1, 0), marker='s', linestyle='-', linewidth=2, label='test')

        plt.xlabel('Epochs', **axis_font)
        plt.ylabel('{}'.format(metric.upper()), **axis_font)

        plt.xticks(size=font_size_axis)
        plt.yticks(size=font_size_axis)

        plt.legend(loc='upper left')
        plt.title(title, **title_font)
        plt.grid(True)

        fig1.savefig(output_history)

    def training_process(self):

        # -- loading the training data and its labels
        all_fnames = self.dataset['all_fnames']
        all_labels = self.dataset['all_labels']
        train_users_idxs = self.dataset['train_users_idxs']

        train_scenarios_idxs = self.dataset['train_scenarios_idxs']

        test_scenarios_idxs = self.dataset['test_scenarios_idxs']
        all_fnames_devel, all_labels_devel = self.dataset['all_fnames'], self.dataset['all_labels']

        x_valid, y_valid  = self.load_data(all_fnames_devel[test_scenarios_idxs['test_grand']],
                                           all_labels_devel[test_scenarios_idxs['test_grand']])
        # x_valid, y_valid = None, None

        for key in sorted(train_scenarios_idxs):

            output_model = os.path.join(self.output_path, key, "model.hdf5")
            if self.force_train or not os.path.exists(output_model):

                print('-- Training a model for the scenario %s' % key, flush=True)
                x_train, y_train = self.load_data(all_fnames[train_scenarios_idxs[key]], all_labels[train_scenarios_idxs[key]])

                self.training(x_train, y_train, x_validation=x_valid, y_validation=y_valid, prefix=key,
                              train_fnames=all_fnames[train_scenarios_idxs[key]], train_users_idxs=train_users_idxs)

                # -- free memory
                del x_train, y_train

    def testing_phase(self):

        self.run_evaluation_train_scenarios(self.dataset, self.dataset_b, prefix='intra')

        # -- cross-dataset
        if self.dataset_b is not None:
            self.run_evaluation_train_scenarios(self.dataset, self.dataset_b, prefix='inter_%s' % self.dataset_b_name)

    def run(self):
        """
        This method implements the whole training and testing process considering the evaluation protocol defined for the dataset.
        """
        # -- start the training process
        self.training_process()

        # -- compute the predicted scores and labels for the testing sets
        self.testing_phase()

    @abstractmethod
    def training(self, x_train, y_train, x_validation=None, y_validation=None, prefix='', train_fnames=None, train_users_idxs=None):
        """ This method implements the training stage.

        The training stage will be implemented by the subclasses, taking into account the particularities of the classification algorithm to be used.

        Args:
            x_train (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to train a classifier.
            y_train (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used during the training stage.
            x_validation (numpy.ndarray, optional): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_validation (numpy.ndarray, optional): A multidimensional array containing the labels refers to the feature vectors that will be used for testing the classification model.
            prefix (str):
        """
        return NotImplemented

    @abstractmethod
    def testing(self, x_test, y_test, prefix='', test_fnames=None, test_labels=None, visualization=False):
        """ This method implements the testing stage.

        The testing stage will be implemented by the subclasses.

        Args:
            x_test (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_test (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used to test the classification model.
            prefix (str):

        Returns:
            A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}
        """
        return NotImplemented

    @abstractmethod
    def extract_features(self, x, output_fname, prefix=''):
        pass
