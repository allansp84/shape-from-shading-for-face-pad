# -*- coding: utf-8 -*-

import json

from abc import ABCMeta
from abc import abstractmethod

from sklearn import metrics
from matplotlib import ticker
from operator import itemgetter

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.measure import *

from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file

import scipy.stats as stats


class BaseMetaClassifier(metaclass=ABCMeta):

    def __init__(self, output_path, predictions_files, meta_classification_from, n_models,
                 selection_algo=0, compute_feature_importance=False,
                 frames_per_video=10, frame_fusion_type='max',
                 prefix='inter'):

        self.verbose = True
        self.output_path = os.path.abspath(output_path)
        self.predictions_files = predictions_files
        self.meta_classification_from = meta_classification_from
        self.n_models = n_models

        self.frames_per_video = frames_per_video
        self.frame_fusion_type = frame_fusion_type
        self.prefix = prefix

        self.selection_algo = selection_algo
        self.compute_feature_importance = compute_feature_importance

        self.train_scenario = 'train_grand-devel_grand'
        # self.train_scenario = 'train_scenario_6-devel_scenario_6'

    def interesting_samples(self, all_fnames, test_sets, class_report, predictions, threshold_type='EER'):
        """ This method persists a dictionary containing interesting samples for later visual assessments, which contains the filenames of
        the samples that were incorrectly classified.

        Args:
            all_fnames (numpy.ndarray): A list containing the filename for each image in the dataset.
            test_sets (dict): A dictionary containing the data and the labels for all testing sets that compose the dataset.
            class_report (dict): A dictionary containing several evaluation measures for each testing set.
            predictions (dict): A dictionary containing the predicted scores and labels for each testing set.
            threshold_type (str): Defines what threshold will be considered for deciding the false acceptance and false rejections.

        """

        int_samples = {}
        predictions_test = predictions.copy()
        predictions_test.pop('train_set')

        for key in predictions_test:

            gt = predictions_test[key]['gt']
            scores = predictions_test[key]['predicted_scores']
            test_idxs = test_sets[key]['idxs']
            int_samples_idxs = get_interesting_samples(gt, scores, class_report[key][threshold_type]['threshold'])

            int_samples[key] = {}

            for key_samples in int_samples_idxs.keys():
                int_samples[key][key_samples] = {'input_fnames': []}
                for idx in int_samples_idxs[key_samples]:
                    int_samples[key][key_samples]['input_fnames'] += [all_fnames[test_idxs[idx]]]

        json_fname = os.path.join(self.output_path, 'int_samples.json')
        with open(json_fname, mode='w') as f:
            print("-- saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(int_samples, indent=4))

    def save_performance_results(self, class_report):
        """ Save the performance results in a .json file.

        Args:
            class_report (dict): A dictionary containing the evaluation results for each testing set.

        """

        print('-- saving the performance results in {0}\n'.format(self.output_path))
        sys.stdout.flush()

        for k in class_report:
            output_dir = os.path.join(self.output_path, k)
            try:
                os.makedirs(output_dir)
            except OSError:
                pass

            json_fname = os.path.join(output_dir, 'results.json')
            with open(json_fname, mode='w') as f:
                print("-- saving results.json file:", json_fname)
                sys.stdout.flush()
                f.write(json.dumps(class_report[k], indent=4))

    def plot_roc_curve(self, false_positive_rate, true_positive_rate, roc_auc, filename, set_name):
        """

        Args:
            false_positive_rate (numpy.ndarray):
            true_positive_rate (numpy.ndarray):
            n_points (int): Number of points considered to build the curve.
            axis_font_size (str): A string specifying the axis font size.
            **kwargs: Optional arguments for the plot function of the matplotlib.pyplot package.

        """
        plt.clf()
        plt.figure(figsize=(10, 10), dpi=100)

        title_font = {'size': '18', 'color': 'black'}
        plt.title("Receiver operating characteristic Curve (%s set)" % set_name, **title_font)
        plt.plot(false_positive_rate, true_positive_rate,
                 color=(0, 0, 0), marker='o', linestyle='-', linewidth=2, label='AUC = %2.4f)' % roc_auc)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(loc="lower right", fontsize=16)
        plt.savefig(filename)

    def plot_score_distributions(self, thresholds, neg_scores, pos_scores, filename, set_name):
        """ Plot the score distribution for a binary classification problem.

        Args:
            thresholds (list): A list of tuples containing the types and the values of the thresholds applied in this work.
            neg_scores (numpy.ndarray): The scores for the negative class.
            pos_scores (numpy.ndarray): The scores for the positive class.
            set_name (str): Name of the set used for computing the scores

        """

        plt.clf()
        plt.figure(figsize=(12, 9), dpi=100)

        title_font = {'size': '18', 'color': 'black'}
        plt.title("Score distributions (%s set)" % set_name, **title_font)

        n, bins, patches = plt.hist(neg_scores, bins=30, normed=True, facecolor='red', alpha=0.5, histtype='bar', label='Negative class')
        na, binsa, patchesa = plt.hist(pos_scores, bins=30, normed=True, facecolor='green', alpha=0.5, histtype='bar', label='Positive class')

        # -- add a line showing the expected distribution
        y = mlab.normpdf(bins, np.mean(neg_scores), np.std(neg_scores))
        _ = plt.plot(bins, y, 'k--', linewidth=1.5)
        y = mlab.normpdf(binsa, np.mean(pos_scores), np.std(pos_scores))
        _ = plt.plot(binsa, y, 'k--', linewidth=1.5)

        for thr_type, thr_value in thresholds:
            plt.axvline(x=thr_value, linewidth=1, color='blue')
            plt.text(thr_value, -3, str(thr_type).upper(), fontsize=14, rotation=90)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plt.xlabel('Scores', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)

        plt.legend(fontsize=16)
        plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0, hspace=0)

        plt.savefig(filename)

    @staticmethod
    def plot_det_curve(neg_scores, pos_scores, filename, key):

        # -- bottom vertical alignment for more space
        title_font = {'size': '18', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
        axis_font = {'size': '14'}
        font_size_axis = 12

        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        plt.clf()
        n_points = 300
        det(neg_scores, pos_scores, n_points, color=(0, 0, 0), marker='o', linestyle='-', linewidth=2, label=key)
        det_axis([0.01, 40, 0.01, 40])
        plt.xlabel('FRR (%)', **axis_font)
        plt.ylabel('FAR (%)', **axis_font)

        plt.xticks(size=font_size_axis)
        plt.yticks(size=font_size_axis)

        plt.legend()
        title = 'DET Curve'
        plt.title(title, **title_font)
        plt.grid(True)

        fig1.savefig(filename)

    @staticmethod
    def plot_crossover_error_rate(neg_scores, pos_scores, filename, n_points=100):
        """ TODO: Not ready yet.

        Args:
            neg_scores (numpy.ndarray):
            pos_scores (numpy.ndarray):
            filename (str):
            n_points (int):
        """

        fars, frrs, thrs = farfrr_curve(neg_scores, pos_scores, n_points=n_points)

        # -- create the general figure
        fig1 = plt.figure(figsize=(12, 8), dpi=300)

        # -- plot the FAR curve
        ax1 = fig1.add_subplot(111)
        ax1.plot(fars, thrs, 'b-')
        plt.ylabel("(BPCER) FAR")

        # -- plot the FRR curve
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(frrs, thrs, 'r-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("(APCER) FRR")

        plt.xticks()
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')

        plt.show()
        plt.savefig(filename)

    @staticmethod
    def classification_results_summary(report):
        """ This method is responsible for printing a summary of the classification results.

        Args:
            report (dict): A dictionary containing the measures (e.g., Acc, APCER)  computed for each test set.

        """

        print('-- Classification Results')

        headers = ['Testing set', 'Threshold (value)', 'AUC', 'ACC', 'Balanced ACC', 'BPCER (FAR)', 'APCER (FRR)', 'HTER']
        header_line = "| {:<20s} | {:<20s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<8s} |\n".format(*headers)
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
                          report[k1][k2]['bpcer'],
                          report[k1][k2]['apcer'],
                          report[k1][k2]['hter'],
                          ]
                line = "| {:<20s} | {:<20s} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<8.4f} |\n".format(*values)
                final_report += line
        final_report += sep_line

        print(final_report)
        sys.stdout.flush()

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

            eer_thr = eer_threshold(attack_scores_devel, genuine_scores_devel)
            thresholds = [('EER', eer_thr),
                          # ('FAR@0.01', far_thr),
                          # ('0.5', 0.5),
                          ]

            print('-- attack_scores_devel', attack_scores_devel, flush=True)
            print('-- genuine_scores_devel', genuine_scores_devel, flush=True)
            print('-- eer_thr', eer_thr, flush=True)

            # -- plotting the score distribution for the training set
            output_path = os.path.join(self.output_path, '{}-{}-{}-'.format(prefix, train_key, devel_key))
            self.plot_score_distributions(thresholds, attack_scores_devel, genuine_scores_devel, 'devel', output_path)

            # -- compute the evaluation metrics for the test sets
            for test_key in sorted(test_scenarios_idxs):

                key = "{}-{}-{}-{}".format(prefix, train_key, devel_key, test_key)

                report[key] = {}
                # ground_truth = predictions[key]['gt']
                # pred_scores = predictions[key]['predicted_scores']

                ground_truth, pred_scores, fuse_idxs = perform_frame_fusion(predictions[key]['gt'],
                                                                            predictions[key]['predicted_scores'],
                                                                            self.frame_fusion_type,
                                                                            self.frames_per_video,
                                                                            )

                neg_scores, pos_scores = split_score_distributions(ground_truth, pred_scores, label_neg=0, label_pos=1)

                # -- compute the Area Under ROC curve
                try:
                    roc_auc = metrics.roc_auc_score(ground_truth, pred_scores)
                except ValueError:
                    roc_auc = 0.

                for thr_type, thr_value in thresholds:

                    # -- compute the FAR and FRR
                    # -- FAR (BPCER) is the rate of Genuine images classified as Presentation Attacks images
                    # -- FRR (APCER) is the rate of Presentation Attack images classified as Genuine images
                    # -- Note: Presentation Attacks images represent the Positive Class (label 1) are the
                    # --       genuine images represent the Negative Class (0)
                    bpcer, apcer = farfrr(neg_scores, pos_scores, thr_value)

                    hter = (bpcer + apcer)/2.

                    # -- compute the ACC and Balanced ACC
                    acc = acc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)
                    bacc = bacc_threshold(ground_truth, pred_scores, thr_value, label_neg=0, label_pos=1)

                    # -- save the results in a dictionary
                    report[key][thr_type] = {'auc': roc_auc, 'acc': acc, 'bacc': bacc, 'threshold': thr_value,
                                             'apcer': apcer, 'bpcer': bpcer, 'hter': hter,
                                             }

                # # -- plotting the score distribution and the Crossover Error Rate (CER) graph
                # self.plot_score_distributions(thresholds, neg_scores, pos_scores, key)
                # self.plot_crossover_error_rate(neg_scores, pos_scores, filename=os.path.join(self.output_path, '%s.cer.png' % key))

                # classification_results_summary(report)

        return report

    def _save_raw_scores(self, predictions, prefix=''):
        """ Save the scores obtained for the trained model.

        Args:
            all_fnames (numpy.ndarray): List of filenames of all images in the dataset.
            test_scenarios_idxs (dict): A dictionary containing the metadata of the dataset.
            predictions (dict): A dictionary containing the obtained scores.
        """

        # output_path = os.path.join(self.output_path, prefix)
        #
        # for key in test_scenarios_idxs:
        #     indexes = test_scenarios_idxs[key]
        #     aux = zip(all_fnames[indexes], predictions[key]['predicted_scores'], predictions[key]['predicted_labels'])
        #     dt = dict(names=('fname', 'predicted_score', 'predicted_label'), formats=('U100', np.float32, np.int32))
        #     results = np.array(list(aux), dtype=dt)
        #     np.savetxt(os.path.join(output_path, '%s.predictions.txt' % key), results, fmt="%s,%.6f,%d", delimiter=',')
        fname = os.path.join(self.output_path, '%s.predictions.pkl' % prefix)
        save_object(predictions, fname)

    def complementary_by_cohen_kappa_score(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_kappa_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):
            kappa_coefs = []
            for b, pred_b in enumerate(lpredictors):
                try:
                    kc = metrics.cohen_kappa_score(pred_a, pred_b)
                except ValueError:
                    raise Exception('Please, use the predicted labels, instead of predicted scores')

                if np.isnan(kc):
                    kc = 0.

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_kappa_coefs += [kappa_coefs]

        return all_kappa_coefs

    def complementary_by_kendalltau(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_kappa_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):
            kappa_coefs = []
            for b, pred_b in enumerate(lpredictors):

                kc, pv = stats.kendalltau(pred_a, pred_b)
                if np.isnan(kc):
                    kc = 0.

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_kappa_coefs += [kappa_coefs]

        return all_kappa_coefs

    def complementary_by_q_stat(self, selected_models_by_importance_idxs, lpredictors):

        lpredictors = np.array(lpredictors)

        all_kappa_coefs = []
        for a, pred_a in zip(selected_models_by_importance_idxs, lpredictors[selected_models_by_importance_idxs]):

            kappa_coefs = []

            for b, pred_b in enumerate(lpredictors):

                cm_a, cm_b, cm_c, cm_d = 0, 0, 0, 0

                pred_a = np.array(pred_a)
                pred_b = np.array(pred_b)
                n_total = len(pred_a)

                for pa, pb in zip(pred_a, pred_b):

                    if pa == pb:
                        if pa == 1:
                            cm_a += 1
                        if pa == 0:
                            cm_d += 1

                    if pa != pb:
                        if pa == 1 and pb == 0:
                            cm_b += 1
                        if pa == 0 and pb == 1:
                            cm_c += 1

                cm_a /= n_total
                cm_b /= n_total
                cm_c /= n_total
                cm_d /= n_total

                numerator = (cm_a * cm_d - cm_b * cm_c)
                denominator = (cm_a * cm_d + cm_b * cm_c)

                kc = 0.
                if denominator:
                    kc = numerator/denominator

                kappa_coefs += [[kc, a, b]]

            if len(kappa_coefs):
                kappa_coefs = sorted(kappa_coefs, key=itemgetter(0), reverse=True)
            all_kappa_coefs += [kappa_coefs]

        return all_kappa_coefs

    def find_complementary_models(self, x_train_pred_labels, x_train_pred_scores, n_models=-1):

        min_value, max_value = .9, 1.0

        selected_models_by_importance_idxs = self.select_classifiers_by_feature_importance(n_models=self.n_models)

        n_selected_models = len(selected_models_by_importance_idxs)

        all_kappa_coefs = self.complementary_by_cohen_kappa_score(selected_models_by_importance_idxs, x_train_pred_labels)
        # all_kappa_coefs = self.complementary_by_q_stat(selected_models_by_importance_idxs, x_train_pred_labels)
        # all_kappa_coefs = self.complementary_by_kendalltau(selected_models_by_importance_idxs, x_train_pred_scores)

        all_selected_models_idxs = []
        for kappa_coefs in all_kappa_coefs:
            selected_models_idxs = []
            print('-- kappa_coefs:', kappa_coefs)
            for k, a, b in kappa_coefs:
                if n_models == 0:
                    # if k > min_value and k < max_value:
                    if k > min_value:
                        selected_models_idxs += [a, b]
                else:
                    selected_models_idxs += [a, b]

            selected_models_idxs = list(unique_everseen(selected_models_idxs))

            if n_models == 0:
                selected_models_idxs = selected_models_idxs
            else:
                selected_models_idxs = selected_models_idxs[:n_models]

            all_selected_models_idxs += [selected_models_idxs]

        # all_selected_models_idxs = list(unique_everseen(np.concatenate(all_selected_models_idxs)))

        final_list = []
        for list_a in all_selected_models_idxs:
            for list_b in all_selected_models_idxs:
                final_list += [np.intersect1d(list_a, list_b)]

        final_list = np.concatenate(final_list)

        counter = np.zeros((final_list.max()+1,),dtype=np.int)
        for elem in final_list:
            counter[elem] += 1

        all_selected_models_idxs = []
        for c, idx  in zip(counter[counter.argsort()[::-1]], counter.argsort()[::-1]):
            if c > 0:
                all_selected_models_idxs += [idx]

        # all_selected_models_idxs = np.concatenate((selected_models_by_importance_idxs, counter.argsort()[::-1]))
        # all_selected_models_idxs = list(unique_everseen(all_selected_models_idxs))

        return all_selected_models_idxs[:n_models*3]

    def select_classifiers_by_feature_importance(self, n_models=-1):

        # -- open json file
        json_fname = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.output_path))), 'predictor_importances.json')
        print("-- reading the predictors importance:", json_fname, flush=True)
        # data = json.load(open(json_fname, 'r'))
        dt = dict(names=('idxs', 'importance'), formats=(np.int, np.float32))
        data = np.loadtxt(json_fname, dtype=dt, delimiter=',')

        data = sorted(data, key=itemgetter(1), reverse=True)
        data = data[:n_models]
        print('-- data', data)

        selected_models_idxs = []
        for idx, importance in data:

            # -- remove the features that has no importance according to random forest
            if importance > 0.:
                selected_models_idxs += [int(idx)]

        print('-- selected_models_idxs', sorted(selected_models_idxs), flush=True)
        selected_models_idxs = np.array(selected_models_idxs)

        return selected_models_idxs[:n_models]

    def run(self):
        """
        Start the classification step.
        """

        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

        all_predictions = []
        for fpred in self.predictions_files:
            all_predictions += [load_object(fpred)]

        x_train_pred_scores, x_train_pred_labels, y_train = [], [], []
        x_devel_pred_scores, x_devel_pred_labels, y_devel = [], [], []
        x_test_pred_scores, x_test_pred_labels, y_test = [], [], []

        for preds in all_predictions:

            x_train_pred_labels += [preds['%s-%s-' % (self.prefix, self.train_scenario)]['predicted_labels']]
            x_devel_pred_labels += [preds['%s-%s-' % (self.prefix, self.train_scenario)]['predicted_labels']]
            x_test_pred_labels += [preds['%s-%s-test_grand' % (self.prefix, self.train_scenario)]['predicted_labels']]

            x_train_pred_scores += [preds['%s-%s-' % (self.prefix, self.train_scenario)]['predicted_scores']]
            x_devel_pred_scores += [preds['%s-%s-' % (self.prefix, self.train_scenario)]['predicted_scores']]
            x_test_pred_scores += [preds['%s-%s-test_grand' % (self.prefix, self.train_scenario)]['predicted_scores']]

            # -- the ground-truth for all models are iquals, so it's necessary to get only the last one
            y_train = [preds['%s-%s-' % (self.prefix, self.train_scenario)]['gt']]
            y_devel = [preds['%s-%s-' % (self.prefix, self.train_scenario)]['gt']]
            y_test = [preds['%s-%s-test_grand' % (self.prefix, self.train_scenario)]['gt']]

        x_train_pred_scores = np.array(x_train_pred_scores)
        x_devel_pred_scores = np.array(x_devel_pred_scores)
        x_test_pred_scores = np.array(x_test_pred_scores)

        x_train_pred_labels = np.array(x_train_pred_labels)
        x_devel_pred_labels = np.array(x_devel_pred_labels)
        x_test_pred_labels = np.array(x_test_pred_labels)

        if 'scores' in self.meta_classification_from:
            x_train = x_train_pred_scores.transpose()
            x_devel = x_devel_pred_scores.transpose()
            x_test = x_test_pred_scores.transpose()
        else:
            x_train = x_train_pred_labels.transpose()
            x_devel = x_devel_pred_labels.transpose()
            x_test = x_test_pred_labels.transpose()

        # if self.n_models != -1:
        #
        #     # -- selecting the most promissing models
        #     print('-- selection_algo', self.selection_algo, flush=True)
        #
        #     if self.selection_algo == 0:
        #         selected_models_idxs = self.select_classifiers_by_feature_importance(n_models=self.n_models)
        #
        #         x_train = x_train[:, selected_models_idxs]
        #         x_devel = x_devel[:, selected_models_idxs]
        #         x_test = x_test[:, selected_models_idxs]
        #
        #         selected_models = {}
        #         for a, fpred in enumerate(np.array(self.predictions_files)[selected_models_idxs]):
        #             selected_models.update({'%d'% a: fpred})
        #             print(a, fpred, flush=True)
        #
        #         json_fname = os.path.join(self.output_path, 'selected_models_for_fusion.json')
        #         with open(json_fname, mode='w') as f:
        #             print("-- saving json file:", json_fname)
        #             sys.stdout.flush()
        #             f.write(json.dumps(selected_models, indent=4))
        #
        #     elif self.selection_algo == 1:
        #         selected_models_idxs = self.find_complementary_models(x_train_pred_labels, x_train_pred_scores,
        #                                                               n_models=self.n_models)
        #
        #         x_train = x_train[:, selected_models_idxs]
        #         x_devel = x_devel[:, selected_models_idxs]
        #         x_test = x_test[:, selected_models_idxs]
        #
        #         selected_models = {}
        #         for a, fpred in enumerate(np.array(self.predictions_files)[selected_models_idxs]):
        #             selected_models.update({'%d'% a: fpred})
        #             print(a, fpred, flush=True)
        #
        #         json_fname = os.path.join(self.output_path, 'selected_models_for_fusion.json')
        #         with open(json_fname, mode='w') as f:
        #             print("-- saving json file:", json_fname)
        #             sys.stdout.flush()
        #             f.write(json.dumps(selected_models, indent=4))
        #
        #     else:
        #         pass

        y_train = np.array(y_train).flatten()
        y_devel = np.array(y_devel).flatten()
        y_test = np.array(y_test).flatten()

        print('-- x_train', x_train.shape)
        print('-- y_train', y_train.shape)

        print('-- x_devel', x_devel.shape)
        print('-- y_devel', y_devel.shape)

        print('-- x_test', x_test.shape)
        print('-- y_test', y_test.shape)

        if 'scores' in self.meta_classification_from:
            x_train_scale = preprocessing.MinMaxScaler().fit(x_train)

            x_train = x_train_scale.transform(x_train)
            x_devel = x_train_scale.transform(x_devel)
            x_test = x_train_scale.transform(x_test)

        # # -- saving for further analysis using other libs
        # dump_svmlight_file(x_train, y_train, os.path.join(self.output_path, 'x_train.txt'))
        # dump_svmlight_file(x_test, y_test, os.path.join(self.output_path, 'x_test.txt'))

        meta_predictions = {}

        # -- start the training process
        self.training(x_train, y_train, x_test)

        # # -- compute the predicted scores and labels for the training set
        # meta_predictions['%s-%s-' % (self.prefix, self.train_scenario)] = self.testing(x_train, y_train,
        #                                                                   x_test_scores=x_train_pred_scores)

        # -- compute the predicted scores and labels for the training set
        meta_predictions['%s-%s-' % (self.prefix, self.train_scenario)] = self.testing(x_devel, y_devel,
                                                                          x_test_scores=x_train_pred_scores)

        # -- compute the predicted scores and labels for the testing sets
        meta_predictions['%s-%s-test_grand' % (self.prefix, self.train_scenario)] = self.testing(x_test, y_test,
                                                                                               x_test_scores=x_test_pred_scores)

        prefix = ''
        train_scenarios_idxs, devel_scenarios_idxs, test_scenarios_idxs = [], [], []

        for key in meta_predictions:
            split_key = key.split('-')

            prefix = split_key[0]
            train_scenarios_idxs += [split_key[1]]
            devel_scenarios_idxs += [split_key[2]]
            test_scenarios_idxs += [split_key[3]]

        train_scenarios_idxs = set(train_scenarios_idxs)
        devel_scenarios_idxs = set(devel_scenarios_idxs)
        test_scenarios_idxs = set(test_scenarios_idxs)
        test_scenarios_idxs.remove('')

        # -- estimating the performance of the classifier
        class_report = self.performance_evaluation(meta_predictions, train_scenarios_idxs, devel_scenarios_idxs, test_scenarios_idxs,
                                                   prefix=prefix)

        # -- saving the performance results
        self.save_performance_results(class_report)

        classification_results_summary(class_report)

        # # -- saving the raw scores
        # self._save_raw_scores(self.dataset.meta_info['all_fnames'], dataset_protocol, meta_predictions)
        # 
        # # -- saving the interesting samples for further analysis
        # self.interesting_samples(self.dataset.meta_info['all_fnames'], dataset_protocol['test_set'], class_report, meta_predictions,
        #                          threshold_type='EER')

    @abstractmethod
    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        return NotImplemented

    @abstractmethod
    def testing(self, x_test, y_test, x_test_scores=None):
        return NotImplemented
