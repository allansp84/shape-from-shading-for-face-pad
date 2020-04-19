# -*- coding: utf-8 -*-

import json

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.metaclassification.basemetaclassifier import BaseMetaClassifier


class MajorityVoteFusion(BaseMetaClassifier):
    """
    """

    def __init__(self, output_path, predictions_files, meta_classification_from, n_models,
                 selection_algo=0, compute_feature_importance=False,
                 frames_per_video=10, frame_fusion_type='max',
                 prefix='inter',
                 force_train=False):

        super(MajorityVoteFusion, self).__init__(output_path, predictions_files, meta_classification_from, n_models,
                                            selection_algo=selection_algo, compute_feature_importance=compute_feature_importance,
                                            frames_per_video=frames_per_video, frame_fusion_type=frame_fusion_type,
                                            prefix=prefix)

        self.verbose = True
        self.output_path = output_path

        self.frames_per_video = frames_per_video
        self.frame_fusion_type = frame_fusion_type

        self.prefix = prefix
        self.force_train = force_train

        self.output_model = os.path.join(self.output_path, "meta_svm_classifier_model.pkl")
        self.output_weights = os.path.join(self.output_path, "weights.hdf5")
        self.model = None

    def training(self, x_train, y_train, x_validation=None, y_validation=None):
        """

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (numpy.ndarray, optional): Testing data. Defaults to None.
            y_validation (numpy.ndarray, optional): Labels of the testing data. Defaults to None.

        """
        pass

    def testing(self, x_test, y_test, x_test_scores=None):
        """ This method is responsible for testing the fitted model.

        Args:
            x_test (numpy.ndarray): Testing data
            y_test (numpy.ndarray): Labels of the Testing data

        Returns:
            dict: A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}

        """

        # predicted_labels = x_test.max(axis=1)

        predicted_labels = x_test.mean(axis=1)
        predicted_labels[np.where(predicted_labels >= 0.5)[0]] = 1.
        predicted_labels[np.where(predicted_labels < 0.5)[0]] = 0.

        neg_idxs = np.where(predicted_labels == 0)[0]
        pos_idxs = np.where(predicted_labels == 1)[0]

        predicted_scores = np.zeros(predicted_labels.shape, dtype=np.float32)

        predicted_scores[neg_idxs] = x_test_scores[:, neg_idxs].min(axis=0)
        predicted_scores[pos_idxs] = x_test_scores[:, pos_idxs].max(axis=0)

        # -- define the output dictionary
        r_dict = {'gt': y_test,
                  'predicted_labels': predicted_labels,
                  'predicted_scores': predicted_scores,
                  }

        return r_dict
