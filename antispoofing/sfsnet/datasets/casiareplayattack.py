# -*- coding: utf-8 -*-

import operator
import itertools
from glob import glob
from functools import reduce
from antispoofing.sfsnet.datasets.dataset import Dataset
from antispoofing.sfsnet.datasets.casia import Casia
from antispoofing.sfsnet.datasets.replayattack import ReplayAttack

from antispoofing.sfsnet.utils import *


class CasiaReplayAttack(Dataset):
    def __init__(self, dataset_path, output_path='./working', face_locations_path=None, file_types=('.avi', '.mov',),
                 operation='crop', max_axis='200', n_channel=3, frame_offset=0, total_n_frames=-1):

        super(CasiaReplayAttack, self).__init__(dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                                      n_channel, frame_offset, total_n_frames)

        self.casia = Casia(os.path.join(dataset_path, 'casia'),
                           output_path=output_path,
                           face_locations_path=os.path.join(dataset_path, 'casia', 'face-locations-v4'),
                           file_types=file_types,
                           operation=operation,
                           max_axis=max_axis,
                           n_channel=n_channel,
                           frame_offset=frame_offset,
                           total_n_frames=total_n_frames,
                           )

        self.ra = ReplayAttack(os.path.join(dataset_path, 'replayattack'),
                               output_path=output_path,
                               face_locations_path=os.path.join(dataset_path, 'casia', 'face-locations'),
                               file_types=file_types,
                               operation=operation,
                               max_axis=max_axis,
                               n_channel=n_channel,
                               frame_offset=frame_offset,
                               total_n_frames=total_n_frames,
                               )

    def build_meta(self, inpath, filetypes, frame_numbers=1):

        in_path_casia = inpath.replace(self.dataset_path.split("/")[-1], os.path.join(self.dataset_path.split("/")[-1], 'casia'))
        r_dict_casia = self.casia.build_meta(in_path_casia, filetypes, frame_numbers)

        in_path_ra = inpath.replace(self.dataset_path.split("/")[-1], os.path.join(self.dataset_path.split("/")[-1], 'replayattack'))
        r_dict_ra = self.ra.build_meta(in_path_ra, filetypes, frame_numbers)

        shift_idx = r_dict_casia['all_idxs'].max() + 1

        all_fnames = np.concatenate((r_dict_casia['all_fnames'], r_dict_ra['all_fnames']))
        all_labels = np.concatenate((r_dict_casia['all_labels'], r_dict_ra['all_labels']))
        all_idxs = np.concatenate((r_dict_casia['all_idxs'], r_dict_ra['all_idxs'] + shift_idx))

        train_idxs = np.concatenate((r_dict_casia['train_idxs'], r_dict_ra['train_idxs'] + shift_idx))
        devel_idxs = np.concatenate((r_dict_casia['devel_idxs'], r_dict_ra['devel_idxs'] + shift_idx))
        test_all_idxs = np.concatenate((r_dict_casia['test_all_idxs'], r_dict_ra['test_all_idxs'] + shift_idx))

        train_grand = np.concatenate((r_dict_casia['train_scenarios_idxs']['train_scenario_6'],
                                      r_dict_ra['train_scenarios_idxs']['train_grand'] + shift_idx))

        devel_grand = np.concatenate((r_dict_casia['devel_scenarios_idxs']['devel_scenario_6'],
                                      r_dict_ra['devel_scenarios_idxs']['devel_grand'] + shift_idx))

        test_grand = np.concatenate((r_dict_casia['test_scenarios_idxs']['test_grand'],
                                     r_dict_ra['test_scenarios_idxs']['test_grand'] + shift_idx))

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'devel_idxs': devel_idxs,
                  'test_all_idxs': test_all_idxs,
                  'train_scenarios_idxs': {'train_grand': train_grand,
                                           },
                  'devel_scenarios_idxs': {'devel_grand': devel_grand,
                                           },
                  'test_scenarios_idxs': {'test_grand': test_grand,
                                          }
                  }

        return r_dict

    def protocol_eval(self):
        """ This method implement validation evaluation protocol for this dataset.

        Args:
            meta_info (dict):

        Returns:
            dict: A dictionary containing the training, development and testing sets.

        """

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        train_idxs = self.meta_info['train_idxs']
        train_scenario_idxs = self.meta_info['train_scenario_idxs']
        test_idxs = self.meta_info['test_idxs']

        train_scenario_set = {}
        for train_id in train_scenario_idxs:
            train_scenario_set[train_id] = {'idxs': train_scenario_idxs[train_id], 'id': train_id, }

        devel_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'idxs': test_idxs[test_id], }
        dataset_protocol = {'train_set': train_scenario_set, 'devel_set': devel_set, 'test_set': test_set}

        return self.meta_info, dataset_protocol
