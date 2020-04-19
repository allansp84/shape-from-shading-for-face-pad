# -*- coding: utf-8 -*-

import operator
import itertools
from glob import glob
from functools import reduce
from antispoofing.sfsnet.datasets.dataset import Dataset
from antispoofing.sfsnet.utils import *


class Casia(Dataset):

    def __init__(self, dataset_path, output_path='./working', face_locations_path=None, file_types=('.avi',),
                 operation='crop', max_axis='200', n_channel=3, frame_offset=0, total_n_frames=-1, protocol_id=0):

        super(Casia, self).__init__(dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                                    n_channel, frame_offset, total_n_frames)

    def build_meta(self, inpath, filetypes, frame_numbers=1):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        train_idxs_scenario_1 = []
        train_idxs_scenario_2 = []
        train_idxs_scenario_3 = []
        train_idxs_scenario_4 = []
        train_idxs_scenario_5 = []
        train_idxs_scenario_6 = []

        test_idxs_scenario_1 = []
        test_idxs_scenario_2 = []
        test_idxs_scenario_3 = []
        test_idxs_scenario_4 = []
        test_idxs_scenario_5 = []
        test_idxs_scenario_6 = []
        test_idxs_scenario_7 = []

        scenario_1 = {"L1", "L2", "L3", "L4"}
        scenario_2 = {"N1", "N2", "N3", "N4"}
        scenario_3 = {"H1", "H2", "H3", "H4"}
        scenario_4 = {"L1", "N1", "H1", "L2", "N2", "H2"}
        scenario_5 = {"L1", "N1", "H1", "L3", "N3", "H3"}
        scenario_6 = {"L1", "N1", "H1", "L4", "N4", "H4"}

        pos_samples = ["1", "2", "HR_1"]
        pos_samples = ["{0}{1}".format(s, file_type) for file_type in filetypes for s in pos_samples]

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))
        train_users_idxs = {}

        for i, folder in enumerate(folders):

            if 'train_release/' in folder or 'test_release/' in folder:

                fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
                fnames = sorted(list(itertools.chain.from_iterable(fnames)))

                for fname in fnames:

                    # -- hack to handle with feature extracted by video
                    if '.avi' in os.path.splitext(fname)[1]:
                        filename = os.path.basename(fname)
                    else:
                        filename = "{0}{1}".format(os.path.split(fname)[0], filetypes[0])
                        filename = os.path.basename(filename)

                    positive_class = [s == filename for s in pos_samples]
                    name_video = os.path.splitext(filename)[0]

                    if 'H' in name_video:
                        video_id = 'H{0}'.format(name_video.split("_")[1])
                    else:
                        if int(name_video) % 2 == 0:
                            idx = np.where(np.arange(2, 9, 2) == int(name_video))[0]
                            idx += 1
                            video_id = 'L%d' % idx[0]

                        else:
                            idx = np.where(np.arange(1, 9, 2) == int(name_video))[0]
                            idx += 1
                            video_id = 'N%d' % idx[0]

                    if 'train_release/' in os.path.relpath(fname, inpath):
                        all_fnames += [fname]
                        all_labels += [int(reduce(operator.or_, positive_class))]
                        all_idxs += [img_idx]
                        train_idxs += [img_idx]

                        if video_id in scenario_1:
                            train_idxs_scenario_1 += [img_idx]
                        if video_id in scenario_2:
                            train_idxs_scenario_2 += [img_idx]
                        if video_id in scenario_3:
                            train_idxs_scenario_3 += [img_idx]
                        if video_id in scenario_4:
                            train_idxs_scenario_4 += [img_idx]
                        if video_id in scenario_5:
                            train_idxs_scenario_5 += [img_idx]
                        if video_id in scenario_6:
                            train_idxs_scenario_6 += [img_idx]

                        img_idx += 1

                    elif 'test_release/':
                        if video_id in scenario_1:
                            test_idxs_scenario_1 += [img_idx]
                        if video_id in scenario_2:
                            test_idxs_scenario_2 += [img_idx]
                        if video_id in scenario_3:
                            test_idxs_scenario_3 += [img_idx]
                        if video_id in scenario_4:
                            test_idxs_scenario_4 += [img_idx]
                        if video_id in scenario_5:
                            test_idxs_scenario_5 += [img_idx]
                        if video_id in scenario_6:
                            test_idxs_scenario_6 += [img_idx]
                        test_idxs_scenario_7 += [img_idx]

                        all_fnames += [fname]
                        all_labels += [int(reduce(operator.or_, positive_class))]
                        all_idxs += [img_idx]

                        img_idx += 1
                    else:
                        pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        train_idxs = np.array(train_idxs)
        train_idxs_scenario_1 = np.array(train_idxs_scenario_1)
        train_idxs_scenario_2 = np.array(train_idxs_scenario_2)
        train_idxs_scenario_3 = np.array(train_idxs_scenario_3)
        train_idxs_scenario_4 = np.array(train_idxs_scenario_4)
        train_idxs_scenario_5 = np.array(train_idxs_scenario_5)
        train_idxs_scenario_6 = np.array(train_idxs_scenario_6)

        # pos_test_idxs = np.where(all_labels[test_idxs_scenario_7] == 1)[0]

        test_idxs_scenario_1 = np.array(test_idxs_scenario_1)
        test_idxs_scenario_2 = np.array(test_idxs_scenario_2)
        test_idxs_scenario_3 = np.array(test_idxs_scenario_3)
        test_idxs_scenario_4 = np.array(test_idxs_scenario_4)
        test_idxs_scenario_5 = np.array(test_idxs_scenario_5)
        test_idxs_scenario_6 = np.array(test_idxs_scenario_6)
        test_idxs_scenario_7 = np.array(test_idxs_scenario_7)

        # train_idxs = self.prune_set(all_labels[train_idxs], train_idxs)
        # train_idxs_scenario_1 = self.prune_set(all_labels[train_idxs_scenario_1], train_idxs_scenario_1)
        # train_idxs_scenario_2 = self.prune_set(all_labels[train_idxs_scenario_2], train_idxs_scenario_2)
        # train_idxs_scenario_3 = self.prune_set(all_labels[train_idxs_scenario_3], train_idxs_scenario_3)
        # train_idxs_scenario_4 = self.prune_set(all_labels[train_idxs_scenario_4], train_idxs_scenario_4)
        # train_idxs_scenario_5 = self.prune_set(all_labels[train_idxs_scenario_5], train_idxs_scenario_5)
        # train_idxs_scenario_6 = self.prune_set(all_labels[train_idxs_scenario_6], train_idxs_scenario_6)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'train_users_idxs': train_users_idxs,
                  'devel_idxs': train_idxs,
                  'test_all_idxs': test_idxs_scenario_7,
                  'train_scenarios_idxs': {
                                           'train_grand': train_idxs,
                                           # 'train_scenario_1': train_idxs_scenario_1,
                                           # 'train_scenario_2': train_idxs_scenario_2,
                                           # 'train_scenario_3': train_idxs_scenario_3,

                                           # 'train_scenario_4': train_idxs_scenario_4,
                                           # 'train_scenario_5': train_idxs_scenario_5,
                                           # 'train_scenario_6': train_idxs_scenario_6,
                                           },
                  'devel_scenarios_idxs': {
                                           'devel_grand': train_idxs,
                                           # 'devel_scenario_1': train_idxs_scenario_1,
                                           # 'devel_scenario_2': train_idxs_scenario_2,
                                           # 'devel_scenario_3': train_idxs_scenario_3,

                                           'devel_scenario_4': train_idxs_scenario_4,
                                           'devel_scenario_5': train_idxs_scenario_5,
                                           'devel_scenario_6': train_idxs_scenario_6,
                                           },

                  'test_scenarios_idxs': {'test_grand': test_idxs_scenario_7,
                                          # 'test_scenario_1': test_idxs_scenario_1,
                                          # 'test_scenario_2': test_idxs_scenario_2,
                                          # 'test_scenario_3': test_idxs_scenario_3,

                                          'test_scenario_4': test_idxs_scenario_4,
                                          'test_scenario_5': test_idxs_scenario_5,
                                          'test_scenario_6': test_idxs_scenario_6,

                                          # 'test_scenario_7': test_idxs_scenario_7,
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

        # train_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}
        train_scenario_set = {}
        for train_id in train_scenario_idxs:
            train_scenario_set[train_id] = {'idxs': train_scenario_idxs[train_id], 'id': train_id,}

        devel_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'idxs': test_idxs[test_id], }
        dataset_protocol = {'train_set': train_scenario_set, 'devel_set': devel_set, 'test_set': test_set}

        # # train_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}
        # train_scenario_set = {}
        # for train_id in train_scenario_idxs:
        #     train_scenario_set[train_id] = {'fnames': all_fnames[train_scenario_idxs[train_id]],
        #                                     'labels': all_labels[train_scenario_idxs[train_id]],
        #                                     'idxs': train_scenario_idxs[train_id],
        #                                     'id': train_id,
        #                                     }
        #
        # devel_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}
        #
        # test_set = {}
        # for test_id in test_scenarios_idxs:
        #     if test_scenarios_idxs[test_id].size:
        #         test_set[test_id] = {'fnames': all_fnames[test_scenarios_idxs[test_id]],
        #                              'labels': all_labels[test_scenarios_idxs[test_id]],
        #                              'idxs': test_scenarios_idxs[test_id],
        #                              }
        # dataset_protocol = {'train_set': train_scenario_set, 'devel_set': devel_set, 'test_set': test_set}

        return self.meta_info, dataset_protocol
