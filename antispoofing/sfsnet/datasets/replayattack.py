# -*- coding: utf-8 -*-

import itertools
from glob import glob
from antispoofing.sfsnet.datasets.dataset import Dataset
from antispoofing.sfsnet.utils import *


class ReplayAttack(Dataset):

    def __init__(self, dataset_path, output_path='./working', face_locations_path=None, file_types=('.mov',),
                 operation='crop', max_axis='200', n_channel=3, frame_offset=0, total_n_frames=-1, protocol_id=0):

        super(ReplayAttack, self).__init__(dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                                           n_channel, frame_offset, total_n_frames)

    def build_meta(self, inpath, filetypes, frame_numbers=1):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        devel_idxs = []
        test_idxs = []
        anon_idxs = []

        train_attack_highdef_idxs = []
        train_attack_mobile_idxs = []
        train_attack_print_idxs = []
        train_attack_fixed_idxs = []
        train_attack_hand_idxs = []

        devel_attack_highdef_idxs = []
        devel_attack_mobile_idxs = []
        devel_attack_print_idxs = []
        devel_attack_fixed_idxs = []
        devel_attack_hand_idxs = []

        test_attack_highdef_idxs = []
        test_attack_mobile_idxs = []
        test_attack_print_idxs = []
        test_attack_fixed_idxs = []
        test_attack_hand_idxs = []

        train_users_idxs = {}

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        for i, folder in enumerate(folders):

            if 'train/' in folder or 'devel/' in folder or 'test/' in folder:

                fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
                fnames = sorted(list(itertools.chain.from_iterable(fnames)))

                for fname in fnames:

                    if 'enroll' not in fname and 'competition_icb2013' not in fname:

                        if 'train/' in os.path.relpath(fname, inpath):
                            all_idxs += [img_idx]
                            train_idxs += [img_idx]
                            all_fnames += [fname]
                            all_labels += [int('real/' in (os.path.relpath(fname, inpath)))]
                            img_idx += 1

                            if 'attack_highdef' in os.path.relpath(fname, inpath):
                                train_attack_highdef_idxs += [img_idx]
                            elif 'attack_mobile' in os.path.relpath(fname, inpath):
                                train_attack_mobile_idxs += [img_idx]
                            elif 'attack_print' in os.path.relpath(fname, inpath):
                                train_attack_print_idxs += [img_idx]
                            else:
                                pass

                            if 'attack/fixed' in os.path.relpath(fname, inpath):
                                train_attack_fixed_idxs += [img_idx]
                            elif 'attack/hand' in os.path.relpath(fname, inpath):
                                train_attack_hand_idxs += [img_idx]
                            else:
                                pass

                        else:
                            if 'devel/' in os.path.relpath(fname, inpath):
                                all_idxs += [img_idx]
                                devel_idxs += [img_idx]
                                all_fnames += [fname]
                                all_labels += [int('real/' in (os.path.relpath(fname, inpath)))]
                                img_idx += 1

                                if 'attack_highdef' in os.path.relpath(fname, inpath):
                                    devel_attack_highdef_idxs += [img_idx]
                                elif 'attack_mobile' in os.path.relpath(fname, inpath):
                                    devel_attack_mobile_idxs += [img_idx]
                                elif 'attack_print' in os.path.relpath(fname, inpath):
                                    devel_attack_print_idxs += [img_idx]
                                else:
                                    pass

                                if 'attack/fixed' in os.path.relpath(fname, inpath):
                                    devel_attack_fixed_idxs += [img_idx]
                                elif 'attack/hand' in os.path.relpath(fname, inpath):
                                    devel_attack_hand_idxs += [img_idx]
                                else:
                                    pass

                            elif 'test/' in os.path.relpath(fname, inpath):

                                if 'attack_highdef' in os.path.relpath(fname, inpath):
                                    test_attack_highdef_idxs += [img_idx]
                                elif 'attack_mobile' in os.path.relpath(fname, inpath):
                                    test_attack_mobile_idxs += [img_idx]
                                elif 'attack_print' in os.path.relpath(fname, inpath):
                                    test_attack_print_idxs += [img_idx]
                                else:
                                    pass

                                if 'attack/fixed' in os.path.relpath(fname, inpath):
                                    test_attack_fixed_idxs += [img_idx]
                                elif 'attack/hand' in os.path.relpath(fname, inpath):
                                    test_attack_hand_idxs += [img_idx]
                                else:
                                    pass

                                all_idxs += [img_idx]
                                test_idxs += [img_idx]
                                all_fnames += [fname]
                                all_labels += [int('real/' in (os.path.relpath(fname, inpath)))]
                                img_idx += 1

                            elif 'competition_icb2013/' in os.path.relpath(fname, inpath):
                                anon_idxs += [img_idx]
                                all_fnames += [fname]
                                all_labels += [int('real/' in (os.path.relpath(fname, inpath)))]
                                img_idx += 1

                            else:
                                pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        train_idxs = np.array(train_idxs)
        devel_idxs = np.array(devel_idxs)
        test_idxs = np.array(test_idxs)

        # -- check if the training and testing sets are disjoint.
        try:
            assert not np.intersect1d(all_fnames[train_idxs], all_fnames[test_idxs]).size
        except AssertionError:
            raise Exception('The training and testing sets are mixed')

        pos_idxs = np.where(all_labels[test_idxs] == 1)[0]

        test_attack_hand_idxs = np.sort(np.concatenate((test_attack_hand_idxs, test_idxs[pos_idxs])))
        test_attack_fixed_idxs = np.sort(np.concatenate((test_attack_fixed_idxs, test_idxs[pos_idxs])))

        test_attack_highdef_idxs = np.sort(np.concatenate((test_attack_highdef_idxs, test_idxs[pos_idxs])))
        test_attack_mobile_idxs = np.sort(np.concatenate((test_attack_mobile_idxs, test_idxs[pos_idxs])))
        test_attack_print_idxs = np.sort(np.concatenate((test_attack_print_idxs, test_idxs[pos_idxs])))

        for img_idx, fname in enumerate(all_fnames[train_idxs]):

            if '.mov' in os.path.splitext(fname)[1]:
                user_id = os.path.splitext(os.path.basename(fname))[0]

                if 'real/' in fname:
                    user_id = user_id.split('_')[0]
                else:
                    user_id = user_id.split('_')[2]
            else:
                user_id = os.path.splitext(os.path.basename(os.path.dirname(fname)))[0]
                if 'real/' in fname:
                    user_id = user_id.split('_')[0]
                else:
                    user_id = user_id.split('_')[2]

            try:
                train_users_idxs[user_id] += [img_idx]
            except KeyError:
                train_users_idxs[user_id] = [img_idx]

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'train_users_idxs': train_users_idxs,
                  'devel_idxs': devel_idxs,
                  'test_all_idxs': test_idxs,
                  'train_scenarios_idxs': {'train_grand': train_idxs,
                                           # 'train_attack_hand': train_attack_hand_idxs,
                                           # 'train_attack_fixed': train_attack_fixed_idxs,
                                           # 'train_attack_highdef': train_attack_highdef_idxs,
                                           # 'train_attack_mobile': train_attack_mobile_idxs,
                                           # 'train_attack_print': train_attack_print_idxs,
                                           },
                  'devel_scenarios_idxs': {'devel_grand': devel_idxs,
                                           # 'devel_attack_hand': devel_attack_hand_idxs,
                                           # 'devel_attack_fixed': devel_attack_fixed_idxs,
                                           # 'devel_attack_highdef': devel_attack_highdef_idxs,
                                           # 'devel_attack_mobile': devel_attack_mobile_idxs,
                                           # 'devel_attack_print': devel_attack_print_idxs,
                                           },
                  'test_scenarios_idxs': {'test_grand': test_idxs,
                                          # 'test_attack_hand': test_attack_hand_idxs,
                                          # 'test_attack_fixed': test_attack_fixed_idxs,
                                          # 'test_attack_highdef': test_attack_highdef_idxs,
                                          # 'test_attack_mobile': test_attack_mobile_idxs,
                                          # 'test_attack_print': test_attack_print_idxs,
                                          }
                  }

        return r_dict

    def protocol_eval(self, fold=0, n_fold=5, test_size=0.5):
        """ This method implement validation evaluation protocol for this dataset.

        Args:
            fold (int): This parameter is not used since this dataset already has the predefined subsets.
            n_fold (int): This parameter is not used since this dataset already has the predefined subsets.
            test_size (float): This parameter is not used since this dataset already has the predefined subsets.

        Returns:
            dict: A dictionary containing the training, development and testing sets.

        """

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        train_idxs = self.meta_info['train_idxs']
        test_idxs = self.meta_info['test_idxs']

        train_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        devel_set = {'fnames': all_fnames[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'fnames': all_fnames[test_idxs[test_id]],
                                     'labels': all_labels[test_idxs[test_id]],
                                     'idxs': test_idxs[test_id],
                                     }

        return {'train_set': train_set, 'devel_set': devel_set, 'test_set': test_set}
