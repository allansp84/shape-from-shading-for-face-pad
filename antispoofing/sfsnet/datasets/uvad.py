# -*- coding: utf-8 -*-

import operator
import itertools
from glob import glob
from functools import reduce
from antispoofing.sfsnet.datasets.dataset import Dataset
from antispoofing.sfsnet.utils import *


class UVAD(Dataset):

    def __init__(self, dataset_path, output_path='./working', face_locations_path=None, file_types=('.MOV', '.MP4',),
                 operation='crop', max_axis='200', n_channel=3, frame_offset=0, total_n_frames=-1, protocol_id=0):

        super(UVAD, self).__init__(dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                                    n_channel, frame_offset, total_n_frames)

    def sampling_training_set(self, train_idxs, all_labels, reduction_rate=0.2):

        print('-- sampling the training set', flush=True)

        # -- reduce the training set
        r_state = np.random.RandomState(7)

        train_pos_idxs = np.where(all_labels[train_idxs] == 1)[0]
        train_neg_idxs = np.where(all_labels[train_idxs] == 0)[0]

        n_samples_neg = int(len(train_neg_idxs) * reduction_rate)

        train_idxs_rand_neg = r_state.permutation(train_neg_idxs)[:n_samples_neg]

        strain_idxs = np.concatenate((train_idxs[train_idxs_rand_neg], train_idxs[train_pos_idxs]))

        return strain_idxs

    def split_data(self, all_labels, all_idxs, training_rate):

        rstate = np.random.RandomState(7)

        pos_idxs = np.where(all_labels[all_idxs] == 1)[0]
        neg_idxs = np.where(all_labels[all_idxs] == 0)[0]

        # -- cross dataset idxs
        n_samples_pos = int(len(all_idxs[pos_idxs]) * training_rate)
        n_samples_neg = int(len(all_idxs[neg_idxs]) * training_rate)

        rand_idxs_pos = rstate.permutation(all_idxs[pos_idxs])
        rand_idxs_neg = rstate.permutation(all_idxs[neg_idxs])

        train_idxs_rand_pos = rand_idxs_pos[:n_samples_pos]
        train_idxs_rand_neg = rand_idxs_neg[:n_samples_neg]
        test_idxs_rand_pos = rand_idxs_pos[n_samples_pos:]
        test_idxs_rand_neg = rand_idxs_neg[n_samples_neg:]

        train_idxs_for_cross = np.concatenate((train_idxs_rand_neg, train_idxs_rand_pos))
        devel_idxs_for_cross = np.concatenate((test_idxs_rand_neg, test_idxs_rand_pos))

        return train_idxs_for_cross, devel_idxs_for_cross

    def build_meta(self, inpath, filetypes, frame_numbers=1):

        img_idx = 0
        training_rate = 0.8

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        test_idxs = []

        canon_idxs = []
        kodac_idxs = []
        nikon_idxs = []
        olympus_idxs = []
        panasonic_idxs = []
        sony_idxs = []

        train_fnames = None
        file_name = os.path.join(self.output_path, 'train_fnames_sampling.txt')
        if os.path.exists(file_name):
            train_fnames = np.loadtxt(file_name, dtype=np.str, delimiter=',')

        cameras = {'sony': [], 'kodac': [], 'olympus': [], 'nikon': [], 'canon': [], 'panasonic': []}

        train_range = ['sony', 'kodac', 'olympus', ]
        test_range = ['nikon', 'canon', 'panasonic']

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        for i, folder in enumerate(folders):

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = sorted(list(itertools.chain.from_iterable(fnames)))

            for fname in fnames:

                rel_path = os.path.relpath(fname, inpath)

                valid_path = True
                if '.png' in os.path.splitext(fname)[1] and train_fnames is not None:
                    valid_path = os.path.dirname(rel_path) in train_fnames

                if valid_path:
                    class_name = rel_path.split('/')[0]
                    camera_name = rel_path.split('/')[1]

                    filename = os.path.basename(fname)
                    name_video = os.path.splitext(filename)[0]
                    video_number = int(''.join([i for i in name_video if i.isdigit()]))

                    if camera_name in train_range:

                        all_fnames += [fname]
                        all_labels += [int('real' in class_name)]
                        all_idxs += [img_idx]
                        train_idxs += [img_idx]
                        img_idx += 1

                        cameras[camera_name] += [img_idx]

                    elif camera_name in test_range:

                        all_fnames += [fname]
                        all_labels += [int('real' in class_name)]
                        all_idxs += [img_idx]
                        test_idxs += [img_idx]
                        img_idx += 1

                        cameras[camera_name] += [img_idx]

                    else:
                        pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)

        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

        # if '.png' not in os.path.splitext(all_fnames[0])[1]:
        #
        #     strain_idxs = self.sampling_training_set(train_idxs, all_labels)
        #
        #     # -- saving the fnames used for training
        #     train_fnames = np.array([os.path.splitext(os.path.relpath(fname, inpath))[0] for fname in all_fnames[strain_idxs]], dtype=np.str)
        #     test_fnames = np.array([os.path.splitext(os.path.relpath(fname, inpath))[0] for fname in all_fnames[test_idxs]], dtype=np.str)
        #
        #     sample_fnames = np.sort(np.concatenate((train_fnames, test_fnames)))
        #
        #     np.savetxt(file_name, sample_fnames, fmt='%s', delimiter=',')

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'train_users_idxs': {},
                  'devel_idxs': train_idxs,
                  'test_all_idxs': test_idxs,
                  'train_scenarios_idxs': {'train_grand': train_idxs,
                                           },
                  'devel_scenarios_idxs': {'devel_grand': train_idxs,
                                           },

                  'test_scenarios_idxs': {'test_grand': test_idxs,
                                          }
                  }

        return r_dict

    def protocol_eval(self):
        pass
