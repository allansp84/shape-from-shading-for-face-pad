# -*- coding: utf-8 -*-

import itertools
from glob import glob
from antispoofing.sfsnet.datasets.dataset import Dataset
from antispoofing.sfsnet.utils import *


class OuluNPU(Dataset):

    def __init__(self, dataset_path, output_path='./working', face_locations_path=None, file_types=('.avi',),
                 operation='crop', max_axis='200', n_channel=3, frame_offset=0, total_n_frames=-1,
                 file_type_face_annotation='.face', delimiter_face_annotation=" ", protocol_id=0):

        super(OuluNPU, self).__init__(dataset_path, output_path, face_locations_path, file_types, operation, max_axis,
                                      n_channel, frame_offset, total_n_frames)

    def build_meta(self, inpath, filetypes, frame_numbers=1):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        devel_idxs = []
        test_idxs = []

        all_user = []
        all_session = []
        all_phone = []
        all_type_of_attack = []

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        for i, folder in enumerate(folders):

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = sorted(list(itertools.chain.from_iterable(fnames)))

            for fname in fnames:

                rel_fname = os.path.relpath(fname, inpath)

                # -- hack to handle with feature extracted by video
                if '.avi' in os.path.splitext(fname)[1]:
                    filename = os.path.splitext(os.path.basename(rel_fname))[0]
                else:
                    filename = os.path.basename(os.path.dirname(rel_fname))

                phone, session, user, type_of_attack = filename.split('_')

                phone = int(phone)
                session = int(session)
                user = int(user)
                type_of_attack = int(type_of_attack)

                if 1 <= user <= 20:
                    all_idxs += [img_idx]
                    train_idxs += [img_idx]
                    all_fnames += [fname]
                    all_labels += [1 if type_of_attack == 1 else 0]
                    img_idx += 1

                elif 21 <= user <= 35:
                        all_idxs += [img_idx]
                        devel_idxs += [img_idx]
                        all_fnames += [fname]
                        all_labels += [1 if type_of_attack == 1 else 0]
                        img_idx += 1

                elif user >= 36:

                    all_idxs += [img_idx]
                    test_idxs += [img_idx]
                    all_fnames += [fname]
                    all_labels += [1 if type_of_attack == 1 else 0]
                    img_idx += 1

                else:
                    pass

                all_user += [user]
                all_session += [session]
                all_phone += [phone]
                all_type_of_attack += [type_of_attack]

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        all_user = np.array(all_user)
        all_session = np.array(all_session)
        all_phone = np.array(all_phone)
        all_type_of_attack = np.array(all_type_of_attack)

        session_3_idxs = np.where(all_session == 3)[0]

        phone_1_idxs = np.where(all_phone == 1)[0]
        phone_2_idxs = np.where(all_phone == 2)[0]
        phone_3_idxs = np.where(all_phone == 3)[0]
        phone_4_idxs = np.where(all_phone == 4)[0]
        phone_5_idxs = np.where(all_phone == 5)[0]
        phone_6_idxs = np.where(all_phone == 6)[0]

        print_attack_idxs = np.where((all_type_of_attack == 1)|(all_type_of_attack == 2)|(all_type_of_attack == 3))[0]
        replay_attack_idxs = np.where((all_type_of_attack == 1)|(all_type_of_attack == 4)|(all_type_of_attack == 5))[0]

        train_idxs = np.array(train_idxs)
        devel_idxs = np.array(devel_idxs)
        test_idxs = np.array(test_idxs)

        # -- defining protocol 0
        test_grand_print_attack_idxs = np.intersect1d(test_idxs, print_attack_idxs)
        test_grand_replay_attack_idxs = np.intersect1d(test_idxs, replay_attack_idxs)

        # -- defining protocol 1
        train_protocol_1_idxs = np.setdiff1d(train_idxs, session_3_idxs)
        devel_protocol_1_idxs = np.setdiff1d(devel_idxs, session_3_idxs)
        test_protocol_1_idxs = np.intersect1d(test_idxs, session_3_idxs)
        test_protocol_1_print_attack_idxs = np.intersect1d(test_protocol_1_idxs, print_attack_idxs)
        test_protocol_1_replay_attack_idxs = np.intersect1d(test_protocol_1_idxs, replay_attack_idxs)

        # -- defining protocol 2
        type_of_attack_1_2_4_idxs = np.where((all_type_of_attack == 1)|(all_type_of_attack == 2)|(all_type_of_attack == 4))[0]
        type_of_attack_1_3_5_idxs = np.where((all_type_of_attack == 1)|(all_type_of_attack == 3)|(all_type_of_attack == 5))[0]

        train_protocol_2_idxs = np.intersect1d(train_idxs, type_of_attack_1_2_4_idxs)
        devel_protocol_2_idxs = np.intersect1d(devel_idxs, type_of_attack_1_2_4_idxs)
        test_protocol_2_idxs = np.intersect1d(test_idxs, type_of_attack_1_3_5_idxs)
        test_protocol_2_print_attack_idxs = np.intersect1d(test_protocol_2_idxs, print_attack_idxs)
        test_protocol_2_replay_attack_idxs = np.intersect1d(test_protocol_2_idxs, replay_attack_idxs)

        # -- defining protocol 3
        train_protocol_3_phone_1_idxs = np.setdiff1d(train_idxs, phone_1_idxs)
        devel_protocol_3_phone_1_idxs = np.setdiff1d(devel_idxs, phone_1_idxs)
        test_protocol_3_phone_1_idxs = np.intersect1d(test_idxs, phone_1_idxs)
        test_protocol_3_phone_1_print_attack_idxs = np.intersect1d(test_protocol_3_phone_1_idxs, print_attack_idxs)
        test_protocol_3_phone_1_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_1_idxs, replay_attack_idxs)

        train_protocol_3_phone_2_idxs = np.setdiff1d(train_idxs, phone_2_idxs)
        devel_protocol_3_phone_2_idxs = np.setdiff1d(devel_idxs, phone_2_idxs)
        test_protocol_3_phone_2_idxs = np.intersect1d(test_idxs, phone_2_idxs)
        test_protocol_3_phone_2_print_attack_idxs = np.intersect1d(test_protocol_3_phone_2_idxs, print_attack_idxs)
        test_protocol_3_phone_2_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_2_idxs, replay_attack_idxs)

        train_protocol_3_phone_3_idxs = np.setdiff1d(train_idxs, phone_3_idxs)
        devel_protocol_3_phone_3_idxs = np.setdiff1d(devel_idxs, phone_3_idxs)
        test_protocol_3_phone_3_idxs = np.intersect1d(test_idxs, phone_3_idxs)
        test_protocol_3_phone_3_print_attack_idxs = np.intersect1d(test_protocol_3_phone_3_idxs, print_attack_idxs)
        test_protocol_3_phone_3_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_3_idxs, replay_attack_idxs)

        train_protocol_3_phone_4_idxs = np.setdiff1d(train_idxs, phone_4_idxs)
        devel_protocol_3_phone_4_idxs = np.setdiff1d(devel_idxs, phone_4_idxs)
        test_protocol_3_phone_4_idxs = np.intersect1d(test_idxs, phone_4_idxs)
        test_protocol_3_phone_4_print_attack_idxs = np.intersect1d(test_protocol_3_phone_4_idxs, print_attack_idxs)
        test_protocol_3_phone_4_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_4_idxs, replay_attack_idxs)

        train_protocol_3_phone_5_idxs = np.setdiff1d(train_idxs, phone_5_idxs)
        devel_protocol_3_phone_5_idxs = np.setdiff1d(devel_idxs, phone_5_idxs)
        test_protocol_3_phone_5_idxs = np.intersect1d(test_idxs, phone_5_idxs)
        test_protocol_3_phone_5_print_attack_idxs = np.intersect1d(test_protocol_3_phone_5_idxs, print_attack_idxs)
        test_protocol_3_phone_5_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_5_idxs, replay_attack_idxs)

        train_protocol_3_phone_6_idxs = np.setdiff1d(train_idxs, phone_6_idxs)
        devel_protocol_3_phone_6_idxs = np.setdiff1d(devel_idxs, phone_6_idxs)
        test_protocol_3_phone_6_idxs = np.intersect1d(test_idxs, phone_6_idxs)
        test_protocol_3_phone_6_print_attack_idxs = np.intersect1d(test_protocol_3_phone_6_idxs, print_attack_idxs)
        test_protocol_3_phone_6_replay_attack_idxs = np.intersect1d(test_protocol_3_phone_6_idxs, replay_attack_idxs)

        # -- defining protocol 4
        train_protocol_4_phone_1_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_1_idxs)
        devel_protocol_4_phone_1_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_1_idxs)
        test_protocol_4_phone_1_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_1_idxs)
        test_protocol_4_phone_1_print_attack_idxs = np.intersect1d(test_protocol_4_phone_1_idxs, print_attack_idxs)
        test_protocol_4_phone_1_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_1_idxs, replay_attack_idxs)

        train_protocol_4_phone_2_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_2_idxs)
        devel_protocol_4_phone_2_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_2_idxs)
        test_protocol_4_phone_2_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_2_idxs)
        test_protocol_4_phone_2_print_attack_idxs = np.intersect1d(test_protocol_4_phone_2_idxs, print_attack_idxs)
        test_protocol_4_phone_2_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_2_idxs, replay_attack_idxs)

        train_protocol_4_phone_3_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_3_idxs)
        devel_protocol_4_phone_3_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_3_idxs)
        test_protocol_4_phone_3_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_3_idxs)
        test_protocol_4_phone_3_print_attack_idxs = np.intersect1d(test_protocol_4_phone_3_idxs, print_attack_idxs)
        test_protocol_4_phone_3_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_3_idxs, replay_attack_idxs)

        train_protocol_4_phone_4_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_4_idxs)
        devel_protocol_4_phone_4_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_4_idxs)
        test_protocol_4_phone_4_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_4_idxs)
        test_protocol_4_phone_4_print_attack_idxs = np.intersect1d(test_protocol_4_phone_4_idxs, print_attack_idxs)
        test_protocol_4_phone_4_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_4_idxs, replay_attack_idxs)

        train_protocol_4_phone_5_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_5_idxs)
        devel_protocol_4_phone_5_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_5_idxs)
        test_protocol_4_phone_5_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_5_idxs)
        test_protocol_4_phone_5_print_attack_idxs = np.intersect1d(test_protocol_4_phone_5_idxs, print_attack_idxs)
        test_protocol_4_phone_5_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_5_idxs, replay_attack_idxs)

        train_protocol_4_phone_6_idxs = np.intersect1d(np.intersect1d(train_protocol_1_idxs, train_protocol_2_idxs), train_protocol_3_phone_6_idxs)
        devel_protocol_4_phone_6_idxs = np.intersect1d(np.intersect1d(devel_protocol_1_idxs, devel_protocol_2_idxs), devel_protocol_3_phone_6_idxs)
        test_protocol_4_phone_6_idxs = np.intersect1d(np.intersect1d(test_protocol_1_idxs, test_protocol_2_idxs), test_protocol_3_phone_6_idxs)
        test_protocol_4_phone_6_print_attack_idxs = np.intersect1d(test_protocol_4_phone_6_idxs, print_attack_idxs)
        test_protocol_4_phone_6_replay_attack_idxs = np.intersect1d(test_protocol_4_phone_6_idxs, replay_attack_idxs)

        train_scenarios_idxs = {}
        devel_scenarios_idxs = {}
        test_scenarios_idxs = {}

        if self.protocol_id == 0:
            train_scenarios_idxs = {'train_grand': train_idxs,
                                    }

            devel_scenarios_idxs = {'devel_grand': devel_idxs,
                                    }

            test_scenarios_idxs = {'test_grand': test_idxs,
                                   # 'test_grand_print_attack_idxs': test_grand_print_attack_idxs,
                                   # 'test_grand_replay_attack_idxs': test_grand_replay_attack_idxs,
                                   }

        elif self.protocol_id == 1:
            train_scenarios_idxs = {'train_protocol_1_idxs': train_protocol_1_idxs,
                                    }

            devel_scenarios_idxs = {'devel_protocol_1_idxs': devel_protocol_1_idxs,
                                    }

            test_scenarios_idxs = {'test_protocol_1_idxs': test_protocol_1_idxs,
                                   # 'test_protocol_1_print_attack_idxs': test_protocol_1_print_attack_idxs,
                                   # 'test_protocol_1_replay_attack_idxs': test_protocol_1_replay_attack_idxs,
                                   }

        elif self.protocol_id == 2:
            train_scenarios_idxs = {'train_protocol_2_idxs': train_protocol_2_idxs,
                                    }

            devel_scenarios_idxs = {'devel_protocol_2_idxs': devel_protocol_2_idxs,
                                    }

            test_scenarios_idxs = {'test_protocol_2_idxs': test_protocol_2_idxs,
                                   # 'test_protocol_2_print_attack_idxs': test_protocol_2_print_attack_idxs,
                                   # 'test_protocol_2_replay_attack_idxs': test_protocol_2_replay_attack_idxs,
                                   }

        elif self.protocol_id == 3:
            train_scenarios_idxs = {'train_protocol_3_phone_1_idxs': train_protocol_3_phone_1_idxs,
                                    'train_protocol_3_phone_2_idxs': train_protocol_3_phone_2_idxs,
                                    'train_protocol_3_phone_3_idxs': train_protocol_3_phone_3_idxs,
                                    'train_protocol_3_phone_4_idxs': train_protocol_3_phone_4_idxs,
                                    'train_protocol_3_phone_5_idxs': train_protocol_3_phone_5_idxs,
                                    'train_protocol_3_phone_6_idxs': train_protocol_3_phone_6_idxs,
                                    }

            devel_scenarios_idxs = {'devel_protocol_3_phone_1_idxs': devel_protocol_3_phone_1_idxs,
                                    'devel_protocol_3_phone_2_idxs': devel_protocol_3_phone_2_idxs,
                                    'devel_protocol_3_phone_3_idxs': devel_protocol_3_phone_3_idxs,
                                    'devel_protocol_3_phone_4_idxs': devel_protocol_3_phone_4_idxs,
                                    'devel_protocol_3_phone_5_idxs': devel_protocol_3_phone_5_idxs,
                                    'devel_protocol_3_phone_6_idxs': devel_protocol_3_phone_6_idxs,
                                    }

            test_scenarios_idxs = {'test_protocol_3_phone_1_idxs': test_protocol_3_phone_1_idxs,
                                   # 'test_protocol_3_phone_1_print_attack_idxs': test_protocol_3_phone_1_print_attack_idxs,
                                   # 'test_protocol_3_phone_1_replay_attack_idxs': test_protocol_3_phone_1_replay_attack_idxs,
                                   'test_protocol_3_phone_2_idxs': test_protocol_3_phone_2_idxs,
                                   # 'test_protocol_3_phone_2_print_attack_idxs': test_protocol_3_phone_2_print_attack_idxs,
                                   # 'test_protocol_3_phone_2_replay_attack_idxs': test_protocol_3_phone_2_replay_attack_idxs,
                                   'test_protocol_3_phone_3_idxs': test_protocol_3_phone_3_idxs,
                                   # 'test_protocol_3_phone_3_print_attack_idxs': test_protocol_3_phone_3_print_attack_idxs,
                                   # 'test_protocol_3_phone_3_replay_attack_idxs': test_protocol_3_phone_3_replay_attack_idxs,
                                   'test_protocol_3_phone_4_idxs': test_protocol_3_phone_4_idxs,
                                   # 'test_protocol_3_phone_4_print_attack_idxs': test_protocol_3_phone_4_print_attack_idxs,
                                   # 'test_protocol_3_phone_4_replay_attack_idxs': test_protocol_3_phone_4_replay_attack_idxs,
                                   'test_protocol_3_phone_5_idxs': test_protocol_3_phone_5_idxs,
                                   # 'test_protocol_3_phone_5_print_attack_idxs': test_protocol_3_phone_5_print_attack_idxs,
                                   # 'test_protocol_3_phone_5_replay_attack_idxs': test_protocol_3_phone_5_replay_attack_idxs,
                                   'test_protocol_3_phone_6_idxs': test_protocol_3_phone_6_idxs,
                                   # 'test_protocol_3_phone_6_print_attack_idxs': test_protocol_3_phone_6_print_attack_idxs,
                                   # 'test_protocol_3_phone_6_replay_attack_idxs': test_protocol_3_phone_6_replay_attack_idxs,
                                   }

        elif self.protocol_id == 4:
            train_scenarios_idxs = {'train_protocol_4_phone_1_idxs': train_protocol_4_phone_1_idxs,
                                    'train_protocol_4_phone_2_idxs': train_protocol_4_phone_2_idxs,
                                    'train_protocol_4_phone_3_idxs': train_protocol_4_phone_3_idxs,
                                    'train_protocol_4_phone_4_idxs': train_protocol_4_phone_4_idxs,
                                    'train_protocol_4_phone_5_idxs': train_protocol_4_phone_5_idxs,
                                    'train_protocol_4_phone_6_idxs': train_protocol_4_phone_6_idxs,
                                    }

            devel_scenarios_idxs = {'devel_protocol_4_phone_1_idxs': devel_protocol_4_phone_1_idxs,
                                    'devel_protocol_4_phone_2_idxs': devel_protocol_4_phone_2_idxs,
                                    'devel_protocol_4_phone_3_idxs': devel_protocol_4_phone_3_idxs,
                                    'devel_protocol_4_phone_4_idxs': devel_protocol_4_phone_4_idxs,
                                    'devel_protocol_4_phone_5_idxs': devel_protocol_4_phone_5_idxs,
                                    'devel_protocol_4_phone_6_idxs': devel_protocol_4_phone_6_idxs,
                                    }

            test_scenarios_idxs = {
                                   # 'test_protocol_1_idxs': test_protocol_1_idxs,
                                   # 'test_protocol_1_print_attack_idxs': test_protocol_1_print_attack_idxs,
                                   # 'test_protocol_1_replay_attack_idxs': test_protocol_1_replay_attack_idxs,
                                   # 'test_protocol_2_idxs': test_protocol_2_idxs,
                                   # 'test_protocol_2_print_attack_idxs': test_protocol_2_print_attack_idxs,
                                   # 'test_protocol_2_replay_attack_idxs': test_protocol_2_replay_attack_idxs,
                                   # 'test_protocol_3_phone_1_idxs': test_protocol_3_phone_1_idxs,
                                   # 'test_protocol_3_phone_1_print_attack_idxs': test_protocol_3_phone_1_print_attack_idxs,
                                   # 'test_protocol_3_phone_1_replay_attack_idxs': test_protocol_3_phone_1_replay_attack_idxs,
                                   # 'test_protocol_3_phone_2_idxs': test_protocol_3_phone_2_idxs,
                                   # 'test_protocol_3_phone_2_print_attack_idxs': test_protocol_3_phone_2_print_attack_idxs,
                                   # 'test_protocol_3_phone_2_replay_attack_idxs': test_protocol_3_phone_2_replay_attack_idxs,
                                   # 'test_protocol_3_phone_3_idxs': test_protocol_3_phone_3_idxs,
                                   # 'test_protocol_3_phone_3_print_attack_idxs': test_protocol_3_phone_3_print_attack_idxs,
                                   # 'test_protocol_3_phone_3_replay_attack_idxs': test_protocol_3_phone_3_replay_attack_idxs,
                                   # 'test_protocol_3_phone_4_idxs': test_protocol_3_phone_4_idxs,
                                   # 'test_protocol_3_phone_4_print_attack_idxs': test_protocol_3_phone_4_print_attack_idxs,
                                   # 'test_protocol_3_phone_4_replay_attack_idxs': test_protocol_3_phone_4_replay_attack_idxs,
                                   # 'test_protocol_3_phone_5_idxs': test_protocol_3_phone_5_idxs,
                                   # 'test_protocol_3_phone_5_print_attack_idxs': test_protocol_3_phone_5_print_attack_idxs,
                                   # 'test_protocol_3_phone_5_replay_attack_idxs': test_protocol_3_phone_5_replay_attack_idxs,
                                   # 'test_protocol_3_phone_6_idxs': test_protocol_3_phone_6_idxs,
                                   # 'test_protocol_3_phone_6_print_attack_idxs': test_protocol_3_phone_6_print_attack_idxs,
                                   # 'test_protocol_3_phone_6_replay_attack_idxs': test_protocol_3_phone_6_replay_attack_idxs,
                                   'test_protocol_4_phone_1_idxs': test_protocol_4_phone_1_idxs,
                                   # 'test_protocol_4_phone_1_print_attack_idxs': test_protocol_4_phone_1_print_attack_idxs,
                                   # 'test_protocol_4_phone_1_replay_attack_idxs': test_protocol_4_phone_1_replay_attack_idxs,
                                   'test_protocol_4_phone_2_idxs': test_protocol_4_phone_2_idxs,
                                   # 'test_protocol_4_phone_2_print_attack_idxs': test_protocol_4_phone_2_print_attack_idxs,
                                   # 'test_protocol_4_phone_2_replay_attack_idxs': test_protocol_4_phone_2_replay_attack_idxs,
                                   'test_protocol_4_phone_3_idxs': test_protocol_4_phone_3_idxs,
                                   # 'test_protocol_4_phone_3_print_attack_idxs': test_protocol_4_phone_3_print_attack_idxs,
                                   # 'test_protocol_4_phone_3_replay_attack_idxs': test_protocol_4_phone_3_replay_attack_idxs,
                                   'test_protocol_4_phone_4_idxs': test_protocol_4_phone_4_idxs,
                                   # 'test_protocol_4_phone_4_print_attack_idxs': test_protocol_4_phone_4_print_attack_idxs,
                                   # 'test_protocol_4_phone_4_replay_attack_idxs': test_protocol_4_phone_4_replay_attack_idxs,
                                   'test_protocol_4_phone_5_idxs': test_protocol_4_phone_5_idxs,
                                   # 'test_protocol_4_phone_5_print_attack_idxs': test_protocol_4_phone_5_print_attack_idxs,
                                   # 'test_protocol_4_phone_5_replay_attack_idxs': test_protocol_4_phone_5_replay_attack_idxs,
                                   'test_protocol_4_phone_6_idxs': test_protocol_4_phone_6_idxs,
                                   # 'test_protocol_4_phone_6_print_attack_idxs': test_protocol_4_phone_6_print_attack_idxs,
                                   # 'test_protocol_4_phone_6_replay_attack_idxs': test_protocol_4_phone_6_replay_attack_idxs,
                                   }

        else:
            raise Exception('Protocol Not Defined Yet!')

        # -- check if the training and testing sets are disjoint.
        try:
            assert not np.intersect1d(all_fnames[train_idxs], all_fnames[test_idxs]).size
        except AssertionError:
            raise Exception('The training and testing sets are mixed')

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'train_idxs': train_idxs,
                  'train_users_idxs': {},
                  'devel_idxs': devel_idxs,
                  'test_all_idxs': test_idxs,
                  'train_scenarios_idxs': train_scenarios_idxs,
                  'devel_scenarios_idxs': devel_scenarios_idxs,
                  'test_scenarios_idxs': test_scenarios_idxs,
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
