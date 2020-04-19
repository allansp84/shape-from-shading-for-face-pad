# -*- coding: utf-8 -*-

import argparse

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.datasets import *
from antispoofing.sfsnet.features.estimation import *
from antispoofing.sfsnet.controller import *
from antispoofing.sfsnet.classification import *
from antispoofing.sfsnet.metaclassification import *


class CommandLineParser(object):
    def __init__(self):

        # -- define the arguments available in the command line execution
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    def parsing(self):
        dataset_options = 'Available dataset interfaces: '
        for k in sorted(registered_datasets.keys()):
            dataset_options += ('%s-%s, ' % (k, registered_datasets[k].__name__))

        map_type_options = 'Type of maps to extracted features: '
        for k in sorted(map_type_dict.keys()):
            map_type_options += ('%s-%s, ' % (k, map_type_dict[k]))

        sfs_algo_options = 'Available Algorithm for Shape from shading: '
        for k in sorted(r_sfs_algo.keys()):
            sfs_algo_options += ('%s-%s, ' % (k, r_sfs_algo[k].__name__))

        ml_algo_options = 'Available Algorithm for Classification: '
        for k in sorted(ml_algo.keys()):
            ml_algo_options += ('%s-%s, ' % (k, ml_algo[k].__name__))

        meta_ml_algo_options = 'Available Algorithm for Meta Classification: '
        for k in sorted(meta_ml_algo.keys()):
            meta_ml_algo_options += ('%s-%s, ' % (k, meta_ml_algo[k].__name__))

        losses_functions_options = 'Available Algorithm for Losses: '
        for k in sorted(losses_functions.keys()):
            if isinstance(losses_functions[k], str):
                losses_functions_options += ('%s-%s, ' % (k, losses_functions[k]))
            else:
                losses_functions_options += ('%s-%s, ' % (k, losses_functions[k].__name__))

        optimizer_methods_options = 'Available Optimizers: '
        for k in sorted(optimizer_methods.keys()):
            optimizer_methods_options += ('%s-%s, ' % (k, optimizer_methods[k]))

        frame_fusion_options = 'Type of fusion of the frames: '
        for k in sorted(fusion_type_dict.keys()):
            frame_fusion_options += ('%s, ' % k)

        color_space_options = 'Color space: '
        for k in sorted(color_space_dict.keys()):
            color_space_options += ('%s, ' % k)

        light_direction_options = ["constant"]

        # -- arguments related to the dataset and to the output
        group_a = self.parser.add_argument_group('Arguments')

        group_a.add_argument('--protocol', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_a.add_argument('--dataset', type=int, metavar='', default=0, choices=range(len(registered_datasets)),
                             help=dataset_options + '(default=%(default)s).')

        group_a.add_argument('--dataset_path', type=str, metavar='', default='',
                             help='Path to the dataset.')

        group_a.add_argument('--output_path', type=str, metavar='', default='working',
                             help='Path where the results will be saved (default=%(default)s).')

        group_a.add_argument('--face_locations_path', type=str, metavar='', default='',
                             help='A .csv file containing the faces locations (default=%(default)s).')

        group_a.add_argument('--dataset_b', type=int, metavar='', default=-1, choices=range(len(registered_datasets)),
                             help=dataset_options + '(default=%(default)s).')

        group_a.add_argument('--dataset_path_b', type=str, metavar='', default='',
                             help='Path to the dataset.')

        group_a.add_argument('--face_locations_path_b', type=str, metavar='', default='',
                             help='A .csv file containing the faces locations for the dataset b (default=%(default)s).')

        group_sfs = self.parser.add_argument_group('Available Parameters for Map Estimation')

        group_sfs.add_argument('--map_extraction', action='store_true',
                               help='Extract maps from frames (default=%(default)s).')

        group_sfs.add_argument('--map_type', type=int, metavar='int', default=4,
                               help=map_type_options + '(default=%(default)s).')

        group_sfs.add_argument('--sfs_algo', type=int, default=0, choices=range(len(r_sfs_algo)),
                               help=sfs_algo_options + '(default=%(default)s).')

        group_sfs.add_argument('--colorspace', type=str, metavar='str', default='grayscale',
                               help=color_space_options + '(default=%(default)s).')

        group_sfs.add_argument('--local_estimation', action='store_true')

        group_sfs.add_argument('--light_direction', type=str, default="constant", metavar="",
                               choices=light_direction_options,
                               help="Method for light direction estimation. " +
                                    "Allowed values are: " + ", ".join(light_direction_options) + " (default=%(default)s)")

        # -- arguments related to the Feature extraction module
        group_b = self.parser.add_argument_group('Available Parameters for Feature Extraction')

        group_b.add_argument('--build_multichannel_input', action='store_true',
                             help='Execute the feature extraction process (default=%(default)s).')

        group_b.add_argument('--feature_extraction', action='store_true',
                             help='Execute the feature extraction process (default=%(default)s).')

        group_b.add_argument("--descriptor", type=str, default="RawImage", metavar="",
                             choices=['RawImage', 'HybridImage', 'RawVideo', 'None'],
                             help="(default=%(default)s)")

        # -- arguments related to the Classification module
        group_c = self.parser.add_argument_group('Available Parameters for Classification')

        group_c.add_argument('--classification', action='store_true',
                             help='Execute the classification process (default=%(default)s).')

        group_c.add_argument('--ml_algo', type=int, metavar='', default=7, choices=ml_algo.keys(),
                             help=ml_algo_options + '(default=%(default)s).')

        group_c.add_argument('--epochs', type=int, metavar='', default=150,
                             help='Number of the epochs considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--bs', type=int, metavar='', default=64,
                             help='The size of the batches (default=%(default)s).')

        group_c.add_argument('--lr', type=float, metavar='', default=0.001,
                             help='The learning rate considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--decay', type=float, metavar='', default=0.0,
                             help='The decay value considered during the learning stage (default=%(default)s).')

        group_c.add_argument('--reg', type=float, metavar='', default=0.0001,
                             help='The value of the L2 regularization method (default=%(default)s).')

        group_c.add_argument('--loss_function', type=int, metavar='', default=0, choices=range(len(losses_functions_options)),
                             help=losses_functions_options + '(default=%(default)s).')

        group_c.add_argument('--optimizer', type=int, metavar='', default=3, choices=range(len(optimizer_methods)),
                             help=optimizer_methods_options + '(default=%(default)s).')

        group_c.add_argument('--fold', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_c.add_argument('--force_train', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--meta_ml_algo', type=int, metavar='', default=0, choices=range(len(meta_ml_algo)),
                             help=meta_ml_algo_options + '(default=%(default)s).')

        group_c.add_argument('--meta_classification', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--compute_feature_importance', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--meta_classification_from', type=str, metavar='', default='scores', choices=['scores', 'labels'],
                             help='(default=%(default)s).')

        group_c.add_argument('--load_weights', type=str, metavar='', default='',
                             help='(default=%(default)s).')

        group_c.add_argument('--n_models', type=int, metavar='', default=21,
                             help='(default=%(default)s).')

        group_c.add_argument('--selection_algo', type=int, metavar='', default=0,
                             help='(default=%(default)s).')

        group_c.add_argument('--fine_tuning', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--testing_best_weights', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--force_testing', action='store_true',
                             help='(default=%(default)s).')

        group_c.add_argument('--feature_visualization', action='store_true',
                             help='(default=%(default)s).')

        group_d = self.parser.add_argument_group('Other options')

        group_d.add_argument('--show_results', action='store_true',
                             help='(default=%(default)s).')

        group_d.add_argument('--operation', type=str, metavar='', default='resize', choices=['', 'crop', 'resize'],
                             help='(default=%(default)s).')

        group_d.add_argument('--n_channel', type=int, metavar='', default=3,
                             help='(default=%(default)s).')

        group_d.add_argument('--frame_offset', type=int, metavar='', default=30,
                             help='(default=%(default)s).')

        group_d.add_argument('--total_n_frames', type=int, metavar='', default=61,
                             help='(default=%(default)s).')

        group_d.add_argument('--load_n_frames', type=int, metavar='', default=-1,
                             help='(default=%(default)s).')

        group_d.add_argument('--n_frames_for_testing', type=int, metavar='', default=-1,
                             help='(default=%(default)s).')

        group_d.add_argument('--frame_fusion_type', type=str, metavar='', default='mean',
                             help=frame_fusion_options + '(default=%(default)s).')

        group_d.add_argument('--max_axis', type=int, metavar='', default=150,
                             help='(default=%(default)s).')

        group_d.add_argument('--device_number', type=str, metavar='', default='0',
                             help='(default=%(default)s).')

        self.parser.add_argument('--n_jobs', type=int, metavar='', default=N_JOBS,
                                 help='Number of jobs to be used during processing (default=%(default)s).')

        self.parser.add_argument('--seed', type=int, metavar='int', default=SEED,
                                 help='(default=%(default)s).')

        self.parser.add_argument('--debug', action='store_true',
                                 help='(default=%(default)s).')

        deprecated = self.parser.add_argument_group('Deprecated arguments')

        deprecated.add_argument('--last_layer', type=str, metavar='', default='linear', choices=['linear', 'softmax'],
                                help='(default=%(default)s).')

        deprecated.add_argument('--layers_name', nargs='+', type=str, metavar='', default=['conv_1'],
                                help='(default=%(default)s).')

        deprecated.add_argument('--fv', action='store_true',
                                help='(default=%(default)s).')

    def get_args(self):
        return self.parser.parse_args()


# -- main function
def main():
    # -- parsing the command line options
    command_line = CommandLineParser()
    command_line.parsing()

    args = command_line.get_args()
    print('ARGS:', args)
    sys.stdout.flush()

    # -- create and execute a Controller object
    control = Controller(args)
    control.run()


if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print('Total time elapsed: {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed))))
    sys.stdout.flush()
