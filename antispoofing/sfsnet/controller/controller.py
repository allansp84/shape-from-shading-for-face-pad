# -*- coding: utf-8 -*-

import json

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.datasets import *
from antispoofing.sfsnet.features import Extraction, HybridImage
from antispoofing.sfsnet.features.estimation import r_sfs_algo
from antispoofing.sfsnet.features.estimation import MapsExtraction
from antispoofing.sfsnet.classification import *
from antispoofing.sfsnet.metaclassification import *


class Controller(object):

    def __init__(self, args):

        self.args = args
        self.data = None
        self.data_b = None
        self.n_jobs = self.args.n_jobs
        self.path_to_maps = ""
        self.features_path = ""
        self.classification_path = ""
        self.path_to_features = ""

    def build_multichannel_input(self):

        start = get_time()

        n_channel = self.args.n_channel * 3

        input_path_albedo = os.path.join(self.path_to_features, str(self.args.n_channel), map_type_dict[0])
        meta_info_feats = self.data.meta_info_feats(input_path_albedo, ['.png'])
        input_fnames_albedo = meta_info_feats['all_fnames'].reshape(-1, 1)

        input_path_depth = os.path.join(self.path_to_features, str(self.args.n_channel), map_type_dict[1])
        meta_info_feats = self.data.meta_info_feats(input_path_depth, ['.png'])
        input_fnames_depth = meta_info_feats['all_fnames'].reshape(-1, 1)

        input_path_reflectance = os.path.join(self.path_to_features, str(self.args.n_channel), map_type_dict[2])
        meta_info_feats = self.data.meta_info_feats(input_path_reflectance, ['.png'])
        input_fnames_reflectance = meta_info_feats['all_fnames'].reshape(-1, 1)

        try:
            assert input_fnames_albedo.shape == input_fnames_depth.shape
            assert input_fnames_albedo.shape == input_fnames_reflectance.shape
        except AssertionError:
            raise Exception('-- inputs are not the same size')

        input_fnames = np.concatenate((input_fnames_depth, input_fnames_reflectance, input_fnames_albedo), axis=1)

        path_to_maps = os.path.join(self.path_to_features, str(self.args.n_channel), map_type_dict[1])

        # --  creates a output filename for each image in the dataset
        output_paths = []
        for fname in input_fnames[:, 0]:
            rel_fname = os.path.relpath(fname, path_to_maps)
            rel_fname = "{0}.npy".format(os.path.splitext(rel_fname)[0])
            output_path = os.path.join(self.path_to_features, str(n_channel), map_type_dict[self.args.map_type], rel_fname)
            output_path = output_path.replace('depth', map_type_dict[self.args.map_type])
            output_paths.append(output_path)
        output_paths = np.array(output_paths)

        # -- creates a pool of extraction object
        tasks = []
        for idx in range(len(input_fnames)):
            tasks += [HybridImage(self.data, input_fnames[idx], output_paths[idx],
                                  n_channel=self.args.n_channel,
                                  )]

        # -- start to execute the objects Extraction by running the method Extraction.run()
        if self.n_jobs > 1:
            print("running %d tasks in parallel" % len(tasks))
            RunInParallel(tasks, self.n_jobs).run()
        else:
            print("running %d tasks in sequence" % len(tasks))
            for idx in range(len(input_fnames)):
                tasks[idx].run()
                progressbar('-- RunInSequence', idx, len(input_fnames))

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def map_extraction(self):

        start = get_time()

        padding = 1
        input_fnames = self.data.meta_info['all_fnames']

        output_paths = []
        for fname in input_fnames:
            rel_fname = os.path.relpath(fname, self.data.dataset_path)
            rel_fname = os.path.splitext(rel_fname)[0]
            output_paths.append(os.path.join(self.path_to_features, str(self.args.n_channel), map_type_dict[self.args.map_type], rel_fname))
        output_paths = np.array(output_paths)

        tasks = []
        for idx in range(len(input_fnames)):
            tasks += [MapsExtraction(self.data, self.data.dataset_path, output_paths[idx], input_fnames[idx],
                                     sfs_algo=r_sfs_algo[self.args.sfs_algo],
                                     n_channel=self.args.n_channel,
                                     frame_numbers=self.args.total_n_frames,
                                     mp=self.args.map_type,
                                     local_estimation=self.args.local_estimation,
                                     light_direction=self.args.light_direction,
                                     padding=padding,
                                     )]

        if self.n_jobs > 1:
            print("running %d tasks in parallel" % len(tasks))
            RunInParallel(tasks, self.n_jobs).run()
        else:
            print("running %d tasks in sequence" % len(tasks))
            for idx in range(len(input_fnames)):
                tasks[idx].run()
                progressbar('-- RunInSequence', idx, len(input_fnames))

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def classification(self):

        start = get_time()

        algo = ml_algo[self.args.ml_algo]

        if self.args.map_type == 4:
            n_channel = self.args.n_channel * 3
            file_type = ".npy"
        else:
            n_channel = self.args.n_channel
            file_type = ".png"

        if self.args.loss_function == 7:
            loss_function_str = 'categorical_focal_loss'
        else:
            loss_function_str = '{}'.format(losses_functions[self.args.loss_function])

        output_fname = "max_axis-{}-ml_algo-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-" \
                       "seed-{}-fold-{}".format(self.args.max_axis,
                                        self.args.ml_algo,
                                        self.args.epochs,
                                        self.args.bs,
                                        loss_function_str,
                                        self.args.lr,
                                        self.args.decay,
                                        optimizer_methods[self.args.optimizer],
                                        self.args.reg,
                                        self.args.seed,
                                        self.args.fold,
                                        )

        output_path = os.path.join(self.classification_path,
                                   str(n_channel),
                                   map_type_dict[self.args.map_type],
                                   output_fname,
                                   )

        input_path = os.path.join(self.path_to_features, str(n_channel), map_type_dict[self.args.map_type])
        meta_info_feats = self.data.meta_info_feats(input_path, [file_type])

        if self.args.dataset_b >= 0:
            input_path = os.path.join(self.path_to_features_b, str(n_channel), map_type_dict[self.args.map_type])
            meta_info_feats_b = self.data_b.meta_info_feats(input_path, [file_type])
        else:
            meta_info_feats_b = None

        linearize_hybrid_imgs = False
        if self.args.map_type == 4 and (self.args.ml_algo in [6, 8, 10, 11, 71]):
            linearize_hybrid_imgs = True

        algo(output_path, meta_info_feats,
             dataset_b=meta_info_feats_b,
             dataset_name=str(self.data.__class__.__name__).lower(),
             dataset_b_name=str(self.data_b.__class__.__name__).lower(),
             input_shape=self.args.max_axis,
             n_channel=n_channel,
             frames_per_video=self.args.total_n_frames,
             n_frames_for_testing=self.args.n_frames_for_testing,
             frame_fusion_type=self.args.frame_fusion_type,
             load_n_frames=self.args.load_n_frames,
             epochs=self.args.epochs,
             batch_size=self.args.bs,
             loss_function=losses_functions[self.args.loss_function],
             lr=self.args.lr,
             decay=self.args.decay,
             optimizer=optimizer_methods[self.args.optimizer],
             regularization=self.args.reg,
             device_number=self.args.device_number,
             force_train=self.args.force_train,
             filter_vis=self.args.fv,
             layers_name=self.args.layers_name,
             fold=self.args.fold,
             fine_tuning=self.args.fine_tuning,
             seed=self.args.seed,
             testing_best_weights=self.args.testing_best_weights,
             force_testing=self.args.force_testing,
             load_weights=self.args.load_weights,
             linearize_hybrid_imgs=linearize_hybrid_imgs,
             feature_visualization=self.args.feature_visualization,
             debug=self.args.debug,
             ).run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def meta_classification(self):

        start = get_time()

        algo = meta_ml_algo[self.args.meta_ml_algo]

        output_path = os.path.join(self.data.output_path,
                                   str(self.args.max_axis),
                                   self.args.light_direction,
                                   "meta_classification",
                                   "selection_algo-{}".format(self.args.selection_algo),
                                   self.args.meta_classification_from,
                                   str(algo.__name__).lower(),
                                   "n_models-{}".format(self.args.n_models),
                                   "ml_algo-{}".format(self.args.ml_algo),
                                   )

        max_axis_to_fuse = [self.args.max_axis]
        light_direction_to_fuse = [self.args.light_direction]

        input_paths = []
        for m_axis in max_axis_to_fuse:
            for light_direction in light_direction_to_fuse:
                input_path = os.path.join(self.data.output_path,
                                          str(m_axis),
                                          light_direction,
                                          "classification",
                                          )
                input_paths += [input_path]

        all_fnames = []
        for input_path in input_paths:
            fnames = retrieve_fnames(os.path.abspath(input_path), '.pkl')
            all_fnames += [fnames]
        all_fnames = np.concatenate(all_fnames)

        if self.args.dataset_b >= 0:
            prefix = 'inter_%s' % str(self.data_b.__class__.__name__).lower()
        else:
            prefix = 'intra'

        predictions_files = [fname for fname in all_fnames if '%s.predictions.pkl' % prefix in fname]

        maps_to_fuse = ['albedo/',
                        'reflectance/',
                        'depth/',
                        # 'hybrid/',
                        ]
        classifier_to_fuse = ["ml_algo-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-" \
                              "seed-{}-fold-{}".format(self.args.ml_algo,
                                                       self.args.epochs,
                                                       self.args.bs,
                                                       losses_functions[self.args.loss_function],
                                                       self.args.lr,
                                                       self.args.decay,
                                                       optimizer_methods[self.args.optimizer],
                                                       self.args.reg,
                                                       self.args.seed,
                                                       self.args.fold)]

        predictions_files = [fname for fname in predictions_files for map in maps_to_fuse if map in fname]
        predictions_files = [fname for fname in predictions_files for algo in classifier_to_fuse if algo in fname]

        print('-- predictions_files', flush=True)
        for k, fn in enumerate(predictions_files):
            print('-- %d:' % k, fn, flush=True)

        algo(output_path, predictions_files,
             meta_classification_from=self.args.meta_classification_from,
             n_models=self.args.n_models,
             selection_algo=self.args.selection_algo,
             compute_feature_importance=self.args.compute_feature_importance,
             frames_per_video=self.args.total_n_frames,
             frame_fusion_type=self.args.frame_fusion_type,
             prefix=prefix,
             force_train=self.args.force_train,
             ).run()

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def show_results(self):
        """ Method responsible for showing the classification results.
        """

        start = get_time()

        if self.args.map_type == 4:
            n_channel = self.args.n_channel * 3
        else:
            n_channel = self.args.n_channel

        if self.args.loss_function == 7:
            loss_function_str = 'categorical_focal_loss'
        else:
            loss_function_str = '{}'.format(losses_functions[self.args.loss_function])

        output_fname = "max_axis-{}-ml_algo-{}-epochs-{}-bs-{}-losses-{}-lr-{}-decay-{}-optimizer-{}-reg-{}-" \
                       "seed-{}-fold-{}".format(self.args.max_axis,
                                        self.args.ml_algo,
                                        self.args.epochs,
                                        self.args.bs,
                                        loss_function_str,
                                        self.args.lr,
                                        self.args.decay,
                                        optimizer_methods[self.args.optimizer],
                                        self.args.reg,
                                        self.args.seed,
                                        self.args.fold,
                                        )

        if self.args.meta_classification:
            algo = meta_ml_algo[self.args.meta_ml_algo]
            input_path = os.path.join(self.data.output_path,
                                      str(self.args.max_axis),
                                      # str(self.args.total_n_frames),
                                      self.args.light_direction,
                                      "meta_classification",
                                      "selection_algo-{}".format(self.args.selection_algo),
                                      self.args.meta_classification_from,
                                      str(algo.__name__).lower(),
                                      "n_models-{}".format(self.args.n_models),
                                      "ml_algo-{}".format(self.args.ml_algo),
                                      )

        else:
            input_path = os.path.join(self.classification_path,
                                      str(n_channel),
                                      map_type_dict[self.args.map_type],
                                      output_fname,
                                      )

        fnames = retrieve_fnames(os.path.abspath(input_path), '.json')
        fnames = [fname for fname in fnames if 'results.%s.json' % self.args.frame_fusion_type in fname]
        fnames = [fname for fname in fnames if 'intra' in fname or 'inter_%s' % str(self.data_b.__class__.__name__).lower() in fname]

        print("-- fnames", fnames)
        report = {}
        for fname in fnames:
            key = os.path.basename(os.path.dirname(fname))
            json_data = json.load(open(fname, 'r'))
            report[key] = json_data

        classification_results_summary(report)

        elapsed = total_time_elapsed(start, get_time())
        print('spent time: {0}!'.format(elapsed))
        sys.stdout.flush()

    def run(self):

        # -- create an object for the main dataset
        dataset = registered_datasets[self.args.dataset]
        self.data = dataset(self.args.dataset_path,
                            output_path=self.args.output_path,
                            face_locations_path=self.args.face_locations_path,
                            operation=self.args.operation,
                            max_axis=self.args.max_axis,
                            n_channel=self.args.n_channel,
                            frame_offset=self.args.frame_offset,
                            total_n_frames=self.args.total_n_frames,
                            protocol_id=self.args.protocol,
                            )

        self.data.output_path = os.path.join(self.args.output_path,
                                             str(self.data.__class__.__name__).lower(),
                                             )

        self.path_to_maps = os.path.join(self.data.output_path,
                                         str(self.args.max_axis),
                                         self.args.light_direction, "maps",
                                         )

        self.path_to_features = os.path.join(self.data.output_path,
                                             str(self.args.max_axis),
                                             self.args.light_direction, "features",
                                             )

        self.classification_path = os.path.join(self.data.output_path,
                                                str(self.args.max_axis),
                                                self.args.light_direction, "classification",
                                                )

        # -- create an object for the second dataset in case the user opts for running the cross-dataset protocol
        if self.args.dataset_b >= 0:
            dataset_b = registered_datasets[self.args.dataset_b]
            self.data_b = dataset_b(self.args.dataset_path_b,
                                    output_path=self.args.output_path,
                                    face_locations_path=self.args.face_locations_path_b,
                                    operation=self.args.operation,
                                    max_axis=self.args.max_axis,
                                    n_channel=self.args.n_channel,
                                    frame_offset=self.args.frame_offset,
                                    total_n_frames=self.args.total_n_frames,
                                    protocol_id=self.args.protocol,
                                    )

            self.data_b.output_path = os.path.join(self.args.output_path,
                                                   str(self.data_b.__class__.__name__).lower(),
                                                   )

            self.path_to_maps_b = os.path.join(self.data_b.output_path,
                                               str(self.args.max_axis),
                                               self.args.light_direction, "maps",
                                               )

            self.path_to_features_b = os.path.join(self.data_b.output_path,
                                                   str(self.args.max_axis),
                                                   self.args.light_direction, "features",
                                                   )

        if self.args.map_extraction:
            print("-- extracting maps ...")
            self.map_extraction()

        # if self.args.feature_extraction:
        #     print("-- extracting features ...")
        #     self.feature_extraction()

        if self.args.build_multichannel_input:
            if self.args.map_type == 4:
                print("-- building hybrid images ...")
                self.build_multichannel_input()

        if self.args.classification:
            print("-- classifying ...")
            self.classification()

        if self.args.meta_classification and not self.args.show_results:
            print("-- meta classification ...")
            self.meta_classification()

        if self.args.show_results:
            print("-- showing the results ...")
            self.show_results()
