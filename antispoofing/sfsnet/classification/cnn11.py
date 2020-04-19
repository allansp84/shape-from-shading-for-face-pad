# -*- coding: utf-8 -*-

import json

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.classification.baseclassifier import BaseClassifier, metric_hter, metric_bal_accuracy
from sklearn.utils import class_weight


class CNN11(BaseClassifier):

    def __init__(self, output_path, dataset, dataset_b=None, dataset_name="", dataset_b_name="", input_shape=200, n_channel=3,
                 frames_per_video=10, n_frames_for_testing=10, frame_fusion_type='max', load_n_frames=-1,
                 epochs=50, batch_size=8, loss_function='categorical_crossentropy', lr=0.01, decay=0.0005, optimizer='SGD', regularization=0.1,
                 device_number=0, force_train=False, filter_vis=False, layers_name=('conv_1',),
                 fold=0, fine_tuning=False, seed=0, testing_best_weights=False, force_testing=True, load_weights='',
                 linearize_hybrid_imgs=False, feature_visualization=False, debug=False):

        super(CNN11, self).__init__(output_path, dataset, dataset_b=dataset_b,
                                   dataset_name=dataset_name, dataset_b_name=dataset_b_name,
                                   input_shape=input_shape, n_channel=n_channel,
                                   frames_per_video=frames_per_video,
                                   n_frames_for_testing=n_frames_for_testing,
                                   load_n_frames=load_n_frames,
                                   force_train=force_train, fold=fold, seed=seed,
                                   testing_best_weights=testing_best_weights,
                                   force_testing=force_testing,
                                   load_weights=load_weights,
                                   linearize_hybrid_imgs=linearize_hybrid_imgs,
                                   feature_visualization=feature_visualization,
                                   debug=debug,
                                   )
        print("-- CNN11 (deep CNN)", flush=True)
        self.verbose = True

        self.dataset = dataset
        self.output_path = output_path
        self.model = None
        self.svm_model = None

        self.input_shape = (input_shape, input_shape, n_channel)
        self.n_channel = n_channel
        self.frames_per_video = frames_per_video
        self.frame_fusion_type = frame_fusion_type

        self.num_classes = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.lr = lr
        self.decay = decay
        self.optimizer = optimizer
        self.regularization = regularization
        self.device_number = device_number
        self.force_train = force_train
        self.filter_vis = filter_vis
        self.layers_name = list(layers_name)

        self.fine_tuning = fine_tuning
        self.seed = seed
        self.testing_best_weights = testing_best_weights
        self.force_testing = force_testing
        self.load_weights = load_weights

    def set_gpu_configuration(self):
        """ This function is responsible for setting up which GPU will be used during the processing and some configurations related
        to GPU memory usage when the TensorFlow is used as backend.
        """

        if self.verbose:
            print('-- setting the GPU configurations', flush=True)

        if 'tensorflow' in keras.backend.backend():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_number

            K.clear_session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True, allocator_type='BFC')
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options,
                                                               allow_soft_placement=True,
                                                               log_device_placement=True)))

    def architecture_definition(self):
        """
        In this method we define the architecture of our CNN.
        """

        input_shape = self.input_shape[:2] + (3,)

        base_model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, classes=self.num_classes)

        x = base_model.output

        x = Flatten(name='flatten')(x)

        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.models.Model(inputs=base_model.input, outputs=predictions)

        if self.load_weights:
            print('-- loading weights:', self.load_weights)
            self.model.load_weights(self.load_weights)

        # # -- first: train only the top layers, i.e., freeze all convolutional Xception layer
        # for layer in base_model.layers:
        #     layer.trainable = False

        if self.verbose:
            print(self.model.summary())

        # -- saving the CNN architecture definition in a .json file
        model_json = json.loads(self.model.to_json())
        json_fname = os.path.join(self.output_path, 'model.json')
        with open(json_fname, mode='w') as f:
            print("--saving json file:", json_fname)
            sys.stdout.flush()
            f.write(json.dumps(model_json, indent=4))

    def add_new_last_layer(self, base_model):
        """ Add last layer to the convnet
            Args:
              base_model: keras model excluding top
              nb_classes: # of classes
            Returns:
              new keras model with last layer
        """

        # -- truncate and replace softmax layer for transfer learning
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

        n_layer_base_model = len(base_model.layers)

        x = base_model.output
        x = Dense(1024, activation='relu', name='dense1024')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        model = keras.models.Model(input=base_model.input, output=predictions)

        for idx in range(n_layer_base_model):
            model.layers[idx].trainable = False

        return model

    def fit_model(self, x_train, y_train, x_validation=None, y_validation=None, class_weights=None, output_path=''):
        """ Fit a model classification.

        Args:
            x_train (numpy.ndarray): A multidimensional array containing the feature vectors (or images) to be used to train a classifier.
            y_train (numpy.ndarray): A multidimensional array containing the labels refers to the feature vectors that will be used during the training stage.
            x_validation (numpy.ndarray, optional): A multidimensional array containing the feature vectors (or images) to be used to test the classifier.
            y_validation (numpy.ndarray, optional): A multidimensional array containing the labels refers to the feature vectors that will be used for testing the classification model.
            class_weights (dict): A dictionary containig class weights for unbalanced datasets.
            output_path (str):
        """

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- define the architecture
        self.architecture_definition()

        # -- choose the optimizer that will be used during the training process
        optimizer_methods = {'SGD': keras.optimizers.SGD,
                             'Adam': keras.optimizers.Adam,
                             'Adagrad': keras.optimizers.Adagrad,
                             'Adadelta': keras.optimizers.Adadelta,
                             }

        try:
            opt = optimizer_methods[self.optimizer]
        except KeyError:
            raise Exception('The optimizer %s is not being considered in this work yet:' % self.optimizer)

        # --  configure the learning process
        self.model.compile(loss=self.loss_function, optimizer=opt(lr=self.lr, decay=self.decay),
                           metrics=['accuracy',
                                    keras.metrics.categorical_crossentropy,
                                    keras.metrics.binary_crossentropy,
                                    keras.losses.categorical_hinge,
                                    metric_bal_accuracy,
                                    metric_hter,
                                    ])

        # r_state = np.random.RandomState(7)
        # rand_idxs = r_state.permutation(len(x_train))

        # -- normalization step
        x_train = x_train/255.

        callbacks = []

        if x_validation is None:
            validation_split = 0.0
            validation_data = None
        else:
            validation_split = 0.0
            x_validation = x_validation/255.
            validation_data = (x_validation, y_validation)

        # -- define the callbacks if the validation set is available

        # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=0, mode='auto')
        # callbacks.append(early_stopping)

        log_dir = '{0}/logs'.format(output_path)
        safe_create_dir(log_dir)

        print('--logdir=%s' % log_dir, flush=True)

        tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                   histogram_freq=0, 
                                                   batch_size=self.batch_size, 
                                                   write_graph=True, 
                                                   write_grads=False,
                                                   write_images=False)
        callbacks.append(tensor_board)

        # file_path = log_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        # check_point = keras.callbacks.ModelCheckpoint(filepath=file_path)
        # callbacks.append(check_point)
        
        # file_path = log_dir + "/weights.best.hdf5"
        # best_weights = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='metric_bal_accuracy', verbose=1, save_best_only=True, mode='max')
        # callbacks.append(best_weights)

        # -- fit a model
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_split=validation_split,
                                 validation_data=validation_data,
                                 shuffle=True,
                                 class_weight=class_weights,
                                 )

        # -- save the training history
        self.saving_training_history(history.history, output_path, 'acc')
        self.saving_training_history(history.history, output_path, 'loss')

    def training(self, x_train, y_train, x_validation=None, y_validation=None, prefix='', train_fnames=None, train_users_idxs=None):
        """ This method implements the training process of our CNN.

        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Labels of the training data
            x_validation (numpy.ndarray, optional): Testing data. Defaults to None.
            y_validation (numpy.ndarray, optional): Labels of the testing data. Defaults to None.
            prefix (str):

        """

        output_path = os.path.join(self.output_path, prefix)
        safe_create_dir(output_path)

        output_model = os.path.join(output_path, "model.hdf5")
        output_weights = os.path.join(output_path, "weights.hdf5")

        if self.force_train or not os.path.exists(output_model) or self.fine_tuning:
            print('-- training ...', flush=True)
            print('-- training size:', x_train.shape, flush=True)

            # -- compute the class weights for unbalanced datasets
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
            print('-- class_weights', class_weights, flush=True)

            # -- convert class vectors to binary class matrices.
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            print('--y_train', y_train)
            if y_validation is not None:
                y_validation = keras.utils.to_categorical(y_validation, self.num_classes)

            # -- fit the model
            self.fit_model(x_train, y_train,
                           x_validation=x_validation, y_validation=y_validation, class_weights=class_weights, output_path=output_path)

            # -- save the fitted model
            print("-- saving model", output_model)
            sys.stdout.flush()

            self.model.save(output_model)
            self.model.save_weights(output_weights)
        else:
            print('-- model already exists in', output_model)
            sys.stdout.flush()

    def testing(self, x_test, y_test, prefix='', test_fnames=None, test_labels=None, visualization=False):
        """ This method is responsible for testing the fitted model.

        Args:
            x_test (numpy.ndarray): Testing data
            y_test (numpy.ndarray): Labels of the Testing data
            prefix (str):

        Returns:
            dict: A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}

        """

        output_path = os.path.join(self.output_path, prefix)
        safe_create_dir(output_path)

        output_model = os.path.join(output_path, "model.hdf5")

        # -- configure the GPU that will be used
        self.set_gpu_configuration()

        # -- load the fitted model
        self.model = keras.models.load_model(output_model,
                                             custom_objects={'categorical_crossentropy': keras.metrics.categorical_crossentropy,
                                                             'binary_crossentropy': keras.metrics.binary_crossentropy,
                                                             'categorical_hinge': keras.losses.categorical_hinge,
                                                             'metric_bal_accuracy': metric_bal_accuracy,
                                                             'metric_hter': metric_hter,
                                                             },
                                             )

        if self.testing_best_weights:
            best_weights = output_path + "/weights.best.hdf5"
            self.model.load_weights(best_weights)

        # -- normalization step
        for i in range(len(x_test)):
            x_test[i] = x_test[i] / 255.

        # -- generates output predictions for the testing data.
        scores = self.model.predict(x_test, batch_size=self.batch_size, verbose=0)

        # -- get the predicted scores and labels for the testing data
        y_pred = np.argmax(scores, axis=1)
        y_scores = scores[:, 1]

        # -- define the output dictionary
        r_dict = {'gt': y_test,
                  'predicted_labels': y_pred,
                  'predicted_scores': y_scores,
                  }

        return r_dict

    def extract_features(self, x, output_fname, prefix=''):
        """ This method is responsible for testing the fitted model.

        Args:
            x (numpy.ndarray):
            output_fname (str):
            prefix (str):

        Returns:
            dict: A dictionary with the ground-truth, the predicted scores and the predicted labels for the testing data, such as {'gt': y_test, 'predicted_labels': y_pred, 'predicted_scores': y_scores}

        """

        output_path = os.path.join(self.output_path, prefix)
        output_model = os.path.join(output_path, "model.hdf5")

        # # -- configure the GPU that will be used
        # self.set_gpu_configuration()

        # -- load the fitted model
        model = keras.models.load_model(output_model)

        # -- remove last layer
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        # -- extract the features
        features = self.model.predict(x, batch_size=self.batch_size, verbose=0)
        np.save(output_fname, features)
