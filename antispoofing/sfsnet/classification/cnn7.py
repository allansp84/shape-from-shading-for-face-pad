# -*- coding: utf-8 -*-

import cv2
import json

from vis.utils import utils
from keras import activations
from sklearn.utils import class_weight
from vis.visualization import visualize_cam
from vis.visualization import visualize_activation
from vis.visualization import visualize_saliency

from antispoofing.sfsnet.utils import *
from antispoofing.sfsnet.classification.baseclassifier import BaseClassifier
from antispoofing.sfsnet.classification.baseclassifier import metric_hter
from antispoofing.sfsnet.classification.baseclassifier import metric_bal_accuracy
from antispoofing.sfsnet.classification.baseclassifier import categorical_focal_loss

# # -- get reproducible results
# os.environ['PYTHONHASHSEED'] = '{0}'.format(0)
# np.random.rand(0)
# rn.seed(0)
# tf.set_random_seed(0)


class CNN7(BaseClassifier):
    """ This class implements a detector for iris-based spoofing using a shallow Convolutional Neural Network (CNN).

    In this class, we define the architecture of our model and we implement the learning stage and the testing stage. Due to the huge
    amount of options available to implement the learning stage, we decide to parametizer the main components of this stage in order to
    find the best learning parameters for achieving a good classification result. In this way, for instance, it's possible to choose the
    loss function, the optimizer, so on, using the command line interface.

    Args:
        output_path (str):
        dataset (Dataset):
        dataset_b (Dataset):
        input_shape (int): Defaults to 200.
        epochs (int): Defaults to 50.
        batch_size (int): Defaults to 8.
        loss_function (str): Defaults to 'categorical_crossentropy'.
        lr (float): Defaults to 0.01.
        decay (float): Defaults to 0.0005.
        optimizer (str): Defaults to 'SGD'.
        regularization (float): Defaults to 0.1.
        device_number (int): Defaults to 0.
        force_train (bool): Defaults to False.
        filter_vis (bool): Defaults to False.
        layers_name (tuple): Defaults to ('conv_1',).
        fold (int):(default: 0)
    """

    def __init__(self, output_path, dataset, dataset_b=None, dataset_name="", dataset_b_name="", input_shape=200, n_channel=3,
                 frames_per_video=10, n_frames_for_testing=10, frame_fusion_type='max', load_n_frames=-1,
                 epochs=50, batch_size=8, loss_function='categorical_crossentropy', lr=0.01, decay=0.0005, optimizer='SGD', regularization=0.1,
                 device_number=0, force_train=False, filter_vis=False, layers_name=('conv_1',),
                 fold=0, fine_tuning=False, seed=0, testing_best_weights=False, force_testing=True, load_weights='',
                 linearize_hybrid_imgs=False, feature_visualization=False, debug=False):

        super(CNN7, self).__init__(output_path, dataset, dataset_b=dataset_b,
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
        print("-- CNN7 (new Shallow CNN)", flush=True)
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
        self.feature_visualization = feature_visualization

    def set_gpu_configuration(self):
        """ This function is responsible for setting up which GPU will be used during the processing and some configurations related
        to GPU memory usage when the TensorFlow is used as backend.
        """

        if self.verbose:
            print('-- setting the GPU configurations', flush=True)

        if 'tensorflow' in keras.backend.backend():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_number

            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(self.seed)
            rn.seed(self.seed)
            tf.set_random_seed(self.seed)

            K.clear_session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True, allocator_type='BFC')
            K.set_session(K.tf.Session(graph=tf.get_default_graph(),
                                       config=K.tf.ConfigProto(gpu_options=gpu_options,
                                                               intra_op_parallelism_threads=1,
                                                               inter_op_parallelism_threads=1,
                                                               allow_soft_placement=True,
                                                               log_device_placement=True)))

    def spoofnet(self, y):

        # -- first layer
        y = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal',
                   )(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(9, 9), strides=(8, 8))(y)

        # -- second layer
        y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal',
                   )(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(9, 9), strides=(8, 8))(y)

        return y

    def spoofnet_residual_block(self, y):

        # -- first layer
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal',
                   )(y)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = keras.layers.LeakyReLU(alpha=0.9)(x)

        # -- second layer
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal',
                   )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = keras.layers.LeakyReLU(alpha=0.9)(x)

        y = keras.layers.add([x, y])
        y = keras.layers.LeakyReLU(alpha=0.9)(y)
        y = Activation('relu')(y)

        return y

    def architecture_definition(self):
        """
        In this method we define the architecture of our CNN.
        """

        img_input = Input(shape=self.input_shape, name='input_1')

        x = self.spoofnet(img_input)
        x = keras.layers.Dropout(rate=0.25, seed=self.seed)(x)
        x = self.spoofnet_residual_block(x)
        x = keras.layers.Dropout(rate=0.25, seed=self.seed)(x)

        # -- flatten the output of the previous layer
        x = Flatten(name='flatten')(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(128, activation='relu')(x)

        output = Dense(self.num_classes, activation='softmax', name='predictions',
                       kernel_regularizer=keras.regularizers.l2(self.regularization),
                       )(x)

        self.model = keras.models.Model(img_input, output, name='mcnn')

        if self.load_weights:
            print('-- loading weights:', self.load_weights)
            self.model.load_weights(self.load_weights)

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

        for layer in model.layers[:n_layer_base_model]:
            layer.trainable = False

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
                                    # keras.metrics.categorical_crossentropy,
                                    # keras.metrics.binary_crossentropy,
                                    # keras.losses.categorical_hinge,
                                    # metric_bal_accuracy,
                                    # metric_hter,''
                                    # categorical_focal_loss,
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

            # log_dir = '{0}/logs'.format(output_path)
            # safe_create_dir(log_dir)
            # print('--logdir=%s' % log_dir, flush=True)
            # tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_grads=True,
            #                                            batch_size=self.batch_size, write_images=False)
            # callbacks.append(tensor_board)

            # file_path = output_path + '/logs/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            # check_point = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True)
            # callbacks.append(check_point)
            #
            # file_path = output_path + "/weights.best.hdf5"
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
                                             # custom_objects={'categorical_crossentropy': keras.metrics.categorical_crossentropy,
                                             #                 'binary_crossentropy': keras.metrics.binary_crossentropy,
                                             #                 'categorical_hinge': keras.losses.categorical_hinge,
                                                             # 'categorical_focal_loss': categorical_focal_loss(alpha=.25, gamma=2),
                                                             # 'metric_bal_accuracy': metric_bal_accuracy,
                                                             # 'metric_hter': metric_hter,
                                                             # },
                                             compile=False,
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

        # attack_idxs = np.where(test_labels==0)[0]
        # img_idx = scores[attack_idxs, 0].argmax()

        # import pdb; pdb.set_trace()

        if visualization:
            for img_idx, fname in enumerate(test_fnames):
                if '025.npy' in fname or '025.png' in fname:
                    self.vis_cam(x_test, scores, img_idx, fname, output_path)
                    self.vis_saliency(x_test, scores, img_idx, fname, output_path)
                    # self.vis_activation(x_test, scores, img_idx, fname, output_path)


        return r_dict

    def plot_map(self, grads, img, classlabel, score, output_fname):
        alpha = 0.6
        shape = grads.shape[:2]
        output = cv2.resize(img, shape)
        output = cv2.addWeighted(grads, alpha, output, 1 - alpha, 0)
        cv2.imwrite(output_fname, output[:,:,::-1])
        # fig, axes = plt.subplots(1,2,figsize=(14,5))
        # axes[0].imshow(img)
        # axes[1].imshow(img)
        # i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
        # fig.colorbar(i)
        # plt.suptitle("Pr(class={}) = {:5.2f}".format(classlabel, score))
        # plt.savefig(output_fname)
        # plt.close('all')

    def vis_cam(self, x_test, scores, img_idx, fname, output_path):
        json_fname = os.path.join(output_path, 'model.json')
        layer_idx = utils.find_layer_idx(self.model, 'predictions')
        self.model.layers[layer_idx].activation = activations.linear
        self.model = utils.apply_modifications(self.model)
        classlabel = ['PA', 'Genuine']

        for class_idx in range(len(classlabel)):

            seed_input = x_test[img_idx]
            fname_original = fname.split('/')[:-6] + ['3', 'origin_image',] + fname.split('/')[-4:]
            fname_original = '/'.join(fname_original)
            fname_original = "{}.png".format(os.path.splitext(fname_original)[0])
            img = cv2.imread(fname_original, cv2.IMREAD_COLOR)[:, :, ::-1]
            output_fname = os.path.join(output_path,
                                        'vis_cam',
                                        classlabel[class_idx],
                                        os.path.relpath(fname, self.output_path).replace('../', ''),
                                        )
            output_fname = "{}.png".format(os.path.splitext(output_fname)[0])
            safe_create_dir(os.path.dirname(output_fname))
            grad_top  = visualize_cam(self.model, layer_idx, class_idx, seed_input, penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)
            print('-- saving visualizations in', output_fname)
            self.plot_map(grad_top, img, classlabel[class_idx], scores[img_idx,class_idx], output_fname)

    def vis_saliency(self, x_test, scores, img_idx, fname, output_path):
        json_fname = os.path.join(self.output_path, 'model.json')
        layer_idx = utils.find_layer_idx(self.model, 'predictions')
        self.model.layers[layer_idx].activation = activations.linear
        self.model = utils.apply_modifications(self.model)
        classlabel = ['PA', 'Genuine']

        for class_idx in range(len(classlabel)):

            seed_input = x_test[img_idx]
            fname_original = fname.split('/')[:-6] + ['3', 'origin_image',] + fname.split('/')[-4:]
            fname_original = '/'.join(fname_original)
            fname_original = "{}.png".format(os.path.splitext(fname_original)[0])
            img = cv2.imread(fname_original, cv2.IMREAD_COLOR)[:, :, ::-1]
            output_fname = os.path.join(output_path,
                                        'vis_saliency',
                                        classlabel[class_idx],
                                        os.path.relpath(fname, output_path).replace('../', ''),
                                        )
            output_fname = "{}.png".format(os.path.splitext(output_fname)[0])
            safe_create_dir(os.path.dirname(output_fname))
            grad_top  = visualize_saliency(self.model, layer_idx, class_idx, seed_input)
            print('-- saving visualizations in', output_fname)
            self.plot_map(grad_top, img, classlabel[class_idx], scores[img_idx,class_idx], output_fname)
            # import pdb; pdb.set_trace()

    def vis_activation(self, x_test, scores, img_idx, fname, output_path):

        json_fname = os.path.join(self.output_path, 'model.json')
        layer_idx = utils.find_layer_idx(self.model, 'predictions')
        self.model.layers[layer_idx].activation = activations.linear
        self.model = utils.apply_modifications(self.model)
        classlabel = ['PA', 'Genuine']

        for class_idx in range(len(classlabel)):

            seed_input = x_test[img_idx]
            fname_original = fname.split('/')[:-6] + ['3', 'origin_image',] + fname.split('/')[-4:]
            fname_original = '/'.join(fname_original)
            fname_original = "{}.png".format(os.path.splitext(fname_original)[0])
            img = cv2.imread(fname_original, cv2.IMREAD_COLOR)[:, :, ::-1]
            output_fname = os.path.join(output_path,
                                        'vis_activation',
                                        classlabel[class_idx],
                                        os.path.relpath(fname, output_path).replace('../', ''),
                                        )
            output_fname = "{}.png".format(os.path.splitext(output_fname)[0])
            safe_create_dir(os.path.dirname(output_fname))

            try:
                grad_top  = visualize_activation(self.model, layer_idx, filter_indices=class_idx, input_range=(0., 1.))
                print('-- saving visualizations in', output_fname)
                self.plot_map(grad_top, img, classlabel[class_idx], scores[img_idx,class_idx], output_fname)
            except:
                pass

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
