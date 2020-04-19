# -*- coding: utf-8 -*-

from antispoofing.sfsnet.classification.svm import SVM
# from antispoofing.sfsnet.classification.cnn0 import CNN0
# from antispoofing.sfsnet.classification.cnn1 import CNN1
# from antispoofing.sfsnet.classification.cnn2 import CNN2
# from antispoofing.sfsnet.classification.cnn3 import CNN3
# from antispoofing.sfsnet.classification.cnn4 import CNN4
# from antispoofing.sfsnet.classification.cnn5 import CNN5
from antispoofing.sfsnet.classification.cnn6 import CNN6
from antispoofing.sfsnet.classification.cnn7 import CNN7
from antispoofing.sfsnet.classification.cnn8 import CNN8
from antispoofing.sfsnet.classification.cnn9 import CNN9
from antispoofing.sfsnet.classification.cnn10 import CNN10
from antispoofing.sfsnet.classification.cnn11 import CNN11
from antispoofing.sfsnet.classification.cnn12 import CNN12

# from antispoofing.sfsnet.classification.cnn30 import CNN30
# from antispoofing.sfsnet.classification.cnn32 import CNN32
# from antispoofing.sfsnet.classification.cnn33 import CNN33
# from antispoofing.sfsnet.classification.cnn34 import CNN34
# from antispoofing.sfsnet.classification.cnn1a import CNN1A
# from antispoofing.sfsnet.classification.cnn1d import CNN1D
# from antispoofing.sfsnet.classification.cnn3e import CNN3E
# from antispoofing.sfsnet.classification.cnn3f import CNN3F
# from antispoofing.sfsnet.classification.cnn3g import CNN3G
# from antispoofing.sfsnet.classification.cnn3h import CNN3H
# from antispoofing.sfsnet.classification.cnn3i import CNN3I
# from antispoofing.sfsnet.classification.cnn3j import CNN3J
# from antispoofing.sfsnet.classification.cnn41 import CNN41
# from antispoofing.sfsnet.classification.cnn42 import CNN42
# from antispoofing.sfsnet.classification.cnn44 import CNN44
# from antispoofing.sfsnet.classification.cnn45 import CNN45

from antispoofing.sfsnet.classification.cnn71 import CNN71
from antispoofing.sfsnet.classification.cnn72 import CNN72
from antispoofing.sfsnet.classification.cnn73 import CNN73
from antispoofing.sfsnet.classification.cnn81 import CNN81
from antispoofing.sfsnet.classification.baseclassifier import loss_hter, categorical_focal_loss
from keras import losses


ml_algo = {
           # 0: CNN0,
           # 1: CNN1,
           # 2: CNN2,
           # 3: CNN3,
           # 4: CNN4,
           # 5: CNN5,
           6: CNN6,
           7: CNN7,
           8: CNN8,
           9: CNN9,
           10: CNN10,
           11: CNN11,
           12: CNN12,

           # 24: CNN3E,
           # 25: CNN3F,
           # 26: CNN3G,
           # 27: CNN3H,
           # 28: CNN3I,
           # 29: CNN3J,
           # 30: CNN30,
           # 32: CNN32,
           # 33: CNN33,
           # 34: CNN34,
           # 41: CNN41,
           # 42: CNN42,
           # 44: CNN44,
           # 45: CNN45,

           71: CNN71,
           72: CNN72,
           73: CNN73,
           81: CNN81,
           }


losses_functions = {0: 'categorical_crossentropy',
                    1: 'sparse_categorical_crossentropy',
                    2: 'binary_crossentropy',
                    3: 'hinge',
                    4: 'categorical_hinge',
                    5: 'cosine_proximity',
                    6: 'kullback_leibler_divergence',
                    7: categorical_focal_loss(alpha=.25, gamma=2),
                    }


optimizer_methods = {0: 'SGD',
                     1: 'Adam',
                     2: 'Adagrad',
                     3: 'Adadelta',
                     4: 'Adamax',
                     }

