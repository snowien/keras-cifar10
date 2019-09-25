from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import os
import json
import warnings
import numpy as np
from keras.applications import backend
from keras.applications import layers
from keras.applications import models
from keras.applications import utils
from keras.datasets import cifar10
from keras.engine.input_layer import Input
from keras.preprocessing import image
from keras.optimizers import adam,sgd,rmsprop
import random
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt
from glob import glob
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':      #查询图像数据格式的值，channels_first或channels_last
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = layers.Conv2D(filters2, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet18(input_shape=None, classes=10, **kwargs):
    # Define the input as a tensor with shape input_shape
    x_input = Input(input_shape)

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #stage 1
    with tf.name_scope('stage1'):
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x_input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        print("stage1:" + str(x.shape))

    #stage 2
    with tf.name_scope('stage2'):
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = identity_block(x, 3, [64, 64], stage=2, block='b')
        x = identity_block(x, 3, [64, 64], stage=2, block='c')
        print("stage2:" + str(x.shape))

    #stage 3
    with tf.name_scope('stage3'):
        x = conv_block(x, 3, [128,128], stage=3, block='a')
        x = identity_block(x, 3, [128, 128], stage=3, block='d')
        print("stage3:" + str(x.shape))

    #stage 4
    with tf.name_scope('stage4'):
        x = conv_block(x, 3, [256,256], stage=4, block='a')
        x = identity_block(x, 3, [256, 256], stage=4, block='c')
        print("stage4:" + str(x.shape))

    #stage 5
    with tf.name_scope('stage5'):
        x = conv_block(x, 3, [512, 512], stage=5, block='a')
        x = identity_block(x, 3, [512, 512], stage=5, block='c')
        print("stage5:" + str(x.shape))

    #full-connected layer
    with tf.name_scope('fc'):
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc10')(x)

    # Create model.
    model = models.Model(x_input, x, name='resnet18')
    return model
