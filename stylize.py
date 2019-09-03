#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    main.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    Description of this file

    :author: Jiayang
    :copyright: (c) 2019, Tungee
    :date created: 2019-07-22

"""
import scipy.io
import numpy as np
import tensorflow as tf


def build_model(h, w):
    net = dict()

    def conv_layer(_input, _w):
        return tf.nn.conv2d(_input, w, [1, 1, 1, 1], 'SAME')

    def relu_layer(_input, _b):
        return tf.nn.relu(_input + _b)

    vgg_rawnet = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_rawnet['layers'][0]

    net['input'] = tf.Variable(np.zeros((1, h, w, 3)), dtype=tf.float32)

    # -------------------------------------------------------------- #

    net['conv-1-1'] = conv_layer(
        net['input'], tf.constant(vgg_layers[0][0][0][2][0][0])
    )
    net['relu-1-1'] = relu_layer(
        net['conv-1-1'], tf.constant(vgg_layers[0][0][0][2][0][1])
    )
    net['conv-1-2'] = conv_layer(
        net['relu-1-1'], tf.constant(vgg_layers[2][0][0][2][0][0])
    )
    net['relu-1-2'] = relu_layer(
        net['conv-1-2'], tf.constant(vgg_layers[2][0][0][2][0][1])
    )
    net['mp-1'] = tf.nn.max_pool(
        net['relu-1-2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'
    )

    # -------------------------------------------------------------- #

    net['conv-2-1'] = conv_layer(
        net['mp-1'], tf.constant(vgg_layers[0][0][0][2][0][0])
    )
    net['relu-2-1'] = relu_layer(
        net['conv-2-1'], tf.constant(vgg_layers[0][0][0][2][0][1])
    )
    net['conv-2-2'] = conv_layer(
        net['relu-2-1'], tf.constant(vgg_layers[2][0][0][2][0][0])
    )
    net['relu-2-2'] = relu_layer(
        net['conv-2-2'], tf.constant(vgg_layers[2][0][0][2][0][1])
    )
    net['mp-2'] = tf.nn.max_pool(
        net['relu-2-2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'
    )

    # -------------------------------------------------------------- #
