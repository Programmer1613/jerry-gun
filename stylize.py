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
from copy import deepcopy

import cv2
import scipy.io
import numpy as np
import tensorflow as tf


def build_model(h, w):
    net = dict()

    def conv_layer(_input, _w):
        return tf.nn.conv2d(_input, _w, [1, 1, 1, 1], 'SAME')

    def relu_layer(_input, _b):
        return tf.nn.relu(tf.add(_input, _b))

    vgg_rawnet = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_rawnet['layers'][0]

    def get_wight(_index):
        return tf.constant(vgg_layers[_index][0][0][2][0][0])

    def get_bias(_index):
        _bias = vgg_layers[_index][0][0][2][0][1]
        return tf.constant(np.reshape(_bias, _bias.size))

    def build_conv_layer(_depth, _args_index, _input):
        for _i in range(len(_args_index)):
            if _i == 0:
                net['conv-%s-%s' % (_depth, _i)] = conv_layer(net[_input], get_wight(_args_index[_i]))
            else:
                net['conv-%s-%s' % (_depth, _i)] = conv_layer(
                    net['relu-%s-%s' % (_depth, _i - 1)], get_wight(_args_index[_i]))
            net['relu-%s-%s' % (_depth, _i)] = relu_layer(
                net['conv-%s-%s' % (_depth, _i)], get_bias(_args_index[_i]))
        net['pool-%s' % _depth] = tf.nn.max_pool2d(
            net['relu-%s-%s' % (_depth, len(_args_index) - 1)],
            [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'
        )

    net['input'] = tf.Variable(np.zeros((1, h, w, 3), dtype=np.float32))

    build_conv_layer(0, [0, 2], 'input')
    build_conv_layer(1, [5, 7], 'pool-0')
    build_conv_layer(2, [10, 12, 14, 16], 'pool-1')
    build_conv_layer(3, [19, 21, 23, 25], 'pool-2')
    build_conv_layer(4, [28, 30, 32, 34], 'pool-3')

    return net


def get_content_loss(sess, input_image, model):
    loss_sum = 0.
    sess.run(model['input'].assign(input_image))

    def _get_loss(_layer_name):
        _a = sess.run(model[_layer_name])
        _x = model[_layer_name]
        _a = tf.convert_to_tensor(_a)

        _, _h, _w, _d = _a.get_shape()
        _loss = tf.reduce_sum(tf.pow(_a - _x, 2))
        _loss *= 0.5 * (_h.value*_w.value) ** 0.5
        return _loss

    loss_sum += 0.2 * _get_loss('conv-4-2')
    return loss_sum


def gram_matrix(mat):
    _, h, w, d = mat.get_shape()
    f = tf.reshape(mat, (h.value*w.value, d.value))
    return tf.matmul(tf.transpose(f), f)


def get_style_loss(sess, input_image, model):
    loss_sum = 0.
    sess.run(model['input'].assign(input_image))

    def _get_loss(_layer_name):
        _a = sess.run(model[_layer_name])
        _x = model[_layer_name]
        _a = tf.convert_to_tensor(_a)

        _a = gram_matrix(_a)
        _x = gram_matrix(_x)
        _h, _w = _a.get_shape()
        _loss = tf.reduce_sum(tf.pow(_a - _x, 2))
        _loss *= 0.5 * (_h.value*_w.value) ** 0.5
        return _loss

    loss_sum += 0.2 * _get_loss('relu-0-0')
    loss_sum += 0.2 * _get_loss('relu-1-0')
    loss_sum += 0.2 * _get_loss('relu-2-0')
    loss_sum += 0.2 * _get_loss('relu-3-0')
    loss_sum += 0.2 * _get_loss('relu-4-0')
    return loss_sum


def read_image(image_path, shape=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None
    image = image.astype(np.float32)
    if shape is not None:
        image = cv2.resize(image, dsize=shape, interpolation=cv2.INTER_AREA)
    image = image[..., :: -1]
    image = image[None, :, :, :]
    image -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return image


def write_image(image, path):
    image = deepcopy(image)
    image += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    image = image[..., ::-1]
    cv2.imwrite(path, image)


def get_optimizer(loss_func):
    return tf.contrib.opt.ScipyOptimizerInterface(
        loss_func, method='L-BFGS-B', options={'maxiter': 1, 'disp': 50})


def stylize_image(content_image_path, style_image_path):
    content_image = read_image(content_image_path)
    _, h, w, _ = content_image.shape
    style_image = read_image(style_image_path, (w, h))

    with tf.Graph().as_default():
        with tf.device('/gpu:0'), tf.Session() as sess:
            # init
            model = build_model(h, w)
            content_loss = get_content_loss(sess, content_image, model)
            style_loss = get_style_loss(sess, style_image, model)

            loss_func = 0.5 * content_loss + 0.5 * style_loss
            optimizer = get_optimizer(loss_func)

            # train
            sess.run(tf.global_variables_initializer())
            sess.run(model['input'].assign(content_image))
            optimizer.minimize(sess)

            # output
            output_image = sess.run(model['input'])
            write_image(output_image, 'output' + content_image)


stylize_image('zio.jpg', 'scream.jpg')

