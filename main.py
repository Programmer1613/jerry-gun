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
import tensorflow as tf

from data_input import gray_input_iter


def train():

    with tf.compat.v1.variable_scope('cnn'):
        image_input = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 32, 32, 1))
        image_input /= 255.0
        conv1_w = tf.compat.v1.get_variable(
            name='conv1_w',
            shape=[3, 3, 1, 32],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv1_b = tf.compat.v1.get_variable(
            name='conv1_b',
            shape=[32],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv1 = tf.nn.relu(tf.nn.conv2d(
            image_input,
            conv1_w + conv1_b,
            [1, 2, 2, 1],
            padding='SAME'
        ))

        conv2_w = tf.compat.v1.get_variable(
            name='conv2_w',
            shape=[3, 3, 32, 64],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv2_b = tf.compat.v1.get_variable(
            name='conv2_b',
            shape=[64],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv2 = tf.nn.relu(tf.nn.conv2d(
            conv1,
            conv2_w + conv2_b,
            [1, 2, 2, 1],
            padding='SAME'
        ))
        conv2_flat = tf.reshape(conv2, [-1, 8*8*64])

        fc1_w = tf.compat.v1.get_variable(
            name='fc1_w',
            shape=[8*8*64, 1024],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        fc1_b = tf.compat.v1.get_variable(
            name='fc1_b',
            shape=[1024],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        fc_1 = tf.nn.relu(tf.matmul(conv2_flat, fc1_w) + fc1_b)

        fc2_w = tf.compat.v1.get_variable(
            name='fc2_w',
            shape=[1024, 512],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        fc2_b = tf.compat.v1.get_variable(
            name='fc2_b',
            shape=[512],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        fc_2 = tf.nn.relu(tf.matmul(fc_1, fc2_w) + fc2_b)

    sess = tf.InteractiveSession()
    for x in gray_input_iter(50):
        sess.run(tf.compat.v1.global_variables_initializer())
        res = sess.run(fc_2, feed_dict={image_input: x})
        break


train()
