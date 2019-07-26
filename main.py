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
from data_input import unpickle
import matplotlib.pyplot as plt
import tensorflow as tf


def train():

    with tf.compat.v1.variable_scope('cnn') as scope:
        image_input = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 32, 32, 3))
        w1 = tf.compat.v1.get_variable(
            name='conv1_w',
            shape=[3, 3, 3, 32],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv1 = tf.nn.relu(tf.nn.conv2d(
            image_input,
            w1,
            [1, 2, 2, 1],
            padding='SAME'
        ))

        w2 = tf.compat.v1.get_variable(
            name='conv2_w',
            shape=[3, 3, 32, 64],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer()
        )
        conv2 = tf.nn.relu(tf.nn.conv2d(
            conv1,
            w2,
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

    x = unpickle('data/data_batch_1')[b'data'][0]
    x = [x.reshape(32, 32, 3)]
    sess = tf.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    res = sess.run(fc_2, feed_dict={image_input: x})
    print(res.shape)
    print(res)


train()
