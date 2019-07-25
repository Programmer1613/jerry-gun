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

    image_input = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=(None, 32, 32, 3))
    W = tf.Variable(tf.random.normal([5, 5, 3, 3]))
    conv = tf.nn.conv2d(
        image_input,
        W,
        [1, 1, 1, 1],
        padding='SAME'
    )
    x = unpickle('data/data_batch_1')[b'data'][0]
    x = [x.reshape(32, 32, 3)]
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    res = sess.run(conv, feed_dict={image_input: x})
    print(res)




train()
