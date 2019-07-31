#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    model.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    Description of this file

    :author: Jiayang
    :copyright: (c) 2019, Tungee
    :date created: 2019-07-31

"""
import tensorflow as tf


class Generator(object):

    def __init__(self):

        self.image_input = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 32, 32, 1))
        self.image_input /= 255.0
        with tf.compat.v1.variable_scope('cnn'):
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
                self.image_input,
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

        with tf.compat.v1.variable_scope('generator'):
            gen_conv_w = tf.compat.v1.get_variable(
                name='gen_conv_w',
                shape=[3, 3, 1, 3],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_conv_b = tf.compat.v1.get_variable(
                name='gen_conv_b',
                shape=[3],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_conv = tf.nn.relu(tf.nn.conv2d(
                self.image_input,
                gen_conv_w + gen_conv_b,
                [1, 1, 1, 1],
                padding='SAME'
            ))
            color_input = tf.split(gen_conv, 3, 3)
            r_input = color_input[0]
            g_input = color_input[1]
            b_input = color_input[2]

            gen_r_conv_w = tf.compat.v1.get_variable(
                name='gen_r_conv_w',
                shape=[3, 3, 1, 1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_r_conv_b = tf.compat.v1.get_variable(
                name='gen_r_conv_b',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_r_conv = tf.nn.relu(tf.nn.conv2d(
                r_input,
                gen_r_conv_w + gen_r_conv_b,
                [1, 1, 1, 1],
                padding='SAME'
            ))
            gen_r_conv_flat = tf.reshape(gen_r_conv, [-1, 32*32*1])

            gen_g_conv_w = tf.compat.v1.get_variable(
                name='gen_g_conv_w',
                shape=[3, 3, 1, 1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_g_conv_b = tf.compat.v1.get_variable(
                name='gen_g_conv_b',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_g_conv = tf.nn.relu(tf.nn.conv2d(
                g_input,
                gen_g_conv_w + gen_g_conv_b,
                [1, 1, 1, 1],
                padding='SAME'
            ))
            gen_g_conv_flat = tf.reshape(gen_g_conv, [-1, 32*32*1])

            gen_b_conv_w = tf.compat.v1.get_variable(
                name='gen_b_conv_w',
                shape=[3, 3, 1, 1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_b_conv_b = tf.compat.v1.get_variable(
                name='gen_b_conv_b',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_b_conv = tf.nn.relu(tf.nn.conv2d(
                b_input,
                gen_b_conv_w + gen_b_conv_b,
                [1, 1, 1, 1],
                padding='SAME'
            ))
            gen_b_conv_flat = tf.reshape(gen_b_conv, [-1, 32*32*1])

            gen_r_w1 = tf.compat.v1.get_variable(
                name='gen_r_w1',
                shape=[32*32*1 + 512, 32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_r_b1 = tf.compat.v1.get_variable(
                name='gen_r_b1',
                shape=[32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            r_params = tf.transpose(
                tf.reshape(tf.concat([
                    tf.reshape(tf.transpose(gen_r_conv_flat), [-1]),
                    tf.reshape(tf.transpose(fc_2), [-1]),
                ], 0), [512 + 32 * 32, -1])
            )
            gen_r_flat = tf.nn.relu(tf.matmul(r_params, gen_r_w1) + gen_r_b1)

            gen_g_w1 = tf.compat.v1.get_variable(
                name='gen_g_w1',
                shape=[32*32*1 + 512, 32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_g_b1 = tf.compat.v1.get_variable(
                name='gen_g_b1',
                shape=[32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            g_params = tf.transpose(
                tf.reshape(tf.concat([
                    tf.reshape(tf.transpose(gen_g_conv_flat), [-1]),
                    tf.reshape(tf.transpose(fc_2), [-1]),
                ], 0), [512 + 32 * 32, -1])
            )
            gen_g_flat = tf.nn.relu(tf.matmul(g_params, gen_g_w1) + gen_g_b1)

            gen_b_w1 = tf.compat.v1.get_variable(
                name='gen_b_w1',
                shape=[32*32*1 + 512, 32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            gen_b_b1 = tf.compat.v1.get_variable(
                name='gen_b_b1',
                shape=[32*32],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer()
            )
            b_params = tf.transpose(
                tf.reshape(tf.concat([
                    tf.reshape(tf.transpose(gen_b_conv_flat), [-1]),
                    tf.reshape(tf.transpose(fc_2), [-1]),
                ], 0), [512 + 32 * 32, -1])
            )
            gen_b_flat = tf.nn.relu(tf.matmul(b_params, gen_b_w1) + gen_b_b1)

            gen_r = tf.reshape(gen_r_flat, [-1, 32, 32, 1])
            gen_g = tf.reshape(gen_g_flat, [-1, 32, 32, 1])
            gen_b = tf.reshape(gen_b_flat, [-1, 32, 32, 1])

            self.gen_output_image = tf.concat([gen_r, gen_g, gen_b], 3)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def generate_picture(self, image_input):
        res = self.sess.run(self.gen_output_image, feed_dict={
            self.image_input: image_input,
        })
        return res
