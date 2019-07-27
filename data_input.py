#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    data_input.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    Description of this file

    :author: Jiayang
    :copyright: (c) 2019, Tungee
    :date created: 2019-07-22

"""
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def gray_input_iter(batch_size):
    result = list()

    objs = unpickle('data/test_batch')

    for x in objs[b'data']:
        if len(result) >= batch_size:
            yield result
            result = list()
        x = x.reshape(3, 32, 32)
        x[0] = x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114
        x[1] = x[0]
        x[2] = x[0]
        x = x[:1]
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)
        result.append(x)
    yield result
