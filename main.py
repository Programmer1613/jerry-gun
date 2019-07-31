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
from model import (
    Generator,
)


def train():

    generator = Generator()

    for x in gray_input_iter(50):
        res = generator.generate_picture(x)
        print(res.shape)


train()
