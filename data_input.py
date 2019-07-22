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
import urllib3


def download_data():
    download_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    rsp = urllib3.PoolManager().request('GET', download_url)
    with open('data/data.tar.gz', 'wb') as f:
        f.write(rsp.data)


def read_data():
    import os

    if not os.path.exists('data/'):
        os.makedirs('data/')

    data_size = os.path.getsize('data/')
    if data_size <= 4096:
        download_data()


read_data()
