#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : linear
@Software       : PyCharm
@Modify Time    : 2021/7/27 19:44
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
from collections import Iterable

import tensorflow as tf


def linear():
    input_layer = tf.keras.layers.Dense(
        units=1,
        activation=None,
        kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer()
    )

