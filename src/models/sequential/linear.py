#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : linear
@Software       : PyCharm
@Modify Time    : 2021/7/27 19:37
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
from collections import Iterable

import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential


def linear() -> Sequential:
    """
    :return: the linear model
    """
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    )
    return model


def columns_linear(feature_columns: Iterable) -> Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.DenseFeatures(feature_columns),
            tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer()
            )
        ]
    )
    return model
