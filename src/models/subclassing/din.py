#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : din
@Software       : PyCharm
@Modify Time    : 2022/5/25 10:33
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
import tensorflow as tf

from . import ModelBaseBuilder


class DIN(ModelBaseBuilder):
    def __init__(self, config, *args, **kwargs):
        super(DIN, self).__init__(*args, **kwargs)
        self.atten_layer = tf.keras.layers.Attention()
        self.dense_layer_1 = tf.keras.layers.Dense(
            units=64, activation='relu'
        )
        self.dense_layer_2 = tf.keras.layers.Dense(
            units=32, activation='relu'
        )
        self.dense_layer_3 = tf.keras.layers.Dense(
            units=16, activation='relu'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=1
        )
