#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : linear
@Software       : PyCharm
@Modify Time    : 2020/8/31 21:23     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the linear model under tf2.0-keras
"""

import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


def sequence_linear():
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    )
    return model
