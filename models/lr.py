#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : lr
@Software       : PyCharm
@Modify Time    : 2020/9/15 08:11     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""

import tensorflow as tf


class Lr(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        output = self.dense(inputs)
        return output


def sequence_lr():
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    )
    return model
