#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : layers
@Software       : PyCharm
@Modify Time    : 2020/9/11 08:13     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the utils functions for layers
"""
import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    """
    the custom layers
    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_variable(name='w', shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b', shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
