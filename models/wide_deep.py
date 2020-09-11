#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : wide_deep
@Software       : PyCharm
@Modify Time    : 2020/9/9 08:08     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the wide and deep model
    1. firstly, implementation the add losses between linear AND DNN
"""

import tensorflow as tf

from models import linear, mlp


class WideAndDeep(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.linear = linear.Linear()
        self.dnn = mlp.DNN()

    def call(self, input):
        linear_output = self.linear(input)
        dnn_output = self.dnn(input)
        return linear_output, dnn_output

