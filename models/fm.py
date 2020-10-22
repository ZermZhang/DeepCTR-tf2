#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : fm
@Software       : PyCharm
@Modify Time    : 2020/10/22 08:44     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the code for FM model
"""

import tensorflow as tf
from models import linear
from utils import feature_columns


class FM(linear.Linear):
    def __init__(self, config, wide_columns, deep_columns):

        super().__init__(wide_columns)
        self.deep_columns = deep_columns

    def call(self, input):
        # get the logits for the first order part
        linear_logits = self.input_layers(input)

        # get the logits for the second order part
        sparse_emb_list, dense_value_list = feature_columns.input_from_feature_columns(input, self.deep_columns)
        sum_square = tf.square(tf.reduce_sum(sparse_emb_list, axis=1, keepdims=True))
        square_sum = tf.reduce_sum(sparse_emb_list * sparse_emb_list, axis=1, keepdims=True)
        second_order = square_sum - sum_square
        second_order_logits = 0.5 * tf.reduce_sum(second_order, axis=2, keepdims=False)

        return second_order_logits + linear_logits
