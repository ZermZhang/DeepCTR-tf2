#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : custom_losses
@Software       : PyCharm
@Modify Time    : 2020/9/22 08:59     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""
import tensorflow as tf


class LossesFunc(object):
    """
    the custom loss function
    """
    def __init__(self, name="reduce_sum_sparse_categorical_crossentropy"):
        self.name = name
        if self.name == "reduce_sum_sparse_categorical_crossentropy":
            self.loss = self.reduce_sum_sparse_categorical_crossentropy

    @staticmethod
    def reduce_sum_sparse_categorical_crossentropy(y_true, y_pred):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def reduce_mean_squre_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    @staticmethod
    def reduce_sum_sigmoid_crossentropy_loss(y_true, y_pred):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred
            )
        )
