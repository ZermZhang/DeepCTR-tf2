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
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.losses import deserialize


@keras_export('keras.losses.get')
def get(self, identifier):
    if identifier is None:
        return None

    # some custom loss function
    if identifier == 'reduce_sum_sparse_categorical_crossentropy':
        return CustomLoss.reduce_sum_sigmoid_crossentropy_loss
    if identifier == 'reduce_mean_squre_loss':
        return CustomLoss.reduce_mean_squre_loss
    if identifier == 'reduce_sum_sigmoid_crossentropy_loss':
        return CustomLoss.reduce_sum_sigmoid_crossentropy_loss

    # the builtin loss function
    if isinstance(identifier, str):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if callable(identifier):
        return identifier
    raise ValueError(f'Could not interpret loss function identifier: {identifier}')


class CustomLoss:
    """
    some custom loss function
    """
    @staticmethod
    def reduce_sum_sparse_categorical_crossentropy(y_true, y_pred):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def reduce_mean_squre_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    @staticmethod
    def reduce_sum_sigmoid_crossentropy_loss(y_true, y_pred):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
