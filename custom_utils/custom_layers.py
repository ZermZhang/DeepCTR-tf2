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


class SequencePoolingLayer(tf.keras.layers.Layer):
    """
    the pooling layers for sequence features like UBS (User Behaviour Sequence)
    """
    def __init__(self, mode='mean', support_masking=False, **kwargs):
        super(SequencePoolingLayer, self).__init__(**kwargs)
        if mode not in ['mean', 'max', 'min']:
            raise Exception("the mode: {} is not supported!".format(mode))

        self.mode = mode
        self.epsilon = tf.constant(-2 ** 32 + 1, tf.float32)
        self.supports_masking = support_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When support_masking=True, input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list
            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return tf.reduce_max(hist, 1, keepdims=True)

        hist = tf.reduce_sum(uiseq_embed_list * mask, 1, keepdims=True)

        if self.mode == 'mean':
            hist = tf.divide(hist, tf.cast(user_behavior_length, tf.float32) + self.epsilon)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return None, 1, input_shape[-1]
        else:
            return None, 1, input_shape[0][-1]

    def get_config(self):
        config = {'mode': self.mode, 'support_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
