#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : custom_metrics
@Software       : PyCharm
@Modify Time    : 2020/9/23 08:28
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
import tensorflow as tf


class SparseCategoticalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total


class MyPrecision(tf.keras.metrics.Metric):
    """
    计算自定义精准度，按照pred排序，计算topK中正样本的占比
    """
    def __init__(self, name='my_precision', **kwargs):
        super(MyPrecision, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, top_k=5):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # cal the precision for pred
        top_idx = tf.nn.top_k(y_pred, top_k).indices
        diag_idx = tf.tile(tf.expand_dims(tf.range(top_k), axis=1), [1,2])

        y_true_top = tf.gather_nd(tf.gather(y_true, axis=-1, indices=top_idx), diag_idx)
        y_pred_top = tf.gather_nd(tf.gather(y_pred, axis=-1, indices=top_idx), diag_idx)

        # cal the all positive cnt
        true_sum = tf.reduce_sum(y_true_top)
        self.total.assign_add(true_sum)

        # cal the all data cnt
        num = tf.cast(tf.size(y_true_top), dtype=tf.float32)
        self.count.assign_add(num)

    def result(self):
        return self.total / self.count
