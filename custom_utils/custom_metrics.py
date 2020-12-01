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
