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

from utils.model_utils import classes_activation_check


class Lr(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.classes = config.model_config['classes']
        self.wide_params = config.wide_model_config
        self.activation = tf.keras.activations.get(self.wide_params['activation'])
        classes_activation_check(self.classes, self.activation)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=self.classes,
            activation=self.activation,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        x = self.flatten(inputs)
        output = self.dense(x)
        return output


def sequence_lr(config):
    classes = config.model_config['classes']
    wide_params = config.wide_model_config
    activation = tf.keras.activations.get(wide_params['activation'])
    classes_activation_check(classes, activation)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=classes,
                activation=activation,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer()
            )
        ]
    )
    return model
