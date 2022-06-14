#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : lr
@Software       : PyCharm
@Modify Time    : 2020/9/15 08:11     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the lr model
"""

import tensorflow as tf

from src.utils.model_utils import classes_activation_check

from . import ModelBaseBuilder


class Lr(ModelBaseBuilder):
    def __init__(self, config, *args, **kwargs):
        super(Lr, self).__init__(config, *args, **kwargs)
        self.config = config

        self.classes = self.config.model_config['classes']
        self.wide_params = self.config.wide_model_config
        self.activation = tf.keras.activations.get(self.wide_params['activation'])

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=self.classes,
            activation=self.activation,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        encoded_features = []
        for feature_name, feature_config in self.config.items():
            if not self.emb_layers[feature_name]:
                encoded_features.append(
                    self.encoder_layers[feature_name](inputs[feature_name])
                )
            else:
                encoded_features.append(
                    self.emb_layers[feature_name](self.encoder_layers[feature_name](inputs[feature_name]))
                )
        x = self.flatten(encoded_features)
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
