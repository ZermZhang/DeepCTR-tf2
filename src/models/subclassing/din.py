#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : din
@Software       : PyCharm
@Modify Time    : 2022/5/25 10:33
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
import tensorflow as tf

from . import ModelBaseBuilder
from src.utils.config import Config


class DINBuilder(ModelBaseBuilder):
    def __init__(self, config_: Config, *args, **kwargs):
        super(DINBuilder, self).__init__(config_, *args, **kwargs)

        self.sequence_feature = [key for key, val in config_.feature_config.items() if val['type'] == 'sequence']

        self.atten_layer = tf.keras.layers.Attention()
        self.dense_layer_1 = tf.keras.layers.Dense(
            units=64, activation='relu'
        )
        self.dense_layer_2 = tf.keras.layers.Dense(
            units=32, activation='relu'
        )
        self.dense_layer_3 = tf.keras.layers.Dense(
            units=16, activation='relu'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=1
        )

    def feature_encoder(self, inputs: dict):
        encoded_features = {}
        for feature_name, feature_config in self.preprocessing_config.items():
            if not self.emb_layers[feature_name]:
                encoded_features[feature_name] = tf.expand_dims(
                    self.encoder_layers[feature_name](inputs[feature_name]),
                    axis=1
                )
            else:
                embedding = self.emb_layers[feature_name](
                    self.encoder_layers[feature_name](inputs[feature_name])
                )
                encoded_features[feature_name] = embedding
        return encoded_features

    def call(self, inputs):
        return 0
