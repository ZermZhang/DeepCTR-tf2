#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Software       : PyCharm
@Modify Time    : 2021/7/27 19:32
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
import tensorflow as tf
from custom_utils.custom_layers import StaticEncodedFeatureBuilder


class ModelBaseBuilder(tf.keras.Model):
    def __init__(self, config: dict, *args, **kwargs):
        super(ModelBaseBuilder, self).__init__(*args, **kwargs)
        # init the feature config info
        self.config = config
        # definite the input layers
        self.encoder_layers = {}
        self.emb_layers = {}
        self.all_inputs = {}
        self.init_input_layer()

    def init_input_layer(self):
        for feature_name, feature_config in self.config.items():
            feature_builder = StaticEncodedFeatureBuilder(feature_name, feature_config)
            self.encoder_layers[feature_name] = feature_builder.feature_encoder
            self.emb_layers[feature_name] = feature_builder.emb_layer
            self.all_inputs[feature_name] = feature_builder.inputs

    def build_graph(self):
        return tf.keras.models.Model(
            inputs=self.all_inputs,
            outputs=self.call(self.all_inputs)
        )





class MLP(tf.keras.Model):
    def __init__(self, config: dict, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        # init the feature config info
        self.config = config
        # definite the layers
        self.dense_layer = tf.keras.layers.Dense(32, activation='relu')
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(1)

        # init the preprocessing layer
        self.encoders = {}
        self.emb_layers = {}
        self.all_inputs = {}

    def build(self):
        for feature_name, feature_config in self.config.items():
            encoder_layer = StaticEncodedFeatureBuilder(feature_name, feature_config)
            self.encoders[feature_name] = encoder_layer.feature_encoder
            self.emb_layers[feature_name] = encoder_layer.emb_layer
            self.all_inputs[feature_name] = encoder_layer.inputs

    def call(self, inputs):
        encoded_features = []
        for feature_name, feature_config in self.config.items():
            if not self.emb_layers[feature_name]:
                encoded_features.append(
                    self.encoders[feature_name](inputs[feature_name])
                )
            else:
                encoded_features.append(
                    self.emb_layers[feature_name](self.encoders[feature_name](inputs[feature_name]))
                )
        x = tf.keras.layers.concatenate(encoded_features)
        x = self.dense_layer(x)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x

    def build_graph(self):
        return tf.keras.models.Model(
            inputs=self.all_inputs,
            outputs=self.call(self.all_inputs)
        )
