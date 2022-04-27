#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : mlp.py
@Software       : PyCharm
@Modify Time    : 2020/9/1 19:47     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""

import tensorflow as tf

from utils.feature_builder import FeatureColumnBuilder
from utils.config import Config
from custom_utils.custom_layers import EncodedFeatureBuilder
from utils.feature_preprocesing import StaticEncodedFeatureBuilder


"""
MLP模型，和utils.feature_preprocessing.EncodedFeatureBuilder联动，已经调通
model = MLP(config)
model.build_graph().summary()

tf.keras.utils.plot_model(model.build_graph(), to_file='./test_model.png', show_shapes=True)

"""


class MLP(tf.keras.Model):
    def __init__(self, config: dict):
        super(MLP, self).__init__()
        # init the feature config info
        self.config = config
        # definite the layers
        # self.feature_encoder_layers = EncodedFeatureBuilder()
        self.dense_layer = tf.keras.layers.Dense(32, activation='relu')
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(1)

        # init the preprocessing layer
        self.encoders = {}
        self.all_inputs = {}
        for feature_name, feature_config in self.config.items():
            encoder_layer = StaticEncodedFeatureBuilder(feature_name, feature_config)
            self.encoders[feature_name] = encoder_layer
            self.all_inputs[feature_name] = encoder_layer.inputs

    def call(self, inputs):
        encoded_features = []
        for feature_name, feature_config in self.config.items():
            encoder_layer = self.encoders[feature_name]
            encoded_features.append(
                encoder_layer.emb_layer(encoder_layer.feature_encoder(inputs[feature_name]))
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


class DNN(tf.keras.Model):
    def __init__(self, config: Config, feature_builder: FeatureColumnBuilder):
        super(DNN, self).__init__()
        self.classes = config.model_config['classes']
        self.deep_params = config.deep_model_config

        self.feature_builder = feature_builder

        self.input_layer = tf.keras.layers.DenseFeatures(self.feature_builder.feature_columns.values())
        self.activation = tf.keras.activations.get(self.deep_params['activation'])
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layers = []
        for unit_num in self.deep_params['units']:
            self.hidden_layers.append(tf.keras.layers.Dense(units=unit_num, activation=self.activation))
        self.output_layers = tf.keras.layers.Dense(units=self.classes, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output = self.output_layers(x)
        return output

    def build_graph(self):
        input_ = self.feature_builder.inputs_list
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


def sequence_mlp():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ]
    )
    return model
