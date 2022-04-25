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


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output

    def build_graph(self, input_shape):
        input_ = tf.keras.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))
# how to summary the model struct
# test_model = MLP()
# test_model.build_graph(input_shape=(16,)).summary()
# how to plot the model struct
# tf.keras.utils.plot_model(
#     test_model.build_graph(input_shape=(16, )),
#     to_file='./test_model.png',
#     show_shapes=True
# )


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

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output = self.output_layers(x)
        return output

    def build_graph(self, input_shape=None):
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
