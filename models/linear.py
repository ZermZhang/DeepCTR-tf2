#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : linear
@Software       : PyCharm
@Modify Time    : 2020/8/31 21:23     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the linear model under tf2.0-keras
"""

import tensorflow as tf

from datas import load_data
from utils import config, feature_columns


class LinearBase(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


class Linear(tf.keras.Model):
    def __init__(self, feature_columns):
        super().__init__()
        self.input_layers = tf.keras.layers.DenseFeatures(feature_columns)

    def call(self, input):
        inputs = self.input_layers(input)
        output = self.dense(inputs)
        return output


def sequence_linear():
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    )
    return model


def columns_sequence_linear(feature_columns):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.DenseFeatures(feature_columns),
            tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer()
            )
        ]
    )
    return model


def tester(CONFIG, wide_columns={}, deep_columns={}):
    config_train = CONFIG.model_config

    if not deep_columns:
        model = sequence_linear()
    else:
        model = columns_sequence_linear(deep_columns.values())

    train_path = CONFIG.read_data_path('train')
    batch_size = CONFIG.read_data_batch_size()
    epochs = CONFIG.read_data_epochs()
    data_load = load_data.CustomDataLoader(CONFIG, train_path).input_fn(batch_size=batch_size, epochs=epochs)

    # training
    model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['mean_absolute_error'], run_eagerly=True)
    model.fit(data_load, epochs=100)

    print(model.variables)
    return 0


def linear_runner():
    """
    the example runner for linear model
    """
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    model = LinearBase()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    print(model.variables)

    # array([[0.16730985],
    #        [1.14391],
    #        [2.1205118]], dtype=float32)
    return 0


if __name__ == "__main__":
    CONFIG = config.Config('./conf/')
    wide_columns, deep_columns = feature_columns.get_feature_columns(CONFIG)
    print(wide_columns, deep_columns)
    if not wide_columns and not deep_columns:
        tester(CONFIG)
    else:
        tester(CONFIG, wide_columns, deep_columns)
