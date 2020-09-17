#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : wide_deep
@Software       : PyCharm
@Modify Time    : 2020/9/9 08:08     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the wide and deep model
    1. firstly, implementation the add losses between linear AND DNN
"""

import tensorflow as tf

from models import lr, mlp
from datas import load_data
from utils import runner, layers
from utils.model_utils import classes_activation_check


class Wide(tf.keras.Model):
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
        return x


class Deep(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.classes = config.model_config['classes']
        self.deep_params = config.deep_model_config
        self.activation = tf.keras.activations.get(self.deep_params['activation'])
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layers = []
        for unit_num in self.deep_params['units']:
            self.hidden_layers.append(tf.keras.layers.Dense(units=unit_num, activation=self.activation))
        self.output_layers = tf.keras.layers.Dense(units=self.classes, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return x


class WideAndDeep(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.classes = config.model_config['classes']
        self.wide = Wide(config)
        self.deep = Deep(config)
        self._output_layers = tf.keras.layers.Dense(units=self.classes, activation='softmax')

    def call(self, input):
        wide_last = self.wide(input)
        deep_last = self.deep(input)
        output = self._output_layers(tf.concat([wide_last, deep_last], axis=-1))
        return output


def tester(CONFIG):
    config_train = CONFIG.model_config

    num_epoches = config_train.get('num_epoches', 3)
    batch_size = config_train.get('batch_size', 100)
    learning_rate = config_train.get('learning_rate', 0.01)

    model = WideAndDeep(CONFIG)
    data_loader = load_data.MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = layers.LossesFunc('reduce_sum_sparse_categorical_crossentropy').loss

    # training
    model = runner.model_train(data_loader, model, loss_func, optimizer, batch_size=batch_size, num_epoches=num_epoches)

    # evaluate
    metrics = ['SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy']
    results = runner.model_evaluate(data_loader, model, metrics, batch_size=batch_size)

    for (name, result) in zip(metrics, results):
        print("the {} evaluate result: {}".format(name, result.result()))

    # batch 2339: loss 15.38129997253418
    # the SparseCategoricalAccuracy evaluate result: 0.9689503312110901
    # the SparseCategoricalCrossentropy evaluate result: 0.10526901483535767
    return 0
