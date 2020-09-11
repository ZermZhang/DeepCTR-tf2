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

from models import linear, mlp
from datas import load_data
from utils import config, runner, layers


class WideAndDeep(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.linear = linear.Linear()
        self.dnn = mlp.DNN(config)

    def call(self, input):
        linear_output = self.linear(input)
        dnn_output = self.dnn(input)
        return linear_output, dnn_output


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
    num_batches = int(data_loader.num_train_data // batch_size * num_epoches)
    for batch_index in range(num_batches):
        features, lables = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            wide_lables_pred, deep_lables_pred = model(features)
            wide_loss = loss_func(lables, wide_lables_pred)
            deep_loss = loss_func(lables, deep_lables_pred)
            loss = wide_loss + deep_loss
            print("batch {batch_index}: loss {loss}".format(batch_index=batch_index, loss=loss))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # evaluate
    metrics = ['SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy']
    results = runner.model_evaluate(data_loader, model, metrics, batch_size=batch_size)

    for (name, result) in zip(metrics, results):
        print("the {} evaluate result: {}".format(name, result.result()))
