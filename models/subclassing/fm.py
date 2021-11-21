#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : fm
@Software       : PyCharm
@Modify Time    : 2020/10/22 08:44
@Author         : zermzhang
@version        : 1.0
@Desciption     : the code for FM model
"""

import tensorflow as tf

from subclassing import linear
from utils import feature_columns, config
from datas import load_data


class FM(linear.Linear):
    def __init__(self, deep_columns):

        super().__init__(deep_columns.values())
        self.deep_columns = deep_columns

    def call(self, input):
        # get the logits for the first order part
        ori_linear_logits = self.input_layers(input)
        linear_logits = self.get_dense_layers(ori_linear_logits)

        # get the logits for the second order part
        dense_value_list, sparse_emb_list = feature_columns.input_from_feature_columns(input, self.deep_columns)

        sparse_emb_list = tf.keras.layers.Concatenate(axis=-1)(sparse_emb_list)
        sparse_emb_list = tf.squeeze(sparse_emb_list, axis=1)

        sum_square = tf.square(tf.reduce_sum(sparse_emb_list, axis=1, keepdims=True))
        square_sum = tf.reduce_sum(sparse_emb_list * sparse_emb_list, axis=1, keepdims=True)
        second_order = square_sum - sum_square
        second_order_logits = 0.5 * second_order

        return second_order_logits + linear_logits


def fm_runner(CONFIG, deep_columns):
    train_path = CONFIG.read_data_path('train')
    batch_size = CONFIG.read_data_batch_size()
    epochs = CONFIG.read_data_epochs()
    data_load = load_data.CustomDataLoader(CONFIG, train_path).input_fn(
        batch_size=batch_size,
        epochs=epochs
    )

    model = FM(deep_columns)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for i in range(100):
        X, y = next(iter(data_load))
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    print(model.variables)

    return 0


if __name__ == "__main__":
    CONFIG = config.Config('../../conf/')
    wide_columns, deep_columns = feature_columns.get_feature_columns(CONFIG)
    fm_runner(CONFIG, deep_columns)
