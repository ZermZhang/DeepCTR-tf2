#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : runner
@Software       : PyCharm
@Modify Time    : 2020/9/7 08:52     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the scheduling logits for different models
"""

import tensorflow as tf


# TODO: too many inputs for wrapper.
# TODO: The ideal wrapper's inputs should be the must inputs, likes optimizer, epochs, loss_func···
# TODO: AND fit the data after get the compiled model
def call_backup(x=None, y=None, optimizer_func=None, epochs=100, loss_func=None):
    """
    wrapper for model build function
    run the model when call the model build function
    """
    def _call_backup(model_func):
        def _wrapper(*args, **kwargs):
            model = model_func(*args, **kwargs)
            for i in range(epochs):
                with tf.GradientTape() as tape:
                    y_pred = model(x)
                    loss = loss_func(y_pred, y)

                grads = tape.gradient(loss, model.variables)
                optimizer_func.apply_gradients(grads_and_vars=zip(grads, model.variables))
            return model
        return _wrapper
    return _call_backup


def model_train(datas, model, loss_func, optimizer_func, **kwargs):
    """
    :param datas: the data loader can get train AND eval datas
    :param model: the tf model
    :param loss_func:  the loss function for model
    :param optimizer_func: the optimizer for model
    :param kwargs:
    :return:
    """
    # train_data, test_data = datas.get("train", None), datas.get("test", None) if isinstance(datas, dict) else (None, None)
    # train_data_num = len(train_data) if train_data is not None else datas.num_train_data

    num_batches = int(datas.num_train_data // kwargs['batch_size'] * kwargs['num_epoches'])
    for batch_index in range(num_batches):
        features, labels = datas.get_batch(kwargs['batch_size'])
        with tf.GradientTape() as tape:
            lables_pred = model(features)
            loss = loss_func(labels, lables_pred)
            print("batch {batch_index}: loss {loss}".format(batch_index=batch_index, loss=loss))

        grads = tape.gradient(loss, model.variables)
        optimizer_func.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return model


def model_evaluate(datas, model, metrics, **kwargs):
    """
    :param datas: the data loader can get eval datas
    :param model: the tf model
    :param metrics: the list for metrics
    :param kwargs:
    :return:
    """
    num_batches = int(datas.num_test_data // kwargs['batch_size'])

    metric_funcs = [tf.keras.metrics.get(metric) for metric in metrics]

    results = []
    for metric in metric_funcs:
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * kwargs['batch_size'], (batch_index + 1) * kwargs['batch_size']
            y_pred = model.predict(datas.test_data[start_index: end_index])
            metric.update_state(y_true=datas.test_label[start_index: end_index], y_pred=y_pred)
        results.append(metric)

    return results
