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


def model_trainer(datas, model, loss, optimizer, **kwargs):
    """
    :param datas: the data loader can get train AND eval datas
    :param model: the tf model
    :param loss:  the loss function for model
    :param optimizer: the optimizer for model
    :param kwargs:
    :return:
    """
    # TODO: waiting Compatible with the normal data format
    # train_data, test_data = datas.get("train", None), datas.get("test", None) if isinstance(datas, dict) else (None, None)
    # train_data_num = len(train_data) if train_data is not None else datas.num_train_data

    num_batches = int(datas.num_train_data // kwargs['batch_size'] * kwargs['num_epoches'])
    for batch_index in range(num_batches):
        features, labels = datas.get_batch(kwargs['batch_size'])
        with tf.GradientTape() as tape:
            lables_pred = model(features)
            loss = loss(labels, lables_pred)
            print("batch {batch_index}: loss {loss}".format(batch_index=batch_index, loss=loss))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return model
