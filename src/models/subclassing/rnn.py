#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : rnn
@Software       : PyCharm
@Modify Time    : 2020/9/10 08:09     
@Author         : zermzhang
@version        : 1.0
@Desciption     : the rnn model
"""
import numpy as np
import tensorflow as tf

from src.datas import load_data


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)  # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)  # 获得 RNN 的初始状态
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)  # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
        logits = self.dense(output)
        if from_logits:  # from_logits 参数控制输出是否通过 softmax 函数进行归一化
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])


def tester(CONFIG):
    config_train = CONFIG.model_config

    num_epoches = config_train.get('num_epoches', 3)
    seq_length = config_train.get('seq_length', 40)
    batch_size = config_train.get('batch_size', 100)
    learning_rate = config_train.get('learning_rate', 0.01)
    optimizer = config_train.get('optimizer', 'Adam')

    data_loader = load_data.RnnDataLoader()
    model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for batch_index in range(num_epoches):
        X, y = data_loader.get_batch(seq_length=seq_length, batch_size=batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_sum(loss)
            print("batch {}: loss {}".format(batch_index, loss.numpy()))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print('diversity: {}'.format(diversity))
        for t in range(400):
            y_pred = model.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=1)
        print("\n")
    return 0
