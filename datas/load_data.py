#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : load_data
@Software       : PyCharm
@Modify Time    : 2020/9/1 19:24     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""
import numpy as np

import tensorflow as tf


class MNISTLoader(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()

        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=1)

        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)

        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        """
        :param batch_size: 对数据进行batch分组的时候，采用的batch大小
        :return: 本次的batch数据
        """
        # 随机采样的满足batch大小的数组下标值
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


class RnnDataLoader(object):
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index: index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)


if __name__ == "__main__":
    data_load = MNISTLoader()
