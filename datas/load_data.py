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
import pandas as pd

import tensorflow as tf

from utils.config import Config


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


# the data loader for all numeric data
class CustomDataLoaderNumeric(object):
    def __init__(self, CONFIG):
        self.dataset_config = CONFIG.dataset_config
        self.train_path = self.dataset_config.get('train_path', None)
        self.test_path = self.dataset_config.get('test_path', None)
        assert self.train_path is not None and self.test_path is not None
        self.label_name = self.dataset_config.get('label_name', None)
        assert self.label_name is not None
        self.params = self.dataset_config.get('params', None)
        (self.train_data, self.train_label) = self._get_csv_data(self.train_path,
                                                                 self.params, self.label_name)
        (self.test_data, self.test_label) = self._get_csv_data(self.test_path,
                                                               self.params, self.label_name)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    @staticmethod
    def _get_csv_data(data_path, params, label_name):
        data = pd.read_csv(data_path, **params)
        features = data[[col for col in data.columns if col != label_name]]
        labels = data[[label_name]]
        return features, labels

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return np.array(self.train_data)[index, :], np.array(self.train_label)[index, :]


class CustomDataLoader(object):
    def __init__(self, CONFIG, data_path, mode='train'):
        self.dataset_config = CONFIG.dataset_config
        self.schema = CONFIG.read_data_schema()
        self.mode = mode
        assert self.mode in {'train', 'test'}, ('the data type {} is not supported'.format(self.mode))

        self.data_path = data_path

        self.params = CONFIG.read_data_params()
        self.label_name = self.dataset_config.get('label_name', None)
        assert self.label_name is not None
        self.column_names, self.column_defaults, self.valid_columns = CONFIG.get_column_default()

    def _parse_csv(self):
        column_names = self.column_names
        column_defaults = self.column_defaults
        field_delim = self.params['sep']

        def parser(value):
            columns = tf.io.decode_csv(value, record_defaults=column_defaults, field_delim=field_delim)
            features = {key: value for key, value in dict(zip(column_names, columns)).items() if key in self.valid_columns}
            label = features.pop(self.label_name)
            return features, label

        return parser

    def input_fn(self, batch_size, epochs):
        dataset = tf.data.TextLineDataset(self.data_path).skip(1) if self.params['header'] is True else tf.data.TextLineDataset(
            self.data_path)

        # parse the csv data
        dataset = dataset.map(self._parse_csv())

        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=10000, seed=123)
            dataset = dataset.repeat(epochs)

        dataset = dataset.prefetch(2 * batch_size).batch(batch_size)
        return dataset


def get_dta():
    # get the origin datas
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)
    dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0., 1.)

    # Drop un-used columns.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

    # generate the dataset
    def df_to_dataset(dataframe_, shuffle=True, batch_size_=32):
        dataframe_ = dataframe_.copy()
        labels = dataframe_.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe_), labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe_))

        ds = ds.batch(batch_size_)
        ds = ds.prefetch(batch_size_)

        return ds

    train_ds = df_to_dataset(dataframe, shuffle=True, batch_size=4)
    return train_ds


if __name__ == "__main__":
    CONFIG = Config('./conf/')
    train_path = CONFIG.read_data_path('train')
    batch_size = CONFIG.read_data_batch_size()
    epochs = CONFIG.read_data_epochs()
    data_load = CustomDataLoader(CONFIG, train_path).input_fn(batch_size=batch_size, epochs=epochs)

    print(list(data_load.as_numpy_iterator())[0])
