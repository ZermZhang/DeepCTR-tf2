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

from src.utils.config import Config


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
    def __init__(self, config_):
        self.dataset_config = config_.dataset_config
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


def get_data():
    # get the origin datas
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='')
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

    train_ds = df_to_dataset(dataframe, shuffle=True, batch_size_=4)
    return train_ds


class CustomDataLoader(object):
    def __init__(self, config_, mode='train', data_path=None):
        self.dataset_config = config_.dataset_config

        # 根据具体的mode，处理需要使用到的数据
        self.mode = mode
        assert self.mode in {'train', 'test'}, ('the data type {} is not supported'.format(self.mode))
        self.data_path = self.dataset_config[f'{self.mode}_path'] if data_path is None else data_path
        self.params = config_.read_data_params()

        # 针对数据的schema和default信息进行处理
        self.label_name = self.dataset_config['label_name']
        assert self.label_name is not None, f'the None label is not supported.'
        self.trainable_features = self.dataset_config['valid_feature_columns']
        self.sequence_columns = self.dataset_config['sequence_columns']

        # 读取数据的时候需要的一些配置参数
        self.field_delim = self.params['sep']
        self.all_column_defaults = self.dataset_config['all_column_defaults']
        self.all_column_names = self.dataset_config['all_column_names']

    def _parse_csv(self):

        def parser(value):
            columns = tf.io.decode_csv(value, record_defaults=self.all_column_defaults,
                                       field_delim='\t')
            features = {key: value for key, value in dict(zip(self.all_column_names, columns)).items()
                        if key in self.trainable_features}
            for feature_name in self.sequence_columns:
                split_value = tf.strings.split(features[feature_name], ',')
                features[feature_name] = split_value

            label = features.pop(self.label_name)
            return features, label

        return parser

    def input_fn(self, batch_size_, epochs_):
        dataset = tf.data.TextLineDataset(self.data_path).skip(1) \
            if self.params['header'] is True else tf.data.TextLineDataset(self.data_path)

        # parse the csv data
        dataset = dataset.map(self._parse_csv())

        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=10000, seed=123)
            dataset = dataset.repeat(epochs_)

        dataset = dataset.prefetch(2 * batch_size_).batch(batch_size_)
        return dataset


if __name__ == "__main__":
    CONFIG = Config('../examples/conf/')
    train_path = CONFIG.read_data_path('train')
    batch_size = CONFIG.read_data_batch_size()
    epochs = CONFIG.read_data_epochs()
    data_load = CustomDataLoader(CONFIG).input_fn(batch_size_=batch_size, epochs_=epochs)

    print(list(data_load.as_numpy_iterator())[0])
