#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : config
@Software       : PyCharm
@Modify Time    : 2020/9/6 20:49
@Author         : zermzhang
@version        : 1.0
@Desciption     : parsing the config file
"""
import os

import yaml


class Config(object):
    def __init__(self, config_path):
        abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        if not os.path.exists(abs_path):
            raise Exception("{config_path} not exists".format(config_path=abs_path))
        else:
            self.config_path = abs_path
            self.__conf_path_check(self.config_path)
            self.config = self.load(os.path.join(self.config_path, 'conf.yaml'))
            self.feature_config = self.load(os.path.join(self.config_path, 'feature.yaml'))

    @staticmethod
    def __conf_path_check(config_path):
        config_files = os.listdir(config_path)
        if 'conf.yaml' not in config_files:
            raise Exception("{} not existed!".format('conf.yaml'))
        if 'feature-old.yaml' not in config_files:
            raise Exception("{} not existed!".format('feature-old.yaml'))

    @staticmethod
    def load(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    # the dataset config info from configure file
    @property
    def dataset_config(self):
        return self.config.get('dataset', {})

    # the model config info from configure file
    @property
    def model_config(self):
        return self.config.get('model', {})

    @property
    def deep_model_config(self):
        return self.model_config.get('deep', {})

    @property
    def wide_model_config(self):
        return self.model_config.get('wide', {})

    # the detail info for data config
    def read_data_path(self, mode='train'):
        dataset_config = self.dataset_config
        if mode == 'train':
            data_path = dataset_config.get('train_path', '')
        elif mode == 'test':
            data_path = dataset_config.get('test_path', '')
        else:
            raise ("the mode {} is not supported".format(mode))
        return data_path

    def read_data_params(self):
        dataset_config = self.dataset_config
        params = dataset_config.get('params', {'head': True, 'sep': ','})
        return params

    def read_data_batch_size(self):
        dataset_config = self.dataset_config
        batch_size = dataset_config.get('batch_size', 128)
        return batch_size

    def read_data_epochs(self):
        dataset_config = self.dataset_config
        epochs = dataset_config.get('epochs', 10)
        return epochs

    # parser the feature column info
    def read_data_schema(self):
        schema = list(self.feature_config.keys())
        return schema

    def get_column_default(self):
        column_names = []
        column_defaults = []
        valid_columns = []

        def get_default_value(dtype):
            if (not dtype) or (dtype == 'string'):
                return ""
            elif dtype == 'int':
                return 0
            elif dtype == 'float':
                return 0.0
            else:
                raise Exception('dtype: {} is not supported'.format(dtype))

        for name, value in self.feature_config.items():
            if value.get('default', None) is not None:
                column_defaults.append(value['default'])
            else:
                column_defaults.append(get_default_value(value['type']))
            if value.get('params', None) is not None:
                valid_columns.append(name)

        return column_names, column_defaults, valid_columns

    def get_continuous_features_config(self):
        return self.feature_config.get('continuous_features', None)

    def get_sparse_features_config(self):
        return self.feature_config.get('sparse_features', None)

    def get_cross_features_config(self):
        return self.feature_config.get('cross_features', None)

    def get_embedding_features_config(self):
        return self.feature_config.get('embedding_features', None)

    def get_shared_embedding_features_config(self):
        return self.feature_config.get('shared_embedding_features', None)


if __name__ == "__main__":

    config_ = Config('./conf/')
    print(config_.config)
    CONFIG_TRAIN = config_.model_config
    print(CONFIG_TRAIN)
    features_config = config_.feature_config
    print(features_config)
