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
        if 'feature.yaml' not in config_files:
            raise Exception("{} not existed!".format('feature.yaml'))

    @staticmethod
    def load(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
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
        data_path = ''
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

    def read_data_schema(self):
        dataset_config = self.dataset_config
        schema = dataset_config.get('schema', {})
        return schema

    def read_data_batch_size(self):
        dataset_config = self.dataset_config
        batch_size = dataset_config.get('batch_size', 128)
        return batch_size

    def read_data_epochs(self):
        dataset_config = self.dataset_config
        epochs = dataset_config.get('epochs', 10)
        return epochs

    def get_column_default(self):
        schema = self.read_data_schema()
        column_names = list(schema.keys())
        column_defaults = []

        def get_default_value(dtype):
            if (not dtype) or (dtype == 'string'):
                return ""
            elif dtype == 'int':
                return 0
            elif dtype == 'float':
                return 0.0
            else:
                raise Exception('dtype: {} is not supported'.format(dtype))

        for name, value in schema.items():
            if value.get('default', None) is not None:
                column_defaults.append(value['default'])
            else:
                column_defaults.append(get_default_value(value['type']))

        return column_names, column_defaults

    # parser the feature column info
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

    CONFIG = Config('./conf/conf.yaml')
    print(CONFIG.config)
    CONFIG_TRAIN = CONFIG.model_config
    print(CONFIG_TRAIN)
