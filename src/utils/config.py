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
            config = yaml.safe_load(f)
        return config

    # the dataset config info from configure file
    @property
    def dataset_config(self):
        base_dataset_config = self.config.get('dataset', {})
        all_column_names, all_column_defaults, valid_feature_columns = self.get_column_default()
        label_name = [key for key, value in self.feature_config.items() if value['type'] == 'label']

        base_dataset_config['all_column_names'] = all_column_names
        base_dataset_config['all_column_defaults'] = all_column_defaults
        base_dataset_config['valid_feature_columns'] = valid_feature_columns
        base_dataset_config['label_name'] = label_name

        return base_dataset_config

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

        def get_default_value(identify):
            if identify in ('hashing', 'sequence', 'pre-trained'):
                return ''
            elif identify in ('numerical', 'label'):
                return 0.
            elif identify == 'no-use':
                return ''
            else:
                raise Exception(f'identify: {identify} is not supported.')

        for name, value in self.feature_config.items():
            column_names.append(name)

            if value.get('default', None) is not None:
                column_defaults.append(value['default'])
            else:
                column_defaults.append(get_default_value(value['type']))

            if value.get('type', None) is not None:
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

    config_ = Config('../examples/conf/')
    print(config_.config)
    train_config = config_.model_config
    print(train_config)
    features_config = config_.feature_config
    print(features_config)
    dataset_config = config_.dataset_config
    print(dataset_config)
