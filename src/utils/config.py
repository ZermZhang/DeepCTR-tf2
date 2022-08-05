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

        if not os.path.exists(config_path):
            raise Exception("{config_path} not exists".format(config_path=config_path))
        else:
            self.config_path = config_path
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

    # =============================================================
    # 配置信息的出口
    # =============================================================
    @property
    def dataset_config(self):
        """
        数据集的相关配置，主要包括：
        1. 数据集的基本配置信息（schema，default_value德国）
        2. 模型训练中需要用到的feature_col信息
        """
        base_dataset_config = self.config.get('dataset', {})
        all_column_names, all_column_defaults, valid_feature_columns, sequence_columns = self.get_column_default()
        label_name = [key for key, value in self.feature_config.items() if value['type'] == 'label'][0]

        base_dataset_config['all_column_names'] = all_column_names
        base_dataset_config['all_column_defaults'] = all_column_defaults
        base_dataset_config['valid_feature_columns'] = valid_feature_columns
        base_dataset_config['label_name'] = label_name
        base_dataset_config['sequence_columns'] = sequence_columns

        return base_dataset_config

    @property
    def model_config(self):
        """
        模型的基本配置，网络结构等信息
        """
        return self.config.get('model', {})

    @property
    def train_config(self):
        """
        训练过程需要的配置，optimizer，loss，learning_rate等
        """
        return self.config.get('train', {})

    @property
    def run_config(self):
        """
        对模型执行train和eval过程中需要使用的一些配置信息
        """
        return self.config.get('run', {})

    @property
    def preprocessing_config(self):
        """
        preprocessing_layer的相关配置，包括preprocessing_layer和embedding_layer
        """
        features_config_ = self.feature_config
        preprocessing_config_ = {}
        for feature_name, config in features_config_.items():
            if config['type'] not in ('label', 'no-use'):
                preprocessing_config_.update({feature_name: config})
        return preprocessing_config_

    # =============================================================
    # =============================================================

    def get_column_default(self):
        column_names = []
        column_defaults = []
        valid_columns = []
        sequence_columns = []

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

            if value.get('type', None) == 'sequence':
                sequence_columns.append(name)

        return column_names, column_defaults, valid_columns, sequence_columns

    # the detail info for data config
    def read_data_path(self, mode='train'):
        dataset_config_ = self.dataset_config
        if mode == 'train':
            data_path = dataset_config_.get('train_path', '')
        elif mode == 'test':
            data_path = dataset_config_.get('test_path', '')
        else:
            raise ("the mode {} is not supported".format(mode))
        return data_path

    def read_data_params(self):
        dataset_config_ = self.dataset_config
        params = dataset_config_.get('params', {'head': True, 'sep': ','})
        return params

    def read_data_batch_size(self):
        dataset_config_ = self.dataset_config
        batch_size = dataset_config_.get('batch_size', 128)
        return batch_size

    def read_data_epochs(self):
        dataset_config_ = self.dataset_config
        epochs = dataset_config_.get('epochs', 10)
        return epochs

    # parser the feature column info
    def read_data_schema(self):
        schema = list(self.feature_config.keys())
        return schema

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
    preprocessing_config = config_.preprocessing_config
    print(len(preprocessing_config))
    print(preprocessing_config)
