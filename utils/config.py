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
            self.config = self.load(self.config_path)

    @staticmethod
    def load(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
        return config

    # the dataset config info from configure file
    @property
    def dataset_config(self):
        return self.config.get('dataset', {})

    # the feature columns config info from configure file
    @property
    def feature_config(self):
        return self.config.get('features', {})

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

    def read_data_schema(self):
        dataset_config = self.dataset_config
        schema = dataset_config.get('schema', {})
        return schema

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
            if value['default'] is not None:
                column_defaults.append(value['default'])
            else:
                column_defaults.append(get_default_value(value['type']))

        return column_names, column_defaults


if __name__ == "__main__":

    CONFIG = Config('./conf/conf.yaml')
    print(CONFIG.config)
    CONFIG_TRAIN = CONFIG.model_config
    print(CONFIG_TRAIN)
