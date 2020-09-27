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

    @property
    def dataset_config(self):
        return self.config.get('dataset', {})

    @property
    def model_config(self):
        return self.config.get('model', {})

    @property
    def deep_model_config(self):
        return self.model_config.get('deep', {})

    @property
    def wide_model_config(self):
        return self.model_config.get('wide', {})


if __name__ == "__main__":

    CONFIG = Config('./conf/conf.yaml')
    print(CONFIG.config)
    CONFIG_TRAIN = CONFIG.model_config
    print(CONFIG_TRAIN)
