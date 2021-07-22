#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : feature_columns_new
@Software       : PyCharm
@Modify Time    : 2021/7/20 19:02
@Author         : zermzhang
@version        : 1.0
@Desciption     :
@Config Description:
    {feature_name}:
        feature_type: {feature_type}
        config:
            dims: {dims}
            dtype: {dtype}
            feature_params:
                {feature_params}
            embed_params:
                {embed_params}
"""
import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import HashedCategoricalColumn, VocabularyFileCategoricalColumn, EmbeddingColumn


class FeatureColumnBuilder:
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.features_column = {}
        self.inputs_list = []

    def feature_builder(self):
        for feature_name, feature_config in self.config.items():
            feature_type = feature_config['feature_type']
            if feature_type == 'hashing':
                pass
            elif feature_type == 'vocabulary':
                pass
            elif feature_type == 'numerical':
                pass
            elif feature_type == 'pre-trained':
                pass
            elif feature_type == 'crossed':
                pass
            else:
                pass
