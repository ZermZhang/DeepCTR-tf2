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
from tensorflow.python.feature_column.feature_column_v2 import (
    CategoricalColumn
)


class FeatureBaseBuilder:
    def __init__(self, feature_name: str, dims: int, batch_size: int,
                 dtype: str, feature_params: dict, embed_params: dict):
        self.feature_name = feature_name
        self.feature_params = {'key': feature_name}
        if feature_params is None:
            pass
        else:
            self.feature_params.update(feature_params)
        self.embed_params = embed_params
        self.dims = dims
        if dtype == 'string':
            self.dtype = tf.string
        elif dtype == 'float':
            self.dtype = tf.float64
        else:
            self.dtype = tf.int16

        self.inputs = tf.keras.Input(shape=(batch_size, dims), dtype=self.dtype)

    def input_layer(self):
        return tf.feature_column.categorical_column_with_hash_bucket(**self.feature_params)

    def __call__(self, *args, **kwargs):
        source_column = self.input_layer()
        if isinstance(source_column, CategoricalColumn):
            return tf.feature_column.embedding_column(source_column, **self.embed_params)
        else:
            return source_column


class VocabEmbeddingBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return tf.feature_column.categorical_column_with_vocabulary_list(**self.feature_params)


class NumericalBuilder(FeatureBaseBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalizer_fn_builder(scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x - normalization_params[0]) / (normalization_params[1] - normalization_params[0])
        elif scaler == 'standard':
            return lambda x: (x - normalization_params[0]) / normalization_params[1]
        elif scaler == 'log':
            return lambda x: tf.math.log(x)
        else:
            return None

    def input_layer(self):
        f_tran, normalization = self.feature_params.get('transformer', None), \
                                self.feature_params.get('normalization', None)

        normalizer_fn = self.normalizer_fn_builder(f_tran, normalization)
        return tf.feature_column.numeric_column(key=self.feature_name, normalizer_fn=normalizer_fn)

    def __call__(self, *args, **kwargs):
        source_column = self.input_layer()
        if self.embed_params is not None:
            return tf.feature_column.bucketized_column(source_column, **self.embed_params)
        else:
            return source_column


class FeatureColumnBuilder:
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.feature_columns = {}
        self.inputs_list = {}
        self.batch_size = kwargs.get('batch_size', 512)

    def feature_builder(self):
        for feature_name, feature_config in self.config.items():
            feature_type = feature_config['feature_type']
            dims = feature_config['dims']
            dtype = feature_config['dtype']
            feature_params = feature_config['params']['feature_params']
            embed_params = feature_config['params']['embed_params']
            if feature_type == 'hashing':
                feature_builder = FeatureBaseBuilder(
                    feature_name=feature_name,
                    dims=dims,
                    batch_size=self.batch_size,
                    dtype=dtype,
                    feature_params=feature_params,
                    embed_params=embed_params
                )
                self.feature_columns[feature_name] = feature_builder()
                self.inputs_list[feature_name] = feature_builder.inputs
            elif feature_type == 'vocabulary':
                feature_builder = VocabEmbeddingBuilder(
                    feature_name=feature_name,
                    dims=dims,
                    batch_size=self.batch_size,
                    dtype=dtype,
                    feature_params=feature_params,
                    embed_params=embed_params
                )
                self.feature_columns[feature_name] = feature_builder()
                self.inputs_list[feature_name] = feature_builder.inputs
            elif feature_type == 'numerical':
                feature_builder = NumericalBuilder(
                    feature_name=feature_name,
                    dims=dims,
                    batch_size=self.batch_size,
                    dtype=dtype,
                    feature_params=feature_params,
                    embed_params=embed_params
                )
                self.feature_columns[feature_name] = feature_builder()
                self.inputs_list[feature_name] = feature_builder.inputs
            elif feature_type == 'pre-trained':
                pass
            elif feature_type == 'crossed':
                pass
            else:
                pass
