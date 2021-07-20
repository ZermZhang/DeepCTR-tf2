#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : preprocesing
@Software       : PyCharm
@Modify Time    : 2021/7/19 14:17
@Author         : zermzhang
@version        : 1.0
@Desciption     : process the input features by keras.layers.preprocessing
    1. Encoder category features by CategoryEncoding
    2. Cross category features by CategoryCrossing
    3. Liscretizate numerical features by Discretization
    4. Load pre-trained embedding features by StringLookup
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
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.python.framework.dtypes import DType


class FeatureBaseBuilder:
    def __init__(self, dims: int, batch_size: int, dtype: DType,
                 feature_params: dict, embed_params: dict):
        self.feature_params = feature_params
        self.embed_params = embed_params
        self.dim = dims
        self.inputs = tf.keras.Input(shape=(batch_size, dims), dtype=dtype)

    def input_layer(self):
        return preprocessing.PreprocessingLayer(**self.feature_params)(self.inputs)

    def __call__(self, *args, **kwargs):
        return layers.Embedding(**self.embed_params)(self.input_layer())


class HashEmbeddingBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.Hashing(**self.feature_params)(self.inputs)


class VocabEmbeddingBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.StringLookup(**self.feature_params)(self.inputs)


class NumericalBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.Discretization(**self.feature_params)(self.inputs)


class FeatureProcess:
    def __init__(self, config: dict, dims: int, **kwargs):
        self.config = config
        self.dims = dims
        self.features_embed_layers = {}

    def feautre_builder(self):
        for feature_name, feature_config in self.config.items():
            if feature_config['feature_type'] == 'hashing':
                self.features_embed_layers[feature_name] = HashEmbeddingBuilder(
                    **feature_config['config']
                )
            elif feature_config['feature_type'] == 'vocabulary':
                self.features_embed_layers[feature_name] = VocabEmbeddingBuilder(
                    **feature_config['config']
                )
            elif feature_config['feature_type'] == 'numerical':
                self.features_embed_layers[feature_name] = NumericalBuilder(
                    **feature_config['config']
                )
            elif feature_config['feature_type'] == 'pre-trained':
                pass
            elif feature_config['feature_type'] == 'crossed':
                pass
            else:
                pass
