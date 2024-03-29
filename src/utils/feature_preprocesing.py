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


class FeatureBaseBuilder:
    """
    feature_params: the params for preprocessingLayer for input features
    emb_layer_params: the params for embedding layer
    use_emb_layer: use the layers.Embedding or not
        when yes: return the embedding features
        when no: return the feature after preprocessing layer
    """

    def __init__(self, feature_params: dict, emb_params: dict = None,
                 use_emb_layer: bool = True):
        self.feature_params = feature_params
        self.use_emb_layer = use_emb_layer
        # init the Embedding layer
        if emb_params:
            self.emb_params = emb_params
        else:
            input_dim = (
                self.feature_params['num_bins']
                if 'num_bins' in self.feature_params else
                len(self.feature_params['bin_boundaries'])
            )

            self.emb_params = {
                'input_dim': input_dim,
                'output_dim': 8
            }

        if self.use_emb_layer:
            self.emb_layer = layers.Embedding(**self.emb_params)

    def input_layer(self):
        preprocessing_layer = preprocessing.PreprocessingLayer(**self.feature_params)
        return preprocessing_layer

    def __call__(self, inputs):
        if self.use_emb_layer:
            return self.emb_layer(self.input_layer()(inputs))
        else:
            return self.input_layer()(inputs)


class HashEmbeddingBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.Hashing(**self.feature_params)


class VocabEmbeddingBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.StringLookup(**self.feature_params)


class NumericalBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return preprocessing.Discretization(**self.feature_params)


class CrossedBuilder(FeatureBaseBuilder):
    def input_layer(self):
        return tf.keras.layers.experimental.preprocessing.HashedCrossing(**self.feature_params)


class EncodedFeatureBuilder:
    def __init__(self, feature_name, feature_config):
        self.feature_name = feature_name
        self.feature_config = feature_config
        self.feature_encoder = self.build_encoded_features(self.feature_config)
        self.inputs = self.build_inputs(self.feature_name, self.feature_config)

    @staticmethod
    def build_encoded_features(feature_config):
        feature_encoder_type = feature_config['type']
        feature_encoder_params = feature_config['config']
        feature_embedding_params = feature_config.get('embed_config', None)
        if feature_encoder_type == 'hashing':
            encoding_layer = HashEmbeddingBuilder(
                feature_params=feature_encoder_params,
                emb_params=feature_embedding_params
            )
            return encoding_layer
        elif feature_encoder_type == 'vocabulary':
            encoding_layer = VocabEmbeddingBuilder(
                feature_params=feature_encoder_params,
                emb_params=feature_embedding_params
            )
            return encoding_layer
        elif feature_encoder_type == 'numerical':
            encoding_layer = NumericalBuilder(
                feature_params=feature_encoder_params,
                emb_params=feature_embedding_params
            )
            return encoding_layer
        elif feature_encoder_type == 'pre-trained':
            raise Exception("There is no preprocessing layer for type: {}".format(feature_encoder_type))
            pass
        elif feature_encoder_type == 'crossed':
            encoding_layer = CrossedBuilder(
                feature_params=feature_encoder_params,
                emb_params=feature_embedding_params
            )
            return encoding_layer
        else:
            raise Exception("There is no preprocessing layer for type: {}".format(feature_encoder_type))
            pass

    @staticmethod
    def build_inputs(feature_name, feature_config):
        feature_encoder_type = feature_config['type']
        if feature_encoder_type in ('hashing', 'vocabulary'):
            input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
            return input_col
        elif feature_encoder_type in {'numerical'}:
            input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.float32)
            return input_col
        else:
            raise Exception("There is no preprocessing layer for type: {}".format(feature_encoder_type))
            pass