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

from datas.load_data import CustomDataLoader
from utils.config import Config


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
        self.emb_params = emb_params
        self.use_emb_layer = use_emb_layer

    def input_layer(self):
        return preprocessing.PreprocessingLayer(**self.feature_params)

    def __call__(self, inputs):
        if self.use_emb_layer:
            if self.emb_params:
                return layers.Embedding(**self.emb_params)(self.input_layer()(inputs))
            else:
                input_dim = (
                    self.feature_params['num_bins']
                    if 'num_bins' in self.feature_params else
                    len(self.feature_params['bin_boundaries'])
                )

                emb_params = {
                    'input_dim': input_dim,
                    'output_dim': 8
                }
                return layers.Embedding(**emb_params)(self.input_layer()(inputs))

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
    def __init__(self, config: dict):
        self.config = config
        self.all_inputs = []
        self.encoded_features = []
        self.feature_builder()

    def feature_builder(self) -> None:
        for feature_name, feature_config in self.config.items():
            if feature_config['type'] == 'hashing':
                input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
                encoding_layer = HashEmbeddingBuilder(feature_params=feature_config['config'])
                self.all_inputs.append(input_col)
                self.encoded_features.append(encoding_layer(input_col))
            elif feature_config['type'] == 'vocabulary':
                input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
                encoding_layer = VocabEmbeddingBuilder(feature_params=feature_config['config'])
                self.all_inputs.append(input_col)
                self.encoded_features.append(encoding_layer(input_col))
            elif feature_config['type'] == 'numerical':
                input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
                encoding_layer = NumericalBuilder(feature_params=feature_config['config'])
                self.all_inputs.append(input_col)
                self.encoded_features.append(encoding_layer(input_col))
            elif feature_config['type'] == 'pre-trained':
                raise Exception("There is no preprocessing layer for type: {}".format(feature_config['feature_type']))
                pass
            elif feature_config['type'] == 'crossed':
                input_col = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
                encoding_layer = CrossedBuilder(feature_params=feature_config['config'])
                self.all_inputs.append(input_col)
                self.encoded_features.append(encoding_layer(input_col))
            else:
                raise Exception("There is no preprocessing layer for type: {}".format(feature_config['feature_type']))
                pass


class FeatureProcess:
    def __init__(self, config: dict):
        self.config = config
        self.features_embed_layers = {}

    def feautre_builder(self) -> None:
        for feature_name, feature_config in self.config.items():
            if feature_config['type'] == 'hashing':
                self.features_embed_layers[feature_name] = HashEmbeddingBuilder(
                    **feature_config['config']
                )
            elif feature_config['type'] == 'vocabulary':
                self.features_embed_layers[feature_name] = VocabEmbeddingBuilder(
                    **feature_config['config']
                )
            elif feature_config['type'] == 'numerical':
                self.features_embed_layers[feature_name] = NumericalBuilder(
                    **feature_config['config']
                )
            elif feature_config['type'] == 'pre-trained':
                raise Exception("There is no preprocessing layer for type: {}".format(feature_config['feature_type']))
                pass
            elif feature_config['type'] == 'crossed':
                self.features_embed_layers[feature_name] = CrossedBuilder(
                    **feature_config['config']
                )
            else:
                raise Exception("There is no preprocessing layer for type: {}".format(feature_config['feature_type']))
                pass


if __name__ == '__main__':
    CONFIG = Config('./conf/')
    train_path = CONFIG.read_data_path('train')
    batch_size = CONFIG.read_data_batch_size()
    epochs = CONFIG.read_data_epochs()
    ds = CustomDataLoader(CONFIG, train_path).input_fn(batch_size=batch_size, epochs=epochs)

    print(list(ds.as_numpy_iterator())[0])
