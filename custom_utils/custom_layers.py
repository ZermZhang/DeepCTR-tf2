#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : layers
@Software       : PyCharm
@Modify Time    : 2020/9/11 08:13
@Author         : zermzhang
@version        : 1.0
@Desciption     : the utils functions for layers
"""
import pickle

import tensorflow as tf
from tensorflow.python.ops import embedding_ops, math_ops

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers


class PreTrainedEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, file_path, **kwargs):
        self.built = None
        self.mask_zero = False
        self.supports_masking = self.mask_zero
        self.dim = dim
        self.file_path = file_path
        self.kernel = None
        self.embeddings = None
        super(PreTrainedEmbedding, self).__init__(**kwargs)

    def build(self):
        self.kernel = self.add_weight(name='weight', shape=(2, self.dim), initializer='glorot_uniform')
        with open(self.file_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        self.embeddings = tf.convert_to_tensor(self.embeddings, dtype=tf.float32)
        self.embeddings = tf.concat((self.kernel, self.embeddings), axis=0)
        self.built = True

    def call(self, inputs):
        out = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return out

    def compute_mask(self, inputs):
        if not self.mask_zero:
            return None
        return math_ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], self.dim)
        return output_shape

    def get_config(self):
        base_config = super(PreTrainedEmbedding, self).get_config()
        base_config['dim'] = self.dim
        return base_config


class LinearLayer(tf.keras.layers.Layer):
    """
    the custom layers
    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_variable(name='w', shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b', shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


class SequencePoolingLayer(tf.keras.layers.Layer):
    """
    the pooling layers for sequence features like UBS (User Behaviour Sequence)
    """
    def __init__(self, mode='mean', support_masking=False, **kwargs):
        super(SequencePoolingLayer, self).__init__(**kwargs)
        self.seq_len_max = None
        if mode not in ['mean', 'max', 'min']:
            raise Exception("the mode: {} is not supported!".format(mode))

        self.mode = mode
        self.epsilon = tf.constant(-2 ** 32 + 1, tf.float32)
        self.supports_masking = support_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, seq_value_len_list, mask=None):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When support_masking=True, input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list
            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return tf.reduce_max(hist, 1, keepdims=True)

        hist = tf.reduce_sum(uiseq_embed_list * mask, 1, keepdims=True)

        if self.mode == 'mean':
            hist = tf.divide(hist, tf.cast(user_behavior_length, tf.float32) + self.epsilon)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return None, 1, input_shape[-1]
        else:
            return None, 1, input_shape[0][-1]

    def get_config(self):
        config = {'mode': self.mode, 'support_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return base_config.update(config)


class StaticEncodedFeatureBuilder:
    def __init__(self, feature_name: str, config: dict):
        """
        feature_params: dict,  preprocessing layer的相关参数
        emb_params: dict = None, Embedding layer的相关参数
        use_emb_layer: bool = True, 是否需要使用Embedding layer
        """
        self.feature_name = feature_name
        self.config = config
        self.use_emb_layer = config.get('use_emb_layer', True)
        self.emb_params = config.get('emb_config', None)
        # init the Embedding layer
        if self.emb_params:
            pass
        else:
            input_dim = (
                self.config['config']['num_bins']
                if 'num_bins' in self.config['config'] else
                len(self.config['config']['bin_boundaries'])
            )

            self.emb_params = {
                'input_dim': input_dim,
                'output_dim': 8
            }

        if self.use_emb_layer:
            self.emb_layer = layers.Embedding(**self.emb_params)
        else:
            self.emb_layer = None

        self.feature_encoder = self.build_encoded_features(self.config)
        self.inputs = self.build_inputs(self.feature_name, self.config)

    @staticmethod
    def build_encoded_features(feature_config):
        feature_encoder_type = feature_config['type']
        feature_encoder_params = feature_config['config']
        if feature_encoder_type == 'hashing':
            encoding_layer = preprocessing.Hashing(**feature_encoder_params)
            return encoding_layer
        elif feature_encoder_type == 'vocabulary':
            encoding_layer = preprocessing.StringLookup(**feature_encoder_params)
            return encoding_layer
        elif feature_encoder_type == 'numerical':
            encoding_layer = preprocessing.Discretization(**feature_encoder_params)
            return encoding_layer
        elif feature_encoder_type == 'pre-trained':
            raise Exception("There is no preprocessing layer for type: {}".format(feature_encoder_type))
            pass
        elif feature_encoder_type == 'crossed':
            encoding_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(**feature_encoder_params)
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
