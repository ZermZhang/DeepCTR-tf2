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
from typing import List, Any

import tensorflow as tf
from tensorflow.python.ops import embedding_ops, math_ops
from tensorflow import Tensor

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers


class InputIdxLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    """
    获取输入特征对应的特征list中的idx信息，方便后续从embeddinglist中获取需要的embdding
    1. 直接映射
        item => idx for item in inputs_list
    2. 间接映射
        item => attr => idx for item's attr in target_list
    """
    def __init__(self, inputs_list: List, mapping_list: List = None,
                 target_list: List = None, **kwargs):
        super(InputIdxLayer, self).__init__(**kwargs)
        self.inputs_list = inputs_list
        self.mapping_list = mapping_list
        self.target_list = target_list
        print(self.inputs_list, self.mapping_list, self.target_list)
        if self.mapping_list is None:
            self.target_table = self.get_table(self.inputs_list,
                                               [i for i in range(len(self.inputs_list))])
        else:
            self.mapping_table = self.get_table(self.inputs_list, self.mapping_list,
                                                default_value='')
            self.target_table = self.get_table(self.target_list,
                                               [i for i in range(len(self.target_list))])

    @staticmethod
    def get_table(keys: List, vals: List, default_value: Any = -1):
        init = tf.lookup.KeyValueTensorInitializer(keys, vals)
        table = tf.lookup.StaticHashTable(init, default_value=default_value)
        return table

    def call(self, inputs):
        if self.mapping_table is None:
            idx_ = self.target_table.lookup(inputs)
            return idx_
        else:
            target_ = self.mapping_table.lookup(inputs)
            idx_ = self.target_table.lookup(target_)
            return idx_


class LoadEmbeddingLayer(tf.keras.layers.Layer):
    """
    get the pre-trained embedding in embdding List
    inpust: the idx_ output from InputIdxLayer
    """
    def __init__(self, embedding, **kwargs):
        super(LoadEmbeddingLayer).__init__(**kwargs)
        self.embedding = tf.constant(embedding)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)


class PreTrainEmbLoadLayer(tf.keras.layers.Layer):
    """
    方便导入预训练的embedding的layer
    可以处理两种embedding导入的方式：
    1. 直接映射
        item => embedding
    2. 间接映射
        item => attr => embedding
    """
    def __init__(self,
                 embedding: List,
                 inputs_list: List,
                 mapping_list: List = None,
                 target_list: List = None,
                 **kwargs):
        super(PreTrainEmbLoadLayer, self).__init__(**kwargs)
        self.embedding = tf.constant(embedding)
        self.inputs_list = inputs_list
        self.mapping_list = mapping_list
        self.target_list = target_list
        self.mapping_table = None
        self.target_table = None

        if self.mapping_list is None:
            self.target_table = self.get_table(self.inputs_list,
                                               [i for i in range(len(self.inputs_list))])
        else:
            self.mapping_table = self.get_table(self.inputs_list, self.mapping_list,
                                                default_value='')
            self.target_table = self.get_table(self.target_list,
                                               [i for i in range(len(self.target_list))])

    @staticmethod
    def get_table(keys: List, vals: List, default_value: Any = -1):
        init = tf.lookup.KeyValueTensorInitializer(keys, vals)
        table = tf.lookup.StaticHashTable(init, default_value=default_value)
        return table

    def call(self, inputs):
        if self.mapping_table is None:
            idx_ = self.target_table.lookup(inputs)
            return tf.nn.embedding_lookup(self.embedding, idx_)
        else:
            target_ = self.mapping_table.lookup(inputs)
            print(target_)
            idx_ = self.target_table.lookup(target_)
            print(idx_)
            return tf.nn.embedding_lookup(self.embedding, idx_)


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


class SeBlockLayer(tf.keras.layers.Layer):
    """
    the SeBlock layer for features
    """
    def __init__(self, out_dim, ratio=4.0, *args, **kwargs):
        super(SeBlockLayer, self).__init__(*args, **kwargs)

        self.out_dim = out_dim
        self.mid_dim = tf.math.ceil(out_dim / ratio)
        self.ratio = ratio

        self.avg_weights = []

    def build(self, input_shape):
        super(SeBlockLayer, self).build(input_shape)

    def call(self, feature_embs: List[Tensor]):
        """
        根据feautre_embs进行SEBlock处理
        注意：feature_embs是list类型，元素可能是不同维度的tensor
        返回结果是经过了加权之后的embedding信息
        """

        for feature_emb in feature_embs:
            self.avg_weights.append(tf.reduce_mean(feature_emb, axis=1, keepdims=True))

        se_inputs = tf.concat(self.avg_weights, axis=-1)
        outputs = tf.keras.layers.Dense(self.mid_dim)(se_inputs)
        outputs = tf.nn.relu(outputs)
        outputs = tf.keras.layers.Dense(self.out_dim)(outputs)
        outputs = tf.nn.sigmoid(outputs)

        se_outputs = self.cal_weighted(feature_embs, outputs)

        return se_outputs

    @staticmethod
    def cal_weighted(inputs: List[Tensor], weights: Tensor):
        """
        对输入的embdding特征进行加权
        因为inputs是一个list，里面每个tensor的维度可能是不一样的
        """
        outputs = []

        for i in range(len(inputs)):
            weight = weights[:, i]
            weight = tf.expand_dims(weight, axis=1)
            weighted_tensor = inputs[i] * weight
            outputs.append(weighted_tensor)

        return outputs


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
