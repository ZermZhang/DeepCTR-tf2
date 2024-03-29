#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : feature_columns
@Software       : PyCharm
@Modify Time    : 2020/9/30 08:54     
@Author         : zermzhang
@version        : 1.0
@Desciption     :
    the continuous features just input for continuous feature column
    the sparse features as the input for the specified feature column AND the specified embedding feature column
"""

import tensorflow as tf

from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
from tensorflow.python.feature_column.feature_column import _SharedEmbeddingColumn
from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
from tensorflow.python.feature_column.feature_column_v2 import SharedEmbeddingColumn


_SUPPORTED_FEATURE_COLUMNS = {
    'continuous': tf.feature_column.numeric_column,
    'vocab_file': tf.feature_column.categorical_column_with_vocabulary_file,
    'vocab_list': tf.feature_column.categorical_column_with_vocabulary_list,
    'identity': tf.feature_column.categorical_column_with_identity,
    'hash_bucket': tf.feature_column.categorical_column_with_hash_bucket,
    'sequence_vocab_file': tf.feature_column.sequence_categorical_column_with_vocabulary_file,
    'sequence_vocab_list': tf.feature_column.sequence_categorical_column_with_vocabulary_list,
    'sequence_identity': tf.feature_column.sequence_categorical_column_with_identity,
    'sequence_hash_bucket': tf.feature_column.sequence_categorical_column_with_hash_bucket,
    'embedding': tf.feature_column.embedding_column,
    'shared_embeding': tf.feature_column.shared_embeddings
}


def is_embedding_column(feature_column):
    return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn, _SharedEmbeddingColumn, SharedEmbeddingColumn))


def is_shared_embedding_column(feature_column):
    return isinstance(feature_column, (_SharedEmbeddingColumn, SharedEmbeddingColumn))


def _normalizer_fn_builder(normalization_name, normalization_params):
    """normalizer_fn builder"""
    if normalization_name == 'min_max':
        return lambda x: (x - normalization_params[0]) / (normalization_params[1] - normalization_params[0])
    elif normalization_name == 'standard':
        return lambda x: (x - normalization_params[0]) / normalization_params[1]
    elif normalization_name == 'log':
        return lambda x: tf.math.log(x)
    else:
        return None


def get_feature_columns(CONFIG):
    feature_config = CONFIG.feature_config
    continuous_features_config = CONFIG.get_continuous_features_config()
    sparse_features_config = CONFIG.get_sparse_features_config()
    cross_features_config = CONFIG.get_cross_features_config()
    embedding_features_config = CONFIG.get_embedding_features_config()

    wide_columns = {}
    deep_columns = {}
    # generate the feature columns for continuous features
    if not continuous_features_config:
        print("the features config for continuous features is NULL!")
    else:
        for feature, params in continuous_features_config.items():
            normalizer = _normalizer_fn_builder(params['normalizer'], params['boundaries'])
            col = _SUPPORTED_FEATURE_COLUMNS['continuous'](
                key=feature,
                shape=(1,),
                default_value=0,
                dtype=tf.float32,
                normalizer_fn=normalizer
            )
            deep_columns[feature] = col

    # generate the feature columns for sparse features
    if not sparse_features_config:
        print('the features config for sparse featues is NULL!')
    else:
        for feature, conf in sparse_features_config.items():
            column_type = conf['column_type']
            column_params = conf['params']

            col = _SUPPORTED_FEATURE_COLUMNS[column_type](
                key=feature,
                **column_params
            )
            wide_columns[feature] = col

    # generate the feature columns for cross features

    # generate the feature columns for embedding features
    if not embedding_features_config:
        print("the feature config for embedding features is NULL!")
    else:
        for feature, conf in embedding_features_config.items():
            params = conf['params']
            assert feature in wide_columns, 'the columns: {} not contain in sparse features'.format(feature)
            emb_col = _SUPPORTED_FEATURE_COLUMNS['embedding'](
                categorical_column=wide_columns[feature],
                **params
            )
            deep_columns[feature] = emb_col

    return wide_columns, deep_columns


def input_from_feature_columns(features, feature_columns):
    dense_value_list = []
    sparse_emb_list = []

    for keys, column in feature_columns.items():
        if is_embedding_column(column):
            sparse_emb = tf.expand_dims(tf.keras.layers.DenseFeatures(column)(features), axis=1)
            sparse_emb_list.append(sparse_emb)
        else:
            dense_value_list.append(tf.keras.layers.DenseFeatures(column)(features))

    return dense_value_list, sparse_emb_list


if __name__ == "__main__":
    conf_dir = './conf'
    from src.utils import config

    CONFIG = config.Config(conf_dir)
    wide_columns, deep_columns = get_feature_columns(CONFIG)
    print(wide_columns)
    print(deep_columns)
