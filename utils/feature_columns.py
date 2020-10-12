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

    wide_columns = {}
    deep_columns = {}
    # generate the feature columns for continuous features
    for feature, params in continuous_features_config:
        normalizer = _normalizer_fn_builder(params['normalizer'], tuple(params['boundaries']))
        col = _SUPPORTED_FEATURE_COLUMNS['continuous'](
            key=feature,
            shape=(1,),
            default_value=0,
            dtype=tf.float32,
            normalizer_fn=normalizer
        )
        deep_columns[feature] = col

    # generate the feature columns for sparse features
    for feature, conf in sparse_features_config:
        column_type = conf['column_type']
        column_params = conf['params']

        col = _SUPPORTED_FEATURE_COLUMNS[column_type](
            key=feature,
            **column_params
        )
        wide_columns[feature] = col

    # generate the feature columns for embedding features

    return 0
