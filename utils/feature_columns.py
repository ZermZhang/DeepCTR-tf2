#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : feature_columns
@Software       : PyCharm
@Modify Time    : 2020/9/30 08:54     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""
import tensorflow as tf

from utils import config


class BaseFeatureColumn(object):
    def __init__(self):
        # the feature column for numeric features
        self.numeric_column = tf.feature_column.numeric_column
        self.sequence_numeric_column = tf.feature_column.sequence_numeric_column
        # the feature column for sparse features
        self.bucketized_column = tf.feature_column.bucketized_column
        self.categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
        self.categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
        self.categorical_column_with_vocabulary_file = tf.feature_column.categorical_column_with_vocabulary_file
        self.categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
        # the feature column for sequence features
        self.sequence_categorical_column_with_hash_bucket = tf.feature_column.sequence_categorical_column_with_hash_bucket
        self.sequence_categorical_column_with_identity = tf.feature_column.sequence_categorical_column_with_identity
        self.sequence_categorical_column_with_vocabulary_file = tf.feature_column.sequence_categorical_column_with_vocabulary_file
        self.sequence_categorical_column_with_vocabulary_list = tf.feature_column.sequence_categorical_column_with_vocabulary_list
        # the feature column for embedding column
        self.shared_embeddings = tf.feature_column.shared_embeddings
        self.crossed_column = tf.feature_column.crossed_column
        self.embedding_column = tf.feature_column.embedding_column
        self.indicator_column = tf.feature_column.indicator_column


class ContinuousFeatureColumn(BaseFeatureColumn):
    """
    the Feature Column for continuous features
    """
    def __init__(self, continuous_feature_config):
        super().__init__()

    @staticmethod
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


class SparseFeatureColumn(BaseFeatureColumn):
    def __init__(self, sparse_feature_config):
        super().__init__()


def get_feature_columns():
    return 0
