#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : mlp.py
@Software       : PyCharm
@Modify Time    : 2021/8/1 11:32  
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""
from typing import List

import tensorflow as tf

from utils.feature_builder import FeatureColumnBuilder


def mlp_model(feature_builder: FeatureColumnBuilder,
              hidden_units: List[int]):
    x = tf.keras.layers.DenseFeatures(feature_builder.feature_columns.values())(
        feature_builder.inputs_list
    )
    for unit in hidden_units:
        x = tf.keras.layers.Dense(unit)(x)

    return tf.keras.models.Model(inputs=feature_builder.inputs_list, outputs=x)
