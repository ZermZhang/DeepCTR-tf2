#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File           : base
@Software       : PyCharm
@Modify Time    : 2022/6/14 08:50
@Author         : zermzhang
@version        : 1.0
@Desciption     :
"""
import tensorflow as tf
from custom_utils.custom_layers import StaticEncodedFeatureBuilder
from custom_utils import custom_losses


class ModelBaseBuilder(tf.keras.Model):
    def __init__(self, config: dict,
                 *args, **kwargs):
        super(ModelBaseBuilder, self).__init__(*args, **kwargs)
        # init the feature config info
        self.config = config
        # definite the input layers
        self.encoder_layers = {}
        self.emb_layers = {}
        self.all_inputs = {}
        self.init_input_layer()

        # definite the optimizer
        self.optimizer_conf = self.config.get('optimizer', {})
        self.optimizer = tf.keras.optimizers.get(self.optimizer_conf.get('name'))(self.optimizer_conf.get('params'))
        # definite the loss function
        self.loss_func = custom_losses.get('reduce_sum_sigmoid_crossentropy_loss')
        # definite the metrics
        self.metrics = tf.keras.metrics.get(self.config.get('metrics', 'AUC'))

    def init_input_layer(self):
        """
        生成模型需要依赖的输入层信息
        self.encoder_layers: preprocessing-layer，处理原始特征，转为编码格式
        self.emb_layers: 对编码格式数据进行embedding处理，方便送入模型
        self.all_inputs: 根据输入特征格式生成格式化输入说明，方便summary()和plot_model()
        """
        for feature_name, feature_config in self.config.items():
            feature_builder = StaticEncodedFeatureBuilder(feature_name, feature_config)
            self.encoder_layers[feature_name] = feature_builder.feature_encoder
            self.emb_layers[feature_name] = feature_builder.emb_layer
            self.all_inputs[feature_name] = feature_builder.inputs

    @tf.function
    def train_step(self, batch_features: dict, batch_labels: dict):
        with tf.GradientTape() as tape:
            batch_preds = self.call(batch_features)
            batch_loss = self.loss_func(y_true=batch_labels, y_pred=batch_preds)

        grads = tape.gradient(batch_loss, self.train_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.train_variables))

    @tf.function
    def eval_step(self, batch_features: dict, batch_labels: dict):
        batch_preds = self.call(batch_features)
        self.metrics.update_state(y_true=batch_labels, y_pred=batch_preds)

    def build_graph(self):
        return tf.keras.models.Model(
            inputs=self.all_inputs,
            outputs=self.call(self.all_inputs)
        )