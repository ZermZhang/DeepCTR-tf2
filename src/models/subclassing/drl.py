#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@File           : drl
@Software       : PyCharm
@Modify Time    : 2020/9/22 08:01     
@Author         : zermzhang
@version        : 1.0
@Desciption     : 
"""
import collections
import random

import numpy as np
import tensorflow as tf
import gym


class DRL(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.classes = config.model_config['classes']
        self.deep_params = config.deep_model_config
        self.activation = tf.keras.activations.get(self.deep_params['activation'])

        self.hidden_layers = []
        for unit_num in self.deep_params['units']:
            self.hidden_layers.append(tf.keras.layers.Dense(units=unit_num, activation=self.activation))
        self.output_layers = tf.keras.layers.Dense(units=self.classes, activation='softmax')

    def call(self, inputs):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(inputs)
        output = self.output_layers(x)
        return output

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


def tester(CONFIG):
    config_train = CONFIG.model_config

    num_episodes = config_train.get('num_episodes', 500)
    num_exploration_episodes = config_train.get('num_exploration_episodes', 100)
    max_len_episode = config_train.get('max_len_episode', 1000)
    batch_size = config_train.get('batch_size', 32)
    learning_rate = config_train.get('learning_rate', 0.001)
    gamma = config_train.get('gamma', 1.0)
    initial_epsilon = config_train.get('initial_epsilon', 1.0)
    final_epsilon = config_train.get('final_epsilon', 0.01)

    env = gym.make('CartPole-v1')
    model = DRL(CONFIG)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = collections.deque(maxlen=10000)
    epsilon = initial_epsilon

    for episode_id in range(num_episodes):
        state = env.reset()
        epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes, initial_epsilon)
        for t in range(max_len_episode):
            env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()
                action = action[0]

            next_state, reward, done, info = env.step(action)
            reward = -10. if done else reward
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            state = next_state

            if done:
                print("episode {}, epsilon {}, score {}".format(episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *random.sample(replay_buffer, batch_size)
                )
                batch_state, batch_reward, batch_next_state, batch_done = [np.array(a, dtype=np.float32) for a in
                                                                           [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)

                q_value = model(batch_next_state)
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                    )

                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return 0
