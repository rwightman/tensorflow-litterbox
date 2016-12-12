# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fabric
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def _block_a(net, endpoints, d=64, scope='BlockA'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def _block_b(net, endpoints, d=256, scope='BlockB'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Conv3'] = layers.conv2d(net, d, [3, 3], scope='Conv3_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def _block_c(net, endpoints, d=256, scope='BlockC'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Conv3'] = layers.conv2d(net, d, [3, 3], scope='Conv3_3x3')
        net = endpoints[scope+'/Conv4'] = layers.conv2d(net, d, [3, 3], scope='Conv4_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def _block_output(net, endpoints, num_classes, dropout_keep_prob=0.5):
    with tf.variable_scope('Output'):
        net = layers.flatten(net, scope='Flatten')

        # 7 x 7 x 512
        net = layers.fully_connected(net, 4096, scope='Fc1')
        net = endpoints['Output/Fc1'] = layers.dropout(net, dropout_keep_prob, scope='Dropout1')

        # 1 x 1 x 4096
        net = layers.fully_connected(net, 4096, scope='Fc2')
        net = endpoints['Output/Fc2'] = layers.dropout(net, dropout_keep_prob, scope='Dropout2')

        logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
        # 1 x 1 x num_classes
        endpoints['Logits'] = logits
    return logits


def _build_vgg16(
        inputs,
        num_classes=1000,
        dropout_keep_prob=0.5,
        is_training=True,
        scope=''):
    """Blah"""

    endpoints = {}
    with tf.name_scope(scope, 'vgg16', [inputs]):
        with arg_scope(
                [layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d], 
                    stride=1,
                    padding='SAME'):

                net = _block_a(inputs, endpoints, d=64, scope='Scale1')
                net = _block_a(net, endpoints, d=128, scope='Scale2')
                net = _block_b(net, endpoints, d=256, scope='Scale3')
                net = _block_b(net, endpoints, d=512, scope='Scale4')
                net = _block_b(net, endpoints, d=512, scope='Scale5')
                logits = _block_output(net, endpoints, num_classes, dropout_keep_prob)

                endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                return logits, endpoints


def _build_vgg19(
        inputs,
        num_classes=1000,
        dropout_keep_prob=0.5,
        is_training=True,
        scope=''):
    """Blah"""

    endpoints = {}
    with tf.name_scope(scope, 'vgg19', [inputs]):
        with arg_scope(
                [layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d],
                    stride=1,
                    padding='SAME'):

                net = _block_a(inputs, endpoints, d=64, scope='Scale1')
                net = _block_a(net, endpoints, d=128, scope='Scale2')
                net = _block_c(net, endpoints, d=256, scope='Scale3')
                net = _block_c(net, endpoints, d=512, scope='Scale4')
                net = _block_c(net, endpoints, d=512, scope='Scale5')
                logits = _block_output(net, endpoints, num_classes, dropout_keep_prob)

                endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                return logits, endpoints


def params_vgg(
        num_layers=16):
    params = {
        'num_layers': num_layers,
        'weight_decay': 0.0005,
        'use_batch_norm': False,
        'dropout_keep_prob': 0.5,
        'output_scopes': ['Output']
    }
    return params


def vgg_arg_scope(
        weight_decay=0.0005,
        use_batch_norm=False):
    """"""
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    normalizer_fn = layers.batch_norm if use_batch_norm else None
    normalizer_params = batch_norm_params if use_batch_norm else None
    l2_regularizer = layers.l2_regularizer(weight_decay)  # 0.00004

    with arg_scope(
            [layers.fully_connected],
            biases_initializer=tf.constant_initializer(0.1),
            weights_initializer=layers.variance_scaling_initializer(factor=1.0),
            weights_regularizer=l2_regularizer,
            activation_fn=tf.nn.relu):
        with arg_scope(
                [layers.conv2d],
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=layers.variance_scaling_initializer(factor=1.0),
                weights_regularizer=l2_regularizer,
                activation_fn=tf.nn.relu) as arg_sc:
            return arg_sc


def build_vgg(
        inputs,
        num_classes=1000,
        params=params_vgg(),
        is_training=True,
        scope=''):
    """"""
    params = fabric.model.merge_params(params_vgg(), params)
    num_layers = params['num_layers']
    weight_decay = params['weight_decay']
    use_batch_norm = params['use_batch_norm']
    dropout_keep_prob = params['dropout_keep_prob']

    with vgg_arg_scope(
        weight_decay=weight_decay,
        use_batch_norm=use_batch_norm,
    ):
        if num_layers == 19:
            logits, endpoints = _build_vgg19(
                inputs,
                num_classes=num_classes,
                dropout_keep_prob=dropout_keep_prob,
                is_training=is_training,
                scope=scope)
        else:
            assert num_layers == 16
            logits, endpoints = _build_vgg16(
                inputs,
                num_classes=num_classes,
                dropout_keep_prob=dropout_keep_prob,
                is_training=is_training,
                scope=scope)

    return logits, endpoints
