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

FLAGS = tf.app.flags.FLAGS


def block_a(net, endpoints, d=64, scope='BlockA'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def block_b(net, endpoints, d=256, scope='BlockB'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Conv3'] = layers.conv2d(net, d, [3, 3], scope='Conv3_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def block_c(net, endpoints, d=256, scope='BlockC'):
    with tf.variable_scope(scope):
        net = endpoints[scope+'/Conv1'] = layers.conv2d(net, d, [3, 3], scope='Conv1_3x3')
        net = endpoints[scope+'/Conv2'] = layers.conv2d(net, d, [3, 3], scope='Conv2_3x3')
        net = endpoints[scope+'/Conv3'] = layers.conv2d(net, d, [3, 3], scope='Conv3_3x3')
        net = endpoints[scope+'/Conv4'] = layers.conv2d(net, d, [3, 3], scope='Conv4_3x3')
        net = endpoints[scope+'/Pool1'] = layers.max_pool2d(net, [2, 2], stride=2, scope='Pool1_2x2/2')
    return net


def block_output(net, endpoints, num_classes, dropout_keep_prob=0.5):
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


def build_vgg16(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
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

                net = block_a(inputs, endpoints, d=64, scope='Scale1')
                net = block_a(net, endpoints, d=128, scope='Scale2')
                net = block_b(net, endpoints, d=256, scope='Scale3')
                net = block_b(net, endpoints, d=512, scope='Scale4')
                net = block_b(net, endpoints, d=512, scope='Scale5')
                logits = block_output(net, endpoints, num_classes, dropout_keep_prob)

                endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                return logits, endpoints


def build_vgg19(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
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

                net = block_a(inputs, endpoints, d=64, scope='Scale1')
                net = block_a(net, endpoints, d=128, scope='Scale2')
                net = block_c(net, endpoints, d=256, scope='Scale3')
                net = block_c(net, endpoints, d=512, scope='Scale4')
                net = block_c(net, endpoints, d=512, scope='Scale5')
                logits = block_output(net, endpoints, num_classes, dropout_keep_prob)

                endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                return logits, endpoints


class ModelVgg(fabric.Model):

    def __init__(self, num_layers=16):
        super(ModelVgg, self).__init__()
        self._layers = num_layers

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build_tower(self, inputs, num_classes, is_training=False, scope=None):

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.9997,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        l2_regularizer = layers.l2_regularizer(0.0005)  #0.00004

        with arg_scope(
                [layers.fully_connected],
                biases_initializer=tf.constant_initializer(0.1),
                weights_initializer=layers.variance_scaling_initializer(factor=1.0),
                weights_regularizer=l2_regularizer,
                activation_fn=tf.nn.relu
        ):
            with arg_scope(
                    [layers.conv2d],
                    #normalizer_fn=layers.batch_norm,
                    #normalizer_params=batch_norm_params,
                    weights_initializer=layers.variance_scaling_initializer(factor=1.0),
                    weights_regularizer=l2_regularizer,
                    activation_fn=tf.nn.relu
            ):

                if self._layers == 19:
                    logits, endpoints = build_vgg19(
                        inputs,
                        num_classes=num_classes,
                        is_training=is_training,
                        scope=scope)
                else:
                    logits, endpoints = build_vgg16(
                        inputs,
                        num_classes=num_classes,
                        is_training=is_training,
                        scope=scope)

        self.add_tower(
            name=scope,
            endpoints=endpoints,
            logits=logits
        )

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        return logits

    def add_tower_loss(self, labels, scope=None):
        """Adds all losses for the model.

        The final loss is not returned, the list of losses are collected by slim.losses.
        The losses are accumulated in tower_loss() and summed to calculate the total loss.

        Args:
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          scope: tower scope of losses to add, ie 'tower_0/', defaults to last added tower if None
        """
        tower = self.tower(scope)
        fabric.loss.loss_softmax_cross_entropy(tower.logits, labels)

    def logit_scopes(self):
        return ['logits/logits']

    @staticmethod
    def eval_loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Shape [batch_size, num_classes], dtype float32/64
          labels: Labels from distorted_inputs or inputs(). batch_size vector with int32/64 values in [0, num_classes).
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')
        loss = tf.reduce_mean(cross_entropy)
        return loss
