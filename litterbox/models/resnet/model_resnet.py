# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" ResNet Model
    * ResNet 18, 34, 50, 101, 152 as per https://arxiv.org/abs/1512.03385
    * ResNet 200 as per https://arxiv.org/pdf/1603.05027v3.pdf
    * Width ideas from https://arxiv.org/pdf/1605.07146.pdf
"""
from .build_resnet import build_resnet

import fabric
import layers as my_layers

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


class ModelResnet(fabric.Model):

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self, num_layers=34, width_factor=1):
        super(ModelResnet, self).__init__()
        self.num_layers = num_layers
        self.width_factor = width_factor

    def build_tower(self, inputs, num_classes, is_training=False, scope=None):

        # layer configs
        if self.num_layers == 16:
            num_blocks = [1, 2, 3, 1]
            bottleneck = False
            # filter output depth = 512
        elif self.num_layers == 18:
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
            # filter output depth = 512
        elif self.num_layers == 34:
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
            # filter output depth = 512
        elif self.num_layers == 50:
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
            # filter output depth 2048
        elif self.num_layers == 101:
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
            # filter output depth 2048
        elif self.num_layers == 151:
            num_blocks = [3, 8, 36, 3]
            bottleneck = True
            # filter output depth 2048
        else:
            assert False, "Invalid number of layers"

        k = self.width_factor
        pre_activation = True

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.9997,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        l2_regularizer = layers.l2_regularizer(0.0004)

        arg_scope_layers = arg_scope(
            [layers.conv2d, my_layers.preact_conv2d, layers.fully_connected],
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=l2_regularizer,
            activation_fn=tf.nn.relu)
        arg_scope_conv = arg_scope(
            [layers.conv2d, my_layers.preact_conv2d],
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params)
        with arg_scope_layers, arg_scope_conv:
            logits, endpoints = build_resnet(
                inputs,
                k=k,
                pre_activation=pre_activation,
                num_classes=num_classes,
                num_blocks=num_blocks,
                bottleneck=bottleneck,
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

    def add_tower_loss(self, labels, batch_size=None, scope=None):
        """Adds all losses for the model.

        Note the final loss is not returned. Instead, the list of losses are collected
        by slim.losses. The losses are accumulated in tower_loss() and summed to
        calculate the total loss.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
          scope: tower scope of losses to add, ie 'tower_0/', defaults to last added tower if None
        """
        if not batch_size:
            batch_size = FLAGS.batch_size

        tower = self.tower(scope)

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = tower.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.softmax_cross_entropy(
            tower.logits, dense_labels, label_smoothing=0.1, weight=1.0)

    def logit_scopes(self):
        return ['Outputs/Logits']

    @staticmethod
    def loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Shape [batch_size, num_classes], dtype float32/64
          labels: Labels from distorted_inputs or inputs(). batch_size vector with int32/64 values in [0, num_classes).
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')
        loss = math_ops.reduce_mean(cross_entropy)
        return loss
