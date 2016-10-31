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

FLAGS = tf.app.flags.FLAGS

resnet_default_params = {
    'num_layers': 34,
    'width_factor': 1,
    'pre_activation': False,
    'num_classes': 1000,
}


class ModelResnet(fabric.Model):

    def __init__(self, params=resnet_default_params):
        super(ModelResnet, self).__init__()
        self.num_layers = params['num_layers']
        self.width_factor = params['width_factor']
        self.pre_activation = params['pre_activation']
        self.num_classes = params['num_classes']

    def build_tower(self, inputs, is_training=False, scope=None):

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
                k=self.width_factor,
                pre_activation=self.pre_activation,
                num_classes=self.num_classes,
                num_blocks=num_blocks,
                bottleneck=bottleneck,
                is_training=is_training,
                scope=scope)

        self.add_tower(
            name=scope,
            endpoints=endpoints,
            outputs=logits
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
        fabric.loss.loss_softmax_cross_entropy(tower.outputs, labels)

    def output_scopes(self):
        return ['Outputs/Logits']

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
