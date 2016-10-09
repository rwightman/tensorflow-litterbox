# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""The Inception model family (v3, V4, Inception-Resnet-V1, Inception-Resnet-V2).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .build_inception_v3 import build_inception_v3
from .build_inception_v4 import build_inception_v4, build_inception_resnet

import fabric
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from enum import Enum

FLAGS = tf.app.flags.FLAGS


class ModelInception(fabric.Model):
    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    # Batch normalization. Constant governing the exponential moving average of
    # the 'global' mean and variance for all activations.
    BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

    class Variant(Enum):
        V3 = 1
        V4 = 2
        ResnetV1 = 3
        ResnetV2 = 4

    def __init__(self, variant=Variant.V4):
        super(ModelInception, self).__init__()
        self.variant = variant

    def build_tower(self, images, num_classes, is_training=False, scope=None):
        """Build Inception v4, Inception-Resnet-V1, and Inception-Resnet-V2 model architectures.

        See here for reference: http://arxiv.org/pdf/1602.07261v1.pdf

        Args:
          images: Images returned from inputs() or distorted_inputs().
          num_classes: number of classes
          is_training: If set to `True`, build the inference model for training.
          scope: optional prefix string identifying the ImageNet tower.

        Returns:
          Logits. 2-D float Tensor.
        """
        # Parameters for BatchNorm.
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': ModelInception.BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        # Set weight_decay for weights in Conv and FC layers.
        weight_decay = 0.00004
        l2_regularizer = layers.l2_regularizer(weight_decay)
        res_scale = 0.3
        activation_fn = tf.nn.relu  # tf.nn.elu

        arg_scope_weights = arg_scope(
            [layers.conv2d, layers.fully_connected],
            weights_initializer=layers.variance_scaling_initializer(factor=1.0),
            weights_regularizer=l2_regularizer
        )
        arg_scope_conv = arg_scope(
            [layers.conv2d],
            activation_fn=activation_fn,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params
        )
        with arg_scope_weights, arg_scope_conv:
            if self.variant == ModelInception.Variant.V3:
                logits, endpoints = build_inception_v3(
                    images,
                    dropout_keep_prob=0.8,
                    num_classes=num_classes,
                    is_training=is_training,
                    scope=scope)
            elif self.variant == ModelInception.Variant.V4:
                logits, endpoints = build_inception_v4(
                    images,
                    dropout_keep_prob=0.8,
                    num_classes=num_classes,
                    is_training=is_training,
                    scope=scope)
            else:
                ver = 1 if self.variant == ModelInception.Variant.ResnetV1 else 2
                logits, endpoints = build_inception_resnet(
                    images,
                    ver=ver,
                    res_scale=res_scale,
                    activation_fn=activation_fn,  # activation_fn used directly in res blocks
                    dropout_keep_prob=0.67,
                    num_classes=num_classes,
                    is_training=is_training,
                    scope=scope)

        if self.variant == ModelInception.Variant.V3:
            # Grab the logits associated with the side head. Employed during training.
            aux_logits = endpoints['AuxLogits']
        else:
            aux_logits = None

        self.add_tower(
            scope,
            endpoints,
            logits,
            aux_logits
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
        if self.variant == ModelInception.Variant.V3:
            aux_logits = tower.aux_logits
        else:
            aux_logits = None
        fabric.loss.loss_softmax_cross_entropy(tower.logits, labels, aux_logits)

    def logit_scopes(self):
        return ['Output/Logits']

    @staticmethod
    def eval_loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')
        loss = tf.reduce_mean(cross_entropy)
        return loss

    @staticmethod
    def default_optimizer_params(self):
        opt_type = 'RMSProp'
        # Default params as in Google's inception v3 model
        opt_params = {
            'decay': 0.9,
            'momentum': 0.9,
            'epsilon': 1.0
        }
        return opt_type, opt_params
