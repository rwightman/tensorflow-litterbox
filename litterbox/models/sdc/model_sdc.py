# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Model wrapper for Google's tensorflow/model/slim models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from fabric import model
from .build_inception_resnet_sdc import *
slim = tf.contrib.slim


class ModelSdc(model.Model):

    def __init__(self, model_name='sdc'):
        super(ModelSdc, self).__init__()

        # model_name must correspond to one of google's network names in nets package,
        # see nets_factory.py for valid names.
        self.model_name = model_name

    def build_tower(self, images, num_classes, is_training=False, scope=None):
        weight_decay = 0.0001

        output, endpoints = build_inception_resnet_sdc_regression_v1(
            images,
            is_training=is_training,
            scope=scope)

        self.model_scope = "InceptionResnetV2"

        self.add_tower(
            scope,
            endpoints,
            logits
        )

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        return logits

    def add_tower_loss(self, labels, scope=None):
        tower = self.tower(scope)
        num_classes = tower.logits.get_shape()[-1].value
        labels = slim.one_hot_encoding(labels, num_classes=num_classes)

        slim.losses.softmax_cross_entropy(
            tower.logits, labels, label_smoothing=0.1, weight=1.0)

        if 'AuxLogits' in tower.endpoints:
            slim.losses.softmax_cross_entropy(
                tower.endpoints['AuxLogits'], labels,
                label_smoothing=0.1, weight=0.4, scope='aux_loss')

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
