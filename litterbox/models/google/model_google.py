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
from models.google.nets import nets_factory
slim = tf.contrib.slim


class ModelGoogle(model.Model):

    def __init__(self, model_name='inception_resnet_v2'):
        super(ModelGoogle, self).__init__()

        # model_name must correspond to one of google's network names in nets package,
        # see nets_factory.py for valid names.
        self.model_name = model_name

    def build_tower(self, images, num_classes, is_training=False, scope=None):
        weight_decay = 0.0001
        network_fn = nets_factory.get_network_fn(
            self.model_name,
            num_classes=num_classes,
            weight_decay=weight_decay,
            is_training=is_training)
        logits, endpoints = network_fn(images)

        # HACK get mode variable scope set by google net code from logits op name so it can
        # be removed for smaller Tensorboard tags
        scope_search = re.search('%s_[0-9]*/(\w+)/' % self.TOWER_PREFIX, logits.op.name)
        if scope_search:
            self.model_scope = scope_search.group(1)

        if 'AuxLogits' in endpoints:
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
