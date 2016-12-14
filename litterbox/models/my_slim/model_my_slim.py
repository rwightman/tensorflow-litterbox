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

import tensorflow as tf
from fabric import model, loss
from collections import OrderedDict
from .nets_factory import get_network_fn

my_default_params = {
    'network': 'inception_resnet_v2',
    'num_classes': 1000,
}


class ModelMySlim(model.Model):

    def __init__(self, params):
        super(ModelMySlim, self).__init__()
        params = model.merge_params(my_default_params, params)
        self.network = params['network']
        self.num_classes = params['num_classes']
        self._params = params  # cache for build fns

    def build_tower(self, inputs, is_training=False, scope=None):

        network_fn = get_network_fn(
            self.network,
            num_classes=self.num_classes,
            params=self._params,
            is_training=is_training)
        logits, endpoints = network_fn(inputs)

        if 'AuxLogits' in endpoints:
            # Grab the logits associated with the auxiliary head if present.
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
        aux_logits = None
        if 'AuxLogits' in tower.endpoints:
            aux_logits = tower.aux_outputs

        loss.loss_softmax_cross_entropy_with_aux(tower.outputs, labels, aux_logits)

    def output_scopes(self, prefix_scope=''):
        # all models currently have their num_class specific FC/output layers under the 'Output' scope
        scopes = ['Output']
        prefix = prefix_scope + '/' if prefix_scope else ''
        return [prefix + x for x in scopes]

    def get_predictions(self, outputs, processor):
        if processor is not None:
            logits = processor.decode_output(outputs)
        else:
            logits = outputs
        return tf.nn.softmax(logits)

    @staticmethod
    def eval_ops(logits, labels, processor):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Shape [batch_size, num_classes], dtype float32/64
          labels: Labels from distorted_inputs or inputs(). batch_size vector with int32/64 values in [0, num_classes).
        """
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)
        loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')
        return OrderedDict([('top 5', top_5_op), ('top 1', top_1_op), ('loss', loss_op)])

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
