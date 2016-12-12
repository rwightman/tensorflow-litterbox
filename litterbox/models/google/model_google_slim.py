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
from collections import OrderedDict
from fabric import model
from models.google.nets import nets_factory
slim = tf.contrib.slim

google_default_params = {
    'network': 'inception_resnet_v2',
    'num_classes': 1000,
}


class ModelGoogleSlim(model.Model):

    def __init__(self, params=google_default_params):
        super(ModelGoogleSlim, self).__init__()
        params = model.merge_params(google_default_params, params)

        # model_name must correspond to one of google's network names in nets package,
        # see nets_factory.py for valid names.
        self.network = params['network']
        assert self.network in nets_factory.networks_map
        self.num_classes = params['num_classes']
        assert self.num_classes > 1

    def build_tower(self, images, is_training=False, scope=None):
        weight_decay = 0.0001
        network_fn = nets_factory.get_network_fn(
            self.network,
            num_classes=self.num_classes,
            weight_decay=weight_decay,
            is_training=is_training)
        logits, endpoints = network_fn(images)

        # HACK get mode variable scope set by google net code from logits op name so it can
        # be removed for smaller Tensorboard tags
        scope_search = re.search('%s_[0-9]*/(\w+)/' % self.TOWER_PREFIX, logits.op.name)
        if scope_search:
            self.model_variable_scope = scope_search.group(1)

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
        num_classes = tower.outputs.get_shape()[-1].value
        labels = slim.one_hot_encoding(labels, num_classes=num_classes)

        slim.losses.softmax_cross_entropy(
            tower.outputs, labels, label_smoothing=0.1, weight=1.0)

        if 'AuxLogits' in tower.endpoints:
            slim.losses.softmax_cross_entropy(
                tower.aux_outputs, labels,
                label_smoothing=0.1, weight=0.4, scope='aux_loss')

    def output_scopes(self):
        scopes = ['logits', 'Logits', 'AuxLogits/Aux_logits', 'AuxLogits/Logits', 'AuxLogits/Conv2d_2b_1x1']
        return [self.model_variable_scope + '/' + x for x in scopes]

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

    def check_norm(self, norm):
        if ('vgg' in self.network or 'resnet' in self.network) and norm != 'caffe_rgb':
            print("WARNING: If you are using the pre-trained weights for Google VGG and Resnet models, "
                  "they were imported from Caffe and expect [0, 255] inputs, not the  default [-1, 1]. "
                  "It is recommended to change the image norm method from '%s' to 'caffe_rgb' with "
                  "the --image_norm param." % norm)

