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

import fabric
import re
import tensorflow as tf
from copy import deepcopy
from .build_inception_resnet_sdc import *
slim = tf.contrib.slim

sdc_default_params = {
    'outputs': {'steer': 1}, #, 'xyz': 2},
}


class ModelSdc(fabric.model.Model):

    def __init__(self, params={}):
        super(ModelSdc, self).__init__()
        params = fabric.model.merge_params(sdc_default_params, params)
        self.output_cfg = params['outputs']

    def build_tower(self, inputs, is_training=False, scope=None):

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            output, endpoints = build_inception_resnet_sdc_regression_v1(
                inputs,
                self.output_cfg,
                is_training=is_training,
                scope=scope)

        #scope_search = re.search('%s_[0-9]*/(\w+)/' % self.TOWER_PREFIX, output['steer'].op.name)
        #print(scope_search)
        #self.model_scope = "InceptionResnetV2"

        aux_output = None
        if 'AuxOutput' in endpoints:
            aux_output = endpoints['AuxOutput']

        self.add_tower(
            scope,
            endpoints=endpoints,
            outputs=output,
            aux_outputs=aux_output,
        )

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        return output

    def add_tower_loss(self, targets, scope=None):
        tower = self.tower(scope)
        assert 'xyz' in self.output_cfg or 'steer' in self.output_cfg
        if 'xyz' in self.output_cfg:
            fabric.loss.loss_huber_with_aux(
                tower.outputs['xyz'], targets[1], aux_predictions=tower.aux_outputs['xyz'])
        if 'steer' in self.output_cfg:
            targets_steer = tf.expand_dims(targets[0], 1)
            print(tower.outputs['steer'].get_shape(), targets_steer.get_shape())
            fabric.loss.loss_huber_with_aux(
                tower.outputs['steer'], targets_steer, aux_predictions=tower.aux_outputs['steer'])

    @staticmethod
    def eval_loss_op(predictions, targets):
        """Generate a simple (non tower based) loss op for use in evaluation.
        """
        with slim.arg_scope([slim.add_loss], loss_collection=None):  # don't add eval loss to collection
            loss = fabric.loss.loss_huber(predictions, targets, scope='loss_huber_eval')
            return loss