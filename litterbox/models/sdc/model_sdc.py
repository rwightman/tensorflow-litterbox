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
import tensorflow as tf
from .build_inception_resnet_sdc import *
from .build_resnet_sdc import *
slim = tf.contrib.slim

sdc_default_params = {
    'outputs': {'steer': 1, 'xyz': 2},
    'network': 'inception_resnet_v2', # or one of other options in network_map
    'regression_loss': 'mse', # or huber
    'version': 2,
}

network_map = {
    'inception_resnet_v2': build_inception_resnet_sdc_regression,
    'resnet_v1_50': build_resnet_v1_50_sdc,
    'resnet_v1_101': build_resnet_v1_101_sdc,
    'resnet_v1_152': build_resnet_v1_152_sdc,
}

arg_scope_map = {
    'inception_resnet_v2': inception_resnet_v2_arg_scope,
    'resnet_v1_50': resnet_arg_scope,
    'resnet_v1_101': resnet_arg_scope,
    'resnet_v1_152': resnet_arg_scope,
}


class ModelSdc(fabric.model.Model):

    def __init__(self, params={}):
        super(ModelSdc, self).__init__()
        params = fabric.model.merge_params(sdc_default_params, params)
        self.output_cfg = params['outputs']
        # model variable scope needs to match google net for pretrained weight compat
        if (params['network'] == 'resnet_v1_152' or
                params['network'] == 'resnet_v1-101' or
                params['network'] == 'resnet_v1_50'):
            self.network = params['network']
            self.model_variable_scope = params['network']
        else:
            assert params['network'] == 'inception_resnet_v2'
            self.network = 'inception_resnet_v2'
            self.model_variable_scope = "InceptionResnetV2"
        self.version = params['version']
        if params['regression_loss'] == 'huber':
            self.regression_loss = fabric.loss.loss_huber_with_aux
        else:
            self.regression_loss = fabric.loss.loss_mse_with_aux

    def build_tower(self, inputs, is_training=False, scope=None):

        with slim.arg_scope(arg_scope_map[self.network]()):
            output, endpoints = network_map[self.network](
                inputs,
                output_cfg=self.output_cfg,
                version=self.version,
                is_training=is_training)

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
            target_xyz = targets[1]
            aux_output_xyz = None
            if tower.aux_outputs:
                aux_output_xyz = tower.aux_outputs['xyz']
            self.regression_loss(
                tower.outputs['xyz'], target_xyz, aux_predictions=aux_output_xyz)

        if 'steer' in self.output_cfg:
            target_steer = targets[0]
            aux_output_steer = None
            if tower.aux_outputs:
                aux_output_steer = tower.aux_outputs['steer']
            if self.output_cfg['steer'] > 1:
                # steer is integer target, one hot output, use softmax
                fabric.loss_softmax_cross_entropy_with_aux(
                    tower.outputs['steer'], target_steer, aux_logits=aux_output_steer)
            else:
                assert self.output_cfg['steer'] == 1
                # steer is float target/output, use regression /w huber loss
                self.regression_loss(
                    tower.outputs['steer'], target_steer, aux_predictions=aux_output_steer)
                
    def get_predictions(self, outputs, remove_background=False):
        #FIXME just pass through for regression, add decoders for specific output cfgs, etc
        return outputs

    def output_scopes(self):
        rel_scopes = ['logits', 'Logits', 'Output', 'Output/OutputXYZ', 'Output/OutputSteer', 'Output/Fc1',
                      'AuxLogits/OutputXYZ', 'AuxLogits/OutputSteer', 'AuxLogits/Fc1']
        abs_scopes = [self.model_variable_scope + '/' + x for x in rel_scopes]
        #abs_scopes = ['[\w]*/' + x for x in rel_scopes]
        return abs_scopes

    @staticmethod
    def eval_ops(predictions, targets, decoders=None):
        """Generate a simple (non tower based) loss op for use in evaluation.
        """
        ops = {}
        if 'steer' in predictions:
            steer_targ = targets[0]
            steer_pred = predictions['steer']

            if steer_pred.get_shape()[-1].value > 1:
                # one hot steering loss (non reduced)
                steer_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    steer_pred, steer_targ, name='steer_xentropy_eval')

                # decode non-linear mapping before mse/pce
                steer_pred = tf.cast(tf.argmax(steer_pred, dimension=1), tf.int32)
                if decoders and 'steer' in decoders:
                    steer_pred = decoders['steer'](steer_pred)
                    steer_targ = decoders['steer'](steer_targ)
            else:
                # linear regression steering loss
                assert steer_pred.get_shape()[-1].value == 1
                steer_loss = fabric.loss.metric_huber(steer_pred, steer_targ)

            steer_mse = tf.squared_difference(steer_pred, steer_targ, name='steer_mse_eval')

            ops['steer_loss'] = steer_loss
            ops['steer_mse'] = steer_mse
            #ops['steer_pred'] = steer_pred
            #ops['steer_targ'] = steer_targ

        if 'xyz' in predictions:
            xyz_targ = targets[1]
            xyz_pred = predictions['xyz']
            xyz_loss = fabric.loss.metric_huber(xyz_pred, xyz_targ)
            xyz_mse = tf.squared_difference(xyz_pred, xyz_targ, name='xyz_mse_eval')
            ops['xyz_loss'] = xyz_loss
            ops['xyz_mse'] = xyz_mse
            ops['xyz_pred'] = xyz_pred
            ops['xyz_targ'] = xyz_targ

        return ops