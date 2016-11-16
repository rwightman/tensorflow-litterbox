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
from .build_nvidia_sdc import *
slim = tf.contrib.slim

sdc_default_params = {
    'outputs': {'steer': 1, 'xyz': 2},
    'network': 'inception_resnet_v2',  # or one of other options in network_map
    'regression_loss': 'mse',  # or huber
    'version': 2,
    'bayesian': False,
}

network_map = {
    'inception_resnet_v2': build_inception_resnet_sdc_regression,
    'resnet_v1_50': build_resnet_v1_50_sdc,
    'resnet_v1_101': build_resnet_v1_101_sdc,
    'resnet_v1_152': build_resnet_v1_152_sdc,
    'nvidia_sdc': build_nvidia_sdc,
}

arg_scope_map = {
    'inception_resnet_v2': inception_resnet_v2_arg_scope,
    'resnet_v1_50': resnet_arg_scope,
    'resnet_v1_101': resnet_arg_scope,
    'resnet_v1_152': resnet_arg_scope,
    'nvidia_sdc': nvidia_style_arg_scope,
}


class ModelSdc(fabric.model.Model):

    def __init__(self, params={}):
        super(ModelSdc, self).__init__()
        params = fabric.model.merge_params(sdc_default_params, params)
        self.output_cfg = params['outputs']
        # model variable scope needs to match google net for pretrained weight compat
        if (params['network'] == 'resnet_v1_152' or
                params['network'] == 'resnet_v1_101' or
                params['network'] == 'resnet_v1_50'):
            self.network = params['network']
            self.model_variable_scope = params['network']
        elif params['network'] == 'inception_resnet_v2':
            self.network = 'inception_resnet_v2'
            self.model_variable_scope = "InceptionResnetV2"
        else:
            assert params['network'] == 'nvidia_sdc'
            self.network = 'nvidia_sdc'
            self.model_variable_scope = "NvidiaSdc"

        self.version = params['version']
        self.bayesian = params['bayesian']

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
                bayesian=self.bayesian,
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
                
    def get_predictions(self, outputs, processor=None):
        if processor:
            for k, v in outputs.items():
                outputs[k] = processor.decode_output(v, key=k)
        return outputs

    def output_scopes(self):
        rel_scopes = ['logits', 'Logits', 'Output', 'Output/OutputXYZ', 'Output/OutputSteer', 'Output/Fc1',
                      'AuxLogits/OutputXYZ', 'AuxLogits/OutputSteer', 'AuxLogits/Fc1']
        abs_scopes = [self.model_variable_scope + '/' + x for x in rel_scopes]
        #abs_scopes = ['[\w]*/' + x for x in rel_scopes]
        return abs_scopes

    @staticmethod
    def eval_ops(predictions, labels, processor=None):
        """Generate a simple (non tower based) loss op for use in evaluation.
        """
        ops = {}
        if 'steer' in predictions:
            steer_label = labels[0]
            steer_prediction = predictions['steer']

            if steer_prediction.get_shape()[-1].value > 1:
                # one hot steering loss (non reduced)
                steer_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    steer_prediction, steer_label, name='steer_xentropy_eval')
                # decode non-linear mapping before mse
                steer_prediction = tf.cast(tf.argmax(steer_prediction, dimension=1), tf.int32)
                if processor:
                    steer_prediction = processor.decode_output(steer_prediction, key='steer')
                    steer_label = processor.decode_output(steer_label, key='steer')
            else:
                # linear regression steering loss
                assert steer_prediction.get_shape()[-1].value == 1
                steer_loss = fabric.loss.metric_huber(steer_prediction, steer_label)
                if processor:
                    steer_prediction = processor.decode_output(steer_prediction, key='steer')
                    steer_label = processor.decode_output(steer_label, key='steer')

            steer_mse = tf.squared_difference(
                steer_prediction, steer_label, name='steer_mse_eval')

            ops['steer_loss'] = steer_loss
            ops['steer_mse'] = steer_mse
            #ops['steer_prediction'] = steer_prediction
            #ops['steer_label'] = steer_label

        if 'xyz' in predictions:
            xyz_labels = labels[1]
            xyz_predictions = predictions['xyz']
            if processor:
                xyz_labels = processor.decode_output(xyz_labels, key='xyz')
                xyz_predictions = processor.decode_output(xyz_predictions, key='xyz')
            xyz_loss = fabric.loss.metric_huber(xyz_predictions, xyz_labels)
            xyz_mse = tf.squared_difference(xyz_predictions, xyz_labels, name='xyz_mse_eval')
            ops['xyz_loss'] = xyz_loss
            ops['xyz_mse'] = xyz_mse
            ops['xyz_prediction'] = xyz_predictions
            ops['xyz_label'] = xyz_labels

        return ops
