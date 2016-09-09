# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import abc
import re


def activation_summary(x, tower_name):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
      tower_name: Name of tower
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % tower_name, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class ModelTower(object):
    def __init__(self, name, endpoints, logits, aux_logits=None):
        self.name = name
        self.endpoints = endpoints
        self.logits = logits
        self.aux_logits = aux_logits


class Model(object):
    __metaclass__ = abc.ABCMeta
    
    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    def __init__(self):
        self._last_tower = None
        self._towers = {}

    def add_tower(self, name, endpoints, logits, aux_logits=None):
        self._last_tower = ModelTower(
            name,
            endpoints,
            logits,
            aux_logits
        )
        self._towers[name] = self._last_tower

    def tower(self, name=None):
        tower = self.last_tower() if name is None else self._towers[name]
        if not tower:
            raise RuntimeError('Invalid tower ' % name)
        return tower

    def last_tower(self):
        if not self._last_tower:
            raise RuntimeError('A valid model tower is required, please build one first')
        return self._last_tower

    def last_scope(self):
        return self._last_tower.name if self._last_tower else ''

    # Return scopes (strings) for logit variables to allow filtering for save/restore
    @abc.abstractmethod
    def logit_scopes(self):
        pass

    # Return list of 'get/create variable' functions used by the model (used for variable scoping).
    # Makes it easier to abstract train code from models using different variable helpers
    def get_variable_fns(self):
        return [tf.contrib.framework.variable]

    # Return a list of model variables to restore for a Saver
    def variables_to_restore(self, restore_logits=True):
        model_variables = tf.contrib.framework.variables.get_model_variables()
        if restore_logits:
            return model_variables
        else:
            model_variable_names = [x.name for x in model_variables]
            filtered_variables = tf.contrib.framework.variables.get_variables_to_restore(
                model_variable_names, self.logit_scopes())
            return filtered_variables

    def activation_summaries(self, name=None):
        tower = self.tower(name)
        with tf.name_scope('summaries'):
            for act in tower.endpoints.values():
                activation_summary(act, Model.TOWER_NAME)

    @staticmethod
    def default_optimizer_params(self):
        opt_type = 'momentum'
        opt_params = {
            'learning_rate': 0.1,
            'momentum': 0.9,
            'use_nesterov': True
        }
        return opt_type, opt_params

    @staticmethod
    def scope_name(self, tower_id=0):
        return '%s_%d' % (Model.TOWER_NAME, tower_id)

