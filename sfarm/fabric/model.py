"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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


class ModelInstance(object):
    def __init__(self, name, endpoints, logits, aux_logits=None):
        self.name = name
        self.endpoints = endpoints
        self.logits = logits
        self.aux_logits = aux_logits


class Model(object):
    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    def __init__(self):
        self.last_instance = None
        self.instances = {}

    def add_instance(self, name, endpoints, logits, aux_logits=None):
        self.last_instance = ModelInstance(
            name,
            endpoints,
            logits,
            aux_logits
        )
        self.instances[name] = self.last_instance

    def instance(self, name=None):
        if not self.last_instance:
            raise RuntimeError('Loss requires a valid model instance, please build one first')
        if not name:
            instance = self.last_instance
        else:
            instance = self.instances[name]
        if not instance:
            raise RuntimeError('Invalid instance')
        return instance

    def last_instance(self):
        if not self.last_instance:
            raise RuntimeError('Loss requires a valid model instance, please build one first')
        return self.last_instance

    def last_scope(self):
        return self.last_instance.name if self.last_instance else ''

    # Return scopes (strings) for logit variables to allow filtering for save/restore
    def logit_scopes(self):
        pass

    # Return list of 'get_variable' functions used by the model (used for variable scoping)
    def get_variable_fn_list(self):
        pass

    # Return a list of model variables to restore for a Saver
    def variables_to_restore(self):
        pass

    def activation_summaries(self, name=None):
        instance = self.instance(name)
        with tf.name_scope('summaries'):
            for act in instance.endpoints.values():
                print(act)
                activation_summary(act, Model.TOWER_NAME)

    @staticmethod
    def scope_name(self, tower_id=0):
        return '%s_%d' % (Model.TOWER_NAME, tower_id)

