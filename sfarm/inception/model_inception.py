"""The Inception network family (v3, V4, resnet-V1, resnetV2).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

from .build_inception_v3 import build_inception_v3
from .build_inception_v4 import build_inception_v4, build_inception_resnet

from fabric import model

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


class ModelInception(model.Model):
    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    # Batch normalization. Constant governing the exponential moving average of
    # the 'global' mean and variance for all activations.
    BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

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
        l2_regularizer = layers.l2_regularizer(0.00004)

        with arg_scope(
                [layers.conv2d, layers.fully_connected],
                weights_initializer=layers.xavier_initializer(),
                weights_regularizer=l2_regularizer):
            with arg_scope(
                    [layers.conv2d],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_norm_params):

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
                    res_scale = 0.1
                    logits, endpoints = build_inception_resnet(
                        images,
                        ver=ver,
                        res_scale=res_scale,
                        dropout_keep_prob=0.8,
                        num_classes=num_classes,
                        is_training=is_training,
                        scope=scope)

        if self.variant == ModelInception.Variant.V3:
            # Grab the logits associated with the side head. Employed during training.
            aux_logits = endpoints['aux_logits']
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

    def add_tower_loss(self, labels, batch_size=None, scope=None):
        """Adds all losses for the model.

        Note the final loss is not returned. Instead, the list of losses are collected.
        The losses are accumulated in tower_loss() and summed to calculate the total loss.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
          scope: tower scope of losses to add, ie 'tower_0/', defaults to last added tower if None
        """
        if not batch_size:
            batch_size = FLAGS.batch_size

        tower = self.tower(scope)

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = tower.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.softmax_cross_entropy(
            tower.logits, dense_labels, label_smoothing=0.1, weight=1.0)

        if self.variant == ModelInception.Variant.V3:
            # Cross entropy loss for the auxiliary softmax head.
            losses.softmax_cross_entropy(
                tower.aux_logits, dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')

    def logit_scopes(self):
        return ['output/logits']

    @staticmethod
    def loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')
        loss = math_ops.reduce_mean(cross_entropy)
        return loss

