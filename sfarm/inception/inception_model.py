# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build the Inception v3 network on ImageNet data set.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

#import sli
from slim import *
#from slim import scopes
#from slim import ops
#from slim import arg_scope
#from slim import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


def build_inception_v3(
        inputs,
        dropout_keep_prob=0.8,
        num_classes=1000,
        is_training=True,
        restore_logits=True,
        scope=''):
    """Latest Inception from http://arxiv.org/abs/1512.00567.

      "Rethinking the Inception Architecture for Computer Vision"

      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
      Zbigniew Wojna

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      dropout_keep_prob: dropout keep_prob.
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
      scope: Optional scope for op_scope.

    Returns:
      a list containing 'logits', 'aux_logits' Tensors.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.

    end_points = {}
    with tf.op_scope([inputs], scope, 'inception_v3'):
        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout], is_training=is_training):
            with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool], stride=1, padding='VALID'):
                # 299 x 299 x 3
                end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2, scope='conv0')
                # 149 x 149 x 32
                end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3], scope='conv1')
                # 147 x 147 x 32
                end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3], padding='SAME', scope='conv2')
                # 147 x 147 x 64
                end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3], stride=2, scope='pool1')
                # 73 x 73 x 64
                end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1], scope='conv3')
                # 73 x 73 x 80.
                end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3], scope='conv4')
                # 71 x 71 x 192.
                end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3], stride=2, scope='pool2')
                # 35 x 35 x 192.
                net = end_points['pool2']
            # Inception blocks
            with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool], stride=1, padding='SAME'):
                # mixed: 35 x 35 x 256.
                with tf.variable_scope('mixed_35x35x256a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = ops.conv2d(net, 48, [1, 1])
                        branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    end_points['mixed_35x35x256a'] = net
                # mixed_1: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = ops.conv2d(net, 48, [1, 1])
                        branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    end_points['mixed_35x35x288a'] = net
                # mixed_2: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = ops.conv2d(net, 48, [1, 1])
                        branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    end_points['mixed_35x35x288b'] = net
                # mixed_3: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                                  stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
                    end_points['mixed_17x17x768a'] = net
                # mixed4: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = ops.conv2d(net, 128, [1, 1])
                        branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
                        branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = ops.conv2d(net, 128, [1, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    end_points['mixed_17x17x768b'] = net
                # mixed_5: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768c'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = ops.conv2d(net, 160, [1, 1])
                        branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    end_points['mixed_17x17x768c'] = net
                # mixed_6: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768d'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = ops.conv2d(net, 160, [1, 1])
                        branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    end_points['mixed_17x17x768d'] = net
                # mixed_7: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768e'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = ops.conv2d(net, 192, [1, 1])
                        branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
                        branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = ops.conv2d(net, 192, [1, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    end_points['mixed_17x17x768e'] = net
                # Auxiliary Head logits
                aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                with tf.variable_scope('aux_logits'):
                    aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3, padding='VALID')
                    aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
                    # Shape of feature map before the final layer.
                    shape = aux_logits.get_shape()
                    aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01, padding='VALID')
                    aux_logits = ops.flatten(aux_logits)
                    aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                                        stddev=0.001, restore=restore_logits)
                    end_points['aux_logits'] = aux_logits
                # mixed_8: 8 x 8 x 1280.
                # Note that the scope below is not changed to not void previous
                # checkpoints.
                # (TODO) Fix the scope when appropriate.
                with tf.variable_scope('mixed_17x17x1280a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(net, 192, [1, 1])
                        branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch7x7x3'):
                        branch7x7x3 = ops.conv2d(net, 192, [1, 1])
                        branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
                        branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
                        branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
                    end_points['mixed_17x17x1280a'] = net
                # mixed_9: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat(3, [ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat(3, [ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                    end_points['mixed_8x8x2048a'] = net
                # mixed_10: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat(3, [ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                        branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat(3, [ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = ops.avg_pool(net, [3, 3])
                        branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                    end_points['mixed_8x8x2048b'] = net
                # Final pooling and prediction
                with tf.variable_scope('logits'):
                    shape = net.get_shape()
                    net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
                    # 1 x 1 x 2048
                    net = ops.dropout(net, dropout_keep_prob, scope='dropout')
                    net = ops.flatten(net, scope='flatten')
                    # 2048
                    logits = ops.fc(net, num_classes, activation=None, scope='logits', restore=restore_logits)
                    # 1000
                    end_points['logits'] = logits
                    end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
            return logits, end_points


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


class ModelInceptionV3(object):
    # If a model is trained using multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    TOWER_NAME = 'tower'

    # Batch normalization. Constant governing the exponential moving average of
    # the 'global' mean and variance for all activations.
    BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self):
        self.endpoints = {}

    def build(self, images, num_classes, for_training=False, restore_logits=True, scope=None):
        """Build Inception v3 model architecture.

        See here for reference: http://arxiv.org/abs/1512.00567

        Args:
          images: Images returned from inputs() or distorted_inputs().
          num_classes: number of classes
          for_training: If set to `True`, build the inference model for training.
            Kernels that operate differently for inference during training
            e.g. dropout, are appropriately configured.
          restore_logits: whether or not the logits layers should be restored.
            Useful for fine-tuning a model with different num_classes.
          scope: optional prefix string identifying the ImageNet tower.

        Returns:
          Logits. 2-D float Tensor.
          Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
        """
        # Parameters for BatchNorm.
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': ModelInceptionV3.BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        # Set weight_decay for weights in Conv and FC layers.
        with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
            with scopes.arg_scope(
                    [ops.conv2d], stddev=0.1, activation=tf.nn.relu, batch_norm_params=batch_norm_params):
                logits, endpoints = build_inception_v3(
                    images,
                    dropout_keep_prob=0.8,
                    num_classes=num_classes,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)

        self.logits = logits
        self.endpoints = endpoints

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        # Grab the logits associated with the side head. Employed during training.
        auxiliary_logits = endpoints['aux_logits']
        self.auxiliary_logits = auxiliary_logits

        return logits, auxiliary_logits

    def loss(self, labels, batch_size=None):
        """Adds all losses for the model.

        Note the final loss is not returned. Instead, the list of losses are collected
        by slim.losses. The losses are accumulated in tower_loss() and summed to
        calculate the total loss.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
        """
        if not batch_size:
            batch_size = FLAGS.batch_size

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = self.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.cross_entropy_loss(self.logits,
                                  dense_labels,
                                  label_smoothing=0.1,
                                  weight=1.0)

        # Cross entropy loss for the auxiliary softmax head.
        losses.cross_entropy_loss(self.auxiliary_logits,
                                  dense_labels,
                                  label_smoothing=0.1,
                                  weight=0.4,
                                  scope='aux_loss')

    @staticmethod
    def loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy2')
        loss = math_ops.reduce_mean(cross_entropy)
        return loss

    def activation_summaries(self):
        with tf.name_scope('summaries'):
            for act in self.endpoints.values():
                activation_summary(act, ModelInceptionV3.TOWER_NAME)
