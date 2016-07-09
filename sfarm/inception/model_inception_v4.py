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
"""The Inception v4 network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


def block_stem(net):
    # NOTE observe endpoints of first 3 layers
    endpoints = {}
    with tf.variable_scope('stem'):
        with tf.variable_scope('st0'):  # stage 0 of stem
            # 299 x 299 x 3
            net = layers.conv2d(net, 32, [3, 3], stride=2)
            endpoints['conv0'] = net
            # 149 x 149 x 32
            net = layers.conv2d(net, 32, [3, 3])
            endpoints['conv1'] = net
            # 147 x 147 x 32
            net = layers.conv2d(net, 64, [3, 3], padding='SAME')
            endpoints['stem0'] = net
        with tf.variable_scope('st1'):  # stage 1 of stem
            # 147 x 147 x 64
            with tf.variable_scope('br0_pool'):
                br0_pool = layers.max_pool2d(net, [3, 3], stride=2)
            with tf.variable_scope('br1_3x3'):
                br1_3x3 = layers.conv2d(net, 96, [3, 3], stride=2)
            net = tf.concat(3, [br0_pool, br1_3x3])
            endpoints['stem1'] = net
            # 73 x 73 x 160
        with tf.variable_scope('st2'):  # stage 2 of stem
            with tf.variable_scope('br0_1x1_3x3'):
                br0 = layers.conv2d(net, 64, [1, 1], padding='SAME')
                br0 = layers.conv2d(br0, 96, [3, 3])
            with tf.variable_scope('br1_1x1_7x1_1x7_3x3'):
                br1 = layers.conv2d(net, 64, [1, 1], padding='SAME')
                br1 = layers.conv2d(br1, 64, [7, 1], padding='SAME')
                br1 = layers.conv2d(br1, 64, [1, 7], padding='SAME')
                br1 = layers.conv2d(br1, 96, [3, 3])
            net = tf.concat(3, [br0, br1])
            endpoints['stem2'] = net
            # 71 x 71 x 192
        with tf.variable_scope('st3'):  # stage 3 of stem
            with tf.variable_scope('br0_3x3'):
                br0_3x3 = layers.conv2d(net, 192, [3, 3], stride=2)
            with tf.variable_scope('br1_pool'):
                br1_pool = layers.max_pool2d(net, [3, 3], stride=2)
            net = tf.concat(3, [br0_3x3, br1_pool])
            endpoints['stem3'] = net
            # 35x35x384
    return net, endpoints


def block_a(net, scope='block_a'):
    # 35 x 35 x 384 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('br0_avg_1x1'):
                br0 = layers.avg_pool2d(net, [3, 3])
                br0 = layers.conv2d(br0, 96, [1, 1])
            with tf.variable_scope('br1_1x1'):
                br1 = layers.conv2d(net, 96, [1, 1])
            with tf.variable_scope('br2_1x1_3x3'):
                br2 = layers.conv2d(net, 64, [1, 1])
                br2 = layers.conv2d(br2, 96, [3, 3])
            with tf.variable_scope('br3_1x1_3x3dbl'):
                br3 = layers.conv2d(net, 64, [1, 1])
                br3 = layers.conv2d(br3, 96, [3, 3])
                br3 = layers.conv2d(br3, 96, [3, 3])
            net = tf.concat(3, [br0, br1, br2, br3])
            # 35 x 35 x 384
    return net


def block_reduce_a(net, k=192, l=224, m=256, n=384, scope='block_reduce_a'):
    # 35 x 35 -> 17 x 17 reduce
    # default padding = VALID
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('br0_max'):
            br0 = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('br1_3x3'):
            br1 = layers.conv2d(net, n, [3, 3], stride=2)
        with tf.variable_scope('br2_1x1_3x3dbl'):
            br2 = layers.conv2d(net, k, [1, 1], padding='SAME')
            br2 = layers.conv2d(br2, l, [3, 3], padding='SAME')
            br2 = layers.conv2d(br2, m, [3, 3], stride=2)
        net = tf.concat(3, [br0, br1, br2])
        # 17 x 17 x input + n + m (384 + 384 + 256=1024 normally?)
    return net


def block_b(net, scope='block_b'):
    # 17 x 17 x 1024 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('br0_avg_1x1'):
                br0 = layers.avg_pool2d(net, [3, 3])
                br0 = layers.conv2d(br0, 128, [1, 1])
            with tf.variable_scope('br1_1x1'):
                br1 = layers.conv2d(net, 384, [1, 1])
            with tf.variable_scope('br2_1x1_1x7_7x1'):
                br2 = layers.conv2d(net, 192, [1, 1])
                br2 = layers.conv2d(br2, 224, [1, 7])
                br2 = layers.conv2d(br2, 256, [7, 1])
            with tf.variable_scope('br3_1x1_1x7_7x1dbl'):
                br3 = layers.conv2d(net, 192, [1, 1])
                br3 = layers.conv2d(br3, 192, [1, 7])
                br3 = layers.conv2d(br3, 224, [7, 1])
                br3 = layers.conv2d(br3, 224, [1, 7])
                br3 = layers.conv2d(br3, 256, [7, 1])
            net = tf.concat(3, [br0, br1, br2, br3])
            # 17 x 17 x 1024
    return net


def block_reduce_b(net, scope='block_reduce_b'):
    # 17 x 17 -> 8 x 8 reduce
    with tf.variable_scope(scope):
        with tf.variable_scope('br0_max'):
            br0 = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('br1_1x1_3x3'):
            br1 = layers.conv2d(net, 192, [1, 1], padding='SAME')
            br1 = layers.conv2d(br1, 192, [3, 3], stride=2)
        with tf.variable_scope('br2_1x1_1x7_7x1_3x3'):
            br2 = layers.conv2d(net, 256, [1, 1], padding='SAME')
            br2 = layers.conv2d(br2, 256, [1, 7], padding='SAME')
            br2 = layers.conv2d(br2, 320, [7, 1], padding='SAME')
            br2 = layers.conv2d(br2, 320, [3, 3], stride=2)
        net = tf.concat(3, [br0, br1, br2])
    return net


def block_c(net, scope='block_c'):
    # 8 x 8 x 1536 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('br0_avg_1x1'):
                br0 = layers.avg_pool2d(net, [3, 3])
                br0 = layers.conv2d(br0, 256, [1, 1])
            with tf.variable_scope('br1_1x1'):
                br1 = layers.conv2d(net, 256, [1, 1])
            with tf.variable_scope('br2_1x1_1x3_3x1'):
                br2 = layers.conv2d(net, 384, [1, 1])
                br2a = layers.conv2d(br2, 256, [1, 3])
                br2b = layers.conv2d(br2, 256, [3, 1])
            with tf.variable_scope('br3_1x1_1x7_7x1dbl'):
                br3 = layers.conv2d(net, 384, [1, 1])
                br3 = layers.conv2d(br3, 448, [1, 7])
                br3 = layers.conv2d(br3, 512, [7, 1])
                br3a = layers.conv2d(br3, 256, [1, 7])
                br3b = layers.conv2d(br3, 256, [7, 1])
            net = tf.concat(3, [br0, br1, br2a, br2b, br3a, br3b])
            # 8 x 8 x 1536
    return net


def stem_res_v1(net, scope='stem'):
    return net


def block_res_a(net, scope='block_res_a', activation_fn=tf.nn.relu):
    # 35x35 grid
    with tf.variable_scope(scope):
        shortcut = tf.identity(net)
        with tf.variable_scope('br0_1x1'):
            br0 = layers.conv2d(net, 32, [1, 1])
        with tf.variable_scope('br1_1x1_3x3'):
            br1 = layers.conv2d(net, 32, [1, 1])
            br1 = layers.conv2d(br1, 32, [3, 3])
        with tf.variable_scope('br1_1x1_3x3dbl'):
            br2 = layers.conv2d(net, 32, [1, 1])
            br2 = layers.conv2d(br2, 48, [3, 3])
            br2 = layers.conv2d(br2, 64, [3, 3])
        net = tf.concat(3, [br0, br1, br2])
        net = layers.conv2d(net, 384, [1, 1], activation_fn=None)
        net = activation_fn(tf.concat(3, [shortcut, net]))
    return net


def block_res_b(net, scope='block_res_b', activation_fn=tf.nn.relu):
    # 17 x 17 grid
    with tf.variable_scope(scope):
        shortcut = tf.identity(net)
        with tf.variable_scope('br0_1x1'):
            br0 = layers.conv2d(net, 192, [1, 1])
        with tf.variable_scope('br1_1x1_1x7_7x1'):
            br1 = layers.conv2d(net, 128, [1, 1])
            br1 = layers.conv2d(br1, 160, [1, 7])
            br1 = layers.conv2d(br1, 192, [7, 1])
        net = tf.concat(3, [br0, br1])
        net = layers.conv2d(net, 1154, [1, 1], activation_fn=None)
        net = activation_fn(tf.concat(3, [shortcut, net]))
    return net


def block_res_c(net, scope='block_res_c', activation_fn=tf.nn.relu):
    with tf.variable_scope(scope):
        shortcut = tf.identity(net)
        with tf.variable_scope('br0_1x1'):
            br0 = layers.conv2d(net, 192, [1, 1])
        with tf.variable_scope('br1_1x1_1x3_3x1'):
            br1 = layers.conv2d(net, 192, [1, 1])
            br1 = layers.conv2d(br1, 224, [1, 3])
            br1 = layers.conv2d(br1, 256, [3, 1])
        net = tf.concat(3, [br0, br1])
        net = layers.conv2d(net, 2048, [1, 1], activation_fn=None)
        net = activation_fn(tf.concat(3, [shortcut, net]))
    return net


def block_res_reduce_b(net, scope='block_res_reduce_b'):
    # 17 x 17 -> 8 x 8 reduce
    with tf.variable_scope(scope):
        with tf.variable_scope('br0_max'):
            br0 = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('br1_1x1_3x3'):
            br1 = layers.conv2d(net, 256, [1, 1], padding='SAME')
            br1 = layers.conv2d(br1, 384, [3, 3], stride=2)
        with tf.variable_scope('br2_1x1_3x3'):
            # 256, 288 in the paper, mistake?
            br2 = layers.conv2d(net, 256, [1, 1], padding='SAME')
            br2 = layers.conv2d(br1, 256, [3, 3], stride=2)
        with tf.variable_scope('br3_1x1_3x3dbl'):
            # 256, 288, 320 in the paper, mistake?
            br3 = layers.conv2d(net, 256, [1, 1], padding='SAME')
            br3 = layers.conv2d(br2, 256, [3, 3], padding='SAME')
            br3 = layers.conv2d(br2, 256, [3, 3], stride=2)
        net = tf.concat(3, [br0, br1, br2, br3])
    return net


def block_output(net, num_classes, dropout_keep_prob=0.5, scope='output'):
    with tf.variable_scope(scope):
        # 8 x 8 x 1536
        shape = net.get_shape()
        net = layers.avg_pool2d(net, shape[1:3])
        # 1 x 1 x 1536
        net = layers.dropout(net, dropout_keep_prob)
        net = layers.flatten(net)
        #FIXME monitor global average pool in endpoints?
        # 1536
        net = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
        # num classes
    return net


def build_inception_v4(
        inputs,
        dropout_keep_prob=0.8,
        num_classes=1000,
        is_training=True,
        scope=''):
    """Inception v4 from http://arxiv.org/abs/
    
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      dropout_keep_prob: dropout keep_prob.
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      scope: Optional scope for op_scope.

    Returns:
      a list containing 'logits', 'aux_logits' Tensors.
    """
    # endpoints will collect relevant activations for external use, for example, summaries or losses.
    endpoints = {}
    with tf.op_scope([inputs], scope, 'inception_v4'):
        with arg_scope([layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], 
                           stride=1, padding='VALID'):
                net, stem_endpoints = block_stem(inputs)
                endpoints.update(stem_endpoints)
                # Inception blocks
                for x in range(4):
                    block_scope = 'block_a%d' % x
                    net = block_a(net, block_scope)
                    endpoints[block_scope] = net
                # 35 x 35 x 384
                net = block_reduce_a(net)
                endpoints['block_reduce_a'] = net
                # 17 x 17 x 1024
                for x in range(7):
                    block_scope = 'block_b%d' % x
                    net = block_b(net, block_scope)
                    endpoints[block_scope] = net
                # 17 x 17 x 1024
                net = block_reduce_b(net)
                endpoints['block_reduce_b'] = net
                # 8 x 8 x 1536
                for x in range(3):
                    block_scope = 'block_c%d' % x
                    net = block_c(net, block_scope)
                    endpoints[block_scope] = net
                # 8 x 8 x 1536
                logits = block_output(net, num_classes, dropout_keep_prob, 'output')
                # num_classes
                endpoints['logits'] = logits
                endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, endpoints


class ModelInceptionV4(model.Model):
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
        super(ModelInceptionV4, self).__init__()

    def build_tower(self, images, num_classes, is_training=False, scope=None):
        """Build Inception v4 model architecture.

        See here for reference:

        Args:
          images: Images returned from inputs() or distorted_inputs().
          num_classes: number of classes
          is_training: If set to `True`, build the inference model for training.
          scope: optional prefix string identifying the ImageNet tower.

        Returns:
          Logits. 2-D float Tensor.
          Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
        """
        # Parameters for BatchNorm.
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': ModelInceptionV4.BATCHNORM_MOVING_AVERAGE_DECAY,
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
                logits, endpoints = build_inception_v4(
                    images,
                    dropout_keep_prob=0.8,
                    num_classes=num_classes,
                    is_training=is_training,
                    scope=scope)

        self.add_tower(
            scope,
            endpoints,
            logits
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

