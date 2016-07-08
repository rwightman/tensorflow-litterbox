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

from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


def stem(net, endpoints):
    # NOTE observe endpoints of first 3 layers
    with tf.variable_scope('stage0'):
        # 299 x 299 x 3
        endpoints['conv0'] = layers.conv2d(net, 32, [3, 3], stride=2, scope='conv0')
        # 149 x 149 x 32
        endpoints['conv1'] = layers.conv2d(endpoints['conv0'], 32, [3, 3], scope='conv1')
        # 147 x 147 x 32
        endpoints['conv2'] = layers.conv2d(endpoints['conv1'], 64, [3, 3], padding='SAME', scope='conv2')
        net = endpoints['conv2']
        # 147 x 147 x 64
    with tf.variable_scope('stage1'):
        branch_pool = layers.max_pool2d(net, [3, 3], stride=2, scope='branch_pool')
        branch_3x3 = layers.conv2d(net, 96, [3, 3], stride=2, scope='branch_3x3')
        net = tf.concat(3, [branch_pool, branch_3x3])
        # 73 x 73 x 160
    with tf.variable_scope('stage2'):
        with tf.variable_scope('branch_a'):
            branch_a = layers.conv2d(net, 64, [1, 1], padding='SAME')
            branch_a = layers.conv2d(branch_a, 96, [3, 3])
        with tf.variable_scope('branch_b'):
            branch_b = layers.conv2d(net, 64, [1, 1], padding='SAME')
            branch_b = layers.conv2d(branch_b, 64, [7, 1], padding='SAME')
            branch_b = layers.conv2d(branch_b, 64, [1, 7], padding='SAME')
            branch_b = layers.conv2d(branch_b, 96, [3, 3])
        net = tf.concat(3, [branch_a, branch_b])
        # 71 x 71 x 192
    with tf.variable_scope('stage3'):
        branch_3x3 = layers.conv2d(net, 192, [3, 3], stride=2, scope='branch_3x3')
        branch_pool = layers.max_pool2d(net, [3, 3], stride=2, scope='branch_pool')
        net = tf.concat(3, [branch_3x3, branch_pool])
        # 35x35x384
    return net


def block_a(net, scope='block_a'):
    # 35 x 35 x 384 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_a'):
                branch_a = layers.avg_pool2d(net, [3, 3])
                branch_a = layers.conv2d(branch_a, 96, [1, 1])
            with tf.variable_scope('branch_b'):
                branch_b = layers.conv2d(net, 96, [1, 1])
            with tf.variable_scope('branch_c'):
                branch_c = layers.conv2d(net, 64, [1, 1])
                branch_c = layers.conv2d(branch_c, 96, [3, 3])
            with tf.variable_scope('branch_d'):
                branch_d = layers.conv2d(net, 64, [1, 1])
                branch_d = layers.conv2d(branch_d, 96, [3, 3])
                branch_d = layers.conv2d(branch_d, 96, [3, 3])
            net = tf.concat(3, [branch_a, branch_b, branch_c, branch_d])
            # 35 x 35 x 384
    return net


def block_reduce_a(net, k=192, l=224, m=256, n=384, scope='block_reduce_a'):
    # 35 x 35 -> 17 x 17 reduce
    # default padding = VALID
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('branch_a'):
            branch_a = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('branch_b'):
            branch_b = layers.conv2d(net, n, [3, 3], stride=2)
        with tf.variable_scope('branch_c'):
            branch_c = layers.conv2d(net, k, [1, 1], padding='SAME')
            branch_c = layers.conv2d(branch_c, l, [3, 3], padding='SAME')
            branch_c = layers.conv2d(branch_c, m, [3, 3], stride=2)
        net = tf.concat(3, [branch_a, branch_b, branch_c])
        # 17 x 17 x input + n + m (384 + 384 + 256=1024 normally?)
    return net


def block_b(net, scope='block_b'):
    # 17 x 17 x 1024 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_a'):
                branch_a = layers.avg_pool2d(net, [3, 3])
                branch_a = layers.conv2d(branch_a, 128, [1, 1])
            with tf.variable_scope('branch_b'):
                branch_b = layers.conv2d(net, 384, [1, 1])
            with tf.variable_scope('branch_c'):
                branch_c = layers.conv2d(net, 192, [1, 1])
                branch_c = layers.conv2d(branch_c, 224, [1, 7])
                branch_c = layers.conv2d(branch_c, 256, [7, 1])
            with tf.variable_scope('branch_d'):
                branch_d = layers.conv2d(net, 192, [1, 1])
                branch_d = layers.conv2d(branch_d, 192, [1, 7])
                branch_d = layers.conv2d(branch_d, 224, [7, 1])
                branch_d = layers.conv2d(branch_d, 224, [1, 7])
                branch_d = layers.conv2d(branch_d, 256, [7, 1])
            net = tf.concat(3, [branch_a, branch_b, branch_c, branch_d])
            # 17 x 17 x 1024
    return net


def block_reduce_b(net, scope='block_reduce_b'):
    # 17 x 17 -> 8 x 8 reduce
    with tf.variable_scope(scope):
        with tf.variable_scope('branch_a'):
            branch_a = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('branch_b'):
            branch_b = layers.conv2d(net, 192, [1, 1], padding='SAME')
            branch_b = layers.conv2d(branch_b, 192, [3, 3], stride=2)
        with tf.variable_scope('branch_c'):
            branch_c = layers.conv2d(net, 256, [1, 1], padding='SAME')
            branch_c = layers.conv2d(branch_c, 256, [1, 7], padding='SAME')
            branch_c = layers.conv2d(branch_c, 320, [7, 1], padding='SAME')
            branch_c = layers.conv2d(branch_c, 320, [3, 3], stride=2)
        net = tf.concat(3, [branch_a, branch_b, branch_c])
    return net


def block_c(net, scope='block_c'):
    # 8 x 8 x 1536 grid
    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_a'):
                branch_a = layers.avg_pool2d(net, [3, 3])
                branch_a = layers.conv2d(branch_a, 256, [1, 1])
            with tf.variable_scope('branch_b'):
                branch_b = layers.conv2d(net, 256, [1, 1])
            with tf.variable_scope('branch_c'):
                branch_c = layers.conv2d(net, 384, [1, 1])
                branch_c_a = layers.conv2d(branch_c, 256, [1, 3])
                branch_c_b = layers.conv2d(branch_c, 256, [3, 1])
            with tf.variable_scope('branch_d'):
                branch_d = layers.conv2d(net, 384, [1, 1])
                branch_d = layers.conv2d(branch_d, 448, [1, 7])
                branch_d = layers.conv2d(branch_d, 512, [7, 1])
                branch_d_a = layers.conv2d(branch_d, 256, [1, 7])
                branch_d_b = layers.conv2d(branch_d, 256, [7, 1])
            net = tf.concat(3, [branch_a, branch_b, branch_c_a, branch_c_b, branch_d_a, branch_d_b])
            # 8 x 8 x 1536
    return net


def build_inception_v4(
        inputs,
        dropout_keep_prob=0.8,
        num_classes=1000,
        is_training=True,
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
    # endpoints will collect relevant activations for external use, for example
    # summaries or losses.

    endpoints = {}
    with tf.op_scope([inputs], scope, 'inception_v4'):
        with arg_scope([layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], stride=1, padding='VALID'):
                net = stem(inputs, endpoints)
                endpoints['stem'] = net
                # Inception blocks
                for x in range(4):
                    block_scope = 'block_a_%d' % x
                    net = block_a(net, block_scope)
                    endpoints[block_scope] = net
                # 35 x 35 x 384
                net = block_reduce_a(net)
                endpoints['block_reduce_a'] = net
                # 17 x 17 x 1024
                for x in range(7):
                    block_scope = 'block_b_%d' % x
                    net = block_b(net, block_scope)
                    endpoints[block_scope] = net
                # 17 x 17 x 1024
                net = block_reduce_b(net)
                endpoints['block_reduce_b'] = net
                # 8 x 8 x 1536
                for x in range(3):
                    block_scope = 'block_c_%d' % x
                    net = block_c(net, block_scope)
                    endpoints[block_scope] = net
                # 8 x 8 x 1536
                with tf.variable_scope('logits'):
                    shape = net.get_shape()
                    print("pool shape", shape)
                    net = layers.avg_pool2d(net, shape[1:3], scope='pool')
                    # 1 x 1 x 1536
                    net = layers.dropout(net, dropout_keep_prob, scope='dropout')
                    net = layers.flatten(net, scope='flatten')
                    # 1536
                    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
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
        #self.logits_scopes = ['logits/logits', 'aux_logits/FC']

    def build(self, images, num_classes, is_training=False, restore_logits=True, scope=None):
        """Build Inception v4 model architecture.

        See here for reference:

        Args:
          images: Images returned from inputs() or distorted_inputs().
          num_classes: number of classes
          is_training: If set to `True`, build the inference model for training.
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

        self.add_instance(
            scope,
            endpoints,
            logits
        )

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        return logits, None

    def loss(self, labels, batch_size=None, scope=None):
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

        instance = self.instance(scope)

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = instance.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.softmax_cross_entropy(
            instance.logits, dense_labels, label_smoothing=0.1, weight=1.0)

    def variables_to_restore(self):
        return tf.contrib.framework.variables.get_model_variables()

    def get_variables_fn_list(self):
        return [tf.contrib.framework.variable]

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

