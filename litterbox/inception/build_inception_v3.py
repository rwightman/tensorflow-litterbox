# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# Based on original Work Copyright 2016 Google Inc. All Rights Reserved.
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
"""Build the Inception v3 network.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Base off Google's 'tfslim' implementation in Tensorflow models repository.
See model_inception_v3_legacy.py. This version is similar but adapted for
newer tf.contrib layers/helpers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def block_stem(inputs, endpoints):
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope('Stem'):
            # 299 x 299 x 3
            endpoints['Conv1'] = layers.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv1')
            # 149 x 149 x 32
            endpoints['Conv2'] = layers.conv2d(endpoints['Conv1'], 32, [3, 3], scope='Conv2')
            # 147 x 147 x 32
            endpoints['Conv3'] = layers.conv2d(endpoints['Conv2'], 64, [3, 3], padding='SAME', scope='Conv3')
            # 147 x 147 x 64
            endpoints['Pool1'] = layers.max_pool2d(endpoints['Conv3'], [3, 3], stride=2, scope='Pool1')
            # 73 x 73 x 64
            endpoints['Conv4'] = layers.conv2d(endpoints['Pool1'], 80, [1, 1], scope='Conv4')
            # 73 x 73 x 80.
            endpoints['Conv5'] = layers.conv2d(endpoints['Conv4'], 192, [3, 3], scope='Conv5')
            # 71 x 71 x 192.
            net = layers.max_pool2d(endpoints['Conv5'], [3, 3], stride=2, scope='Pool2')
            # 35 x 35 x 192.
            endpoints['Pool2'] = net
    return net


def block_a(net, endpoints, k=32, scope='BlockA'):
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_1x1'):
            branch1x1 = layers.conv2d(net, 64, [1, 1], scope='Conv1')
        with tf.variable_scope('Br2_5x5'):
            branch5x5 = layers.conv2d(net, 48, [1, 1], scope='Conv1')
            branch5x5 = layers.conv2d(branch5x5, 64, [5, 5], scope='Conv2')
        with tf.variable_scope('Br3_3x3Dbl'):
            branch3x3dbl = layers.conv2d(net, 64, [1, 1], scope='Conv1')
            branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3], scope='Conv2')
            branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3], scope='Conv3')
        with tf.variable_scope('Br4_Pool'):
            branch_pool = layers.avg_pool2d(net, [3, 3], scope='Pool1')
            branch_pool = layers.conv2d(branch_pool, k, [1, 1], scope='Conv1')
        net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
        endpoints[scope] = net
    return net


def block_a_reduce(net, endpoints, scope='BlockReduceA'):
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_3x3'):
            branch3x3 = layers.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv1')
        with tf.variable_scope('Br2_3x3Dbl'):
            branch3x3dbl = layers.conv2d(net, 64, [1, 1], scope='Conv1')
            branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3], scope='Conv2')
            branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3], stride=2, padding='VALID', scope='Conv3')
        with tf.variable_scope('Br3_Pool'):
            branch_pool = layers.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='Conv1')
        net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
        endpoints[scope] = net
    return net


def block_b(net, endpoints, k=128, scope='BlockB'):
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_1x1'):
            branch1x1 = layers.conv2d(net, 192, [1, 1], scope='Conv1')
        with tf.variable_scope('Br2_7x7'):
            branch7x7 = layers.conv2d(net, k, [1, 1], scope='Conv1')
            branch7x7 = layers.conv2d(branch7x7, k, [1, 7], scope='Conv2')
            branch7x7 = layers.conv2d(branch7x7, 192, [7, 1], scope='Conv3')
        with tf.variable_scope('Br3_7x7Dbl'):
            branch7x7dbl = layers.conv2d(net, k, [1, 1], scope='Conv1')
            branch7x7dbl = layers.conv2d(branch7x7dbl, k, [7, 1], scope='Conv2')
            branch7x7dbl = layers.conv2d(branch7x7dbl, k, [1, 7], scope='Conv3')
            branch7x7dbl = layers.conv2d(branch7x7dbl, k, [7, 1], scope='Conv4')
            branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7], scope='Conv5')
        with tf.variable_scope('Br4_Pool'):
            branch_pool = layers.avg_pool2d(net, [3, 3], scope='Pool1')
            branch_pool = layers.conv2d(branch_pool, 192, [1, 1], scope='Conv1')
        net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
        endpoints[scope] = net
    return net


def block_b_reduce(net, endpoints, scope='BlockReduceB'):
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_3x3'):
            branch3x3 = layers.conv2d(net, 192, [1, 1], scope='Conv1')
            branch3x3 = layers.conv2d(branch3x3, 320, [3, 3], stride=2, padding='VALID', scope='Conv2')
        with tf.variable_scope('Br2_7x7x3'):
            branch7x7x3 = layers.conv2d(net, 192, [1, 1], scope='Conv1')
            branch7x7x3 = layers.conv2d(branch7x7x3, 192, [1, 7], scope='Conv2')
            branch7x7x3 = layers.conv2d(branch7x7x3, 192, [7, 1], scope='Conv3')
            branch7x7x3 = layers.conv2d(branch7x7x3, 192, [3, 3], stride=2, padding='VALID', scope='Conv4')
        with tf.variable_scope('Br3_Pool'):
            branch_pool = layers.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='Conv1')
        net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
        endpoints[scope] = net
    return net


def block_aux(net, endpoints, num_classes):
    with tf.variable_scope('Aux'):
        aux_logits = layers.avg_pool2d(net, [5, 5], stride=3, padding='VALID', scope='Pool1')
        aux_logits = layers.conv2d(aux_logits, 128, [1, 1], scope='Conv1')

        shape = aux_logits.get_shape()
        #stddev=0.01,
        aux_logits = layers.conv2d(aux_logits, 768, shape[1:3], padding='VALID', scope='Conv2')
        aux_logits = layers.flatten(aux_logits)

        #stddev=0.001
        aux_logits = layers.fully_connected(aux_logits, num_classes, activation_fn=None, scope='AuxLogits')
        endpoints['AuxLogits'] = aux_logits
    return aux_logits


def block_c(net, endpoints, scope='BlockC'):
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_1x1'):
            branch1x1 = layers.conv2d(net, 320, [1, 1], scope='Conv1')
        with tf.variable_scope('Br2_3x3'):
            branch3x3 = layers.conv2d(net, 384, [1, 1], scope='Conv1')
            branch3x3 = tf.concat(3, [layers.conv2d(branch3x3, 384, [1, 3], scope='Conv2a'),
                                      layers.conv2d(branch3x3, 384, [3, 1], scope='Conv2b')])
        with tf.variable_scope('Br3_3x3Dbl'):
            branch3x3dbl = layers.conv2d(net, 448, [1, 1], scope='Conv1')
            branch3x3dbl = layers.conv2d(branch3x3dbl, 384, [3, 3], scope='Conv2')
            branch3x3dbl = tf.concat(3, [layers.conv2d(branch3x3dbl, 384, [1, 3], scope='Conv3a'),
                                         layers.conv2d(branch3x3dbl, 384, [3, 1], scope='Conv3b')])
        with tf.variable_scope('Br4_Pool'):
            branch_pool = layers.avg_pool2d(net, [3, 3], scope='Pool1')
            branch_pool = layers.conv2d(branch_pool, 192, [1, 1], scope='Conv1')
        net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
        endpoints[scope] = net
    return net


def block_output(net, endpoints, num_classes, dropout_keep_prob=0.5):
    with tf.variable_scope('Output'):
        shape = net.get_shape()
        net = layers.avg_pool2d(net, shape[1:3], padding='VALID', scope='Pool1_Global')
        endpoints['Output/Pool1'] = net
        # 1 x 1 x 2048
        net = layers.dropout(net, dropout_keep_prob)
        net = layers.flatten(net)
        # 2048
        logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='Logits')

        # num_classes
        endpoints['Logits'] = logits
    return logits


def stack(net, endpoints, fn=None, count=1, **kwargs):
    scope = kwargs.pop('scope')
    for i in range(count):
        block_scope = '%s%d' % (scope, (i+1))
        kwargs['scope'] = block_scope
        net = fn(net, **kwargs)
        endpoints[block_scope] = net
    print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def build_inception_v3(
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
    activation_fn = tf.nn.relu
    stack_counts = [3, 4, 2]
    endpoints = {}
    # Inception blocks
    op_scope_net = tf.op_scope([inputs], scope, 'inception_v3')
    arg_scope_conv = arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], stride=1, padding='SAME')
    arg_scope_train = arg_scope([layers.batch_norm, layers.dropout], is_training=is_training)
    with op_scope_net, arg_scope_conv, arg_scope_train:

        net = block_stem(inputs, endpoints)
        # 35 x 35 x 384 (v2)

        with tf.variable_scope('Scale1'):
            # mixed: 35 x 35 x 256.
            stack_a_args = [32, 64, 64]
            net = block_a(net, endpoints, k=32, scope='BlockA1')
            # mixed_1: 35 x 35 x 288.
            net = block_a(net, endpoints, k=64, scope='BlockA2')
            # mixed_2: 35 x 35 x 288.
            net = block_a(net, endpoints, k=64, scope='BlockA3')

            #net = stack(
            #    net, endpoints, fn=block_a, count=stack_counts[0],
            #    scope='BlockA', activation_fn=activation_fn)
            # 35 x 35 x 384

        with tf.variable_scope('Scale2'):
            # mixed_3: 17 x 17 x 768.
            stack_b_args = [128, 160, 160, 192]
            net = block_a_reduce(net, endpoints, scope='BlockReduceA')
            # mixed4: 17 x 17 x 768.
            net = block_b(net, endpoints, k=128, scope='BlockB1')
            # mixed_5: 17 x 17 x 768.
            net = block_b(net, endpoints, k=160, scope='BlockB2')
            # mixed_6: 17 x 17 x 768.
            net = block_b(net, endpoints, k=160, scope='BlockB3')
            # mixed_7: 17 x 17 x 768.
            net = block_b(net, endpoints, k=192, scope='BlockB4')

            #net = stack(
            #    net, endpoints, fn=block_b, count=stack_counts[1],
            #    scope='BlockB', activation_fn=activation_fn)
            # 17 x 17 x 896 v1, 1152 v2

            # Auxiliary Head logits
            aux_logits = tf.identity(net)
            block_aux(aux_logits, endpoints, num_classes)

        with tf.variable_scope('Scale3'):
            # mixed_8: 8 x 8 x 1280.
            block_b_reduce(net, endpoints, scope='BlockReduceB')
            # mixed_9: 8 x 8 x 2048.
            block_c(net, endpoints, scope='BlockC1')
            # mixed_10: 8 x 8 x 2048.
            block_c(net, endpoints, scope='BlockC2')

        # Final pooling and prediction
        logits = block_output(net, endpoints, num_classes, dropout_keep_prob)
        endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, endpoints
