# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" The Inception v4 network.
    Implementation of Inception V4, Inception-Resnet-V1, Inception-Resnet-V2
    based on https://arxiv.org/abs/1602.07261
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def block_stem(net, endpoints, scope='Stem'):
    # Stem shared by inception-v4 and inception-resnet-v2 (resnet-v1 uses simpler stem below)
    # NOTE observe endpoints of first 3 layers
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope(scope):
            # 299 x 299 x 3
            net = layers.conv2d(net, 32, [3, 3], stride=2, scope='Conv1_3x3/2')
            endpoints[scope + '/Conv1'] = net
            # 149 x 149 x 32
            net = layers.conv2d(net, 32, [3, 3], scope='Conv2_3x3')
            endpoints[scope + '/Conv2'] = net
            # 147 x 147 x 32
            net = layers.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv3_3x3')
            endpoints[scope + '/Conv3'] = net
            # 147 x 147 x 64
            with tf.variable_scope('Br1A_Pool'):
                br1a = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3/2')
            with tf.variable_scope('Br1B_3x3'):
                br1b = layers.conv2d(net, 96, [3, 3], stride=2, scope='Conv4_3x3/2')
            net = tf.concat(3, [br1a, br1b], name='Concat1')
            endpoints[scope + '/Concat1'] = net
            # 73 x 73 x 160
            with tf.variable_scope('Br2A_3x3'):
                br2a = layers.conv2d(net, 64, [1, 1], padding='SAME', scope='Conv5_1x1')
                br2a = layers.conv2d(br2a, 96, [3, 3], scope='Conv6_3x3')
            with tf.variable_scope('Br2B_7x7x3'):
                br2b = layers.conv2d(net, 64, [1, 1], padding='SAME', scope='Conv5_1x1')
                br2b = layers.conv2d(br2b, 64, [7, 1], padding='SAME', scope='Conv6_7x1')
                br2b = layers.conv2d(br2b, 64, [1, 7], padding='SAME', scope='Conv7_1x7')
                br2b = layers.conv2d(br2b, 96, [3, 3], scope='Conv8_3x3')
            net = tf.concat(3, [br2a, br2b], name='Concat2')
            endpoints[scope + '/Concat2'] = net
            # 71 x 71 x 192
            with tf.variable_scope('Br3A_3x3'):
                br3a = layers.conv2d(net, 192, [3, 3], stride=2, scope='Conv9_3x3/2')
            with tf.variable_scope('Br3B_Pool'):
                br3b = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool2_3x3/2')
            net = tf.concat(3, [br3a, br3b], name='Concat3')
            endpoints[scope + '/Concat3'] = net
            print('%s output shape: %s' % (scope, net.get_shape()))
            # 35x35x384
    return net


def block_a(net, scope='BlockA'):
    # 35 x 35 x 384 grid
    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_Pool'):
            br1 = layers.avg_pool2d(net, [3, 3], scope='Pool1_3x3')
            br1 = layers.conv2d(br1, 96, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_1x1'):
            br2 = layers.conv2d(net, 96, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br3_3x3'):
            br3 = layers.conv2d(net, 64, [1, 1], scope='Conv1_1x1')
            br3 = layers.conv2d(br3, 96, [3, 3], scope='Conv2_3x3')
        with tf.variable_scope('Br4_3x3Dbl'):
            br4 = layers.conv2d(net, 64, [1, 1], scope='Conv1_1x1')
            br4 = layers.conv2d(br4, 96, [3, 3], scope='Conv2_3x3')
            br4 = layers.conv2d(br4, 96, [3, 3], scope='Conv3_3x3')
        net = tf.concat(3, [br1, br2, br3, br4], name='Concat1')
        # 35 x 35 x 384
    return net


def block_a_reduce(net, endpoints, k=192, l=224, m=256, n=384, scope='BlockReduceA'):
    # 35 x 35 -> 17 x 17 reduce
    # inception-v4: k=192, l=224, m=256, n=384
    # inception-resnet-v1: k=192, l=192, m=256, n=384
    # inception-resnet-v2: k=256, l=256, m=384, n=384
    # default padding = VALID
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope(scope):
            with tf.variable_scope('Br1_Pool'):
                br1 = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3/2')
                # 17 x 17 x input
            with tf.variable_scope('Br2_3x3'):
                br2 = layers.conv2d(net, n, [3, 3], stride=2, scope='Conv1_3x3/2')
                # 17 x 17 x n
            with tf.variable_scope('Br3_3x3Dbl'):
                br3 = layers.conv2d(net, k, [1, 1], padding='SAME', scope='Conv1_1x1')
                br3 = layers.conv2d(br3, l, [3, 3], padding='SAME', scope='Conv2_3x3')
                br3 = layers.conv2d(br3, m, [3, 3], stride=2, scope='Conv3_3x3/2')
                # 17 x 17 x m
            net = tf.concat(3, [br1, br2, br3], name='Concat1')
            # 17 x 17 x input + n + m
            # 1024 for v4 (384 + 384 + 256)
            # 896 for res-v1 (256 + 384 +256)
            # 1152 for res-v2 (384 + 384 + 384)
            endpoints[scope] = net
            print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def block_b(net, scope='BlockB'):
    # 17 x 17 x 1024 grid
    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_Pool'):
            br1 = layers.avg_pool2d(net, [3, 3], scope='Pool1_3x3')
            br1 = layers.conv2d(br1, 128, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_1x1'):
            br2 = layers.conv2d(net, 384, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br3_7x7'):
            br3 = layers.conv2d(net, 192, [1, 1], scope='Conv1_1x1')
            br3 = layers.conv2d(br3, 224, [1, 7], scope='Conv2_1x7')
            br3 = layers.conv2d(br3, 256, [7, 1], scope='Conv3_7x1')
        with tf.variable_scope('Br4_7x7Dbl'):
            br4 = layers.conv2d(net, 192, [1, 1], scope='Conv1_1x1')
            br4 = layers.conv2d(br4, 192, [1, 7], scope='Conv2_1x7')
            br4 = layers.conv2d(br4, 224, [7, 1], scope='Conv3_7x1')
            br4 = layers.conv2d(br4, 224, [1, 7], scope='Conv4_1x7')
            br4 = layers.conv2d(br4, 256, [7, 1], scope='Conv5_7x1')
        net = tf.concat(3, [br1, br2, br3, br4], name='Concat1')
        # 17 x 17 x 1024
    return net


def block_b_reduce(net, endpoints, scope='BlockReduceB'):
    # 17 x 17 -> 8 x 8 reduce
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope(scope):
            with tf.variable_scope('Br1_Pool'):
                br1 = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3/2')
            with tf.variable_scope('Br2_3x3'):
                br2 = layers.conv2d(net, 192, [1, 1], padding='SAME', scope='Conv1_1x1')
                br2 = layers.conv2d(br2, 192, [3, 3], stride=2, scope='Conv2_3x3/2')
            with tf.variable_scope('Br3_7x7x3'):
                br3 = layers.conv2d(net, 256, [1, 1], padding='SAME', scope='Conv1_1x1')
                br3 = layers.conv2d(br3, 256, [1, 7], padding='SAME', scope='Conv2_1x7')
                br3 = layers.conv2d(br3, 320, [7, 1], padding='SAME', scope='Conv3_7x1')
                br3 = layers.conv2d(br3, 320, [3, 3], stride=2, scope='Conv4_3x3/2')
            net = tf.concat(3, [br1, br2, br3], name='Concat1')
            endpoints[scope] = net
            print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def block_c(net, scope='BlockC'):
    # 8 x 8 x 1536 grid
    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('Br1_Pool'):
            br1 = layers.avg_pool2d(net, [3, 3], scope='Pool1_3x3')
            br1 = layers.conv2d(br1, 256, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_1x1'):
            br2 = layers.conv2d(net, 256, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br3_3x3'):
            br3 = layers.conv2d(net, 384, [1, 1], scope='Conv1_1x1')
            br3a = layers.conv2d(br3, 256, [1, 3], scope='Conv2_1x3')
            br3b = layers.conv2d(br3, 256, [3, 1], scope='Conv3_3x1')
        with tf.variable_scope('Br4_7x7Dbl'):
            br4 = layers.conv2d(net, 384, [1, 1], scope='Conv1_1x1')
            br4 = layers.conv2d(br4, 448, [1, 7], scope='Conv2_1x7')
            br4 = layers.conv2d(br4, 512, [7, 1], scope='Conv3_7x1')
            br4a = layers.conv2d(br4, 256, [1, 7], scope='Conv4a_1x7')
            br4b = layers.conv2d(br4, 256, [7, 1], scope='Conv4b_7x1')
        net = tf.concat(3, [br1, br2, br3a, br3b, br4a, br4b], name='Concat1')
        # 8 x 8 x 1536
    return net


def block_stem_res(net, endpoints, scope='Stem'):
    # Simpler stem for inception-resnet-v1 network
    # NOTE observe endpoints of first 3 layers
    # default padding = VALID
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope(scope):
            # 299 x 299 x 3
            net = layers.conv2d(net, 32, [3, 3], stride=2, scope='Conv1_3x3/2')
            endpoints[scope + '/Conv1'] = net
            # 149 x 149 x 32
            net = layers.conv2d(net, 32, [3, 3], scope='Conv2_3x3')
            endpoints[scope + '/Conv2'] = net
            # 147 x 147 x 32
            net = layers.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv3_3x3')
            endpoints[scope + '/Conv3'] = net
            # 147 x 147 x 64
            net = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3/2')
            # 73 x 73 x 64
            net = layers.conv2d(net, 80, [1, 1], padding='SAME', scope='Conv4_1x1')
            # 73 x 73 x 80
            net = layers.conv2d(net, 192, [3, 3], scope='Conv5_3x3')
            # 71 x 71 x 192
            net = layers.conv2d(net, 256, [3, 3], stride=2, scope='Conv6_3x3/2')
            # 35 x 35 x 256
            endpoints[scope] = net
            print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def block_a_res(net, ver=2, res_scale=None, scope='BlockA', activation_fn=tf.nn.relu):
    # 35x35 grid

    # configure branch filter numbers
    br3_num = 32
    if ver == 1:
        br3_inc = 0
    else:
        br3_inc = 16

    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        shortcut = tf.identity(net, name='Shortcut')
        if res_scale:
            shortcut = tf.mul(shortcut, res_scale)  # scale residual
        with tf.variable_scope('Br1_1x1'):
            br1 = layers.conv2d(net, 32, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_3x3'):
            br2 = layers.conv2d(net, 32, [1, 1], scope='Conv1_1x1')
            br2 = layers.conv2d(br2, 32, [3, 3], scope='Conv2_3x3')
        with tf.variable_scope('Br3_3x3Dbl'):
            br3 = layers.conv2d(net, br3_num, [1, 1], scope='Conv1_1x1')
            br3 = layers.conv2d(br3, br3_num + 1*br3_inc, [3, 3], scope='Conv2_3x3')
            br3 = layers.conv2d(br3, br3_num + 2*br3_inc, [3, 3], scope='Conv3_3x3')
        net = tf.concat(3, [br1, br2, br3], name='Concat1')
        net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None, scope='Conv4_1x1')
        net = activation_fn(tf.add(shortcut, net, name='Sum1'))
        # 35 x 35 x 256 res-v1, 384 res-v2
    return net


def block_b_res(net, ver=2, res_scale=None, scope='BlockB', activation_fn=tf.nn.relu):
    # 17 x 17 grid

    # configure branch filter numbers
    if ver == 1:
        br1_num = 128
        br2_num = 128
        br2_inc = 0
    else:
        br1_num = 192
        br2_num = 128
        br2_inc = 32

    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        shortcut = tf.identity(net, name='Shortcut')
        if res_scale:
            shortcut = tf.mul(shortcut, res_scale)  # scale residual
        with tf.variable_scope('Br1_1x1'):
            br1 = layers.conv2d(net, br1_num, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_7x7'):
            br2 = layers.conv2d(net, br2_num, [1, 1], scope='Conv1_1x1')
            br2 = layers.conv2d(br2, br2_num + 1*br2_inc, [1, 7], scope='Conv2_1x7')
            br2 = layers.conv2d(br2, br2_num + 2*br2_inc, [7, 1], scope='Conv3_7x1')
        net = tf.concat(3, [br1, br2], name='Concat1')
        net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None, scope='Conv4_1x1')
        # 17 x 17 x 896 res-v1, 1152 res-v2. Typo in paper, 1152, not 1154
        net = activation_fn(tf.add(shortcut, net, name='Sum1'))
    return net


def block_b_reduce_res(net, endpoints, ver=2, scope='BlockReduceB'):
    # 17 x 17 -> 8 x 8 reduce

    # configure branch filter numbers
    br3_num = 256
    br4_num = 256
    if ver == 1:
        br3_inc = 0
        br4_inc = 0
    else:
        br3_inc = 32
        br4_inc = 32

    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='VALID'):
        with tf.variable_scope(scope):
            with tf.variable_scope('Br1_Pool'):
                br1 = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3/2')
            with tf.variable_scope('Br2_3x3'):
                br2 = layers.conv2d(net, 256, [1, 1], padding='SAME', scope='Conv1_1x1')
                br2 = layers.conv2d(br2, 384, [3, 3], stride=2, scope='Conv2_3x3/2')
            with tf.variable_scope('Br3_3x3'):
                br3 = layers.conv2d(net, br3_num, [1, 1], padding='SAME', scope='Conv1_1x1')
                br3 = layers.conv2d(br3, br3_num + br3_inc, [3, 3], stride=2, scope='Conv2_3x3/2')
            with tf.variable_scope('Br4_3x3Dbl'):
                br4 = layers.conv2d(net, br4_num, [1, 1], padding='SAME', scope='Conv1_1x1')
                br4 = layers.conv2d(br4, br4_num + 1*br4_inc, [3, 3], padding='SAME', scope='Conv2_3x3')
                br4 = layers.conv2d(br4, br4_num + 2*br4_inc, [3, 3], stride=2, scope='Conv3_3x3/2')
            net = tf.concat(3, [br1, br2, br3, br4], name='Concat1')
            # 8 x 8 x 1792 v1, 2144 v2 (paper indicates 2048 but only get this if we use a v1 config for this block)
            endpoints[scope] = net
            print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def block_c_res(net, ver=2, res_scale=None, scope='BlockC', activation_fn=tf.nn.relu):
    # 8 x 8 grid

    # configure branch filter numbers
    br2_num = 192
    if ver == 1:
        br2_inc = 0
    else:
        br2_inc = 32

    # default padding = SAME
    # default stride = 1
    with tf.variable_scope(scope):
        shortcut = tf.identity(net, name='Shortcut')
        if res_scale:
            shortcut = tf.mul(shortcut, res_scale)  # scale residual
        with tf.variable_scope('Br1_1x1'):
            br1 = layers.conv2d(net, 192, [1, 1], scope='Conv1_1x1')
        with tf.variable_scope('Br2_3x3'):
            br2 = layers.conv2d(net, br2_num, [1, 1], scope='Conv1_1x1')
            br2 = layers.conv2d(br2, br2_num + 1*br2_inc, [1, 3], scope='Conv2_1x3')
            br2 = layers.conv2d(br2, br2_num + 2*br2_inc, [3, 1], scope='Conv3_3x1')
        net = tf.concat(3, [br1, br2], name='Concat1')
        net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None, scope='Conv4_1x1')
        # 1792 res-1, 2144 (2048?) res-2
        net = activation_fn(tf.add(shortcut, net, name='Sum1'))
    return net


def block_output(net, endpoints, num_classes=1000, dropout_keep_prob=0.5, scope='Output'):
    with tf.variable_scope(scope):
        # 8 x 8 x 1536
        shape = net.get_shape()
        net = layers.avg_pool2d(net, shape[1:3], padding='VALID', scope='Pool1_Global')
        endpoints['Output/Pool1'] = net
        # 1 x 1 x 1536
        net = layers.dropout(net, dropout_keep_prob)
        net = layers.flatten(net)
        # 1536
        net = layers.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
        # num classes
        endpoints['Logits'] = net
    return net


def stack(net, endpoints, fn=None, count=1, **kwargs):
    scope = kwargs.pop('scope')
    for i in range(count):
        block_scope = '%s%d' % (scope, (i+1))
        kwargs['scope'] = block_scope
        net = fn(net, **kwargs)
        endpoints[block_scope] = net
    print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def build_inception_v4(
        inputs,
        stack_counts=[4, 7, 3],
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
    op_scope_net = tf.op_scope([inputs], scope, 'Inception_v4')
    arg_scope_train = arg_scope([layers.batch_norm, layers.dropout], is_training=is_training)
    arg_scope_conv = arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], stride=1, padding='SAME')
    with op_scope_net, arg_scope_train, arg_scope_conv:

        net = block_stem(inputs, endpoints)
        # 35 x 35 x 384

        with tf.variable_scope('Scale1'):
            net = stack(net, endpoints, fn=block_a, count=stack_counts[0], scope='BlockA')
            # 35 x 35 x 384

        with tf.variable_scope('Scale2'):
            net = block_a_reduce(net, endpoints)
            # 17 x 17 x 1024
            net = stack(net, endpoints, fn=block_b, count=stack_counts[1], scope='BlockB')
            # 17 x 17 x 1024

        with tf.variable_scope('Scale3'):
            net = block_b_reduce(net, endpoints)
            # 8 x 8 x 1536
            net = stack(net, endpoints, fn=block_c, count=stack_counts[2], scope='BlockC')
            # 8 x 8 x 1536

        logits = block_output(net, endpoints, num_classes, dropout_keep_prob, scope='Output')
        # num_classes
        endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, endpoints


def build_inception_resnet(
        inputs,
        stack_counts=[5, 10, 5],
        ver=2,
        res_scale=None,
        activation_fn=tf.nn.relu,
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
      'logits' tensor
      'endpoints' dict
    """
    # endpoints will collect relevant activations for external use, for example, summaries or losses.
    assert ver == 1 or ver == 2
    endpoints = {}
    network_name = 'inception_resnet_v%d' % ver
    print("Building %s" % network_name)

    op_scope_net = tf.op_scope([inputs], scope, network_name)
    arg_scope_train = arg_scope([layers.batch_norm, layers.dropout], is_training=is_training)
    arg_scope_conv = arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], stride=1, padding='SAME')
    with op_scope_net, arg_scope_train, arg_scope_conv:

        net = block_stem_res(inputs, endpoints) if ver == 1 else block_stem(inputs, endpoints)
        # 35 x 35 x 384 (v2)

        with tf.variable_scope('Scale1'):
            net = stack(
                net, endpoints, fn=block_a_res, count=stack_counts[0], scope='BlockA',
                ver=ver, res_scale=res_scale, activation_fn=activation_fn)
            # 35 x 35 x 384

        with tf.variable_scope('Scale2'):
            k, l, m, n = (192, 192, 256, 384) if ver == 1 else (256, 256, 384, 384)
            net = block_a_reduce(net, endpoints, k=k, l=l, m=m, n=n)
            # 17 x 17 x 896 v1, 1152 v2

            net = stack(
                net, endpoints, fn=block_b_res, count=stack_counts[1], scope='BlockB',
                ver=ver, res_scale=res_scale, activation_fn=activation_fn)
            # 17 x 17 x 896 v1, 1152 v2

        with tf.variable_scope('Scale3'):
            net = block_b_reduce_res(net, endpoints, ver=ver)
            # 8 x 8 x 1792 v1, 2144 v2

            net = stack(
                net, endpoints, fn=block_c_res, count=stack_counts[2], scope='BlockC',
                ver=ver, res_scale=res_scale, activation_fn=activation_fn)
            # 8 x 8 x 1792 v1, 2144 v2

        logits = block_output(net, endpoints, num_classes, dropout_keep_prob, 'Output')
        # num_classes
        endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, endpoints
