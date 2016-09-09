# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" ResNet building blocks for ResNet models
"""
import layers as my_layers

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

FLAGS = tf.app.flags.FLAGS


def calc_num_filters_out(num_filters_internal, bottleneck=False):
    # If bottleneck, num_filters_internal*4 filters are output.
    # num_filters_internal is how many filters the 3x3 convolutions output.
    return 4 * num_filters_internal if bottleneck else num_filters_internal


def stem(inputs, endpoints, num_filters=64):
    with tf.variable_scope('Stem'):
        net = layers.conv2d(inputs, num_filters, [7, 7], stride=2, scope='Conv1_7x7')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3')
    endpoints['Stem'] = net
    print("Stem output size: ", net.get_shape())
    return net


#@add_arg_scope
def block_original(net, num_filters_internal, block_stride,
                   bottleneck=False, res_scale=None, scope='Block', activation_fn=tf.nn.relu):
    """Definition of the original Resnet 'basic' (double 3x3) and 'bottleneck' (1x1 + 3x3 + 1x1) block"""
    # default padding=SAME
    # default stride=1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = calc_num_filters_out(num_filters_internal, bottleneck)

    with tf.variable_scope(scope):
        with tf.variable_scope('Shortcut'):
            if num_filters_out != num_filters_in or block_stride != 1:
                shortcut = layers.conv2d(
                    net, num_filters_out, [1, 1],
                    stride=block_stride, activation_fn=None, padding='VALID', scope='Conv1_1x1')
            else:
                shortcut = tf.identity(net)

            if res_scale:
                shortcut = tf.mul(shortcut, res_scale)  # scale the residual by constant if param is set

        if bottleneck:
            net = layers.conv2d(net, num_filters_internal, [1, 1], stride=block_stride, scope='Conv1_1x1')
            net = layers.conv2d(net, num_filters_internal, [3, 3], scope='Conv2_3x3')
            net = layers.conv2d(net, num_filters_out, [1, 1], activation_fn=None, scope='Conv3_1x1')
        else:
            net = layers.conv2d(net, num_filters_internal, [3, 3], stride=block_stride, scope='Conv1_3x3')
            net = layers.conv2d(net, num_filters_out, [3, 3], activation_fn=None, scope='Conv2_3x3')

        net = tf.add(net, shortcut)
        return activation_fn(net)


#@add_arg_scope
def block_preact(net, num_filters_internal, block_stride,
                 bottleneck=False, res_scale=None, scope='Block', activation_fn=tf.nn.relu):
    """Definition of the pre-activation Resnet 'basic' (double 3x3 and 'bottleneck' (1x1 + 3x3 + 1x1) block"""
    # default padding=SAME
    # default stride=1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = calc_num_filters_out(num_filters_internal, bottleneck)

    with tf.variable_scope(scope):
        # shared norm + activation for both convolution and residual shortcut branch
        net = layers.batch_norm(net, activation_fn=activation_fn)

        with tf.variable_scope('Shortcut'):
            if num_filters_out != num_filters_in or block_stride != 1:
                shortcut = my_layers.preact_conv2d(
                    net, num_filters_out, [1, 1], stride=block_stride,
                    normalizer_fn=None, padding='VALID', scope='Conv1_1x1')
            else:
                shortcut = tf.identity(net)

            if res_scale:
                shortcut = tf.mul(shortcut, res_scale)  # scale the residual by constant if param is set

        if bottleneck:
            net = my_layers.preact_conv2d(
                net, num_filters_internal, [1, 1], stride=block_stride, normalizer_fn=None, scope='Conv1_1x1')
            net = my_layers.preact_conv2d(net, num_filters_internal, [3, 3], scope='Conv2_3x3')
            net = my_layers.preact_conv2d(net, num_filters_out, [1, 1], scope='Conv3_1x1')
        else:
            net = my_layers.preact_conv2d(
                net, num_filters_internal, [3, 3], stride=block_stride, normalizer_fn=None, scope='Conv1_3x3')
            net = my_layers.preact_conv2d(net, num_filters_out, [3, 3], scope='Conv2_3x3')

        return tf.add(net, shortcut)


def stack(net, endpoints, num_blocks, num_filters, stack_stride=1, pre_act=False, **kwargs):
    scope = kwargs.pop('scope')
    with tf.variable_scope(scope):
        for i in range(num_blocks):
            block_stride = stack_stride if i == 0 else 1
            block_scope = 'Block' + str(i + 1)
            kwargs['scope'] = block_scope
            if pre_act:
                net = block_preact(net, num_filters, block_stride, **kwargs)
            else:
                net = block_original(net, num_filters, block_stride, **kwargs)
            endpoints[scope + block_scope] = net
        print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def output(net, endpoints, num_classes, pre_act=False):
    with tf.variable_scope('Output'):
        if pre_act:
            net = layers.batch_norm(net, activation_fn=tf.nn.relu)
        shape = net.get_shape()
        net = layers.avg_pool2d(net, shape[1:3], scope='Pool1_Global')
        endpoints['OutputPool1'] = net
        net = layers.flatten(net)
        net = layers.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
        endpoints['Logits'] = net
        # num_classes
    return net


def build_resnet(
        inputs,
        k=1,  # width factor
        pre_activation=False,
        num_classes=1000,
        num_blocks=[3, 4, 6, 3],
        is_training=True,
        bottleneck=True,
        scope=''):
    """Blah"""

    endpoints = {}  # A dictionary of endpoints to observe (activations, extra stats, etc)
    op_scope_net = tf.op_scope([inputs], scope, 'ResNet')
    arg_scope_train = arg_scope([layers.batch_norm, layers.dropout], is_training=is_training)
    arg_scope_conv = arg_scope(
        [layers.conv2d, my_layers.preact_conv2d, layers.max_pool2d, layers.avg_pool2d],
        stride=1, padding='SAME')
    with op_scope_net, arg_scope_train, arg_scope_conv:

        # 224 x 224
        net = stem(inputs, endpoints, 64 * k)
        # 56 x 56

        net = stack(
            net, endpoints, num_blocks[0], num_filters=64 * k, stack_stride=1,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale1')
        # 56 x 56

        net = stack(
            net, endpoints, num_blocks[1], num_filters=128 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale2')
        # 28 x 28

        net = stack(
            net, endpoints, num_blocks[2], num_filters=256 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale3')
        # 14 x 14

        net = stack(
            net, endpoints, num_blocks[3], num_filters=512 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale4')
        # 7 x 7

        logits = output(net, endpoints, num_classes, pre_act=pre_activation)
        endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, endpoints
