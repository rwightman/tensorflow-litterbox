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
import tensorflow as tf
import layers as my_layers
from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def _calc_num_filters_out(num_filters_internal, bottleneck=False):
    # If bottleneck, num_filters_internal*4 filters are output.
    # num_filters_internal is how many filters the 3x3 convolutions output.
    return 4 * num_filters_internal if bottleneck else num_filters_internal


def _stem(inputs, endpoints, num_filters=64):
    with tf.variable_scope('Stem'):
        net = layers.conv2d(inputs, num_filters, [7, 7], stride=2, scope='Conv1_7x7')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='Pool1_3x3')
    endpoints['Stem'] = net
    print("Stem output size: ", net.get_shape())
    return net


#@add_arg_scope
def _block_original(net, num_filters_internal, block_stride,
                    bottleneck=False, res_scale=None, scope='Block', activation_fn=tf.nn.relu):
    """ Definition of the original Resnet
        'basic' (double 3x3) and 'bottleneck' (1x1 + 3x3 + 1x1) block
    """
    # default padding=SAME
    # default stride=1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = _calc_num_filters_out(num_filters_internal, bottleneck)

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
def _block_preact(net, num_filters_internal, block_stride,
                  bottleneck=False, res_scale=None, scope='Block', activation_fn=tf.nn.relu):
    """ Definition of the pre-activation Resnet
        'basic' (double 3x3 and 'bottleneck' (1x1 + 3x3 + 1x1) block
    """
    # default padding=SAME
    # default stride=1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = _calc_num_filters_out(num_filters_internal, bottleneck)

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


def _stack(net, endpoints, num_blocks, num_filters, stack_stride=1, pre_act=False, **kwargs):
    scope = kwargs.pop('scope')
    with tf.variable_scope(scope):
        for i in range(num_blocks):
            block_stride = stack_stride if i == 0 else 1
            block_scope = 'Block' + str(i + 1)
            kwargs['scope'] = block_scope
            if pre_act:
                net = _block_preact(net, num_filters, block_stride, **kwargs)
            else:
                net = _block_original(net, num_filters, block_stride, **kwargs)
            endpoints[scope + block_scope] = net
        print('%s output shape: %s' % (scope, net.get_shape()))
    return net


def _output(net, endpoints, num_classes, pre_act=False):
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


def build_resnet_modules(
        inputs,
        k=1,  # width factor
        pre_activation=False,
        num_classes=1000,
        num_blocks=[3, 4, 6, 3],
        is_training=True,
        bottleneck=True,
        scope=''):
    """

    Args:
        inputs:
        k:
        pre_activation:
        num_classes:
        num_blocks:
        is_training:
        bottleneck:
        scope:

    Returns:

    """

    endpoints = {}  # A dictionary of endpoints to observe (activations, extra stats, etc)
    name_scope_net = tf.name_scope(scope, 'Resnet', [inputs])
    arg_scope_train = arg_scope(
        [layers.batch_norm, layers.dropout],
        is_training=is_training)
    arg_scope_conv = arg_scope(
        [layers.conv2d, my_layers.preact_conv2d, layers.max_pool2d, layers.avg_pool2d],
        stride=1, padding='SAME')
    with name_scope_net, arg_scope_train, arg_scope_conv:

        # 224 x 224
        net = _stem(inputs, endpoints, 64 * k)
        # 56 x 56

        net = _stack(
            net, endpoints, num_blocks[0], num_filters=64 * k, stack_stride=1,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale1')
        # 56 x 56

        net = _stack(
            net, endpoints, num_blocks[1], num_filters=128 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale2')
        # 28 x 28

        net = _stack(
            net, endpoints, num_blocks[2], num_filters=256 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale3')
        # 14 x 14

        net = _stack(
            net, endpoints, num_blocks[3], num_filters=512 * k, stack_stride=2,
            bottleneck=bottleneck, pre_act=pre_activation, scope='Scale4')
        # 7 x 7

        logits = _output(net, endpoints, num_classes, pre_act=pre_activation)
        endpoints['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, endpoints


def params_resnet(num_layers=50, pre_activation=False):
    params = {
        'num_layers': num_layers,
        'pre_activation': pre_activation,
        'width_factor': 1,
    }
    return params


def resnet_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
    }
    l2_regularizer = layers.l2_regularizer(weight_decay)

    arg_scope_layers = arg_scope(
        [layers.conv2d, my_layers.preact_conv2d, layers.fully_connected],
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=l2_regularizer,
        activation_fn=tf.nn.relu)
    arg_scope_conv = arg_scope(
        [layers.conv2d, my_layers.preact_conv2d],
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params)
    with arg_scope_layers, arg_scope_conv as arg_sc:
        return arg_sc


def build_resnet(
        inputs,
        num_classes=1000,
        params=params_resnet(),
        is_training=True,
):
    params = model.merge_params(params_resnet(), params)
    num_layers = params['num_layers']
    pre_activation = params['pre_activation']
    width_factor = params['width_factor']

    # layer configs
    if num_layers == 16:
        # not official size, my 'minimal' resnet
        num_blocks = [1, 2, 3, 1]
        bottleneck = False
        # filter output depth = 512
    elif num_layers == 18:
        num_blocks = [2, 2, 2, 2]
        bottleneck = False
        # filter output depth = 512
    elif num_layers == 34:
        num_blocks = [3, 4, 6, 3]
        bottleneck = False
        # filter output depth = 512
    elif num_layers == 50:
        num_blocks = [3, 4, 6, 3]
        bottleneck = True
        # filter output depth 2048
    elif num_layers == 101:
        num_blocks = [3, 4, 23, 3]
        bottleneck = True
        # filter output depth 2048
    elif num_layers == 152:
        num_blocks = [3, 8, 36, 3]
        bottleneck = True
        # filter output depth 2048
    elif num_layers == 200:
        num_blocks = [3, 24, 36, 3]
        bottleneck = True
        # filter output depth 2048
    else:
        assert False, "Invalid number of layers"

    version = 2 if pre_activation else 1
    name_scope = 'resnet_v%d_%d' % (version, num_layers)

    with arg_scope(resnet_arg_scope()):
        logits, endpoints = build_resnet_modules(
            inputs,
            k=width_factor,
            pre_activation=pre_activation,
            num_classes=num_classes,
            num_blocks=num_blocks,
            bottleneck=bottleneck,
            is_training=is_training,
            scope=name_scope)

    return logits, endpoints
