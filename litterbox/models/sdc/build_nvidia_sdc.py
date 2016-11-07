from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def build_nvidia_sdc(
        inputs,
        output_cfg={'steer': 1},
        version=1,
        is_training=True,
        bayesian=False,
        dropout_keep_prob=0.7,
        reuse=None,
        scope='NvidiaSdc'):

    endpoints = {}
    var_scope = tf.variable_scope(scope, 'NvidiaSdc', [inputs], reuse=reuse)
    arg_scope_train = slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training)
    arg_scope_conv = slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID')
    with var_scope, arg_scope_train, arg_scope_conv:
        #160 x 120 3 x
        print(inputs.get_shape())
        net = slim.conv2d(inputs, 24, 5, stride=2, scope='Conv1_5x5')
        endpoints['Conv1_5x5'] = net
        print(net.get_shape())
        net = slim.conv2d(net, 36, 5, stride=2, scope='Conv2_5x5')
        endpoints['Conv2_5x5'] = net
        print(net.get_shape())
        net = slim.conv2d(net, 48, 5, stride=2, scope='Conv3_5x5')
        endpoints['Conv3_5x5'] = net
        print(net.get_shape())
        net = slim.conv2d(net, 64, 3, scope='Conv4_3x3')
        endpoints['Conv4_3x3'] = net
        print(net.get_shape())
        net = slim.conv2d(net, 64, 3, scope='Conv5_3x3')
        endpoints['Conv5_3x3'] = net
        print(net.get_shape())

        with tf.variable_scope('Output'):
            net = slim.conv2d(net, 1152, net.get_shape()[1:3], scope='Fc1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout1')
            print(net.get_shape())
            net = slim.conv2d(net, 144, 1, scope='Fc2')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout2')
            print(net.get_shape())
            net = slim.conv2d(net, 72, 1, scope='Fc3')
            print(net.get_shape())
            net = slim.flatten(net)
            output = {}
            if 'xyz' in output_cfg:
                output['xyz'] = slim.fully_connected(
                    net, output_cfg['xyz'], activation_fn=None, scope='OutputXYZ')
            if 'steer' in output_cfg:
                output['steer'] = slim.fully_connected(
                    net, output_cfg['steer'], activation_fn=None, scope='OutputSteer')
            endpoints['Output'] = output

    return output, endpoints


def nvidia_style_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001):
    """Yields the scope with the default parameters for inception_resnet_v2.

    Args:
      weight_decay: the weight decay for weights variables.
      batch_norm_decay: decay for the moving average of batch_norm momentums.
      batch_norm_epsilon: small float added to variance to avoid dividing by zero.

    Returns:
      a arg_scope with the parameters needed for inception_resnet_v2.
    """
    # Set weight_decay for weights in conv2d and fully_connected layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.elu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
            return scope

