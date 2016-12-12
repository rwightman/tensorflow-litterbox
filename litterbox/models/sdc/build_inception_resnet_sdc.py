from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        mixed = tf.concat(3, [tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
        mixed = tf.concat(3, [tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def build_inception_resnet_sdc_regression(
        inputs,
        output_cfg={'steer': 1},
        version=1,
        is_training=True,
        bayesian=False,
        dropout_keep_prob=0.7,
        reuse=None,
        scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model.

    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      logits: the logits outputs of the model.
      endpoints: the set of endpoints from the inception model.
    """
    endpoints = {}
    var_scope = tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse)
    arg_scope_train = slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training)
    arg_scope_conv = slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME')
    with var_scope, arg_scope_train, arg_scope_conv:
        # 149 x 149 x 32
        net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        endpoints['Conv2d_1a_3x3'] = net
        # 147 x 147 x 32
        net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
        endpoints['Conv2d_2a_3x3'] = net
        # 147 x 147 x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        endpoints['Conv2d_2b_3x3'] = net
        # 73 x 73 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
        endpoints['MaxPool_3a_3x3'] = net
        # 73 x 73 x 80
        net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
        endpoints['Conv2d_3b_1x1'] = net
        # 71 x 71 x 192
        net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
        endpoints['Conv2d_4a_3x3'] = net
        # 35 x 35 x 192
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
        endpoints['MaxPool_5a_3x3'] = net

        # 35 x 35 x 320
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
                tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
            net = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1])

        endpoints['Mixed_5b'] = net
        net = slim.repeat(net, 10, block35, scale=0.17)

        # 17 x 17 x 1024
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat(3, [tower_conv, tower_conv1_2, tower_pool])

        endpoints['Mixed_6a'] = net
        net = slim.repeat(net, 20, block17, scale=0.10)

        # Auxillary tower
        with tf.variable_scope('AuxLogits'):
            aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID', scope='Conv2d_1a_3x3')
            aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
            print('AuxLogits/Conv2d_1b_1x1', aux.get_shape())

            #FIXME slice this layer off?
            aux = slim.conv2d(aux, 768, aux.get_shape()[1:3], padding='VALID', scope='Conv2d_2a_5x5')
            aux = slim.flatten(aux)
            print('AuxLogits/Conv2d_2a_5x5', aux.get_shape())

            # Here to end of AuxLogits scope added for SDC regression task
            if version <= 2:
                aux = slim.fully_connected(aux, 768, scope='Fc1')
            aux = slim.dropout(aux, dropout_keep_prob, is_training=bayesian or is_training, scope='Dropout')

            # regression aux outputs
            aux_output = {}
            if 'xyz' in output_cfg:
                aux_output['xyz'] = slim.fully_connected(
                    aux, output_cfg['xyz'], activation_fn=None, scope='OutputXYZ')
            if 'steer' in output_cfg:
                aux_output['steer'] = slim.fully_connected(
                    aux, output_cfg['steer'], activation_fn=None, scope='OutputSteer')
            endpoints['AuxOutput'] = aux_output

        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_3'):
                tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat(3, [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool])

        endpoints['Mixed_7a'] = net

        net = slim.repeat(net, 9, block8, scale=0.20)
        net = block8(net, activation_fn=None)
        print('block8', net.get_shape())

        net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
        # For bayesian network
        if bayesian:
            net = slim.dropout(net, dropout_keep_prob, is_training=True, scope='Dropout')
        endpoints['Conv2d_7b_1x1'] = net
        print('Conv2d_7b_1x1', net.get_shape())

        # Outputs scope is added for SDC regression task
        with tf.variable_scope('Output'):
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_8x8')
            print('AvgPool_8x8', net.get_shape())

            if version == 3:
                net = slim.conv2d(net, 1536, net.get_shape()[1:3], activation_fn=tf.nn.elu, scope='Fc1')
                print('Fc1', net.get_shape())
                net = slim.dropout(net, dropout_keep_prob, is_training=bayesian or is_training, scope='Dropout')
                net = slim.conv2d(net, 768, 1, activation_fn=tf.nn.elu, scope='Fc2')
                net = tf.squeeze(net, squeeze_dims=[1, 2])
                print('Fc2', net.get_shape())
            elif version == 2:
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1536, scope='Fc1')
                net = slim.dropout(net, dropout_keep_prob, is_training=bayesian or is_training, scope='Dropout')
                net = slim.fully_connected(net, 768, scope='Fc2')
            else:
                net = slim.flatten(net)
                net = slim.fully_connected(net, 2048, scope='Fc1')
                net = slim.dropout(net, dropout_keep_prob, is_training=bayesian or is_training, scope='Dropout')

            output = {}
            if 'xyz' in output_cfg:
                output['xyz'] = slim.fully_connected(
                    net, output_cfg['xyz'], activation_fn=None, scope='OutputXYZ')
            if 'steer' in output_cfg:
                output['steer'] = slim.fully_connected(
                    net, output_cfg['steer'], activation_fn=None, scope='OutputSteer')

            endpoints['Output'] = output

    return output, endpoints


build_inception_resnet_sdc_regression.default_image_size = 299


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
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
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
            return scope

