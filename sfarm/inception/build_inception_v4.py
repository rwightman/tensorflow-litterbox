"""The Inception v4 network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers


def block_stem(net):
    # Stem shared by inception-v4 and inception-resnet-v2 (resnet-v1 uses simpler stem below)
    # NOTE observe endpoints of first 3 layers
    endpoints = {}
    with tf.variable_scope('stem'):
        with tf.variable_scope('st0'):  # stage 0 of stem
            # 299 x 299 x 3
            net = layers.conv2d(net, 32, [3, 3], stride=2)
            endpoints['stem_conv0'] = net
            # 149 x 149 x 32
            net = layers.conv2d(net, 32, [3, 3])
            endpoints['stem_conv1'] = net
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
    # inception-v4: k=192, l=224, m=256, n=384
    # inception-resnet-v1: k=192, l=192, m=256, n=384
    # inception-resnet-v2: k=256, l=256, m=384, n=384
    # default padding = VALID
    # default stride = 1
    with tf.variable_scope(scope):
        with tf.variable_scope('br0_max'):
            br0 = layers.max_pool2d(net, [3, 3], stride=2)
            # 17 x 17 x input
        with tf.variable_scope('br1_3x3'):
            br1 = layers.conv2d(net, n, [3, 3], stride=2)
            # 17 x 17 x n
        with tf.variable_scope('br2_1x1_3x3dbl'):
            br2 = layers.conv2d(net, k, [1, 1], padding='SAME')
            br2 = layers.conv2d(br2, l, [3, 3], padding='SAME')
            br2 = layers.conv2d(br2, m, [3, 3], stride=2)
            # 17 x 17 x m
        net = tf.concat(3, [br0, br1, br2])
        # 17 x 17 x input + n + m
        # 1024 for v4 (384 + 384 + 256)
        # 896 for res-v1 (256 + 384 +256)
        # 1152 for res-v2 (384 + 384 + 384)
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


def block_stem_res(net):
    # Simpler stem for inception-resnet-v1 network
    # NOTE observe endpoints of first 3 layers
    # default padding = VALID
    # default stride = 1
    endpoints = {}
    with tf.variable_scope('stem'):
        # 299 x 299 x 3
        net = layers.conv2d(net, 32, [3, 3], stride=2)
        endpoints['stem_conv0'] = net
        # 149 x 149 x 32
        net = layers.conv2d(net, 32, [3, 3])
        endpoints['stem_conv1'] = net
        # 147 x 147 x 32
        net = layers.conv2d(net, 64, [3, 3], padding='SAME')
        endpoints['stem_conv2'] = net
        # 147 x 147 x 64
        net = layers.max_pool2d(net, [3, 3], stride=2)
        # 73 x 73 x 64
        net = layers.conv2d(net, 80, [1, 1], padding='SAME')
        # 73 x 73 x 80
        net = layers.conv2d(net, 192, [3, 3])
        # 71 x 71 x 192
        net = layers.conv2d(net, 256, [3, 3], stride=2)
        # 35 x 35 x 256
        endpoints['stem'] = net
    return net, endpoints


def block_res_a(net, ver=2, res_scale=None, scope='block_res_a', activation_fn=tf.nn.relu):
    # 35x35 grid

    # configure branch filter numbers
    br2_num = 32
    if ver == 1:
        br2_inc = 0
    else:
        br2_inc = 16

    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            shortcut = tf.identity(net)
            if res_scale:
                shortcut = tf.mul(shortcut, res_scale)  # scale residual
            with tf.variable_scope('br0_1x1'):
                br0 = layers.conv2d(net, 32, [1, 1])
            with tf.variable_scope('br1_1x1_3x3'):
                br1 = layers.conv2d(net, 32, [1, 1])
                br1 = layers.conv2d(br1, 32, [3, 3])
            with tf.variable_scope('br1_1x1_3x3dbl'):
                br2 = layers.conv2d(net, br2_num, [1, 1])
                br2 = layers.conv2d(br2, br2_num + 1*br2_inc, [3, 3])
                br2 = layers.conv2d(br2, br2_num + 2*br2_inc, [3, 3])
            net = tf.concat(3, [br0, br1, br2])
            net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None)
            net = activation_fn(tf.add(shortcut, net))
            # 35 x 35 x 256 res-v1, 384 res-v2
    return net


def block_res_b(net, ver=2, res_scale=None, scope='block_res_b', activation_fn=tf.nn.relu):
    # 17 x 17 grid

    # configure branch filter numbers
    if ver == 1:
        br0_num = 128
        br1_num = 128
        br1_inc = 0
    else:
        br0_num = 192
        br1_num = 128
        br1_inc = 32

    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            shortcut = tf.identity(net)
            if res_scale:
                shortcut = tf.mul(shortcut, res_scale)  # scale residual
            with tf.variable_scope('br0_1x1'):
                br0 = layers.conv2d(net, br0_num, [1, 1])
            with tf.variable_scope('br1_1x1_1x7_7x1'):
                br1 = layers.conv2d(net, br1_num, [1, 1])
                br1 = layers.conv2d(br1, br1_num + 1*br1_inc, [1, 7])
                br1 = layers.conv2d(br1, br1_num + 2*br1_inc, [7, 1])
            net = tf.concat(3, [br0, br1])
            net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None)
            # 17 x 17 x 896 res-v1, 1152 res-v2. Typo in paper, 1152, not 1154
            net = activation_fn(tf.add(shortcut, net))
        return net


def block_res_reduce_b(net, ver=2, scope='block_res_reduce_b'):
    # 17 x 17 -> 8 x 8 reduce

    # configure branch filter numbers
    br2_num = 256
    br3_num = 256
    if ver == 1:
        br2_inc = 0
        br3_inc = 0
    else:
        br2_inc = 32
        br3_inc = 32

    with tf.variable_scope(scope):
        with tf.variable_scope('br0_max'):
            br0 = layers.max_pool2d(net, [3, 3], stride=2)
        with tf.variable_scope('br1_1x1_3x3'):
            br1 = layers.conv2d(net, 256, [1, 1], padding='SAME')
            br1 = layers.conv2d(br1, 384, [3, 3], stride=2)
        with tf.variable_scope('br2_1x1_3x3'):
            br2 = layers.conv2d(net, br2_num, [1, 1], padding='SAME')
            br2 = layers.conv2d(br2, br2_num + br2_inc, [3, 3], stride=2)
        with tf.variable_scope('br3_1x1_3x3dbl'):
            br3 = layers.conv2d(net, br3_num, [1, 1], padding='SAME')
            br3 = layers.conv2d(br3, br3_num + 1*br3_inc, [3, 3], padding='SAME')
            br3 = layers.conv2d(br3, br3_num + 2*br3_inc, [3, 3], stride=2)
        net = tf.concat(3, [br0, br1, br2, br3])
        # 8 x 8 x 1792 v1, 2144 v2 (paper indicates 2048 but only get this if we use a v1 config for this block)
    return net


def block_res_c(net, ver=2, res_scale=None, scope='block_res_c', activation_fn=tf.nn.relu):
    # 8 x 8 grid

    # configure branch filter numbers
    br1_num = 192
    if ver == 1:
        br1_inc = 0
    else:
        br1_inc = 32

    # default padding = SAME
    # default stride = 1
    with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
        with tf.variable_scope(scope):
            shortcut = tf.identity(net)
            if res_scale:
                shortcut = tf.mul(shortcut, res_scale)  # scale residual
            with tf.variable_scope('br0_1x1'):
                br0 = layers.conv2d(net, 192, [1, 1])
            with tf.variable_scope('br1_1x1_1x3_3x1'):
                br1 = layers.conv2d(net, br1_num, [1, 1])
                br1 = layers.conv2d(br1, br1_num + 1*br1_inc, [1, 3])
                br1 = layers.conv2d(br1, br1_num + 2*br1_inc, [3, 1])
            net = tf.concat(3, [br0, br1])
            net = layers.conv2d(net, shortcut.get_shape()[-1], [1, 1], activation_fn=None)
            # 1792 res-1, 2144 (2048?) res-2
            net = activation_fn(tf.add(shortcut, net))
    return net


def block_output(net, num_classes, dropout_keep_prob=0.5, scope='output'):
    with tf.variable_scope(scope):
        # 8 x 8 x 1536
        shape = net.get_shape()
        net = layers.avg_pool2d(net, shape[1:3])
        # 1 x 1 x 1536
        net = layers.dropout(net, dropout_keep_prob)
        net = layers.flatten(net)
        #FIXME track global average pool in endpoints?
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

                for x in range(4):
                    block_scope = 'block_a%d' % x
                    net = block_a(net, scope=block_scope)
                    endpoints[block_scope] = net
                # 35 x 35 x 384

                net = block_reduce_a(net)
                endpoints['block_reduce_a'] = net
                # 17 x 17 x 1024

                for x in range(7):
                    block_scope = 'block_b%d' % x
                    net = block_b(net, scope=block_scope)
                    endpoints[block_scope] = net
                # 17 x 17 x 1024

                net = block_reduce_b(net)
                endpoints['block_reduce_b'] = net
                # 8 x 8 x 1536

                for x in range(3):
                    block_scope = 'block_c%d' % x
                    net = block_c(net, scope=block_scope)
                    endpoints[block_scope] = net
                # 8 x 8 x 1536

                logits = block_output(net, num_classes, dropout_keep_prob, 'output')
                # num_classes
                endpoints['logits'] = logits
                endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, endpoints


def build_inception_resnet(
        inputs,
        ver=2,
        res_scale=None,
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
    network_name = 'inception_resnet_v%d' % ver
    print("Building %s" % network_name)
    endpoints = {}
    with tf.op_scope([inputs], scope, network_name):
        with arg_scope([layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d],
                           stride=1, padding='VALID'):

                net, stem_endpoints = block_stem_res(inputs) if ver == 1 else block_stem(inputs)
                endpoints.update(stem_endpoints)
                print('Stem output shape: ', net.get_shape())

                for x in range(5):
                    block_scope = 'block_res_a%d' % x
                    net = block_res_a(net, ver=ver, res_scale=res_scale, scope=block_scope)
                    endpoints[block_scope] = net
                print('Block A output shape: ', net.get_shape())
                # 35 x 35 x 384

                k, l, m, n = (192, 192, 256, 384) if ver == 1 else (256, 256, 384, 384)
                net = block_reduce_a(net, k=k, l=l, m=m, n=n)
                endpoints['block_reduce_a'] = net
                print('Block Reduce A output shape : ', net.get_shape())
                # 17 x 17 x 896 v1, 1152 v2

                for x in range(10):
                    block_scope = 'block_res_b%d' % x
                    net = block_res_b(net, ver=ver, res_scale=res_scale, scope=block_scope)
                    endpoints[block_scope] = net
                print('Block B output shape: ', net.get_shape())
                # 17 x 17 x 896 v1, 1152 v2

                net = block_res_reduce_b(net, ver=ver)
                endpoints['block_res_reduce_b'] = net
                print('Block Reduce B output shape: ', net.get_shape())
                # 8 x 8 x 1792 v1, 2144 v2

                for x in range(5):
                    block_scope = 'block_res_c%d' % x
                    net = block_res_c(net, ver=ver, res_scale=res_scale, scope=block_scope)
                    endpoints[block_scope] = net
                print('Block C output shape: ', net.get_shape())
                # 8 x 8 x 1792 v1, 2144 v2

                logits = block_output(net, num_classes, dropout_keep_prob, 'output')
                # num_classes
                endpoints['logits'] = logits
                endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, endpoints
