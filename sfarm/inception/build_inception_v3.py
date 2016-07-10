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

    endpoints = {}
    with tf.op_scope([inputs], scope, 'inception_v3'):
        with arg_scope(
                [layers.batch_norm, layers.dropout],
                is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d, layers.avg_pool2d],
                    stride=1, padding='VALID'):
                # 299 x 299 x 3
                endpoints['conv0'] = layers.conv2d(inputs, 32, [3, 3], stride=2, scope='conv0')
                # 149 x 149 x 32
                endpoints['conv1'] = layers.conv2d(endpoints['conv0'], 32, [3, 3], scope='conv1')
                # 147 x 147 x 32
                endpoints['conv2'] = layers.conv2d(endpoints['conv1'], 64, [3, 3], padding='SAME', scope='conv2')
                # 147 x 147 x 64
                endpoints['pool1'] = layers.max_pool2d(endpoints['conv2'], [3, 3], stride=2, scope='pool1')
                # 73 x 73 x 64
                endpoints['conv3'] = layers.conv2d(endpoints['pool1'], 80, [1, 1], scope='conv3')
                # 73 x 73 x 80.
                endpoints['conv4'] = layers.conv2d(endpoints['conv3'], 192, [3, 3], scope='conv4')
                # 71 x 71 x 192.
                endpoints['pool2'] = layers.max_pool2d(endpoints['conv4'], [3, 3], stride=2, scope='pool2')
                # 35 x 35 x 192.
                net = endpoints['pool2']
            # Inception blocks
            with arg_scope([layers.conv2d, layers.max_pool2d, layers.avg_pool2d], stride=1, padding='SAME'):
                # mixed: 35 x 35 x 256.
                with tf.variable_scope('mixed_35x35x256a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = layers.conv2d(net, 48, [1, 1])
                        branch5x5 = layers.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 64, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 32, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    endpoints['mixed_35x35x256a'] = net
                # mixed_1: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = layers.conv2d(net, 48, [1, 1])
                        branch5x5 = layers.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 64, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    endpoints['mixed_35x35x288a'] = net
                # mixed_2: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = layers.conv2d(net, 48, [1, 1])
                        branch5x5 = layers.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 64, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                    endpoints['mixed_35x35x288b'] = net
                # mixed_3: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = layers.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 64, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 96, [3, 3],
                                                  stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.max_pool2d(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
                    endpoints['mixed_17x17x768a'] = net
                # mixed4: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = layers.conv2d(net, 128, [1, 1])
                        branch7x7 = layers.conv2d(branch7x7, 128, [1, 7])
                        branch7x7 = layers.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = layers.conv2d(net, 128, [1, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 128, [1, 7])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    endpoints['mixed_17x17x768b'] = net
                # mixed_5: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768c'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = layers.conv2d(net, 160, [1, 1])
                        branch7x7 = layers.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = layers.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = layers.conv2d(net, 160, [1, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    endpoints['mixed_17x17x768c'] = net
                # mixed_6: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768d'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = layers.conv2d(net, 160, [1, 1])
                        branch7x7 = layers.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = layers.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = layers.conv2d(net, 160, [1, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    endpoints['mixed_17x17x768d'] = net
                # mixed_7: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768e'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = layers.conv2d(net, 192, [1, 1])
                        branch7x7 = layers.conv2d(branch7x7, 192, [1, 7])
                        branch7x7 = layers.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = layers.conv2d(net, 192, [1, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = layers.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                    endpoints['mixed_17x17x768e'] = net
                # Auxiliary Head logits
                aux_logits = tf.identity(endpoints['mixed_17x17x768e'])
                with tf.variable_scope('aux_logits'):
                    aux_logits = layers.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID')
                    aux_logits = layers.conv2d(aux_logits, 128, [1, 1], scope='proj')
                    # Shape of feature map before the final layer.
                    shape = aux_logits.get_shape()
                    #stddev=0.01,
                    aux_logits = layers.conv2d(aux_logits, 768, shape[1:3], padding='VALID')
                    aux_logits = layers.flatten(aux_logits)
                    #stddev=0.001
                    aux_logits = layers.fully_connected(aux_logits, num_classes, activation_fn=None)
                    print(tf.get_variable_scope().name)
                    endpoints['aux_logits'] = aux_logits
                # mixed_8: 8 x 8 x 1280.
                # Note that the scope below is not changed to not void previous
                # checkpoints.
                # (TODO) Fix the scope when appropriate.
                with tf.variable_scope('mixed_17x17x1280a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = layers.conv2d(net, 192, [1, 1])
                        branch3x3 = layers.conv2d(branch3x3, 320, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch7x7x3'):
                        branch7x7x3 = layers.conv2d(net, 192, [1, 1])
                        branch7x7x3 = layers.conv2d(branch7x7x3, 192, [1, 7])
                        branch7x7x3 = layers.conv2d(branch7x7x3, 192, [7, 1])
                        branch7x7x3 = layers.conv2d(branch7x7x3, 192, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.max_pool2d(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
                    endpoints['mixed_17x17x1280a'] = net
                # mixed_9: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = layers.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat(3, [layers.conv2d(branch3x3, 384, [1, 3]),
                                                  layers.conv2d(branch3x3, 384, [3, 1])])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 448, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat(3, [layers.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     layers.conv2d(branch3x3dbl, 384, [3, 1])])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                    endpoints['mixed_8x8x2048a'] = net
                # mixed_10: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = layers.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = layers.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat(3, [layers.conv2d(branch3x3, 384, [1, 3]),
                                                  layers.conv2d(branch3x3, 384, [3, 1])])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = layers.conv2d(net, 448, [1, 1])
                        branch3x3dbl = layers.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat(3, [layers.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     layers.conv2d(branch3x3dbl, 384, [3, 1])])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = layers.avg_pool2d(net, [3, 3])
                        branch_pool = layers.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                    endpoints['mixed_8x8x2048b'] = net
                # Final pooling and prediction
                with tf.variable_scope('logits'):
                    shape = net.get_shape()
                    net = layers.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
                    # 1 x 1 x 2048
                    net = layers.dropout(net, dropout_keep_prob, scope='dropout')
                    net = layers.flatten(net, scope='flatten')
                    # 2048
                    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
                    print(tf.get_variable_scope().name)

                    # 1000
                    endpoints['logits'] = logits
                    endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                    return logits, endpoints
