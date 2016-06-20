import tensorflow as tf
from tensorflow.contrib.slim import scopes
from tensorflow.contrib.slim import ops
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim import losses
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS

VGG_MEAN = [103.939, 116.779, 123.68]


def build_vgg_16(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
        is_training=True,
        restore_logits=True,
        scope=''):
    """Blah"""

    end_points = {}
    with tf.op_scope([inputs], scope, 'vgg16'):
        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout], is_training=is_training):
            with scopes.arg_scope([ops.conv2d, ops.max_pool], stride=1, padding='SAME'):
                # 224 x 224 x 3
                end_points['conv1_1'] = ops.conv2d(inputs, 64, [3, 3], scope='conv1_1')
                # 224 x 224 x 64
                end_points['conv1_2'] = ops.conv2d(end_points['conv1_1'], 64, [3, 3], scope='conv1_2')
                end_points['pool1'] = ops.max_pool(end_points['conv1_2'], [2, 2], stride=2, scope='pool1')

                # 112 x 112 x 64
                end_points['conv2_1'] = ops.conv2d(end_points['pool1'], 128, [3, 3], scope='conv2_1')
                # 112 x 112 x 128
                end_points['conv2_2'] = ops.conv2d(end_points['conv2_1'], 128, [3, 3], scope='conv2_2')
                end_points['pool2'] = ops.max_pool(end_points['conv2_2'], [2, 2], stride=2, scope='pool2')

                # 56 x 56 x 128
                end_points['conv3_1'] = ops.conv2d(end_points['pool2'], 256, [3, 3], scope='conv3_1')
                # 56 x 56 x 256
                end_points['conv3_2'] = ops.conv2d(end_points['conv3_1'], 256, [3, 3], scope='conv3_2')
                end_points['conv3_3'] = ops.conv2d(end_points['conv3_2'], 256, [3, 3], scope='conv3_3')
                end_points['pool3'] = ops.max_pool(end_points['conv3_3'], [2, 2], stride=2, scope='pool3')

                # 28 x 28 x 256
                end_points['conv4_1'] = ops.conv2d(end_points['pool3'], 512, [3, 3], scope='conv4_1')
                # 28 x 28 x 512
                end_points['conv4_2'] = ops.conv2d(end_points['conv4_1'], 512, [3, 3], scope='conv4_2')
                end_points['conv4_3'] = ops.conv2d(end_points['conv4_2'], 512, [3, 3], scope='conv4_3')
                end_points['pool4'] = ops.max_pool(end_points['conv4_3'], [2, 2], stride=2, scope='pool4')

                # 14 x 14 x 512
                end_points['conv5_1'] = ops.conv2d(end_points['pool4'], 512, [3, 3], scope='conv5_1')
                # 14 x 14 x 512
                end_points['conv5_2'] = ops.conv2d(end_points['conv5_1'], 512, [3, 3], scope='conv5_2')
                end_points['conv5_3'] = ops.conv2d(end_points['conv5_2'], 512, [3, 3], scope='conv5_3')
                end_points['pool5'] = ops.max_pool(end_points['conv5_3'], [2, 2], stride=2, scope='pool5')
                net = end_points['pool5']

                net = ops.flatten(net, scope='flatten')

                # 7 x 7 x 512
                end_points['fc6'] = ops.fc(net, 4096, scope='fc6')
                net = ops.dropout(net, dropout_keep_prob, scope='fc6_dropout')

                # 1 x 1 x 4096
                end_points['fc7'] = ops.fc(net, 4096, scope='fc7')
                net = ops.dropout(net, dropout_keep_prob, scope='fc7_dropout')

                logits = ops.fc(end_points['fc7'], 1000, scope='logits', restore=restore_logits)
                # 1 x 1 x 1000
                end_points['logits'] = logits
                end_points['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, end_points


class ModelVgg16(object):

    def __init__(self):
        self.endpoints = {}

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, inputs_rgb, train=False):
        rgb_scaled = inputs_rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        inputs_bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert inputs_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
            with scopes.arg_scope([ops.conv2d], stddev=0.1, activation=tf.nn.relu):
                logits, endpoints = build_vgg_16(inputs_bgr, is_training=train)

        self.logits = logits
        self.endpoints = endpoints

    def losses(self, labels, batch_size=None):
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

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = self.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.softmax_cross_entropy(self.logits,
                                     dense_labels,
                                     label_smoothing=0.1,
                                     weight=1.0)
