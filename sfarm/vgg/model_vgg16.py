import tensorflow as tf

from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses

from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS


def build_vgg16(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
        is_training=True,
        restore_logits=True,
        scope=''):
    """Blah"""

    endpoints = {}
    with tf.op_scope([inputs], scope, 'vgg16'):
        with arg_scope(
                [layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d], 
                    stride=1,
                    padding='SAME'):
                # 224 x 224 x 3
                endpoints['conv1_1'] = layers.conv2d(inputs, 64, [3, 3], scope='conv1_1')
                # 224 x 224 x 64
                endpoints['conv1_2'] = layers.conv2d(endpoints['conv1_1'], 64, [3, 3], scope='conv1_2')
                endpoints['pool1'] = layers.max_pool2d(endpoints['conv1_2'], [2, 2], stride=2, scope='pool1')

                # 112 x 112 x 64
                endpoints['conv2_1'] = layers.conv2d(endpoints['pool1'], 128, [3, 3], scope='conv2_1')
                # 112 x 112 x 128
                endpoints['conv2_2'] = layers.conv2d(endpoints['conv2_1'], 128, [3, 3], scope='conv2_2')
                endpoints['pool2'] = layers.max_pool2d(endpoints['conv2_2'], [2, 2], stride=2, scope='pool2')

                # 56 x 56 x 128
                endpoints['conv3_1'] = layers.conv2d(endpoints['pool2'], 256, [3, 3], scope='conv3_1')
                # 56 x 56 x 256
                endpoints['conv3_2'] = layers.conv2d(endpoints['conv3_1'], 256, [3, 3], scope='conv3_2')
                endpoints['conv3_3'] = layers.conv2d(endpoints['conv3_2'], 256, [3, 3], scope='conv3_3')
                endpoints['pool3'] = layers.max_pool2d(endpoints['conv3_3'], [2, 2], stride=2, scope='pool3')

                # 28 x 28 x 256
                endpoints['conv4_1'] = layers.conv2d(endpoints['pool3'], 512, [3, 3], scope='conv4_1')
                # 28 x 28 x 512
                endpoints['conv4_2'] = layers.conv2d(endpoints['conv4_1'], 512, [3, 3], scope='conv4_2')
                endpoints['conv4_3'] = layers.conv2d(endpoints['conv4_2'], 512, [3, 3], scope='conv4_3')
                endpoints['pool4'] = layers.max_pool2d(endpoints['conv4_3'], [2, 2], stride=2, scope='pool4')

                # 14 x 14 x 512
                endpoints['conv5_1'] = layers.conv2d(endpoints['pool4'], 512, [3, 3], scope='conv5_1')
                # 14 x 14 x 512
                endpoints['conv5_2'] = layers.conv2d(endpoints['conv5_1'], 512, [3, 3], scope='conv5_2')
                endpoints['conv5_3'] = layers.conv2d(endpoints['conv5_2'], 512, [3, 3], scope='conv5_3')
                net = layers.max_pool2d(endpoints['conv5_3'], [2, 2], stride=2, scope='pool5')
                endpoints['pool5'] = net

                net = layers.flatten(net, scope='flatten')

                # 7 x 7 x 512
                net = layers.fully_connected(net, 4096, scope='fc6')
                net = layers.dropout(net, dropout_keep_prob, scope='fc6_dropout')
                endpoints['fc6'] = net

                # 1 x 1 x 4096
                net = layers.fully_connected(net, 4096, scope='fc7')
                net = layers.dropout(net, dropout_keep_prob, scope='fc7_dropout')
                endpoints['fc7'] = net

                with tf.variable_scope('logits'):
                    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits') #restore=restore_logits
                    # 1 x 1 x num_classes
                    endpoints['logits'] = logits
                    endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                    return logits, endpoints


class ModelVgg16(model.Model):

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self):
        super(ModelVgg16, self).__init__()

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, inputs, num_classes, is_training=False, restore_logits=True, scope=None):

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.9997,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        l2_regularizer = layers.l2_regularizer(0.0005) #0.00004

        with arg_scope(
                [layers.fully_connected],
                biases_initializer=tf.constant_initializer(0.1),
                weights_initializer=layers.variance_scaling_initializer(factor=1.0),
                weights_regularizer=l2_regularizer,
                activation_fn=tf.nn.relu
        ):
            with arg_scope(
                    [layers.conv2d],
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_norm_params,
                    weights_initializer=layers.variance_scaling_initializer(factor=1.0),
                    weights_regularizer=l2_regularizer,
                    activation_fn=tf.nn.relu
            ):

                logits, endpoints = build_vgg16(
                    inputs,
                    num_classes=num_classes,
                    is_training=is_training,
                    restore_logits=restore_logits,
                    scope=scope)

        self.add_instance(
            name=scope,
            endpoints=endpoints,
            logits=logits
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
        losses.softmax_cross_entropy(instance.logits,
                                     dense_labels,
                                     label_smoothing=0.1,
                                     weight=1.0)

    def get_variables_fn_list(self):
        return [tf.contrib.framework.variable]

    @staticmethod
    def loss_op(logits, labels):
        """Generate a simple (non tower based) loss op for use in evaluation.

        Args:
          logits: List of logits from inference(). Shape [batch_size, num_classes], dtype float32/64
          labels: Labels from distorted_inputs or inputs(). batch_size vector with int32/64 values in [0, num_classes).
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy_eval')

        loss = math_ops.reduce_mean(cross_entropy)
        return loss
