import tensorflow as tf

from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses

from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS

#@layers.add_arg_scope
def block(net, num_filters_internal, block_stride, bottleneck=True, activation_fn=tf.nn.relu):

    # If bottleneck, num_filters_internal*4 filters are output.
    # num_filters_internal is how many filters the 3x3 convolutions output.
    m = 4 if bottleneck else 1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = m * num_filters_internal

    shortcut = net  # identity

    if bottleneck:
        net = layers.conv2d(net, num_filters_internal, [1, 1], stride=block_stride, scope='a')
        net = layers.conv2d(net, num_filters_internal, [3, 3], scope='b')
        net = layers.conv2d(net, num_filters_out, [1, 1], activation_fn=None, scope='c')
    else:
        net = layers.conv2d(net, num_filters_internal, [3, 3], stride=block_stride, scope='A')
        net = layers.conv2d(net, num_filters_out, [3, 3], activation_fn=None, scope='B')

    if num_filters_out != num_filters_in or block_stride != 1:
        shortcut = layers.conv2d(
            shortcut, num_filters_out, [1, 1],
            stride=block_stride, activation_fn=None, padding='VALID', scope='shortcut')

    return activation_fn(net + shortcut)


def stack(net, num_blocks, num_filters_internal, stack_stride=1, bottleneck=True):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            net = block(net, num_filters_internal, block_stride, bottleneck=bottleneck)
    return net


def build_resnet(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
        num_blocks=[3, 4, 6, 3],
        is_training=True,
        bottleneck=True,
        restore_logits=True,
        scope=''):
    """Blah"""

    endpoints = {}
    with tf.op_scope([inputs], scope, 'resnet'):
        with arg_scope(
                [layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d, layers.avg_pool2d],
                    stride=1,
                    padding='SAME'):

                with tf.variable_scope('scale1'):
                    net = layers.conv2d(inputs, 64, [7, 7], stride=2)
                    net = layers.max_pool2d(net, [3, 3], stride=2)
                    endpoints['scale1'] = net
                    print('scale1', net.get_shape())

                with tf.variable_scope('scale2'):
                    net = stack(net, num_blocks[0], 64, stack_stride=1, bottleneck=bottleneck)
                    endpoints['scale2'] = net
                    print('scale2', net.get_shape())

                with tf.variable_scope('scale3'):
                    net = stack(net, num_blocks[1], 128, stack_stride=2, bottleneck=bottleneck)
                    endpoints['scale3'] = net
                    print('scale3', net.get_shape())

                with tf.variable_scope('scale4'):
                    net = stack(net, num_blocks[2], 256, stack_stride=2, bottleneck=bottleneck)
                    endpoints['scale4'] = net
                    print('scale4', net.get_shape())

                with tf.variable_scope('scale5'):
                    net = stack(net, num_blocks[3], 512, stack_stride=2, bottleneck=bottleneck)
                    endpoints['scale5'] = net
                    print('scale5', net.get_shape())

                #net = tf.reduce_mean(net, reduction_indices=[1, 2], name="avg_pool")
                net = layers.avg_pool2d(net, [7, 7], scope='avg_pool')
                net = layers.flatten(net, scope='flatten')
                print('avg_pool', net.get_shape())
                endpoints['avg_pool'] = net

                with tf.variable_scope('logits'):
                    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits') #restore=restore_logits
                    # 1 x 1 x num_classes
                    endpoints['logits'] = logits
                    endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                    return logits, endpoints


class ModelResnet(model.Model):

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self):
        super(ModelResnet, self).__init__()

    def build(self, inputs, num_classes, variant='34', is_training=False, restore_logits=True, scope=None):

        # [layer] config params
        # [18]  = {{2, 2, 2, 2}, 512, basic},
        # [34]  = {{3, 4, 6, 3}, 512, basic},
        # [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
        # [101] = {{3, 4, 23, 3}, 2048, bottleneck},
        # [152] = {{3, 8, 36, 3}, 2048, bottleneck},

        # 34 layer config
        bottleneck=False  # basic
        num_blocks = [3, 4, 6, 3]

        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.9997,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        l2_regularizer = layers.l2_regularizer(0.0004)

        with arg_scope(
                [layers.conv2d, layers.fully_connected],
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=l2_regularizer,
                activation_fn=tf.nn.relu):
            with arg_scope(
                    [layers.conv2d],
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_norm_params):

                logits, endpoints = build_resnet(
                    inputs,
                    num_classes=num_classes,
                    num_blocks=num_blocks,
                    bottleneck=bottleneck,
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

    def variables_to_restore(self):
        return tf.contrib.framework.variables.get_model_variables()

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
