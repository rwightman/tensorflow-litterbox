import tensorflow as tf

from fabric import model
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib import losses

from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS

#@layers.add_arg_scope
def block(net, num_filters_internal, block_stride, bottleneck=True, scope='block', activation_fn=tf.nn.relu):

    # If bottleneck, num_filters_internal*4 filters are output.
    # num_filters_internal is how many filters the 3x3 convolutions output.
    m = 4 if bottleneck else 1
    num_filters_in = net.get_shape()[-1]
    num_filters_out = m * num_filters_internal

    with tf.variable_scope(scope):
        shortcut = tf.identity(net)

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


def stack(net, num_blocks, num_filters_internal, stack_stride=1, bottleneck=True, scope='stack'):
    with tf.variable_scope(scope):
        for n in range(num_blocks):
            block_stride = stack_stride if n == 0 else 1
            block_scope = 'block%d' % (n + 1)
            net = block(net, num_filters_internal, block_stride, bottleneck=bottleneck, scope=block_scope)
    return net


def output(net, num_classes):
    #FIXME temporary hack for model checkpoint compatibility
    if True:
        net = layers.avg_pool2d(net, [7, 7], scope='avg_pool')
        net = layers.flatten(net, scope='flatten')
        with tf.variable_scope('logits'):
            net = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
            # num_classes
    else:
        with tf.variable_scope('output'):
            net = layers.avg_pool2d(net, [7, 7])
            net = layers.flatten(net)
            #FIXME monitor global average pool in endpoints?
            net = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
            # num_classes
    return net


def build_resnet(
        inputs,
        dropout_keep_prob=0.5,
        num_classes=1000,
        num_blocks=[3, 4, 6, 3],
        is_training=True,
        bottleneck=True,
        scope=''):
    """Blah"""

    endpoints = {}  # A dictionary of endpoints to observe (activations, extra stats, etc)
    with tf.op_scope([inputs], scope, 'resnet'):
        with arg_scope([layers.batch_norm, layers.dropout], is_training=is_training):
            with arg_scope(
                    [layers.conv2d, layers.max_pool2d, layers.avg_pool2d],
                    stride=1,
                    padding='SAME'):

                with tf.variable_scope('scale1'):
                    net = layers.conv2d(inputs, 64, [7, 7], stride=2)
                    net = layers.max_pool2d(net, [3, 3], stride=2)
                endpoints['scale1'] = net

                net = stack(net, num_blocks[0], 64, stack_stride=1, bottleneck=bottleneck, scope='scale2')
                endpoints['scale2'] = net

                net = stack(net, num_blocks[1], 128, stack_stride=2, bottleneck=bottleneck, scope='scale3')
                endpoints['scale3'] = net

                net = stack(net, num_blocks[2], 256, stack_stride=2, bottleneck=bottleneck, scope='scale4')
                endpoints['scale4'] = net

                net = stack(net, num_blocks[3], 512, stack_stride=2, bottleneck=bottleneck, scope='scale5')
                endpoints['scale5'] = net

                logits = output(net, num_classes)
                endpoints['logits'] = logits
                endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, endpoints


class ModelResnet(model.Model):

    # The decay to use for the moving average.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self):
        super(ModelResnet, self).__init__()

    def build_tower(self, inputs, num_classes, num_layers=34, is_training=False, scope=None):

        # layer configs
        if num_layers == 18:
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
        elif num_layers == 151:
            num_blocks = [3, 8, 36, 3]
            bottleneck = True
            # filter output depth 2048
        else:
            assert False, "invalid number of layers"

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
                    scope=scope)

        self.add_tower(
            name=scope,
            endpoints=endpoints,
            logits=logits
        )

        # Add summaries for viewing model statistics on TensorBoard.
        self.activation_summaries()

        return logits

    def add_tower_loss(self, labels, batch_size=None, scope=None):
        """Adds all losses for the model.

        Note the final loss is not returned. Instead, the list of losses are collected
        by slim.losses. The losses are accumulated in tower_loss() and summed to
        calculate the total loss.

        Args:
          logits: List of logits from inference(). Each entry is a 2-D float Tensor.
          labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
          batch_size: integer
          scope: tower scope of losses to add, ie 'tower_0/', defaults to last added tower if None
        """
        if not batch_size:
            batch_size = FLAGS.batch_size

        tower = self.tower(scope)

        # Reshape the labels into a dense Tensor of
        # shape [FLAGS.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = tower.logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

        # Cross entropy loss for the main softmax prediction.
        losses.softmax_cross_entropy(tower.logits,
                                     dense_labels,
                                     label_smoothing=0.1,
                                     weight=1.0)

    def logit_scopes(self):
        return ['outputs/logits']

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
