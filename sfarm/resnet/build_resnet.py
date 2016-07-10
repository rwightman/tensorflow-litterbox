import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

FLAGS = tf.app.flags.FLAGS


#@layers.add_arg_scope
def block(net, num_filters_internal, block_stride, bottleneck=True, scope='block', activation_fn=tf.nn.relu):
    # default padding=SAME
    # default stride=1

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

        return activation_fn(tf.add(net, shortcut))


def stack(net, num_blocks, num_filters_internal, stack_stride=1, bottleneck=True, scope='stack'):
    with tf.variable_scope(scope):
        for n in range(num_blocks):
            block_stride = stack_stride if n == 0 else 1
            block_scope = 'block%d' % n
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
        k=1,  # width factor
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

                # 224 x 224
                with tf.variable_scope('stage1'):
                    net = layers.conv2d(inputs, 64 * k, [7, 7], stride=2)
                    net = layers.max_pool2d(net, [3, 3], stride=2)
                endpoints['stage1'] = net
                print("Stage 1 output size: ", net.get_shape())
                # 56 x 56

                net = stack(net, num_blocks[0], 64 * k, stack_stride=1, bottleneck=bottleneck, scope='stage2')
                endpoints['stage2'] = net
                print("Stage 2 output size: ", net.get_shape())
                # 56 x 56

                net = stack(net, num_blocks[1], 128 * k, stack_stride=2, bottleneck=bottleneck, scope='stage3')
                endpoints['stage3'] = net
                print("Stage 3 output size: ", net.get_shape())
                # 28 x 28

                net = stack(net, num_blocks[2], 256 * k, stack_stride=2, bottleneck=bottleneck, scope='stage4')
                endpoints['stage4'] = net
                print("Stage 4 output size: ", net.get_shape())
                # 14 x 14

                net = stack(net, num_blocks[3], 512 * k, stack_stride=2, bottleneck=bottleneck, scope='stage5')
                endpoints['stage5'] = net
                print("Stage 5 output size: ", net.get_shape())
                # 7 x 7

                logits = output(net, num_classes)
                endpoints['logits'] = logits
                endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

                return logits, endpoints
