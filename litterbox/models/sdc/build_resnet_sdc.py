from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
  
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
  
    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.
  
    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
  
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
  
    Note that
  
       net = conv2d_same(inputs, num_outputs, 3, stride=stride)
  
    is equivalent to
  
       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)
  
    whereas
  
       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
  
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
  
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
  
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                           padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.
  
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
  
    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.
  
    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.
  
    Control of the output feature density is implemented by atrous convolution.
  
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the ResNet.
        For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      outputs_collections: Collection to add the ResNet block outputs.
  
    Returns:
      net: Output tensor with stride equal to the specified output_stride.
  
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1
    block_outputs = []

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit

                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=1,
                                            rate=rate)
                        rate *= unit_stride
                    else:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=unit_stride,
                                            rate=1)
                        current_stride *= unit_stride
                block_outputs.append(net)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net, block_outputs


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride,
                                   activation_fn=None, scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.
  
    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.
  
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
  
    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def _build_resnet_root(
        net,
        block_cfg,
        global_pool=True,
        output_stride=None,
        lock_root=False):
    """

    Args:
        net:
        block_cfg:
        global_pool:
        output_stride:
        lock_root:

    Returns:

    """
    root_args_trainable = slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=not lock_root)
    with root_args_trainable:
        if output_stride is not None:
            if output_stride % 4 != 0:
                raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
        net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net, block_outputs = stack_blocks_dense(net, block_cfg, output_stride)
        print('Blocks', net.get_shape())

        if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            print('Global pool', net.get_shape())

    return net, block_outputs


def _build_output(
        nets,
        output_cfg,
        version,
        is_training=False,
        bayesian=False,
        dropout_keep_prob=0.6):
    """

    Args:
        nets:
        output_cfg:
        version:
        is_training:
        bayesian:
        dropout_keep_prob:

    Returns:

    """
    assert len(nets) >= 1
    net = nets[0]

    with tf.variable_scope('Output'):
        if version == 1:
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048, scope='Fc1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout1')
            net = slim.fully_connected(net, 128, scope='Fc2')
        elif version == 2:
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048, activation_fn=tf.nn.elu, scope='Fc1')
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout1')
            net = slim.fully_connected(net, 512, activation_fn=tf.nn.elu, scope='Fc2')
        elif version == 3 or version == 4:
            # Version 4 used for mu-law + tanh experiment
            do_dropout = bayesian or is_training
            out_scope = slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu) if version == 3 else \
                slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu, normalizer_fn=None)
            with out_scope:
                net = slim.conv2d(net, 2048, net.get_shape()[1:3], padding='VALID', scope='Fc1')
                print('Fc1', net.get_shape())
                net = slim.dropout(net, dropout_keep_prob, is_training=do_dropout, scope='Dropout')
                net = slim.conv2d(net, 1024, 1, scope='Fc2')
                print('Fc2', net.get_shape())
                net = tf.squeeze(net, squeeze_dims=[1, 2])
        elif version == 5:
            # Version 5, add extra FC layer and support for siamese conv
            do_dropout = bayesian or is_training
            out_scope = slim.arg_scope(
                [slim.conv2d], activation_fn=tf.nn.elu, normalizer_fn=None)
            with out_scope:
                if len(nets) > 1:
                    net = tf.concat(3, nets)
                    print('Siamese concat', net.get_shape())
                net = slim.dropout(
                    net, min(1.0, dropout_keep_prob * 1.2), is_training=do_dropout, scope='Dropout1')
                net = slim.conv2d(net, 2048, net.get_shape()[1:3], padding='VALID', scope='Fc1')
                print('Fc1', net.get_shape())
                net = slim.dropout(net, dropout_keep_prob, is_training=do_dropout, scope='Dropout2')
                net = slim.conv2d(net, 1024, 1, scope='Fc2')
                print('Fc2', net.get_shape())
                net = slim.conv2d(net, 512, 1, scope='Fc3')
                print('Fc3', net.get_shape())
                net = tf.squeeze(net, squeeze_dims=[1, 2])
        elif version == 6:
            # Version 6 used for multi-resolution feature map experiment
            assert len(nets) == 2
            net2 = nets[1]
            do_dropout = bayesian or is_training
            out_scope = slim.arg_scope(
                [slim.conv2d], activation_fn=tf.nn.elu, normalizer_fn=None)
            with out_scope:
                net = slim.conv2d(net, 2048, net.get_shape()[1:3], padding='VALID', scope='Fc1a')
                net2 = slim.conv2d(net2, 1024, net.get_shape()[1:3], padding='VALID', scope='Fc1b')
                net = tf.concat(3, [net, net2])
                net = slim.dropout(
                    net, min(1.0, dropout_keep_prob * 1.2), is_training=do_dropout, scope='Dropout1')
                print('Fc1', net.get_shape())
                net = slim.dropout(net, dropout_keep_prob, is_training=do_dropout, scope='Dropout2')
                net = slim.conv2d(net, 1024, 1, scope='Fc2')
                print('Fc2', net.get_shape())
                net = slim.conv2d(net, 512, 1, scope='Fc3')
                print('Fc3', net.get_shape())
                net = tf.squeeze(net, squeeze_dims=[1, 2])

        print('Pre-output', net.get_shape())
        output = {}
        if 'xyz' in output_cfg:
            output['xyz'] = slim.fully_connected(
                net, output_cfg['xyz'], activation_fn=None, scope='OutputXYZ')
        if 'steer' in output_cfg:
            output['steer'] = slim.fully_connected(
                net, output_cfg['steer'],
                activation_fn=tf.nn.tanh if version == 4 else None,
                scope='OutputSteer')

        return output


def resnet_v1_sdc(
        inputs,
        blocks,
        output_cfg,
        version,
        dropout_keep_prob=0.6,
        bayesian=False,
        is_training=True,
        global_pool=True,
        output_stride=None,
        lock_root=False,
        reuse=None,
        scope=None):
    """Generator for v1 ResNet models.
  
    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a Block object describing the units in the block.
      is_training: whether is training or not.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      output: Dict of rank-4 tensors of size [batch, height_out, width_out, channels_out].
      endpoints: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        endpoints_collection = sc.name + '_end_points'
        arg_scope_ep = slim.arg_scope(
            [slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=endpoints_collection)
        arg_scope_train = slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training)
        with arg_scope_ep, arg_scope_train:
            nets = []
            siamese = True if len(inputs.get_shape()) == 5 else False
            if siamese:
                with tf.variable_scope(sc, values=[inputs], reuse=reuse) as scs:
                    # siamese, multi-image config
                    unpacked_inputs = tf.unpack(inputs, axis=1)
                    for i, x in enumerate(unpacked_inputs):
                        branch_scope = 'Branch_%d' % i
                        with tf.name_scope(branch_scope):
                            net, _ = _build_resnet_root(
                                x,
                                block_cfg=blocks,
                                global_pool=global_pool,
                                output_stride=output_stride,
                                lock_root=lock_root)
                        scs.reuse_variables()
                        nets.append(net)
            else:
                # normal config
                net, block_outputs = _build_resnet_root(
                    inputs,
                    block_cfg=blocks,
                    global_pool=global_pool,
                    output_stride=output_stride,
                    lock_root=lock_root)
                nets.append(net)

                if version == 6:
                    #  version 6 variant takes an additional global pool from earlier block before the last stride
                    net2 = tf.reduce_mean(block_outputs[11], [1, 2], name='pool5a', keep_dims=True)
                    print('Global pool 2', net2.get_shape())
                    nets.append(net2)

            output = _build_output(
                nets,
                output_cfg=output_cfg,
                version=version,
                is_training=is_training,
                bayesian=bayesian,
                dropout_keep_prob=dropout_keep_prob)

            endpoints = slim.utils.convert_collection_to_dict(endpoints_collection)

            return output, endpoints


resnet_v1_sdc.default_image_size = 224


def build_resnet_v1_50_sdc(
        inputs,
        output_cfg={},
        version=1,
        is_training=True,
        bayesian=False,
        global_pool=True,
        lock_root=False,
        output_stride=None,
        reuse=None,
        scope='resnet_v1_50'):
    
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1_sdc(
        inputs, blocks, output_cfg, version=version, is_training=is_training,
        bayesian=bayesian, global_pool=global_pool, output_stride=output_stride,
        lock_root=lock_root, reuse=reuse, scope=scope)


def build_resnet_v1_101_sdc(
        inputs,
        output_cfg={},
        version=1,
        is_training=True,
        bayesian=False,
        global_pool=True,
        lock_root=False,
        output_stride=None,
        reuse=None,
        scope='resnet_v1_101'):
    
    """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1_sdc(
        inputs, blocks, output_cfg, version=version, is_training=is_training,
        bayesian=bayesian, global_pool=global_pool, output_stride=output_stride,
        lock_root=lock_root, reuse=reuse, scope=scope)


def build_resnet_v1_152_sdc(
        inputs,
        output_cfg={},
        version=1,
        is_training=True,
        bayesian=False,
        global_pool=True,
        lock_root=False,
        output_stride=None,
        reuse=None,
        scope='resnet_v1_152'):
    
    """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1_sdc(
        inputs, blocks, output_cfg, version=version, is_training=is_training,
        bayesian=bayesian, global_pool=global_pool, output_stride=output_stride,
        lock_root=lock_root, reuse=reuse, scope=scope)
