
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn


@add_arg_scope
def preact_conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        padding='SAME',
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    """Adds a 2D convolution preceded by batch normalization and activation.
    """
    with variable_scope.variable_op_scope([inputs], scope, 'Conv', reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        if normalizer_fn:
            normalizer_params = normalizer_params or {}
            inputs = normalizer_fn(inputs, activation_fn=activation_fn, **normalizer_params)
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        weights_shape = [kernel_h, kernel_w, num_filters_in, num_outputs]
        weights_collections = utils.get_variable_collections(variables_collections, 'weights')
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        outputs = nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=padding)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)