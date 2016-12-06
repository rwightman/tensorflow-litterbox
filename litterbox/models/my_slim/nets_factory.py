from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
from fabric import model
from .build_inception_v4 import *
from .build_resnet import *
from .build_vgg import *

networks_map = {
    'vgg_16': build_vgg,
    'vgg_19': build_vgg,
    'inception_v4': build_inception,
    'inception_resnet_v1': build_inception,
    'inception_resnet_v2': build_inception,
    'resnet_v1_18': build_resnet,
    'resnet_v1_34': build_resnet,
    'resnet_v1_50': build_resnet,
    'resnet_v1_101': build_resnet,
    'resnet_v1_152': build_resnet,
    'resnet_v1_200': build_resnet,
    'resnet_v2_50': build_resnet,
    'resnet_v2_101': build_resnet,
    'resnet_v2_152': build_resnet,
    'resnet_v2_200': build_resnet,
}

params_map = {
    'vgg_16': params_vgg(num_layers=16),
    'vgg_19': params_vgg(num_layers=19),
    'inception_v4': params_inception(version=4, residual=False),
    'inception_resnet_v1': params_inception(version=1, residual=True),
    'inception_resnet_v2': params_inception(version=2, residual=True),
    'resnet_v1_18': params_resnet(num_layers=18),
    'resnet_v1_34': params_resnet(num_layers=34),
    'resnet_v1_50': params_resnet(num_layers=50),
    'resnet_v1_101': params_resnet(num_layers=101),
    'resnet_v1_152': params_resnet(num_layers=152),
    'resnet_v1_200': params_resnet(num_layers=200),
    'resnet_v2_50': params_resnet(num_layers=50, pre_activation=True),
    'resnet_v2_101': params_resnet(num_layers=101, pre_activation=True),
    'resnet_v2_152': params_resnet(num_layers=152, pre_activation=True),
    'resnet_v2_200': params_resnet(num_layers=200, pre_activation=True),
}


def get_network_fn(name, num_classes, params, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    params = model.merge_params(params_map[name], params)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(inputs):
        return func(inputs, num_classes, params, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
