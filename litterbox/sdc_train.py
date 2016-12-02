# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from fabric import util
from fabric import exec_train
from fabric import DatasetRecord
from models import ModelSdc
from processors import ProcessorSdc
from feeds import FeedImagesWithLabels

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'validation', 'train', 'test'""")

tf.app.flags.DEFINE_string('root_network', 'resnet_v1_50',
                           """Either resnet_v1_50, resnet_v1_101, resnet_v1_152, inception_resnet_v2, nvidia_sdc""")

tf.app.flags.DEFINE_integer('top_version', 5,
                           """Top level network version, specifies output layer variations. See model code.""")

tf.app.flags.DEFINE_boolean('lock_root', False, 'Lock root convnet parameters')


class SdcData(DatasetRecord):
    """Self-driving car dataset."""

    def __init__(self):
        super(SdcData, self).__init__('sdc', FLAGS.subset)

    def num_classes(self):
        return 0

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        if self.subset == 'train':
            return 92643 #431627 #319814
        elif self.subset == 'validation':
            return 16709 #43134 #57557


def main(_):
    util.check_tensorflow_version()

    processor = ProcessorSdc()
    processor.standardize_input = 'frame'
    #processor.num_input_images = 2

    feed = FeedImagesWithLabels(dataset=SdcData(), processor=processor)

    model_params = {
        'outputs': {
            'steer': 1,
        #    'xyz': 2,
        },

        #'network': 'resnet_v1_50',
        #'version': 5,

        'network': FLAGS.root_network,
        'version': FLAGS.top_version,
        'bayesian': False,
        'lock_root': FLAGS.lock_root,
        'regression_loss': 'mse',
    }
    model = ModelSdc(params=model_params)

    exec_train.train(feed, model)

if __name__ == '__main__':
    tf.app.run()
