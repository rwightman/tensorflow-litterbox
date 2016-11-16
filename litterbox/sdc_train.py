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


class SdcData(DatasetRecord):
    """StateFarm data set."""

    def __init__(self):
        super(SdcData, self).__init__('sdc', FLAGS.subset)

    def num_classes(self):
        return 0

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        if self.subset == 'train':
            return 431627 #319814 #124200  #964809
        elif self.subset == 'validation':
            return 16709 #43134  #57557 # 39000


def main(_):
    util.check_tensorflow_version()

    feed = FeedImagesWithLabels(dataset=SdcData(), processor=ProcessorSdc())

    model_params = {
        'outputs': {'steer': 1},

        #'network': 'resnet_v1_152',

        #'network': 'nvidia_sdc',
        #'version': 2,
        #'regression_loss': 'mse',

        'network': 'resnet_v1_50',
        'version': 3,
        'regression_loss': 'mse',
    }
    model = ModelSdc(params=model_params)

    exec_train.train(feed, model)

if __name__ == '__main__':
    tf.app.run()
