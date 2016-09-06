# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" A binary to train on the StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import ModelInception
from resnet import ModelResnet
from vgg import ModelVgg16
from fabric.train import *
from sfarm_data import *

FLAGS = tf.app.flags.FLAGS


def main(_):
    dataset = StateFarmData(subset=FLAGS.subset)
    assert dataset.data_files()
    #model = ModelInception(variant=ModelInception.Variant.ResnetV2)
    model = ModelResnet(num_layers=18, width_factor=1)

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train(dataset, model)

if __name__ == '__main__':
    tf.app.run()
