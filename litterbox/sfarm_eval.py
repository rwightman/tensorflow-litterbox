# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""A binary to evaluate Inception on SFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import ModelInception
from resnet import ModelResnet
from vgg import ModelVgg16
from fabric.eval import *
from sfarm_data import *

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    dataset = StateFarmDataFile(subset=FLAGS.subset)
    #model = ModelInception(variant=ModelInception.Variant.ResnetV2)
    model = ModelResnet(num_layers=16, width_factor=2)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate(dataset, model)


if __name__ == '__main__':
    tf.app.run()
