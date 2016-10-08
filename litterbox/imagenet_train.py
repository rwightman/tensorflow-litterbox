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

from fabric.train import *
from imagenet_data import *
from models import ModelInception

FLAGS = tf.app.flags.FLAGS


def main(_):
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    model = ModelInception(variant=ModelInception.Variant.ResnetV2)
    #model = ModelResnet(num_layers=50, width_factor=1)
    #model = ModelVgg(19)

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train(dataset, model)

if __name__ == '__main__':
    tf.app.run()
