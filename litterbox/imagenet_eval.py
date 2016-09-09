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

from inception import ModelInception
from resnet import ModelResnet
from fabric.eval import *
from imagenet_data import *

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    dataset = ImagenetData(subset=FLAGS.subset)
    model = ModelInception(variant=ModelInception.Variant.ResnetV2)

    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    evaluate(dataset, model)


if __name__ == '__main__':
    tf.app.run()
