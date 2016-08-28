#
"""A binary to evaluate Inception on SFarm data set.

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
