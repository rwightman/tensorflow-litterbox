#
"""A binary to train Inception on the StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import ModelInceptionV3
from inception import ModelInceptionV4
from resnet import ModelResnet
from vgg import ModelVgg16
from fabric.train import *
from sfarm_data import *

FLAGS = tf.app.flags.FLAGS


def main(_):
    dataset = StateFarmData(subset=FLAGS.subset)
    assert dataset.data_files()
    model = ModelInceptionV4()

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train(dataset, model)

if __name__ == '__main__':
    tf.app.run()
