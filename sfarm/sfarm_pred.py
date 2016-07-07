#
"""
    Predict classes for test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import ModelInceptionV3
from resnet import ModelResnet
from vgg import ModelVgg16
from fabric.predict import *
from sfarm_data import *

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    dataset = StateFarmDataFile(subset='')
    model = ModelVgg16()
    assert dataset.data_files()
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    predict(dataset, model)


if __name__ == '__main__':
    tf.app.run()
