#
"""A binary to evaluate Inception on SFarm data set.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sfarm_data import StateFarmData
from fabric import DatasetFile
from fabric.eval import evaluate
from inception import ModelInceptionV3
from vgg import ModelVgg16

FLAGS = tf.app.flags.FLAGS


class StateFarmDataFile(DatasetFile):
    """StateFarm data set."""

    def __init__(self, subset):
        super(StateFarmDataFile, self).__init__('SFarm', subset)

    def num_classes(self):
        return 10


def main(unused_argv=None):
    dataset = StateFarmDataFile(subset=FLAGS.subset)
    #model = ModelInceptionV3()
    model = ModelVgg16()
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate(dataset, model)


if __name__ == '__main__':
    tf.app.run()
