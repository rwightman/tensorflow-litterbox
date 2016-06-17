#
"""
    Predict classes for test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sfarm_data import StateFarmData
from fabric import DatasetFile
from fabric.predict import predict
from inception import ModelInceptionV3

FLAGS = tf.app.flags.FLAGS


class StateFarmDataFile(DatasetFile):
    """StateFarm data set."""

    def __init__(self, subset):
        super(StateFarmDataFile, self).__init__('SFarm', subset)

    def num_classes(self):
        return 10


def main(unused_argv=None):
    dataset = StateFarmDataFile(subset='')
    model = ModelInceptionV3()
    assert dataset.data_files()
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    predict(dataset, model)


if __name__ == '__main__':
    tf.app.run()
