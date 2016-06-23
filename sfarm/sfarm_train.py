#
"""A binary to train Inception on the StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sfarm_data import StateFarmData
from fabric import DatasetFile
from fabric.train import train
from inception import ModelInceptionV3
from resnet import ModelResnet
from vgg import ModelVgg16

FLAGS = tf.app.flags.FLAGS


class StateFarmDataFile(DatasetFile):
    """StateFarm data set."""

    def __init__(self, subset):
        super(StateFarmDataFile, self).__init__('SFarm', subset)

    def num_classes(self):
        return 10

def main(_):
    dataset = StateFarmData(subset=FLAGS.subset)
    assert dataset.data_files()
    #model = ModelInceptionV3()
    model = ModelResnet()
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train(dataset, model)

if __name__ == '__main__':
    tf.app.run()
