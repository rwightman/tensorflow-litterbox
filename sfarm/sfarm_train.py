#
"""A binary to train Inception on the StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sfarm_data import StateFarmData
from fabric.train import train
from inception import ModelInceptionV3

FLAGS = tf.app.flags.FLAGS

def main(_):
    dataset = StateFarmData(subset=FLAGS.subset)
    assert dataset.data_files()
    model = ModelInceptionV3()
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train(dataset, model)

if __name__ == '__main__':
    tf.app.run()
