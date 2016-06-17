#
"""A binary to evaluate Inception on SFarm data set.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sfarm_data import StateFarmData
from fabric.eval import evaluate
from inception import ModelInceptionV3

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    dataset = StateFarmData(subset=FLAGS.subset)
    model = ModelInceptionV3()
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate(dataset, model)


if __name__ == '__main__':
    tf.app.run()
