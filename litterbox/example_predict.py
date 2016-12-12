# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" Predict classes for test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import os

import tensorflow as tf
from fabric import util, exec_predict
from fabric.dataset_file import DatasetFile
from feeds import FeedImagesWithLabels
from processors import ProcessorImagenet
from models import ModelMySlim, ModelGoogleSlim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean(
    'output_prob', False,
    """Set true to output per-class softmax output probabilities instead of class id.""")

tf.app.flags.DEFINE_integer(
    'num_classes', 1001,
    """Number of class labels""")

tf.app.flags.DEFINE_integer(
    'output_offset', 0,
    """Offset of output prediction. Set to 1 if network trained with background and you want output without.""")

tf.app.flags.DEFINE_boolean(
    'my', False,
    """Enable my variants of the image classification models""")

tf.app.flags.DEFINE_string(
    'network', 'resnet_v1_50',
    """See models/google/nets/nets_factory.py or models/my_slim/nets_factory.py""")


class ExampleData(DatasetFile):
    # Example dataset for feeding folder of images into model

    def __init__(self, subset):
        super(ExampleData, self).__init__('Example', subset)

    def num_classes(self):
        return FLAGS.num_classes


def main(_):
    util.check_tensorflow_version()

    dataset = ExampleData(subset='')

    processor = ProcessorImagenet()
    processor.output_offset = FLAGS.output_offset

    feed = FeedImagesWithLabels(dataset=dataset, processor=processor)

    model_params = {
        'num_classes': feed.num_classes_for_network(),
        'network': FLAGS.network,
    }
    if FLAGS.my:
        # My variants of Resnet, Inception, and VGG networks
        model = ModelMySlim(params=model_params)
    else:
        # Google's tf.slim models
        model = ModelGoogleSlim(params=model_params)
        model.check_norm(processor.normalize)

    output, num_entries = exec_predict.predict(feed, model)

    output_columns = ['Img']
    if FLAGS.output_prob:
        # Dump class probabilities to CSV file.
        class_labels = []
        for c in range(dataset.num_classes()):
            class_labels.append("c%s" % c)
        output_columns += class_labels
        output = np.vstack([np.column_stack([o[1], o[0]]) for o in output])
    else:
        # Dump class index to CSV file
        output_columns += ['Class']
        output = np.vstack([np.column_stack([o[1], np.argmax(o[0], axis=1)]) for o in output])

    df = pd.DataFrame(output, columns=output_columns)
    df.Img = df.Img.apply(lambda x: os.path.basename(x.decode()))
    df.to_csv('./output.csv', index=False)

if __name__ == '__main__':
    tf.app.run()