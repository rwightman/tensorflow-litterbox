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
from fabric import util
from fabric import exec_eval
from feeds import FeedImagesWithLabels
from processors import ProcessorImagenet
from models import ModelMySlim, ModelGoogleSlim
from imagenet_data import ImagenetData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'subset', 'validation',
    """Either 'validation', 'train', 'test'""")

tf.app.flags.DEFINE_boolean(
    'my', False,
    """Enable my variants of the image classification models""")

tf.app.flags.DEFINE_string(
    'network', 'resnet_v1_50',
    """See models/google/nets/nets_factory.py or models/my_slim/nets_factory.py""")

tf.app.flags.DEFINE_integer(
    'label_offset', 0,
    """Offset of labels in dataset. Set to 1 if network trained without background but dataset includes it.""")


def main(_):
    util.check_tensorflow_version()

    dataset = ImagenetData(subset=FLAGS.subset)

    processor = ProcessorImagenet()
    processor.label_offset = FLAGS.label_offset

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

    exec_eval.evaluate(feed=feed, model=model)

if __name__ == '__main__':
    tf.app.run()
