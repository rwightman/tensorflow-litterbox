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
import fabric
import math
from .parse_proto_imagenet import parse_proto_imagenet
from .image_processing_imagenet import image_preprocess_imagenet
from fabric.image_processing_common import *  # FIXME for annoying flags

class ProcessorImagenet(fabric.Processor):

    def __init__(self):
        super(ProcessorImagenet, self).__init__()

        # For aspect based image size, short edge set to FLAGS.image_size
        if FLAGS.image_aspect == 0.0 or FLAGS.image_aspect == 1.0:
            self.width = FLAGS.image_size
            self.height = FLAGS.image_size
        elif FLAGS.image_aspect < 1.0:
            self.width = math.floor(FLAGS.image_size * FLAGS.image_aspect)
            self.height = FLAGS.image_size
        else:
            self.width = FLAGS.image_size
            self.height = math.floor(FLAGS.image_size / FLAGS.image_aspect)

        self.depth = 3
        self.caffe_fmt = True if FLAGS.image_col == 'caffe' else False

    def parse_example(self, serialized_example):
        parsed = parse_proto_imagenet(serialized_example)
        # image_buffer, bbox, file name, class name, class label
        return parsed

    def process_example(self, data, mode='eval', thread_id=0):
        train = (mode == 'train')
        image_buffer, bbox, name, _, label_index = data
        image_processed = image_preprocess_imagenet(
            image_buffer, height=self.height, width=self.width,
            bbox=bbox, train=train, thread_id=thread_id)
        return image_processed, name, label_index

    def reshape_batch(self, batch_data, batch_size, num_splits=0):
        images, names, labels = batch_data
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        names = tf.reshape(names, [batch_size])
        labels = tf.reshape(labels, [batch_size])

        if num_splits > 0:
            images = tf.split(0, num_splits, images)
            names = tf.split(0, num_splits, names)
            labels = tf.split(0, num_splits, labels)

        return images, names, labels
