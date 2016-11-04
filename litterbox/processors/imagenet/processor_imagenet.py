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
        if FLAGS.image_aspect == 0.0:
            self.width = FLAGS.image_size
            self.height = FLAGS.image_size
        elif FLAGS.image_aspect > 1.0:
            self.width = math.ceil(FLAGS.image_size * FLAGS.image_aspect)
            self.height = FLAGS.image_size
        else:
            self.width = FLAGS.image_size
            self.height = math.ceil(FLAGS.image_size / FLAGS.image_aspect)

        self.depth = 3
        self.caffe_fmt = True if FLAGS.image_col == 'caffe' else False

    def parse_example(self, serialized_example):
        parsed = parse_proto_imagenet(serialized_example)
        # image_buffer, label_index, bbox, _, name
        return parsed

    def process_data(self, data, train=False, thread_id=0):
        image_buffer, label_index, bbox, _, name = data
        image_processed = image_preprocess_imagenet(
            image_buffer, height=self.height, width=self.width, train=train, thread_id=thread_id)
        return image_processed, label_index, name

    def reshape_batch(self, batch_data, batch_size, num_splits=0):
        images, labels, names = batch_data
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        labels = tf.reshape(labels, [batch_size])
        names = tf.reshape(names, [batch_size])

        if num_splits > 0:
            images = tf.split(0, num_splits, images)
            labels = tf.split(0, num_splits, labels)
            names = tf.split(0, num_splits, names)

        return images, labels, names

    def map_inputs(self, tensor_list, split_index=None):
        images, labels, names = tensor_list
        if split_index is None:
            return images, labels, names
        else:
            return images[split_index], labels[split_index], names[split_index]
