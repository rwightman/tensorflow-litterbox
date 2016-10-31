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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_float('image_aspect', 0.0, """Aspect ratio based sizing, square image_size*image_size if 0""")
tf.app.flags.DEFINE_string('image_fmt', 'default',
                            """Either 'default' RGB [-1,1] or 'caffe' BGR [0,255]""")


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
        self.caffe_fmt = True if FLAGS.image_fmt == 'caffe' else False

    def parse_example(self, serialized_example):
        parsed = parse_proto_imagenet(serialized_example)
        # image_buffer, label_index, bbox, _, name
        return parsed

    def process_data(self, data, train=False, thread_id=0):
        processed = image_preprocess_imagenet(
            data, height=self.height, width=self.width, train=train, thread_id=thread_id)
        return processed

    def reshape_batch(self, batch_data, batch_size):
        images, labels, names = batch_data
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])

        return images, tf.reshape(labels, [batch_size]), tf.reshape(names, [batch_size])
