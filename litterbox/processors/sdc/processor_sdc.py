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
from processors.sdc.parse_proto_sdc import *
from processors.sdc.image_processing_sdc import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_float('image_aspect', 0.0, """Aspect ratio based sizing, square image_size*image_size if 0""")


class ProcessorSdc(fabric.Processor):

    def __init__(self):
        super(ProcessorSdc, self).__init__()

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

    def parse_example(self, serialized_example):
        parsed = parse_proto_sdc(serialized_example)
        #image_buffer, camera_id, image_timestamp, steer_angle, gps
        return parsed

    def process_data(self, data, train=False, thread_id=0):
        image, camera_id, image_timestamp, steering_angle, gps_coord = data
        processed_image = image_preprocess_sdc(
            image, camera_id, height=self.height, width=self.width, train=train, thread_id=thread_id)
        return processed_image, image_timestamp, steering_angle, gps_coord

    def reshape_batch(self, batch_data, batch_size, num_splits=0):
        images, timestamps, steering_angles, gps_coords = batch_data

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        timestamps = tf.reshape(timestamps, [batch_size])
        steering_angles = tf.reshape(steering_angles, [batch_size])
        gps_coords = tf.reshape(gps_coords, [batch_size, 2])

        if num_splits > 0:
            images = tf.split(0, num_splits, images)
            timestamps = tf.split(0, num_splits, timestamps)
            steering_angles = tf.split(0, num_splits, steering_angles)
            gps_coords = tf.split(0, num_splits, gps_coords)

        return images, timestamps, steering_angles, gps_coords

    def map_inputs(self, tensor_list, split_index=0):
        images, timestamps, steering_angles, gps_coords = tensor_list
        targets = [steering_angles[split_index], gps_coords[split_index]]
        return images[split_index], timestamps[split_index], targets
