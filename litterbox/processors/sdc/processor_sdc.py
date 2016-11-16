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
from processors.sdc.mu_law import *
from fabric.image_processing_common import *  # FIXME for annoying flags


STEERING_STD = 0.3  # rounded
GPS_MEAN = [37.5, -122.3]  # rounded
GPS_STD = [0.2, 0.2]  # approx


class ProcessorSdc(fabric.Processor):

    def __init__(self):
        super(ProcessorSdc, self).__init__()

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
        self.image_fmt = FLAGS.image_fmt
        self.depth = 3
        self.standardize_labels = True
        self.mu_law_steering = False

    def parse_example(self, serialized_example):
        parsed = parse_proto_sdc(serialized_example)
        return parsed

    def process_example(self, tensors, mode='eval', thread_id=0):
        train = (mode == 'train')
        image, camera_id, image_timestamp = tensors[:3]
        processed_image = image_preprocess_sdc(
            image, camera_id,
            height=self.height, width=self.width, image_fmt=self.image_fmt,
            train=train, thread_id=thread_id)
        if mode != 'pred':
            steering_angle, gps_coord = tensors[-2:]
            if steering_angle is not None:
                if self.standardize_labels:
                    steering_angle /= STEERING_STD
                elif self.mu_law_steering:
                    print("Encode angles")
                    steering_angle = mu_law_steering_enc(steering_angle)
            if gps_coord is not None and self.standardize_labels:
                gps_coord = (gps_coord - GPS_MEAN) / GPS_STD
            return processed_image, image_timestamp, steering_angle, gps_coord
        else:
            return processed_image, image_timestamp, tf.zeros((1,)), tf.zeros((2,))

    def reshape_batch(self, batch_tensors, batch_size, num_splits=0):
        images, timestamps, steering_angles, gps_coords = batch_tensors

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        timestamps = tf.reshape(timestamps, [batch_size])
        steering_angles = tf.reshape(steering_angles, [batch_size, 1])
        gps_coords = tf.reshape(gps_coords, [batch_size, 2])

        if num_splits > 0:
            images = tf.split(0, num_splits, images)
            timestamps = tf.split(0, num_splits, timestamps)
            steering_angles = tf.split(0, num_splits, steering_angles)
            gps_coords = tf.split(0, num_splits, gps_coords)

        return images, timestamps, [steering_angles, gps_coords]

    # decode model 'output' values, ie predictions or target labels
    def decode_output(self, value, key=None):
        if key and key == 'steer':
            print('Decoding', key, value)
            if self.standardize_labels:
                return value * STEERING_STD
            elif self.mu_law_steering:
                return mu_law_steering_dec(value)
            else:
                return value
        elif key and key == 'xyz':
            print('Decoding', key, value)
            if self.standardize_labels:
                return value * GPS_STD + GPS_MEAN
            else:
                return value
        else:
            return value
