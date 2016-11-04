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
from fabric.image_processing_common import *  # FIXME for annoying flags


def mu_law_enc(data, input_range=2**14, output_range=2**8, mu=255):
    with tf.name_scope('mu_enc'):
        mu = tf.cast(mu, tf.float32)
        data = tf.cast(data, tf.float32)
        output_range = tf.cast(output_range, tf.float32)
        scaled = 2. * data / input_range
        companded = tf.sign(scaled) * tf.log(1. + mu * tf.abs(scaled)) / tf.log(1. + mu)
        return tf.cast(tf.floor((companded + 1) * output_range / 2), tf.int32)


def mu_law_dec(data, input_range=2**8, output_range=2**14, mu=255):
    with tf.name_scope('mu_dec'):
        mu = tf.cast(mu, tf.float32)
        data = tf.cast(data, tf.float32)
        output_range = tf.cast(output_range, tf.float32)
        scaled = (2. * data + 1) / input_range - 1
        uncompanded = tf.sign(scaled) * (tf.pow(1. + mu, tf.abs(scaled)) - 1.) / mu
        return tf.cast(tf.ceil(uncompanded * output_range / 2.), tf.int32)


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

        self.image_fmt = FLAGS.image_fmt

        self.depth = 3

    def parse_example(self, serialized_example):
        parsed = parse_proto_sdc(serialized_example)
        #image_buffer, camera_id, image_timestamp, steer_angle, gps
        return parsed

    def process_data(self, data, train=False, thread_id=0):
        image, camera_id, image_timestamp, steering_angle, gps_coord = data
        processed_image = image_preprocess_sdc(
            image, camera_id,
            height=self.height, width=self.width, image_fmt=self.image_fmt,
            train=train, thread_id=thread_id)
        if False:
            # convert steering to integer encoding
            steering_angle_i64 = tf.cast(tf.round(steering_angle / .00174533), tf.int64)
            # mu-law encode steering int
            steering_angle = mu_law_enc(steering_angle_i64, input_range=9600, output_range=500, mu=127)
        return processed_image, steering_angle, gps_coord, image_timestamp

    def reshape_batch(self, batch_data, batch_size, num_splits=0):
        images, steering_angles, gps_coords, timestamps = batch_data

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        steering_angles = tf.reshape(steering_angles, [batch_size, 1])
        gps_coords = tf.reshape(gps_coords, [batch_size, 2])
        timestamps = tf.reshape(timestamps, [batch_size])

        if num_splits > 0:
            images = tf.split(0, num_splits, images)
            steering_angles = tf.split(0, num_splits, steering_angles)
            gps_coords = tf.split(0, num_splits, gps_coords)
            timestamps = tf.split(0, num_splits, timestamps)

        return images, steering_angles, gps_coords, timestamps

    def map_inputs(self, tensor_list, split_index=None):
        images, steering_angles, gps_coords, timestamps = tensor_list
        if split_index is None:
            targets = [steering_angles, gps_coords]
            return images, targets, timestamps
        else:
            targets = [steering_angles[split_index], gps_coords[split_index]]
            return images[split_index], targets, timestamps[split_index]

    def decode_steering(self, steering_angle_int):
        decoded = mu_law_dec(steering_angle_int, input_range=500, output_range=9600, mu=127)
        return tf.cast(decoded, tf.float32) * .00174533
