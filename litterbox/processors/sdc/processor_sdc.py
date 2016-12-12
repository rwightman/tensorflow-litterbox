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

    def __init__(self, params={}):
        super(ProcessorSdc, self).__init__()

        image_aspect = params['image_aspect'] if 'image_aspect' in params else FLAGS.image_aspect
        image_size = params['image_size'] if 'image_size' in  params else FLAGS.image_size
        image_fmt = params['image_fmt'] if 'image_fmt' in params else FLAGS.image_fmt
        image_norm = params['image_norm'] if 'image_norm' in params else FLAGS.image_norm

        # For aspect based image size, short edge set to FLAGS.image_size
        if image_aspect == 0.0 or image_aspect == 1.0:
            self.width = image_size
            self.height = image_size
        elif image_aspect < 1.0:
            self.width = math.floor(image_size * image_aspect)
            self.height = image_size
        else:
            self.width = image_size
            self.height = math.floor(image_size / image_aspect)
        self.image_fmt = image_fmt
        self.depth = 3
        self.standardize_input = image_norm
        self.standardize_labels = True
        self.mu_law_steering = False
        self.num_input_images = 1

    def get_input_shape(self, batch_size=0):
        shape = [self.height, self.width, self.depth]
        if self.num_input_images > 1:
            shape = [self.num_input_images] + shape
        if batch_size:
            shape = [batch_size] + shape
        return shape

    def parse_example(self, serialized_example):
        parsed = parse_proto_sdc(serialized_example)
        return parsed

    def process_example(self, tensors, mode='eval', thread_id=0):
        train = (mode == 'train')
        image, image_timestamp, camera_id = tensors[:3]

        #FIXME push single/multi image handling into image_process_sdc if we want to share random augmentations
        if self.num_input_images > 1:
            assert(len(image.get_shape()) > 0)
            print('Multi image', image.get_shape())
            split_image = tf.unpack(image)
            split_processed = []
            for i, x in enumerate(split_image):
                suffix = '%d' % i
                xp, _ = image_preprocess_sdc(
                    x, camera_id,
                    height=self.height, width=self.width, image_fmt=self.image_fmt,
                    normalize=self.standardize_input, train=train, summary_suffix=suffix, thread_id=thread_id)
                split_processed.append(xp)
            processed_image = tf.pack(split_processed)
            #FIXME need to sort out flip across mult-images
            flip_coeff = tf.constant(1.0, dtype=tf.float32)
        else:
            print('Single image')
            processed_image, flip_coeff = image_preprocess_sdc(
                image, camera_id,
                height=self.height, width=self.width, image_fmt=self.image_fmt,
                normalize=self.standardize_input, train=train, thread_id=thread_id)

        if mode != 'pred':
            steering_angle, gps_coord = tensors[-2:]
            if steering_angle is not None:
                steering_angle = tf.mul(steering_angle, flip_coeff)
                if self.standardize_labels:
                    steering_angle /= STEERING_STD
                elif self.mu_law_steering:
                    print("Encode mu-law angles")
                    steering_angle = mu_law_steering_enc(steering_angle)
            if gps_coord is not None and self.standardize_labels:
                gps_coord = (gps_coord - GPS_MEAN) / GPS_STD
            return processed_image, image_timestamp, steering_angle, gps_coord
        else:
            return processed_image, image_timestamp, tf.zeros((1,)), tf.zeros((2,))

    def reshape_batch(self, batch_tensors, batch_size, num_splits=0):
        images, timestamps, steering_angles, gps_coords = batch_tensors

        images = tf.cast(images, tf.float32)
        if self.num_input_images > 1:
            images = tf.reshape(images, shape=[batch_size, self.num_input_images, self.height, self.width, self.depth])
        else:
            images = tf.reshape(images, shape=[batch_size, self.height, self.width, self.depth])
        timestamps = tf.reshape(timestamps, [batch_size])
        steering_angles = tf.reshape(steering_angles, [batch_size, 1])
        gps_coords = tf.reshape(gps_coords, [batch_size, 2])

        if num_splits > 0:
            # Split tensors for multi-gpu training
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
