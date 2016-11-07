# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# Based on original Work Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def parse_proto_sdc(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 480
      image/width: 640
      image/colorspace: 'RGB'
      image/channels: 3
      image/format: 'JPEG'
      image/encoded: <JPEG encoded string>
      steering/angle: 615
      gps/lat: 37.999
      gps/long: 122.300

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.

    """

    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/frame_id': tf.FixedLenFeature([], dtype=tf.string, default_value='center_camera'),
        'image/timestamp': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'steer/angle': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        'steer/speed': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        'steer/timestamp': tf.FixedLenFeature([2], dtype=tf.int64, default_value=[-1, -1]),
        'gps/lat': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        'gps/long': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        'gps/timestamp': tf.FixedLenFeature([2], dtype=tf.int64, default_value=[-1, -1]),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    camera_id = tf.cast(features['image/frame_id'], tf.string)
    image_timestamp = tf.cast(features['image/timestamp'], tf.int64)

    # FIXME for some reason I decided to interpolate in tensorflow, re-thinking that decision...
    zero_const = tf.constant(0.0, dtype=tf.float64)

    steering_timestamp = features['steer/timestamp']
    steering_angle = features['steer/angle']
    if False:
        # interpolate
        steering_angle_delta = tf.cast(steering_angle[1] - steering_angle[0], tf.float64)
        steering_time_delta = tf.cast(steering_timestamp[1] - steering_timestamp[0], tf.float64)
        steering_image_time_delta = tf.cast(image_timestamp - steering_timestamp[0], tf.float64)
        steering_slope = tf.cond(
            tf.equal(steering_time_delta, zero_const),
            lambda: zero_const,
            lambda: steering_angle_delta / steering_time_delta)
        steering_angle = tf.cast(steering_angle[0], tf.float64) + steering_slope * steering_image_time_delta
        steering_angle_f32 = tf.cast(steering_angle, tf.float32)
    else:
        # latest sample
        steering_angle_f32 = tf.cast(steering_angle[1], tf.float32)

    gps_timestamp = features['gps/timestamp']
    gps_lat = features['gps/lat']
    gps_long = features['gps/long']
    if True:
        gps_lat_delta = tf.cast(gps_lat[1] - gps_lat[0], tf.float64)
        gps_long_delta = tf.cast(gps_long[1] - gps_long[0], tf.float64)
        gps_time_delta = tf.cast(gps_timestamp[1] - gps_timestamp[0], tf.float64)
        gps_lat_slope = tf.cond(
            tf.equal(gps_time_delta, zero_const), lambda: zero_const, lambda: gps_lat_delta / gps_time_delta)
        gps_long_slope = tf.cond(
            tf.equal(gps_time_delta, zero_const), lambda: zero_const, lambda: gps_long_delta / gps_time_delta)
        gps_image_time_delta = tf.cast(image_timestamp - gps_timestamp[0], tf.float64)
        gps_lat_interpolated = tf.cast(gps_lat[0], tf.float64) + gps_lat_slope * gps_image_time_delta
        gps_long_interpolated = tf.cast(gps_long[0], tf.float64) + gps_long_slope * gps_image_time_delta
        gps_f32 = tf.concat(0, [tf.cast(gps_lat_interpolated, tf.float32), tf.cast(gps_long_interpolated, tf.float32)])
    else:
        gps_f32 = tf.pack([tf.cast(gps_lat[1], tf.float32), tf.cast(gps_long[1], tf.float32)])

    return features['image/encoded'], camera_id, image_timestamp, steering_angle_f32, gps_f32
