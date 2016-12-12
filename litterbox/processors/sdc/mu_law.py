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


def mu_law_enc(data, mu=255):
    with tf.name_scope('mu_enc'):
        mu = tf.cast(mu, tf.float32)
        data = tf.cast(data, tf.float32)
        companded = tf.sign(data) * tf.log(1. + mu * tf.abs(data)) / tf.log(1. + mu)
        return companded


def mu_law_dec(data, mu=255):
    with tf.name_scope('mu_dec'):
        mu = tf.cast(mu, tf.float32)
        data = tf.cast(data, tf.float32)
        uncompanded = tf.sign(data) * (tf.pow(1. + mu, tf.abs(data)) - 1.) / mu
        return uncompanded


def mu_law_steering_enc(angle_float, discrete=False):
    if discrete:
        # transform to discrete integers based on steering granularity of
        # source and then mu-law to one hot compatible discrete ints
        steering_angle_i64 = tf.cast(tf.round(angle_float / .00174533), tf.int64)
        input_range = tf.cast(9600, tf.float32)
        output_range = tf.cast(500, tf.float32)
        scaled = 2. * steering_angle_i64 / input_range
        encoded = mu_law_enc(scaled, mu=127)
        encoded = tf.cast(tf.floor((encoded + 1) * output_range / 2), tf.int32)
    else:
        # encode from steering float range to companded -1.0 to 1.0f
        input_range = tf.cast(8.5, tf.float32)  # half range, full is +ve - -ve
        scaled = angle_float / input_range
        encoded = mu_law_enc(scaled, mu=255)
    return encoded


def mu_law_steering_dec(angle_enc, discrete=False):
    if discrete:
        # transform back from one hot discrete ints to steering floats
        input_range = tf.cast(500, tf.float32)
        output_range = tf.cast(9600, tf.float32)
        scaled = (2. * angle_enc + 1) / input_range - 1
        decoded = mu_law_dec(scaled, mu=127)
        decoded = tf.cast(tf.ceil(decoded * output_range / 2.), tf.int32)
        decoded = tf.cast(decoded, tf.float32) * .00174533
    else:
        # decode from -1.0 to 1.0f back to float range
        output_range = tf.cast(8.5, tf.float32)
        decoded = mu_law_dec(angle_enc, mu=255)
        decoded = decoded * output_range
    return decoded
