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
"""Read and preprocess image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from fabric.image_processing_common import *


distort_params_sdc = {
    'h_flip': False,
    'v_flip': False,
    'elastic_distortion': False,
    'affine_distortion': True,
    'aspect_ratio_range': [0.88889, 1.125],
    'area_range': [0.7, 1.0],
    'min_object_covered': 0.8,
}


def image_preprocess_sdc(image_buffer, camera_id, height, width, img_format='jpg', train=False, thread_id=0):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: JPEG encoded string Tensor
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image
    """
    if not height or not width:
        raise ValueError('Please specify target image height & width.')

    image = decode_compressed_image(image_buffer, img_format)

    if train:
        left_string = tf.constant('left_camera', tf.string)
        right_string = tf.constant('right_camera', tf.string)
        left_camera = tf.equal(camera_id, left_string)
        right_camera = tf.equal(camera_id, right_string)
        bbox_left = tf.constant([0.0, 0.0, 1.0, 0.75], dtype=tf.float32, shape=[1, 1, 4])
        bbox_right = tf.constant([0.0, 0.25, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        bbox_center = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        case_pairs = [(left_camera, lambda: bbox_left), (right_camera, lambda: bbox_right)]
        bbox = tf.case(case_pairs, lambda: bbox_center, exclusive=False, name='case')

        distort_params = distort_params_default
        distort_params.update(distort_params_sdc)
        image = process_for_train(
            image, height=height, width=width, bbox=bbox, params=distort_params, thread_id=thread_id)
    else:
        image = process_for_eval(image, height, width)

    # Rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    return image
