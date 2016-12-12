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
from copy import deepcopy
from fabric.image_processing_common import *

#SDC_MEAN = [0.2087998337, 0.240992306, 0.284853019]
#SDC_STD = [0.2160449662, 0.2489588968, 0.2898496487]
SDC_MEAN = [0.2956688423, 0.3152727451, 0.3687327858]
SDC_STD = [0.2538597152, 0.2642534638, 0.277498978]
SDC_MEAN_STD = [SDC_MEAN, SDC_STD]

distort_params_sdc = {
    'h_flip': True,
    'v_flip': False,
    'elastic_distortion': False,
    'affine_distortion': False,
    'aspect_ratio_range': [0.909, 1.1],
    'area_range': [0.75, 1.0],
    'min_object_covered': 0.85,
    'hue_delta': 0.1,
    'angle_range': 1.5,
}


def _random_hflip(image, uniform_random):
    """Randomly flip an image horizontally (left to right).
    """
    image = tf.convert_to_tensor(image, name='image')
    mirror = tf.less(tf.pack([1.0, uniform_random, 1.0]), 0.5)
    return tf.reverse(image, mirror)


def image_preprocess_sdc(
        image_buffer, camera_id,
        height, width, image_fmt='jpg',
        normalize='global', train=False, summary_suffix='', thread_id=0):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: encoded string Tensor
      camera_id: string identifier of source camera
      height: image target height
      width: image target width
      image_fmt: encode format of eimage
      normalize: boolean, standardize to dataset mean/std deviation vs rescale
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image
    """
    if not height or not width:
        raise ValueError('Please specify target image height & width.')

    flip_coeff = tf.constant(1.0, dtype=tf.float32)
    if image_buffer.dtype == tf.string:
        # compressed image bytes passed as string
        image = decode_compressed_image(image_buffer, image_fmt)
    else:
        # raw image pixels passed as 3D [H, W, C] tensor in RGB format
        image = image_buffer
    assert len(image.get_shape()) == 3

    if train:
        left_string = tf.constant('left_camera', tf.string)
        right_string = tf.constant('right_camera', tf.string)
        left_camera = tf.equal(camera_id, left_string)
        right_camera = tf.equal(camera_id, right_string)

        # bbox are 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        # where each coordinate is [0, 1) and the coordinates are arranged as [ymin, xmin, ymax, xmax].
        # for this code we want to bias the bbox for left camera to the right side, for the right camera
        # to the left side, and leave center as center

        bbox_left = tf.constant([0.05, 0.0, 0.95, 0.9], dtype=tf.float32, shape=[1, 1, 4])
        bbox_right = tf.constant([0.05, 0.1, 0.95, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        bbox_center = tf.constant([0.05, 0.05, 0.95, 0.95], dtype=tf.float32, shape=[1, 1, 4])
        case_pairs = [(left_camera, lambda: bbox_right), (right_camera, lambda: bbox_left)]
        bbox = tf.case(case_pairs, lambda: bbox_center, exclusive=False, name='case')

        distort_params = deepcopy(distort_params_default)
        distort_params.update(deepcopy(distort_params_sdc))
        h_flip = distort_params['h_flip']
        distort_params['h_flip'] = False  # do not perform h-flip in common processing
        distort_params['aspect_ratio_range'][0] *= (width / height)
        distort_params['aspect_ratio_range'][1] *= (width / height)

        image = process_for_train(
            image,
            height=height, width=width, bbox=bbox,
            params=distort_params, summary_suffix=summary_suffix, thread_id=thread_id)

        if h_flip:
            uniform_random = tf.random_uniform([], 0, 1.0)
            image = _random_hflip(image, uniform_random)
            flip_coeff = tf.cond(uniform_random < 0.5, lambda: tf.mul(flip_coeff, -1.0), lambda: flip_coeff)
    else:
        image = process_for_eval(image, height, width)

    image = image_normalize(image, method=normalize, global_mean_std=SDC_MEAN_STD)

    return image, flip_coeff
