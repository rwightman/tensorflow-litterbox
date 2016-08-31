# Copyright 2016 Google Inc. All Rights Reserved.
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

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in pararllel across mulitple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

try:
    from skimage import transform
    has_skimage = True
except ImportError:
    has_skimage = False

IMAGENET_MEAN_CAFFE = [103.939, 116.779, 123.68]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_jpeg(image_buffer, depth=3, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=depth)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_affine_cv2(image, alpha_affine=15, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    distorted_image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    return distorted_image


def distort_affine_skimage(image, angle=10, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    rot = np.deg2rad(random_state.randint(-angle, angle))
    sheer = np.deg2rad(random_state.randint(-angle, angle))

    shape = image.shape
    shape_size = shape[:2]
    center = np.float32(shape_size) / 2. - 0.5

    pre = transform.SimilarityTransform(translation=-center)
    affine = transform.AffineTransform(rotation=rot, shear=sheer, translation=center)
    tform = pre + affine

    distorted_image = transform.warp(image, tform.params, mode='reflect')

    return distorted_image.astype(np.float32)


def distort_elastic_cv2(image, alpha=100, sigma=20, random_state=None):
    """Elastic deformation of images as per [Simard2003].
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape_size = image.shape[:2]
    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*shape_size) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*shape_size) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y, 
        borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    return distorted_img


def distort_image(image, height, width, bbox=None, thread_id=0, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    h_flip = True  # FIXME make distortion configuration
    v_flip = False
    elastic_distortion = False
    affine_distortion = True

    with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
            tf.image_summary('image_with_bounding_boxes', image_with_box)

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.67, 1.33],
            area_range=[0.1, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distort_bbox)
            tf.image_summary('images_with_distorted_bounding_box', image_with_distorted_box)

        if affine_distortion:
            if has_cv2:
                image = tf.py_func(distort_affine_cv2, [image], [tf.float32])[0]
            elif has_skimage:
                image = tf.py_func(distort_affine_skimage, [image], [tf.float32])[0]
            else:
                print('Affine image distortion disabled, no cv2 or skimage module present.')
            image.set_shape([height, width, 3])

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        resize_method = tf.image.ResizeMethod.BILINEAR
        distorted_image = tf.image.resize_images(distorted_image, height, width, resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])

        if not thread_id:
            tf.image_summary('cropped_resized_image', tf.expand_dims(distorted_image, 0))

        if elastic_distortion:
            if has_cv2:
                distorted_image = tf.py_func(distort_elastic_cv2, [distorted_image], [tf.float32])[0]
            else:
                print('Elastic image distortion disabled, no cv2 module present.')
            distorted_image.set_shape([height, width, 3])

        # Randomly flip the image horizontally.
        if h_flip:
            distorted_image = tf.image.random_flip_left_right(distorted_image)

        if v_flip:
            distorted_image = tf.image.random_flip_up_down(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.image_summary('final_distorted_image', tf.expand_dims(distorted_image, 0))

        return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
        # Crop the central region of the image
        image = tf.image.central_crop(image, central_fraction=0.975)

        # Resize the image to the network height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])

        return image


def image_preprocess(image_buffer, height, width, bbox=None, caffe_fmt=False, train=False, thread_id=0):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: JPEG encoded string Tensor
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if not height or not width:
        raise ValueError('Please specify target image height & width.')

    image = decode_jpeg(image_buffer)

    if train:
        if bbox is None:
            bbox = tf.zeros([1, 1, 4], "float")
        image = distort_image(image, height=height, width=width, bbox=bbox, thread_id=thread_id)
    else:
        image = eval_image(image, height, width)

    if caffe_fmt:
        # Rescale to [0, 255]
        image = tf.mul(image, 255.0)
        # Convert RGB to BGR
        red, green, blue = tf.split(2, 3, image)
        image = tf.concat(2, [
            blue - IMAGENET_MEAN[0],
            green - IMAGENET_MEAN[1],
            red - IMAGENET_MEAN[2],
            ])
    else:
        image = tf.sub(image, IMAGENET_MEAN)
        image = tf.div(image, IMAGENET_STD)
    #else:
    #    # Rescale to [-1,1] instead of [0, 1)
    #    image = tf.sub(image, 0.5)
    #    image = tf.mul(image, 2.0)

    return image
