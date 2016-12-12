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
"""Feed class
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import abc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")

tf.app.flags.DEFINE_integer('sample', 0, '')


class Feed(object):

    def __init__(
            self, dataset, processor, batch_size=None, sample=None,
            num_preprocess_threads=None, num_readers=None):

        if not dataset:
            raise ValueError('Please provide a dataset')
        self.dataset = dataset

        if not processor:
            raise ValueError('Please provide a data preprocessor')
        self.processor = processor

        self.batch_size = FLAGS.batch_size if not batch_size else batch_size

        self.sample = FLAGS.sample if sample is None else sample

        self.num_preprocess_threads = FLAGS.num_preprocess_threads \
            if not num_preprocess_threads else num_preprocess_threads

        if self.num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', self.num_preprocess_threads)

        self.num_readers = FLAGS.num_readers if not num_readers else num_readers
        if self.num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

    def num_batches_per_epoch(self):
        return math.ceil(self.num_examples_per_epoch() / self.batch_size)

    def num_examples_per_epoch(self):
        return FLAGS.num_examples if FLAGS.num_examples else self.dataset.num_examples_per_epoch()

    def inputs_for_eval(self, num_splits=0):
        """Generate batches of undistorted examples with labels for evaluation.
        See _batch_inputs.
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            inputs, _, labels = self._batch_inputs(num_splits, mode='eval')
        return inputs, labels

    def inputs_for_train(self, num_splits=0):
        """Generate batches of distorted examples with labels for training.
        See _batch_inputs
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            inputs, _, labels = self._batch_inputs(num_splits, mode='train')
        return inputs, labels

    def inputs_for_predict(self):
        """Generate batches of undistorted examples for inference
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            inputs, identities, _ = self._batch_inputs(0, mode='pred')
        return inputs, identities

    def _batch_inputs(self, num_splits=0, mode='eval'):
        """Construct batches of training or evaluation examples from the image dataset.

        Returns:
          tuple of lists of tensors/lists (of tensors)/dicts (of tensors) containing a
           batch of input examples
        """
        with tf.name_scope('batch_processing'):
            if self.dataset.is_record:
                inputs = self._batch_inputs_record(mode)
            else:
                inputs = self._batch_inputs_file(mode)

            batch_queue_capacity = 2 * self.num_preprocess_threads * self.batch_size
            batch_data = tf.train.batch_join(
                inputs,
                enqueue_many=self.sample > 0,
                batch_size=self.batch_size,
                capacity=batch_queue_capacity)

            return self.processor.reshape_batch(batch_data, self.batch_size, num_splits)

    @abc.abstractmethod
    def _batch_inputs_record(self, mode):
        """Construct batches of training or evaluation examples from the dataset TF records.
        """
        assert False, 'Calling virtual method'

    @abc.abstractmethod
    def _batch_inputs_file(self, mode):
        """Construct batches of training or evaluation examples from dataset files.
        """
        assert False, 'Calling virtual method'
