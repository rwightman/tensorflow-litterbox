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

# Images are preprocessed asynchronously using multiple threads specifed by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 8,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


class Feed(object):

    def __init__(
            self, dataset, processor, batch_size=None, sample=0,
            num_preprocess_threads=None, num_readers=None):

        self.dataset = dataset
        if not dataset:
            raise ValueError('Please provide a dataset')

        self.processor = processor
        if not processor:
            raise ValueError('Please provide a data preprocessor')

        self.batch_size = FLAGS.batch_size if not batch_size else batch_size

        self.sample = sample

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

    def num_classes(self, with_background=False):
        return self.dataset.num_classes_with_background() if with_background \
            else self.dataset.num_classes()

    def inputs_for_eval(self, num_splits=0):
        """Generate batches of images for evaluation.
        See _batch_inputs.
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            eval_inputs = self._batch_inputs(num_splits, mode='eval')

        return eval_inputs

    def inputs_for_train(self, num_splits=0):
        """Generate batches of distorted images for training.
        See _batch_inputs
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            train_inputs = self._batch_inputs(num_splits, mode='train')

        return train_inputs

    def inputs_for_predict(self):
        """
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            train_inputs = self._batch_inputs(0, mode='pred')

        return train_inputs

    def _batch_inputs(self, num_splits, mode='eval'):
        """Construct batches of training or evaluation examples from the image dataset.

        Returns:
          images: Images. 4D tensor of size [batch_size, image_size, image_size, 3].
          labels: 1-D integer Tensor of [batch_size].
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

    def _batch_inputs_record(self, mode):
        """Construct batches of training or evaluation examples from the image dataset.

        Returns:
          images: 4-D float Tensor of a batch of images
          labels: 1-D integer Tensor of [batch_size].

        Raises:
          ValueError: if data is not found
        """
        data_files = self.dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if mode == 'train':
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle=False, capacity=1)

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        if mode == 'train':
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 4 * self.batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 4 * self.batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if self.num_readers > 1:
            enqueue_ops = []
            for _ in range(self.num_readers):
                reader = self.dataset.reader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = self.dataset.reader()
            _, example_serialized = reader.read(filename_queue)

        inputs = []
        for thread_id in range(self.num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            parsed = self.processor.parse_example(example_serialized)
            processed = self.processor.process_data(parsed, mode=mode, thread_id=thread_id)
            if self.sample > 0:
                processed = [tf.gather(tf.expand_dims(x, 0), [0] * self.sample) for x in processed]
            inputs.append(list(processed))

        return inputs

    def _batch_inputs_file(self, mode):
        """Construct batches of training or evaluation examples from filenames.

        Returns:
          images: 4-D float Tensor of a batch of images
          labels: 1-D integer Tensor of [batch_size].

        Raises:
          ValueError: if data is not found
        """
        data_files = self.dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        #FIXME make this configurable, not all datasets have labels like this
        data_labels = self.dataset.label_indices()

        if mode == 'pred':
            combo = list(zip(data_files, data_labels))
            combo.sort()
            data_files, data_labels = zip(*combo)
            data_files = list(data_files)
            data_labels = list(data_labels)

        filename_tensor = tf.convert_to_tensor(data_files, dtype=tf.string)
        label_tensor = tf.convert_to_tensor(data_labels, dtype=tf.int32)

        min_after_dequeue = 64
        capacity = min_after_dequeue + 3 * self.batch_size
        input_queue = tf.train.slice_input_producer(
            [filename_tensor, label_tensor],
            num_epochs=None,
            shuffle=mode == 'train',
            capacity=capacity)

        prefetch_queue = tf.FIFOQueue(
            capacity=3 * self.batch_size,
            shapes=[(), (), ()],
            dtypes=[tf.string, tf.string, tf.int32])

        enqueue_ops = [prefetch_queue.enqueue(
            (input_queue[0], tf.read_file(input_queue[0]), input_queue[1]))
        ]
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(prefetch_queue, enqueue_ops))
        prefetch_head = prefetch_queue.dequeue()

        inputs = []
        num_threads = 1 if mode == 'pred' else self.num_preprocess_threads
        for thread_id in range(num_threads):
            filename = prefetch_head[0]
            input_buffer = prefetch_head[1]
            label_index = prefetch_head[2]

            #FIXME hack, need to fix this so we can have processing pipeline that doesn't need
            #to handle labels for inference and a file pipeline that can actually training with
            #non-integer index labels

            data_packed = [input_buffer, label_index, filename]
            processed = self.processor.process_data(data_packed, mode=mode, thread_id=thread_id)
            if self.sample > 0:
                processed = [tf.gather(tf.expand_dims(x, 0), [0] * self.sample) for x in processed]
            inputs.append(list(processed))

        return inputs
