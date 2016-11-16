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

import tensorflow as tf
import fabric
from processors import ProcessorImagenet

FLAGS = tf.app.flags.FLAGS

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


class FeedImagesWithLabels(fabric.Feed):

    def __init__(
            self, dataset, processor=None, batch_size=None, sample=None,
            num_preprocess_threads=None, num_readers=None):

        assert dataset.data_files()

        if processor is None:
            processor = ProcessorImagenet()

        super(FeedImagesWithLabels, self).__init__(
            dataset=dataset, processor=processor, batch_size=batch_size,
            sample=sample, num_preprocess_threads=num_preprocess_threads,
            num_readers=num_readers)

    def num_classes(self, with_background=False):
        return self.dataset.num_classes_with_background() if with_background \
            else self.dataset.num_classes()

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

        input_examples = []
        for thread_id in range(self.num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            parsed = self.processor.parse_example(example_serialized)
            processed = self.processor.process_example(parsed, mode=mode, thread_id=thread_id)
            if self.sample > 0:
                processed = [tf.gather(tf.expand_dims(x, 0), [0] * self.sample) for x in processed]
            input_examples.append(list(processed))

        return input_examples

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
            combo = sorted(zip(data_files, data_labels))
            data_files, data_labels = map(list, zip(*combo))

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

        input_examples = []
        num_threads = 1 if mode == 'pred' else self.num_preprocess_threads
        for thread_id in range(num_threads):
            filename = prefetch_head[0]
            input_buffer = prefetch_head[1]
            label_index = prefetch_head[2]

            #FIXME hack, need to fix this so we can have processing pipeline that doesn't need
            #to handle labels for inference and a file pipeline that can actually training with
            #non-integer index labels

            data_packed = [input_buffer, label_index, filename]
            processed = self.processor.process_example(data_packed, mode=mode, thread_id=thread_id)
            if self.sample > 0:
                processed = [tf.gather(tf.expand_dims(x, 0), [0] * self.sample) for x in processed]
            input_examples.append(list(processed))

        return input_examples
