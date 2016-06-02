from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf

from inception.feed import Feed
from inception.image_processing import image_preprocessing





def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(0, [ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


class FeedRecord(Feed):
    def __init__(self):
        super().__init__()
        print("FeedRecord")

    def batch_inputs(self, train):
        """Construct batches of training or evaluation examples from the image dataset.

        Args:
          dataset: instance of Dataset class specifying the dataset.
            See dataset.py for details.
          batch_size: integer
          train: boolean
          num_preprocess_threads: integer, total number of preprocessing threads
          num_readers: integer, number of parallel readers

        Returns:
          images: 4-D float Tensor of a batch of images
          labels: 1-D integer Tensor of [batch_size].

        Raises:
          ValueError: if data is not found
        """
        with tf.name_scope('batch_processing'):
            data_files = self.dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            # Create filename_queue
            if train:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=True,
                                                                capacity=16)
            else:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=False,
                                                                capacity=1)

            # Approximate number of examples per shard.
            examples_per_shard = 1024
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
            if train:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * self.batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * self.batch_size,
                    dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if self.num_readers > 1:
                enqueue_ops = []
                for _ in range(self.num_readers):
                    reader = self.dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = self.dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(self.num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image_buffer, label_index, bbox, _ = parse_example_proto(example_serialized)
                image = image_preprocessing(image_buffer, bbox, train, thread_id)
                images_and_labels.append([image, label_index])

            images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            # Reshape images into these desired dimensions.
            height = FLAGS.image_size
            width = FLAGS.image_size
            depth = 3

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[self.batch_size, height, width, depth])

            # Display the training images in the visualizer.
            tf.image_summary('images', images)

            return images, tf.reshape(label_index_batch, [self.batch_size])