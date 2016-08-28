"""Feed class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from .parse_proto_imagenet import parse_imagenet_proto

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_float('image_aspect', 0.0, """Aspect ratio based sizing, square image_size*image_size if 0""")
tf.app.flags.DEFINE_string('image_fmt', 'default',
                            """Either 'default' RGB [-1,1] or 'caffe' BGR [0,255]""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

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

    def __init__(self, dataset, image_preprocess, batch_size=None, num_preprocess_threads=None, num_readers=None):

        self.dataset = dataset
        if not dataset:
            raise ValueError('Please provide a dataset')

        self.image_preprocess=image_preprocess
        if not image_preprocess:
            raise ValueError('Please provide an image preprocessor')

        self.proto_parser = parse_imagenet_proto

        self.batch_size = FLAGS.batch_size if not batch_size else batch_size

        self.num_preprocess_threads = FLAGS.num_preprocess_threads \
            if not num_preprocess_threads else num_preprocess_threads

        if self.num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', self.num_preprocess_threads)

        self.num_readers = FLAGS.num_readers if not num_readers else num_readers
        if self.num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # For aspect based image size, short edge set to FLAGS.image_size
        if FLAGS.image_aspect == 0.0:
            self.width = FLAGS.image_size
            self.height = FLAGS.image_size
        elif FLAGS.image_aspect > 1.0:
            self.width = math.ceil(FLAGS.image_size * FLAGS.image_aspect)
            self.height = FLAGS.image_size
        else:
            self.width = FLAGS.image_size
            self.height = math.ceil(FLAGS.image_size / FLAGS.image_aspect)

        self.depth = 3
        self.caffe_fmt = True if FLAGS.image_fmt == 'caffe' else False

    def num_batches_per_epoch(self):
        return self.dataset.num_examples_per_epoch() / self.batch_size

    def num_examples_per_epoch(self):
        return self.dataset.num_examples_per_epoch()

    def num_classes(self, with_background=False):
        return self.dataset.num_classes_with_background() if with_background \
            else self.dataset.num_classes()

    def inputs(self):
        """Generate batches of images for evaluation.
        See _batch_inputs.
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            images, labels, names = self._batch_inputs(train=False)

        return images, labels, names

    def distorted_inputs(self):
        """Generate batches of distorted images for training.
        See _batch_inputs
        """
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            images, labels, names = self._batch_inputs(train=True)

        return images, labels, names

    def _batch_inputs(self, train=False):
        """Construct batches of training or evaluation examples from the image dataset.

        Args:
          dataset: instance of Dataset class specifying the dataset.
          batch_size: integer, number of examples in batch
          num_preprocess_threads: integer, total number of preprocessing threads but
            None defaults to FLAGS.num_preprocess_threads.

        Returns:
          images: Images. 4D tensor of size [batch_size, FLAGS.image_size, image_size, 3].
          labels: 1-D integer Tensor of [FLAGS.batch_size].
        """
        with tf.name_scope('batch_processing'):

            inputs = self._batch_inputs_record(train) if self.dataset.is_record \
                else self._batch_inputs_file(train)

            images, label_batch, name_batch = tf.train.batch_join(
                inputs,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, self.depth])

            # Display the training images in the visualizer.
            # tf.image_summary('images', images)

            return images, \
                   tf.reshape(label_batch, [self.batch_size]), \
                   tf.reshape(name_batch, [self.batch_size])

    def _batch_inputs_record(self, train):
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
        data_files = self.dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
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
        if train:
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
            image_buffer, label_index, bbox, _, name = self.proto_parser(example_serialized)
            image = self.image_preprocess(
                image_buffer, self.height, self.width, bbox, self.caffe_fmt, train, thread_id)
            inputs.append([image, label_index, name])

        return inputs

    def _batch_inputs_file(self, train):
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
        data_files = self.dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        data_labels = self.dataset.label_indices()
        if data_labels is None:
            raise ValueError('No data labels found for this dataset')

        filename_tensor = tf.convert_to_tensor(data_files, dtype=tf.string)
        label_tensor = tf.convert_to_tensor(data_labels, dtype=tf.int32)

        min_after_dequeue = 128
        capacity = min_after_dequeue + 3 * self.batch_size
        input_queue = tf.train.slice_input_producer(
            [filename_tensor, label_tensor],
            num_epochs=1 if train else None,
            shuffle=train,
            capacity=capacity)

        inputs = []
        for thread_id in range(self.num_preprocess_threads):
            filename = input_queue[0]
            label_index = input_queue[1]
            image_buffer = tf.read_file(filename)
            image = self.image_preprocess(
                image_buffer,
                height=self.height, width=self.width, caffe_fmt=self.caffe_fmt,
                train=train, thread_id=thread_id)
            inputs.append([image, label_index, filename])

        return inputs
