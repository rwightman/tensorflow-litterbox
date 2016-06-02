"""Feed class"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
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
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")



class Feed(object):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size=None, num_preprocess_threads=None, num_readers=None):

        if not batch_size:
            self.batch_size = FLAGS.batch_size
        else:
            self.batch_size = batch_size

        if not num_preprocess_threads:
            self.num_preprocess_threads = FLAGS.num_preprocess_threads
        else:
            self.num_preprocess_threads = num_preprocess_threads
        if self.num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', self.num_preprocess_threads)

        if not num_readers:
            self.num_readers = FLAGS.num_readers
        else:
            self.num_readers = num_readers
        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        print("Feed")

    @abstractmethod
    def batch_inputs(self, train=False):
        """Construct batches of training or evaluation examples from the image dataset.

        Args:
          dataset: instance of Dataset class specifying the dataset.
          batch_size: integer, number of examples in batch
          num_preprocess_threads: integer, total number of preprocessing threads but
            None defaults to FLAGS.num_preprocess_threads.

        Returns:
          images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                             image_size, 3].
          labels: 1-D integer Tensor of [FLAGS.batch_size].
        """
        pass

    def inputs(self):
        """Generate batches of images for evaluation.

        See batch_inputs.
        """

        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            images, labels = self.batch_inputs(train=False)

        return images, labels

    def distorted_inputs(self):
        """Generate batches of distorted images for training.

        See batch_inputs
        """

        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            images, labels = self.batch_inputs(train=True)
        return images, labels

