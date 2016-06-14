#
"""A library to predict using Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime

import numpy as np
import os.path
import tensorflow as tf

from .image_processing import *
from .feed import Feed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")

def _predict(feed, saver, softmax_op, filenames_op):
    """Runs prediction
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Restores from checkpoint with relative path.
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt.model_checkpoint_path))

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_examples = FLAGS.num_examples if FLAGS.num_examples else feed.num_examples_per_epoch()
            num_iter = int(math.ceil(num_examples / feed.batch_size))
            # Counts the number of correct predictions.

            total_sample_count = num_iter * feed.batch_size
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            outputs = []
            files = []
            while step < num_iter and not coord.should_stop():
                output_batch, file_batch = sess.run([softmax_op, filenames_op])

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = feed.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()

                #output logits to file
                outputs.append(output_batch)
                files.append(file_batch)
                for x in zip(file_batch, output_batch):
                    print(x)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def predict(dataset, model):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        feed = Feed(dataset, image_preprocess, batch_size=FLAGS.batch_size)
        images, _, filenames = feed.inputs()

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

        # Build a Graph that computes the logits predictions from the inference model.
        logits, _ = model.build(images, num_classes)
        softmax_output = tf.nn.softmax(tf.slice(logits, [0, 1], [-1, -1]))

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        _predict(feed, saver, softmax_output, filenames)
