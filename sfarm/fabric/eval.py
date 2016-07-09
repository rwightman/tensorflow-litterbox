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
"""A library to evaluate Inception on a single GPU.
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

from fabric.image_processing import image_preprocess
from fabric.feed import Feed
from fabric import util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/imagenet_train',
                           """Directory or file where to read model checkpoint(s).""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")

tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")


def _eval_once(feed, saver, summary_writer, top_1_op, top_5_op, loss_op, summary_op):
    """Runs Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      loss_op: Cross entropy loss op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        checkpoint_path, global_step = util.resolve_checkpoint_path(FLAGS.checkpoint_path)
        if not checkpoint_path:
            print('No checkpoint file found at %s' % FLAGS.checkpoint_path)
            return
        saver.restore(sess, checkpoint_path)
        print('Successfully loaded model from %s at step=%d.' % (checkpoint_path, global_step))

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_examples = FLAGS.num_examples if FLAGS.num_examples else feed.num_examples_per_epoch()
            num_iter = int(math.ceil(num_examples / feed.batch_size))
            # Counts the number of correct predictions.
            count_top_1 = 0.0
            count_top_5 = 0.0
            count_loss = 0.0
            total_sample_count = num_iter * feed.batch_size
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                top_1, top_5, loss = sess.run([top_1_op, top_5_op, loss_op])
                count_top_1 += np.sum(top_1)
                count_top_5 += np.sum(top_5)
                count_loss += loss
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = feed.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # Compute precision @ 1.
            precision_at_1 = count_top_1 / total_sample_count
            recall_at_5 = count_top_5 / total_sample_count
            loss = count_loss / num_iter
            print('%s: precision @ 1 = %.4f, recall @ 5 = %.4f, loss = %.4f [%d examples]' %
                  (datetime.now(), precision_at_1, recall_at_5, loss, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
            summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
            summary.value.add(tag='Loss', simple_value=loss)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset, model):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        feed = Feed(dataset, image_preprocess, batch_size=FLAGS.batch_size)
        images, labels, _ = feed.inputs()

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes_with_background()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.build_tower(images, num_classes)

        # Calculate predictions.
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)
        loss_op = model.loss_op(logits, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph=tf.get_default_graph())

        while True:
            _eval_once(feed, saver, summary_writer, top_1_op, top_5_op, loss_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
