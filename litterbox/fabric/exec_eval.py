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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from fabric import util
from fabric.feed import Feed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/imagenet_eval',
    """Directory where to write event logs.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 60 * 5,
    """How often to run the eval.""")

tf.app.flags.DEFINE_boolean(
    'run_once', False,
    """Whether to run eval only once.""")

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/imagenet_train',
    """Directory or file where to read model checkpoint(s).""")

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')


def _eval_once(feed, saver, summary_writer, eval_ops, summary_op):
    """Runs Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      eval_ops: dict of evaluation metric ops
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)

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

            eval_ops_list = []
            eval_names_list = []
            if isinstance(eval_ops, dict):
                for name, op in eval_ops.items():
                    eval_ops_list.append(op)
                    eval_names_list.append(name)
            else:
                assert isinstance(eval_ops, list)
                eval_ops_list = eval_ops
                for op in eval_ops:
                    eval_names_list.append(op.name)

            num_examples = feed.num_examples_per_epoch()
            num_iter = int(math.ceil(num_examples / feed.batch_size))
            eval_totals = [np.float64(0.0)] * len(eval_ops_list)
            example_count = 0
            step = 0
            print('%s: starting evaluation on (%s).' % (datetime.now(), feed.dataset.subset))
            start_time = time.time()
            try:
                while step < num_iter and not coord.should_stop():
                    eval_results = sess.run(eval_ops_list)
                    remaining_count = num_examples - example_count
                    example_count += min(feed.batch_size, remaining_count)

                    for i, result in enumerate(eval_results):
                        if remaining_count < feed.batch_size:
                            result = result[:remaining_count]
                        eval_totals[i] += np.sum(result, dtype=np.float64)
                    step += 1

                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = feed.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                              % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                        start_time = time.time()
            except KeyboardInterrupt:
                pass

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            print('%s:' % datetime.now(), end=" ")
            for i, val in enumerate(eval_totals):
                mean_val = val / example_count
                print('%s = %.6f' % (eval_names_list[i], mean_val), end=" ")
                summary.value.add(tag=eval_names_list[i], simple_value=mean_val)
            print('[%d examples]' % example_count)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset, processor, model):
    """Evaluate model on Dataset for a number of steps."""

    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    with tf.Graph().as_default():

        # Get images and labels from the dataset.
        feed = Feed(dataset, processor=processor, batch_size=FLAGS.batch_size)
        eval_inputs = feed.inputs_for_eval()
        inputs, labels, _ = feed.processor.map_inputs(eval_inputs)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        outputs = model.build_tower(inputs)

        # Calculate predictions.
        eval_ops = model.eval_ops(outputs, labels, processor=processor)

        # Restore the moving average version of the learned variables for eval.
        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
        else:
            variables_to_restore = tf.contrib.framework.get_model_variables()

        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph=tf.get_default_graph())

        while True:
            _eval_once(feed, saver, summary_writer, eval_ops, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
