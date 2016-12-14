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
import tensorflow as tf

from fabric import util
from .feed import Feed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'predict_dir', '/tmp/imagenet_predict',
    """Directory where to write event logs.""")

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/imagenet_train',
    """Directory or file where to read model checkpoint(s).""")

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')


def truncate_batch(batch_outputs, remaining):
    truncated_outputs = []
    for o in batch_outputs:
        if isinstance(o, list):
            truncated_outputs.append([v[:remaining] for v in o])
        elif isinstance(o, dict):
            dict_out = {k: v[:remaining] for k, v in o.items()}
            truncated_outputs.append(dict_out)
        else:
            truncated_outputs.append(o[:remaining])
    return truncated_outputs


def _predict(feed, saver, output_op, names_op):
    """Runs prediction
    """
    predictions = []
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        checkpoint_path, global_step = util.resolve_checkpoint_path(FLAGS.checkpoint_path)
        if not checkpoint_path:
            print('No checkpoint file found at %s' % FLAGS.checkpoint_path)
            return predictions, 0
        saver.restore(sess, checkpoint_path)
        print('Successfully loaded model from %s at step=%d.' % (checkpoint_path, global_step))

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        examples_count = 0
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            batch_size = feed.batch_size
            if FLAGS.sample:
                batch_size //= FLAGS.sample
            num_examples = feed.num_examples_per_epoch()
            num_iter = int(math.ceil(num_examples / batch_size))
            print('%s: starting inference on %d examples in (%s).' %
                  (datetime.now(), num_examples, feed.dataset.subset))
            step = 0
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                batch_outputs = sess.run([output_op, names_op])
                remaining_count = num_examples - examples_count
                examples_count += min(batch_size, remaining_count)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()
                if remaining_count < batch_size:
                    batch_outputs = truncate_batch(batch_outputs, remaining_count)
                predictions.append(batch_outputs)
        except KeyboardInterrupt:
            pass
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return predictions, examples_count


def _reduce_vals(values, factor):
    v_split = tf.split(0, values.get_shape()[0] // factor, values)
    v_mean = [tf.reduce_mean(x, reduction_indices=[0], keep_dims=True) for x in v_split]
    return tf.concat(0, v_mean)


def _reduce_batch(outputs, identities, batch_size, f=8):
    ratio = batch_size // f
    if isinstance(outputs, list):
        outputs = [_reduce_vals(v, f) for v in outputs]
    elif isinstance(outputs, dict):
        outputs = {k: _reduce_vals(v, f) for k, v in outputs.items()}
    else:
        outputs = _reduce_vals(outputs, f)
    idx = f * np.arange(0, ratio)
    return outputs, tf.gather(identities, idx)


def predict(feed, model, raw_outputs=False):
    """Predict/infer outputs for dataset using model."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        inputs, identities = feed.inputs_for_predict()

        # Build a Graph that computes the predictions from the inference model.
        outputs = model.build_tower(inputs)
        if raw_outputs:
            predictions = outputs
        else:
            predictions = model.get_predictions(outputs, processor=feed.processor)

        if feed.sample:
            predictions, identities = _reduce_batch(
                predictions, identities, feed.batch_size, f=feed.sample)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
        else:
            variables_to_restore = tf.contrib.framework.get_model_variables()
        saver = tf.train.Saver(variables_to_restore)

        prediction_values = _predict(feed, saver, predictions, identities)
        return prediction_values
