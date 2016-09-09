#
"""A library to predict using Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .image_processing import image_preprocess
from .feed import Feed
from fabric import util

import math
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")

tf.app.flags.DEFINE_string('subset', 'test',
                           """Either 'validation', 'train', 'test'""")


def _predict(feed, saver, output_op, names_op):
    """Runs prediction
    """
    predictions = []
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
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_examples = FLAGS.num_examples if FLAGS.num_examples else feed.num_examples_per_epoch()
            num_iter = int(math.ceil(num_examples / feed.batch_size))

            print('%s: starting inference on %d examples in (%s).' % (datetime.now(), num_examples, FLAGS.subset))
            step = 0
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                output_batch, name_batch = sess.run([output_op, names_op])
                name_batch = np.expand_dims(name_batch, axis=1)
                batch = np.concatenate([name_batch, output_batch], axis=1)

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = feed.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                          % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()

                predictions.append(batch)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return np.vstack(predictions)


def predict(dataset, model):
    """Predict/infer outputs for dataset using model."""
    with tf.Graph().as_default():
        # Number of classes in the dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes_with_background()

        # Get images and labels from the dataset.
        feed = Feed(dataset, image_preprocess, batch_size=FLAGS.batch_size)
        images, _, names = feed.inputs()

        # Build a Graph that computes the logits predictions from the inference model.
        logits = model.build_tower(images, num_classes)
        if dataset.has_background_class:
            softmax_output = tf.nn.softmax(tf.slice(logits, [0, 1], [-1, -1]))
        else:
            softmax_output = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        predictions = _predict(feed, saver, softmax_output, names)

        return predictions
