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

"""A library to train Inception using multiple GPU's with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from .feed import Feed
from .processor import select_split
from .opt_param_scheduler import OptParamScheduler

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")

tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_float('grad_clip', 5.0,
                          """Clip gradients to this value.""")

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')


def _add_tower_loss(inputs, labels, model, scope):
    """Calculate the total loss on a single tower running the ImageNet model.

    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.

    Args:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                         FLAGS.image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
      num_classes: number of classes
      scope: unique prefix string identifying the ImageNet tower, e.g.
        'tower_0'.

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Build inference Graph.
    model.build_tower(inputs, is_training=True, scope=scope)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    model.add_tower_loss(labels)

    # Assemble all of the losses for the current tower only.
    tower_losses = tf.contrib.losses.get_losses(scope)

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(tower_losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(tower_losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in tower_losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on TensorBoard.
        loss_name = model.strip_common_scope(l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary('losses/' + loss_name + ' (raw)', l)
        tf.scalar_summary('losses/' + loss_name, loss_averages.average(l))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
        output_loss = tf.identity(tower_losses[0])

    return total_loss, output_loss


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def _build_train_graph(feed, model):
    global_step = tf.contrib.framework.get_or_create_global_step()

    opt_param_sched = OptParamScheduler(
        global_step_tensor=global_step,
        num_steps_per_epoch=feed.num_batches_per_epoch())

    opt = opt_param_sched.opt

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, 'Batch size must be divisible by number of GPUs'
    num_gpus = FLAGS.num_gpus

    train_examples = feed.inputs_for_train(num_splits=num_gpus)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in range(num_gpus):
        inputs, labels = select_split(train_examples, i)
        with tf.device('/gpu:%d' % i):
            with tf.name_scope(model.scope_name(i)) as scope:
                # Force all Variables to reside on the CPU.
                if num_gpus > 1:
                    with tf.contrib.framework.arg_scope(model.get_variable_fns(), device='/cpu:0'):
                        # Calculate the loss for one tower of the ImageNet model. This
                        # function constructs the entire ImageNet model but shares the
                        # variables across all towers.
                        tower_losses = _add_tower_loss(inputs, labels, model, scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                else:
                    tower_losses = _add_tower_loss(inputs, labels, model, scope)

                # Calculate the gradients for the batch of data on this ImageNet tower.
                grads = opt.compute_gradients(tower_losses[0])

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

    # Retain the summaries from the final tower.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, model.last_scope())

    # Retain the Batch Normalization updates operations only from the
    # final tower. Ideally, we should grab the updates from all towers
    # but these stats accumulate extremely fast so we can ignore the
    # other stats from the other towers without significant detriment.
    batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, model.last_scope())

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = _average_gradients(tower_grads) if num_gpus > 1 else tower_grads[0]

    if FLAGS.grad_clip:
        g, v = zip(*grads)
        g, _ = tf.clip_by_global_norm(g, FLAGS.grad_clip)
        grads = list(zip(g, v))
    
    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', opt_param_sched.learning_rate_tensor))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.histogram_summary(model.strip_common_scope(var.op.name) + '/gradients', grad))

    update_ops = []

    # Apply the gradients to adjust the shared variables.
    apply_gradient_update = opt.apply_gradients(grads, global_step=global_step)
    update_ops.append(apply_gradient_update)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.histogram_summary(model.strip_common_scope(var.op.name), var))

    if FLAGS.moving_average_decay:
        moving_average_variables = (tf.trainable_variables() + tf.moving_average_variables())
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_update = variable_averages.apply(moving_average_variables)
        update_ops.append(variables_averages_update)

    batch_norm_updates = tf.group(*batch_norm_updates)
    update_ops.append(batch_norm_updates)

    # Group all updates to into a single train op.
    train_op = tf.group(*update_ops)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

    return train_op, init_op, summary_op, tower_losses


def restore_pretrained_variables(sess, model, model_path, restore_outputs=True):
    assert tf.gfile.Exists(model_path)
    checkpoint_variable_set = set()
    if tf.gfile.IsDirectory(model_path):
        model_path = tf.train.latest_checkpoint(model_path)
    else:
        model_path = model_path
        reader = tf.train.NewCheckpointReader(model_path)
        checkpoint_variable_set = set(reader.get_variable_to_shape_map().keys())
    variables_to_restore = model.variables_to_restore(restore_outputs, checkpoint_variable_set)
    tf.train.Saver(variables_to_restore).restore(sess, model_path)
    print('%s: Pre-trained model restored from %s' % (datetime.now(), model_path))


def train(feed, model):

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.gfile.DeleteRecursively(FLAGS.train_dir)

    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        train_op, init_op, summary_op, tower_losses = _build_train_graph(feed, model)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        tf.train.write_graph(sess.graph_def, FLAGS.train_dir, 'network.pb.txt', as_text=True)

        sess.run(init_op)

        # When fine-tuning a model, we do not restore the outputs but instead we
        # randomly initialize them.
        if FLAGS.pretrained_model_path:
            restore_pretrained_variables(
                sess, model, model_path=FLAGS.pretrained_model_path, restore_outputs=not FLAGS.fine_tune)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, total_loss_value, output_loss_value = sess.run([train_op, tower_losses[0], tower_losses[1]])
            duration = time.time() - start_time

            assert not np.isnan(total_loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                epoch = 1 + (step * FLAGS.batch_size) // feed.num_examples_per_epoch()
                format_str = '%s: step %d, epoch %d, loss = %.2f total; ' \
                             '%.4f output (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, epoch, total_loss_value,
                                    output_loss_value, examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
