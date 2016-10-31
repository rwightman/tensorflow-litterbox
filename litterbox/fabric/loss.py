from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss_softmax_cross_entropy_with_aux(logits, labels, aux_logits=None):
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, num_classes].
    batch_size = labels.get_shape()[0].value
    num_classes = logits.get_shape()[-1].value

    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    # Cross entropy loss for the main softmax prediction.
    tf.contrib.losses.softmax_cross_entropy(logits, dense_labels, label_smoothing=0.1, weight=1.0)

    if aux_logits is not None:
        # Cross entropy loss for the auxiliary head.
        tf.contrib.losses.softmax_cross_entropy(
            aux_logits, dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')


def loss_huber(predictions, targets, delta=1.0, scope=None):
    with tf.name_scope(scope, "huber_loss", [predictions, targets]):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())
        predictions = tf.to_float(predictions)
        targets = tf.to_float(targets)
        delta = tf.to_float(delta)

        diff = predictions - targets
        diff_abs = tf.abs(diff)
        delta_fact = 0.5 * tf.square(delta)

        condition = tf.less(diff_abs, delta)
        left_opt = 0.5 * tf.square(diff)
        right_opt = delta * diff_abs - delta_fact
        losses_val = tf.select(condition, left_opt, right_opt)

        return tf.contrib.losses.compute_weighted_loss(losses_val)


def loss_huber_with_aux(predictions, targets, delta=1.0, aux_predictions=None):
    loss_huber(predictions, targets, delta=delta)
    if aux_predictions is not None:
        loss_huber(aux_predictions, targets, delta=delta, scope='aux_huber_loss')