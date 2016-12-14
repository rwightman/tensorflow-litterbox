from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss_softmax_cross_entropy_with_aux(logits, labels, aux_logits=None):
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, num_classes].
    num_classes = logits.get_shape()[-1].value
    dense_labels = tf.contrib.layers.one_hot_encoding(labels, num_classes)

    # Cross entropy loss for the main softmax prediction.
    tf.contrib.losses.softmax_cross_entropy(logits, dense_labels, label_smoothing=0.1, weights=1.0)

    if aux_logits is not None:
        # Cross entropy loss for the auxiliary head.
        tf.contrib.losses.softmax_cross_entropy(
            aux_logits, dense_labels, label_smoothing=0.1, weights=0.4, scope='aux_loss')


# Math for calculating huber loss
def _compute_huber(predictions, labels, delta=1.0):
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    delta = tf.to_float(delta)

    diff = predictions - labels
    diff_abs = tf.abs(diff)
    delta_fact = 0.5 * tf.square(delta)
    condition = tf.less(diff_abs, delta)
    left_opt = 0.5 * tf.square(diff)
    right_opt = delta * diff_abs - delta_fact
    losses_val = tf.select(condition, left_opt, right_opt)
    return losses_val


# Returns non-reduced tensor of unweighted losses with batch dimension matching inputs
def metric_huber(predictions, labels, delta=1.0, scope=None):
    with tf.name_scope(scope, "huber_metric", [predictions, labels]):
        return _compute_huber(predictions, labels, delta)


# Returns reduced loss, applies weights, and adds loss to collections
def loss_huber(predictions, labels, delta=1.0, weights=1.0, scope=None):
    with tf.name_scope(scope, "huber_loss", [predictions, labels]):
        losses_val = _compute_huber(predictions, labels, delta)
        return tf.contrib.losses.compute_weighted_loss(losses_val, weights=weights)


def loss_huber_with_aux(predictions, labels, delta=1.0, weight=1.0, aux_predictions=None):
    loss_huber(predictions, labels, delta=delta, weights=weight)
    if aux_predictions is not None:
        loss_huber(aux_predictions, labels, delta=delta, weights=weight*0.4, scope='aux_huber_loss')


def loss_mse_with_aux(predictions, labels, aux_predictions=None):
    tf.contrib.losses.mean_squared_error(predictions, labels=labels, weights=1.0)
    if aux_predictions is not None:
        tf.contrib.losses.mean_squared_error(aux_predictions, labels=labels, weights=0.4, scope='aux_loss')