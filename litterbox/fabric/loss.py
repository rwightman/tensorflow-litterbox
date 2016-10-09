from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import losses


def loss_softmax_cross_entropy(logits, labels, aux_logits=None):
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, num_classes].
    batch_size = labels.get_shape()[0].value
    num_classes = logits.get_shape()[-1].value

    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    # Cross entropy loss for the main softmax prediction.
    losses.softmax_cross_entropy(logits, dense_labels, label_smoothing=0.1, weight=1.0)

    if aux_logits is not None:
        # Cross entropy loss for the auxiliary head.
        losses.softmax_cross_entropy(
            aux_logits, dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')

