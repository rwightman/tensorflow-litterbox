# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
import tensorflow as tf

_default_initializer_params = {
    'stddev': 0.1,
    'dtype': tf.float32,
}


def bidir_lstm(
        inputs,
        num_units,
        num_layers=1,
        initializer_fn=tf.truncated_normal,
        initializer_params=_default_initializer_params,
        dtype=tf.float32,
        scope=None
):
    print('input shape', inputs.get_shape())
    shape = inputs.get_shape().as_list()
    batch_size = shape[0]
    inputs_unpacked = tf.unpack(inputs, axis=1)

    cell_fw = tf.contrib.rnn.python.ops.lstm_ops.LSTMBlockCell(num_units=num_units)
    cell_bw = tf.contrib.rnn.python.ops.lstm_ops.LSTMBlockCell(num_units=num_units)
    print('cell state size', cell_fw.state_size)

    if num_layers > 1:
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers)

    initializer_params = initializer_params or {}
    initializer_params['dtype'] = dtype
    if isinstance(cell_fw.state_size, tuple):
        initial_state_fw = tuple(initializer_fn([batch_size, s]) for s in cell_fw.state_size)
        initial_state_bw = tuple(initializer_fn([batch_size, s]) for s in cell_bw.state_size)
    else:
        initial_state_fw = initializer_fn(shape=[batch_size, cell_fw.state_size], **initializer_params)
        initial_state_bw = initializer_fn(shape=[batch_size, cell_bw.state_size], **initializer_params)

    outputs, _, _ = tf.nn.bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs_unpacked,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        dtype=dtype,
        scope=scope)

    outputs = tf.pack(outputs, axis=1)
    print('output shape', outputs.get_shape())

    return outputs


def lstm(
        inputs,
        num_units,
        num_layers=1,
        initializer_fn=tf.truncated_normal,
        initializer_params=_default_initializer_params,
        dtype=tf.float32,
        scope=None
):
    print('input shape', inputs.get_shape())
    shape = inputs.get_shape().as_list()
    batch_size = shape[0]
    inputs_unpacked = tf.unpack(inputs, axis=1)

    cell = tf.contrib.rnn.python.ops.lstm_ops.LSTMBlockCell(num_units=num_units)
    print('cell state size', cell.state_size)

    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    initializer_params = initializer_params or {}
    initializer_params['dtype'] = dtype
    if isinstance(cell.state_size, tuple):
        initial_state = tuple(initializer_fn([batch_size, s]) for s in cell.state_size)
    else:
        initial_state = initializer_fn(shape=[batch_size, cell.state_size], **initializer_params)

    outputs, _, _ = tf.nn.rnn(
        cell,
        inputs_unpacked,
        initial_state=initial_state,
        dtype=dtype,
        scope=scope)

    outputs = tf.pack(outputs, axis=1)
    print('output shape', outputs.get_shape())

    return outputs