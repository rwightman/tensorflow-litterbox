# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import os
from copy import deepcopy
from fabric import util
from models import ModelSdc
from processors import ProcessorSdc
from collections import defaultdict


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'root_network', 'resnet_v1_50',
    """Either resnet_v1_50, resnet_v1_101, resnet_v1_152, inception_resnet_v2, nvidia_sdc""")

tf.app.flags.DEFINE_integer(
    'top_version', 5,
    """Top level network version, specifies output layer variations. See model code.""")

tf.app.flags.DEFINE_boolean(
    'bayesian', False, """Activate dropout layers for inference.""")

tf.app.flags.DEFINE_integer(
    'samples', 0, """Activate dropout layers for inference.""")

tf.app.flags.DEFINE_string(
    'checkpoint_path', '', """Checkpoint file for model.""")

tf.app.flags.DEFINE_string(
    'ensemble_path', '', """CSV file with ensemble specification. Use as alternative to single model checkpoint.""")

tf.app.flags.DEFINE_string(
    'name', 'model', """Name prefix for outputs of exported artifacts.""")


def _weighted_mean(outputs_list, weights_tensor):
    assert isinstance(outputs_list[0], tf.Tensor)
    print(outputs_list)
    outputs_tensor = tf.concat(1, outputs_list)
    print('outputs concat', outputs_tensor.get_shape())
    if len(outputs_list) > 1:
        weighted_outputs = outputs_tensor * weights_tensor
        print('weighted outputs ', weighted_outputs.get_shape())
        outputs_tensor = tf.reduce_mean(weighted_outputs)
    else:
        outputs_tensor = tf.squeeze(outputs_tensor)
    return outputs_tensor


def _merge_outputs(outputs, weights):
    assert outputs

    merged = defaultdict(list)
    weights_tensor = tf.pack(weights)
    print('weights ', weights_tensor.get_shape())

    # recombine multiple model outputs by dict key or list position under output name based dict
    if isinstance(outputs[0], dict):
        for o in outputs:
            for name, tensor in o.items():
                merged['output_%s' % name].append(tensor)
    elif isinstance(outputs[0], list):
        for o in outputs:
            for index, tensor in enumerate(o):
                merged['output_%d' % index].append(tensor)
    else:
        merged['output'] = outputs

    reduced = {name: _weighted_mean(value_list, weights_tensor) for name, value_list in merged.items()}
    for k, v in reduced.items():
        print(k, v, v.get_shape())

    return reduced


def build_export_graph(models, batch_size=1, export_scope=''):
    assert models

    inputs = tf.placeholder(tf.uint8, [None, None, 3], name='input_placeholder')
    print("Graph Inputs: ")
    print(inputs.name, inputs.get_shape())

    with tf.device('/gpu:0'):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.div(inputs, 255)
        input_tensors = [inputs, tf.zeros(shape=()), tf.constant('', dtype=tf.string)]
        model_outputs_list = []
        weights_list = []
        for m in models:
            with tf.variable_scope(m['name'], values=input_tensors):
                model, processor = m['model'], m['processor']
                processed_inputs = processor.process_example(input_tensors, mode='pred')
                if batch_size > 1:
                    processed_inputs = [tf.gather(tf.expand_dims(x, 0), [0] * batch_size) for x in processed_inputs]
                processed_inputs = processor.reshape_batch(processed_inputs, batch_size=batch_size)
                model_outputs = model.build_tower(
                    processed_inputs[0], is_training=False, summaries=False)
                model_outputs_list += [model.get_predictions(model_outputs, processor)]
                weights_list += [m['weight']]

        merged_outputs = _merge_outputs(model_outputs_list, weights_list)

        print("Graph Outputs: ")
        outputs = []
        for name, output in merged_outputs.items():
            outputs += [tf.identity(output, name)]
        [print(x.name, x.get_shape()) for x in outputs]

        return inputs, outputs


def main(_):
    util.check_tensorflow_version()
    assert os.path.isfile(FLAGS.checkpoint_path) or os.path.isfile(FLAGS.ensemble_path)

    model_args_list = []
    if FLAGS.checkpoint_path:
        model_args_list.append(
            {
                'root_network': FLAGS.root_network,
                'top_version': FLAGS.top_version,
                'image_norm': FLAGS.image_norm,
                'image_size': FLAGS.image_size,
                'image_aspect': FLAGS.image_aspect,
                'checkpoint_path': FLAGS.checkpoint_path,
                'bayesian': FLAGS.bayesian,
                'weight': 1.0,
            }
        )
    else:
        ensemble_df = pd.DataFrame.from_csv(FLAGS.ensemble_path, index_col=None)
        model_args_list += ensemble_df.to_dict('records')

    model_params_common = {
        'outputs': {
            'steer': 1,
        #    'xyz': 2,
        },
    }
    model_list = []
    for i, args in enumerate(model_args_list):
        print(args)
        model_name = 'model_%d' % i
        model_params = deepcopy(model_params_common)
        model_params['network'] = args['root_network']
        model_params['version'] = args['top_version']
        model_params['bayesian'] = FLAGS.bayesian
        model = ModelSdc(params=model_params)

        processor_params = {}
        processor_params['image_norm'] = args['image_norm']
        processor_params['image_size'] = args['image_size']
        processor_params['image_aspect'] = args['image_aspect']
        processor = ProcessorSdc(params=processor_params)

        model_list.append({
            'model': model,
            'processor': processor,
            'weight': args['weight'],
            'name': model_name,
            'checkpoint_path': args['checkpoint_path']
        })

    name_prefix = FLAGS.name
    with tf.Graph().as_default() as g:
        batch_size = 1 if not FLAGS.samples else FLAGS.samples
        build_export_graph(models=model_list, batch_size=batch_size)
        model_variables = tf.contrib.framework.get_model_variables()
        saver = tf.train.Saver(model_variables)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            sess.run(init_op)

            g_def = g.as_graph_def(add_shapes=True)
            tf.train.write_graph(g_def, './', name='%s-graph_def.pb.txt' % name_prefix)

            for m in model_list:
                checkpoint_variable_set = set()
                checkpoint_path, global_step = util.resolve_checkpoint_path(m['checkpoint_path'])
                if not checkpoint_path:
                    print('No checkpoint file found at %s' % m['checkpoint_path'])
                    return
                reader = tf.train.NewCheckpointReader(checkpoint_path)
                checkpoint_variable_set.update(reader.get_variable_to_shape_map().keys())
                variables_to_restore = m['model'].variables_to_restore(
                    restore_outputs=True,
                    checkpoint_variable_set=checkpoint_variable_set,
                    prefix_scope=m['name'])

                saver_local = tf.train.Saver(variables_to_restore)
                saver_local.restore(sess, checkpoint_path)
                print('Successfully loaded model from %s at step=%d.' % (checkpoint_path, global_step))

            saver.export_meta_graph('./%s-meta_graph.pb.txt' % name_prefix, as_text=True)
            saver.save(sess, './%s-checkpoint' % name_prefix, write_meta_graph=True)

if __name__ == '__main__':
    tf.app.run()

