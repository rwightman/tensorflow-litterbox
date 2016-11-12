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
from fabric import util
from fabric import exec_eval
from sdc_data import SdcData
from models import ModelSdc
from processors import ProcessorSdc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation', 'train', 'test'""")


def main(_):
    util.check_tensorflow_version()

    dataset = SdcData(subset=FLAGS.subset)
    processor = ProcessorSdc()
    model_params = {
        'outputs': {'steer': 1},

        #'network': 'resnet_v1_152',
        #'version': 1,

        #'network': 'nvidia_sdc',   # 160x120
        #'version': 2,

        #'network': 'resnet_v1_101',  # 192x128
        #'version': 3,

        'network': 'resnet_v1_50',  # 192x128
        'version': 3,

        #'network': 'inception_resnet_v2',  # 199x149
        #'version': 3,
    }
    model = ModelSdc(params=model_params)
    exec_eval.evaluate(dataset, processor, model)

if __name__ == '__main__':
    tf.app.run()
