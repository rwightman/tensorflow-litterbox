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
from fabric import exec_train
from imagenet_data import ImagenetData
from models import ModelInception
from models import ModelGoogle
from processors import ProcessorImagenet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'validation', 'train', 'test'""")

def main(_):
    util.check_tensorflow_version()

    processor = ProcessorImagenet()

    dataset = ImagenetData(subset=FLAGS.subset)
    model = ModelGoogle(network='resnet_v1_152')
    #model = ModelInception(variant=ModelInception.Variant.ResnetV2)
    #model = ModelResnet(num_layers=50, width_factor=1)
    #model = ModelVgg(19)

    exec_train.train(dataset, processor=processor, model=model)

if __name__ == '__main__':
    tf.app.run()
