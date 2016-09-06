# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
""" Predict classes for test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os

from inception import ModelInception
from resnet import ModelResnet
from vgg import ModelVgg16
from fabric.predict import *
from sfarm_data import *

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    dataset = StateFarmDataFile(subset='')
    #model = ModelVgg16()
    model = ModelResnet(num_layers=16, width_factor=2)
    assert dataset.data_files()
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)

    output = predict(dataset, model)

    df = pd.DataFrame(output, columns=['Img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.Img = df.Img.apply(lambda x: os.path.basename(x.decode()))
    df.to_csv('./output3.csv', index=False)

if __name__ == '__main__':
    tf.app.run()
