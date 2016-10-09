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

import tensorflow as tf
from fabric import util, exec_predict
from fabric.dataset_file import DatasetFile
from models import ModelInception, ModelGoogle


class ExampleData(DatasetFile):
    """StateFarm data set."""

    def __init__(self, subset):
        super(ExampleData, self).__init__('Example', subset)
        self.has_background_class = True

    def num_classes(self):
        return 1000


def main(_):
    util.check_tensorflow_version()

    dataset = ExampleData(subset='')
    model = ModelGoogle()
    output = exec_predict.predict(dataset, model)

    # Dumps raw class probabilities to CSV file.
    class_labels = []
    for c in range(dataset.num_classes()):
        class_labels.append("c%s" % c)
    df = pd.DataFrame(output, columns=['Img']+class_labels)
    df.Img = df.Img.apply(lambda x: os.path.basename(x.decode()))
    df.to_csv('./output.csv', index=False)

if __name__ == '__main__':
    tf.app.run()