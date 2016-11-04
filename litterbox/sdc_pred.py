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

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from fabric import util, exec_predict
from fabric.dataset_file import DatasetFile
from models import ModelSdc
from processors import ProcessorSdc


class Challenge2Data(DatasetFile):
    # Example dataset for feeding folder of images into model

    def __init__(self, subset):
        super(Challenge2Data, self).__init__(
            'Challenge2', subset, types=('.png',))



def main(_):
    util.check_tensorflow_version()

    dataset = Challenge2Data(subset='')
    processor = ProcessorSdc()
    model_params = {
        'outputs': {'steer': 1},
        'network': 'resnet_v1_152',
    }
    model = ModelSdc(params=model_params)
    output, num_entries = exec_predict.predict(dataset, processor, model)
    filenames = []
    steering_angles = []
    for o in output:
        filenames.extend([int(os.path.splitext(os.path.basename(f))[0]) for f in o[1]])
        steering_angles.extend(np.squeeze(o[0]['steer']))
    # Dumps raw class probabilities to CSV file.
    columns = ['frame_id', 'steering_angle']
    df = pd.DataFrame(data={columns[0]: filenames, columns[1]: steering_angles}, columns=columns)
    df = df.head(num_entries)
    #df.Img = df.Img.apply(lambda x: os.path.basename(x.decode()))
    df.to_csv('./output.csv', index=False)

if __name__ == '__main__':
    tf.app.run()