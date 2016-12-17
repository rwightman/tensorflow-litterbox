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
from feeds import FeedImagesWithLabels
from models import ModelSdc
from processors import ProcessorSdc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'root_network', 'resnet_v1_50',
    """Either resnet_v1_50, resnet_v1_101, resnet_v1_152, inception_resnet_v2, nvidia_sdc""")

tf.app.flags.DEFINE_integer(
    'top_version', 5,
    """Top level network version, specifies output layer variations. See model code.""")


class Challenge2Data(DatasetFile):
    # Example dataset for feeding folder of images into model

    def __init__(self, subset):
        super(Challenge2Data, self).__init__(
            'Challenge2', subset, types=('.png', '.jpg'))


def main(_):
    util.check_tensorflow_version()

    processor = ProcessorSdc()
    #processor.mu_law_steering = True
    #processor.standardize_labels = False

    feed = FeedImagesWithLabels(
        dataset=Challenge2Data(subset=''),
        processor=processor)

    model_params = {
        'outputs': {
            'steer': 1,
        #    'xyz': 2,
        },

        'network': FLAGS.root_network,
        'version': FLAGS.top_version,
        'bayesian': False,
    }
    model = ModelSdc(params=model_params)

    output, num_entries = exec_predict.predict(feed, model)

    filenames = []
    steering_angles = []
    coords = []
    for o in output:
        filenames.extend([int(os.path.splitext(os.path.basename(f))[0]) for f in o[1]])
        if 'steer' in o[0]:
            steering_angles.extend(np.squeeze(o[0]['steer'], axis=1))
        if 'xyz' in o[0]:
            coords.extend(o[0]['xyz'])
    if coords:
        coords = np.vstack(coords)

    if steering_angles:
        columns_ang = ['frame_id', 'steering_angle']
        df_ang = pd.DataFrame(data={columns_ang[0]: filenames, columns_ang[1]: steering_angles}, columns=columns_ang)
        df_ang = df_ang.head(num_entries).sort_values(by='frame_id')
        df_ang.to_csv('./output_angle.csv', index=False)

    if isinstance(coords, np.ndarray) and coords.shape[0]:
        columns_loc = ['frame_id', 'longitude', 'latitude']
        df_loc = pd.DataFrame(
            data={columns_loc[0]: filenames, columns_loc[1]: coords[:, 0], columns_loc[2]: coords[:, 1]},
            columns=columns_loc)
        df_loc = df_loc.head(num_entries)
        df_loc = df_loc.sort_values(by='frame_id')
        df_loc.to_csv('./output_coords.csv', index=False)
        df_loc.ix[:, -2:].to_csv('./output_coords_only.csv', index=False)

if __name__ == '__main__':
    tf.app.run()