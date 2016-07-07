# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to a data set.

Methods of Data class:
  data_files: Returns a python list of all (sharded) data set files.
  num_examples_per_epoch: Returns the number of examples in the data set.
  num_classes: Returns the number of classes in the data set.
  reader: Return a reader for a single entry from the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from collections import Counter
#import pandas as pd
from abc import ABCMeta
from abc import abstractmethod

from .dataset import Dataset

FLAGS = tf.app.flags.FLAGS

def get_image_files_and_labels(folder, types=('.jpg', '.jpeg')):
    label_counts = Counter()
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        label = os.path.relpath(root, folder) if (root != folder) else ''
        for f in files:
            if os.path.splitext(f)[1].lower() in types:
                label_counts.update([label])
                filenames.append(os.path.join(root,f))
                labels.append(label)
    return label_counts, labels, filenames


class DatasetFile(Dataset):
    """A simple class for handling file (non-record) data sets."""
    metaclass__ = ABCMeta

    def __init__(self, name, subset, add_background_class=False):
        """Initialize dataset using a subset and the path to the data."""
        super(DatasetFile, self).__init__(name, subset, is_record=False)
        self.file_folder = os.path.join(FLAGS.data_dir, subset)
        self.label_counts, self.image_label_names, self.image_filenames = \
            get_image_files_and_labels(self.file_folder)
        self.num_examples = sum(self.label_counts.values())

        self.label_names = []
        if add_background_class:
            self.label_names += ['background']
            self.has_background_class = True

        # Generate label mappings
        # TODO This could be passed in if defined externally?
        # NOTE Currently assumes lexical order for labels (aside from 'background')
        self.label_names += sorted(self.label_counts.keys())
        self.label_name_to_index = {v: k for (k, v) in enumerate(self.label_names)}
        self.image_label_indices = [self.label_name_to_index[x] for x in self.image_label_names]

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        return self.num_examples

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validations', 'test', '']

    def data_files(self):
        """Returns a python list of all data files.

        Returns:
          python list of all data set files.
        Raises:
          ValueError: if there are no data files matching the subset.
        """
        return self.image_filenames

    def label_names(self):
        """Return label names for list of files"""
        return self.image_label_names

    def label_indices(self):
        """Return label indices for list of files"""
        return self.image_label_indices

    def reader(self):
        """Return a reader for a single entry from the data set.

        See io_ops.py for details of Reader class.

        Returns:
          Reader object that reads the data set.
        """
        return tf.WholeFileReader()
