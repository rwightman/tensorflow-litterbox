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
import pandas as pd
from abc import ABCMeta
from abc import abstractmethod

from .dataset import Dataset

FLAGS = tf.app.flags.FLAGS


def get_image_paths(folder):
    file_list = []
    dir_list = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder)
        jpeg_files = [os.path.join(rel_path, f) for f in files if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg')]
        if rel_path and (jpeg_files or subdirs):
            dir_list.append(rel_path)
        if jpeg_files:
            file_list.append(jpeg_files)
    return file_list, dir_list[::-1]


class DatasetFile(Dataset):
    """A simple class for handling file (non-record) data sets."""
    metaclass__ = ABCMeta

    def __init__(self, name, subset):
        """Initialize dataset using a subset and the path to the data."""
        super(DatasetFile, self).__init__(name, subset, record=False)
        self.labels = {}
        self.num_examples = 0
        self.data = pd.DataFrame()

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return len(self.labels)

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        return self.num_examples

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'val']

    def data_files(self):
        """Returns a python list of all data files.

        Returns:
          python list of all data set files.
        Raises:
          ValueError: if there are no data files matching the subset.
        """

        return ['']

    def label_names(self):
        """Return label names for list of files"""
        return ['']


    def label_indices(self):
        """Return label indices for list of files"""
        return [0]


    def reader(self):
        """Return a reader for a single entry from the data set.

        See io_ops.py for details of Reader class.

        Returns:
          Reader object that reads the data set.
        """
        return tf.WholeFileReader()
