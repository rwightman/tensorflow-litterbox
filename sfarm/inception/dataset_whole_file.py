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


class DatasetWholeFile(Dataset):
  """A simple class for handling data sets."""
  __metaclass__ = ABCMeta

  def __init__(self, name, subset):
    """Initialize dataset using a subset and the path to the data."""
    super(Dataset, self).__init__(name, subset)

  @abstractmethod
  def num_classes(self):
    """Returns the number of classes in the data set."""
    pass

  @abstractmethod
  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    pass

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
    file_pattern = os.path.join(FLAGS.data_dir, '%s/*' % self.subset)
    data_files = tf.gfile.Glob(file_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        FLAGS.data_dir))

      exit(-1)
    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    return tf.WholeFileReader()
