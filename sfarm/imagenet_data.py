#
"""StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fabric.dataset_record import DatasetRecord
from fabric.dataset_file import DatasetFile


class ImagenetData(DatasetRecord):
    """StateFarm data set."""

    def __init__(self, subset):
        super(ImagenetData, self).__init__('Imagenet', subset)
        self.has_background_class = True

    def num_classes(self):
        return 1000

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        if self.subset == 'train':
            return 1281167
        elif self.subset == 'validation':
            return 50000

