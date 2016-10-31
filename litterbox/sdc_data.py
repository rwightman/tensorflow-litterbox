# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""StateFarm data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fabric.dataset_record import DatasetRecord


class SdcData(DatasetRecord):
    """StateFarm data set."""

    def __init__(self, subset):
        super(SdcData, self).__init__('sdc', subset)

    def num_classes(self):
        return 0

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        #FIXME figure out counts for final train/val datasets
        if self.subset == 'train':
            return 1200000
        elif self.subset == 'validation':
            return 0

