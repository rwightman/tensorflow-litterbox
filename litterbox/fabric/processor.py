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
import abc


class Processor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def parse_example(self, serialized_example):
        pass

    def process_data(self, data):
        pass
