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


def select_split(inputs, split_index=None):
    if split_index is None:
        return inputs
    inputs_out = []
    for x in inputs:
        if isinstance(x, list):
            assert x
            if isinstance(x[0], list):
                inputs_out.append([t[split_index] for t in x])
            else:
                inputs_out.append(x[split_index])
        elif isinstance(x, dict):
            inputs_out.append({tk: tv[split_index] for tk, tv in x})
        else:
            assert False, 'Unexpected split format, ' \
                          'expecting list of splits, list of list of splits, or dict of splits'
    return inputs_out


class Processor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def parse_example(self, serialized_example):
        pass

    @abc.abstractmethod
    def process_example(self, data, mode):
        pass
