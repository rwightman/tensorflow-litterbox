# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Optimizer(object):
    def __init__(self):
        print('blah')


class OptimizerSGD(Optimizer):
    def __init__(self):
        super(OptimizerSGD, self).__init__()


class OptimizerRMSProp(Optimizer):
    def __init__(self):
        super(OptimizerRMSProp, self).__init__()


class OptimizerAdaGrad(Optimizer):
    def __init__(self):
        super(OptimizerAdaGrad, self).__init__()