from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf

from inception.feed import Feed

class FeedWholeFile(Feed):
    def __init__(self):
        super().__init__()
        print("FeedWholeFile")