# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_file', '../output/output-best.csv',
    """Path to output csv to boost""")

tf.app.flags.DEFINE_string(
    'output_file', '../output/output-boost.csv',
    """Path to output file""")

def boost(vals):
    vals_arr = vals.ix[1:].values
    vals_sort = np.argsort(vals_arr)
    sorted = vals.ix[1:][vals_sort][::-1].values
    topn_sum = np.sum(sorted[0:3])

    out_arr = np.zeros(len(vals) -1)
    for idx in vals_sort[::-1][0:3]:
        out_arr[idx] = vals_arr[idx]/topn_sum
    vals[1:] = out_arr

    return vals



def main():

    input_path = FLAGS.input_file

    orig = pd.read_csv(input_path, header=0, index_col=False)

    boosted = orig.apply(boost, axis=1)

    boosted.to_csv(FLAGS.output_file, index=False)


if __name__ == "__main__":
    main()

