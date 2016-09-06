# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
import argparse
import numpy as np
import pandas as pd
import collections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', metavar='out_file', type=str, default='ensemble.csv')
    parser.add_argument('-w', '--weights', metavar='weight_file', type=str, default='')
    parser.add_argument('files', metavar='filename', type=str, nargs='*')
    args = parser.parse_args()
    args = vars(args)

    files = args['files']
    outfile = args['outfile']
    weights_file = args['weights']

    # use files in weights csv if it is defined, otherwise use filenames from positional cmd line args
    if weights_file:
        wf = pd.read_csv(weights_file, names=['file', 'weight'], header=None, index_col=False)
        files, weights = wf.file.tolist(), wf.weight.tolist()
    else:
        counted = collections.Counter(files)
        files, weights = counted.keys(), counted.values()

    frames = []
    for file, weight in zip(files, weights):
        df = pd.read_csv(file, header=0, index_col=False)
        df.iloc[:, 1:] = df.iloc[:, 1:] * weight
        frames.append(df)
    merged = pd.concat(frames)
    # group by first column which is assumed to be the image name/id across all models
    avg = merged.groupby(merged.iloc[:, 0]).sum()
    avg = avg / sum(weights)
    avg.to_csv(outfile)

if __name__ == "__main__":
    main()

