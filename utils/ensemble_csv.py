import argparse
import numpy as np
import pandas as pd
import collections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', metavar='out_file', type=str, default='ensemble.csv')
    parser.add_argument('-w', '--weights', metavar='weight_file', type=str, default='')
    parser.add_argument('-m', '--mean', type=str, default='arith')
    parser.add_argument('files', metavar='filename', type=str, nargs='*')
    args = parser.parse_args()
    args = vars(args)

    files = args['files']
    outfile = args['outfile']
    weights_file = args['weights']
    mean_type = args['mean']

    if weights_file:
        wf = pd.read_csv(weights_file, names=['file', 'weight'], header=None, index_col=False)
        files, weights = wf.file.tolist(), wf.weight.tolist()
    else:
        counted = collections.Counter(files)
        files, weights = counted.keys(), counted.values()

    frames = []
    for file, weight in zip(files, weights):
        df = pd.read_csv(file, header=0, index_col=False)
        if mean_type == 'geom':
            df.iloc[:, 1:] = df.iloc[:, 1:].pow(weight)
        else:
            df.iloc[:, 1:] = df.iloc[:, 1:] * weight
        frames.append(df)
    
    merged = pd.concat(frames)
    weights_sum = sum(weights)
    if mean_type == 'geom':
        result = merged.groupby(merged.Img).prod()
        result = result.pow(1/weights_sum)
        result = result.div(result.sum(axis=1), axis=0)
    else:
        result = merged.groupby(merged.Img).sum()
        result = result / weights_sum
    
    result.to_csv(outfile)

if __name__ == "__main__":
    main()

