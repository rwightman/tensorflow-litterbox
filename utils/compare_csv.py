import argparse
import numpy as np
import pandas as pd


def calc_rmse(predictions, targets):
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='filename', type=str, nargs='*')
    args = parser.parse_args()
    args = vars(args)

    files = args['files']

    assert len(files) == 2

    targets_df = pd.read_csv(files[0], header=0, index_col=False)
    predict_df = pd.read_csv(files[1], header=0, index_col=False)

    column = targets_df.columns[1]

    targets = targets_df.as_matrix(columns=[column])

    #predict_df[column] = pd.ewma(predict_df[column], com=1, adjust=False)
    predictions = predict_df.as_matrix(columns=[column])

    rmse, mse = calc_rmse(predictions, targets)
    print("RMSE: %f, MSE: %f" % (rmse, mse))

if __name__ == "__main__":
    main()

