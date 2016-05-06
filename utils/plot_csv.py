import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    input_path = args.input

    df = pd.read_csv(input_path, header=0, index_col=False).ix[:,1:]

    plt.figure()

    df.plot(secondary_y=['TrainLoss', 'TestLoss'])

    plt.savefig('blah.png')


if __name__ == "__main__":
    main()

