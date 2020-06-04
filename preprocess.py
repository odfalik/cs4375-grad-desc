import numpy as np
import pandas as pd
import requests, io

def main():

    # get DF containing raw data
    url = 'http://utdallas.edu/~oxf170130/cs4375-grad-desc/auto-mpg.csv'

    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),
        usecols=['mpg', 'weight', 'horsepower']
    )

    # split into training and testing datasets
    # split_mask = np.random.rand(len(df)) < 0.8

    df.to_csv('mpg_training.csv', index=False)


    # df[split_mask].to_csv('mpg_training.csv', index=False)
    # df[~split_mask].to_csv('mpg_testing.csv', index=False)

if __name__ == "__main__":
    main()