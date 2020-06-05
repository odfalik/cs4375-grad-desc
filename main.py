import numpy as np
import pandas as pd
import requests, io
from part1 import main as p1
from part2 import main as p2

# retrieves dataset and performs any necessary preprocessing
def preprocess(x_feature_names, y_feature_name):
    # get DF containing raw data
    url = 'http://utdallas.edu/~oxf170130/cs4375-grad-desc/auto-mpg.csv'

    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),
        usecols=x_feature_names+[y_feature_name]
    )

    # split into training and testing datasets
    split_mask = np.random.rand(len(df)) < 0.8
    
    return {
        'training_x_df': df[split_mask][x_feature_names],
        'training_y_df': df[split_mask][[y_feature_name]],
        'testing_x_df': df[~split_mask][x_feature_names],
        'testing_y_df': df[~split_mask][[y_feature_name]]
    }

def main():
    x_feature_names = ['weight', 'horsepower']
    y_feature_name = 'mpg'

    datasets = preprocess(x_feature_names, y_feature_name)

    p1(datasets)

    # p2(datasets)
    

if __name__ == "__main__":
    main()