import numpy as np
import pandas as pd
from urllib.request import Request, urlopen
import requests, io

def main():

    # get DF containing raw data
    url = 'http://utdallas.edu/~oxf170130/cs4375-grad-desc/forestfires.csv'
    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),
        usecols = ['month', 'day', 'wind', 'rain', 'area']  # filter data for only these columns
    )

    # convert months and days to numerical values
    month_map = { 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12 }
    day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7 }
    df.month = df.month.map(month_map)
    df.day = df.day.map(day_map)

    # split into training and testing datasets
    split_mask = np.random.rand(len(df)) < 0.8

    df[split_mask].to_csv('forestfires_training.csv', index=False)
    df[~split_mask].to_csv('forestfires_testing.csv', index=False)





if __name__ == "__main__":
    main()