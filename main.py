import numpy as np
import pandas as pd
import requests, io
import part1, part2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# retrieves dataset and performs any necessary preprocessing
def getDatasets(regressors, regressand):
    try:
        # get DF containing raw data
        url = 'http://utdallas.edu/~oxf170130/cs4375-grad-desc/auto-mpg.csv'
        raw_data = io.StringIO(requests.get(url).content.decode('utf-8'))

        if (regressors and regressand):
            df = pd.read_csv(raw_data, sep=',', usecols=regressors+[regressand])
        else:
            df = pd.read_csv(raw_data, sep=',')

        # split into training and testing datasets
        split_mask = np.random.rand(len(df)) < 0.8
        return {
            'training_x_df': df[split_mask][regressors],
            'training_y_df': df[split_mask][[regressand]],
            'testing_x_df': df[~split_mask][regressors],
            'testing_y_df': df[~split_mask][[regressand]]
        }
    except:
        print('Unable to retrieve data')
        quit(code=1)

def main():
    regressors = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']
    regressand = 'mpg'
    datasets = getDatasets(regressors, regressand)


    # Part 1 Training and Testing
    p1_model = part1.LinRegModel()
    p1_model.train(
        training_x_df=datasets['training_x_df'],
        training_y_df=datasets['training_y_df'],
        iterations=3, learning_rate=0.000000005, delta_weight_threshold=0.000005
    )
    p1_model_predictions = p1_model.test(
        testing_x_df=datasets['testing_x_df'],
        testing_y_df=datasets['testing_y_df']
    )
    print(f'P1 -- MSE: %.2f with weights {p1_model.weights_v}' % mean_squared_error(datasets['testing_y_df'], p1_model_predictions))

    # Part 2 Training and Testing
    p2_model = linear_model.LinearRegression()
    p2_model.fit(datasets['training_x_df'], datasets['training_y_df'])
    p2_model_preditions = p2_model.predict(datasets['testing_x_df'])

    weights = p2_model.intercept_ + p2_model.coef_[0]
    print(f'P2 -- MSE: %.2f with weights {p2_model.coef_[0]}' % mean_squared_error(datasets['testing_y_df'], p2_model_preditions))
    

if __name__ == "__main__":
    main()