import numpy as np
import pandas as pd
import requests, io, sys
import part1, part2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data_url    = 'http://utdallas.edu/~oxf170130/cs4375-grad-desc/auto-mpg.csv'
regressors  = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']
regressand  = 'mpg'
draw_plots  = not (len(sys.argv) >= 2 and sys.argv[1] == 'noplot')

# retrieves dataset and performs any necessary preprocessing
def getDatasets(regressors, regressand):
    try:
        # get DF containing raw data
        raw_data = io.StringIO(requests.get(data_url).content.decode('utf-8'))

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


# presents 3d scatter plot of select dimensions, along with both models' 
def plotRegressions(datasets, p1_weights_v, p2_weights_v):

    testing_x_df = datasets['testing_x_df']

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        xs=testing_x_df[:,0],
        ys=testing_x_df[:,1],
        zs=testing_x_df[:,4]
    )
    plt.show()


def main():

    datasets = getDatasets(regressors, regressand)

    # Part 1 Training and Testing
    print('P1 -- Start')
    p1_model =                          part1.LinRegModel()
    p1_training_mse, p1_weights_v =     part1.train(p1_model, datasets)
    p1_testing_mse =                    part1.test(p1_model, datasets)
    print('P1 -- Testing MSE: %.2f with weights %s' % (p1_testing_mse, p1_weights_v))


    # Part 2 Training and Testing
    print('P2 -- Start')
    p2_model =                          linear_model.LinearRegression()
    p2_training_mse, p2_weights_v =     part2.train(p2_model, datasets)
    p2_testing_mse =                    part2.test(p2_model, datasets)
    print('P2 -- Testing MSE: %.2f with weights %s' % (p2_testing_mse, p2_weights_v))

    # if (draw_plots):
    #     plotRegressions(datasets, p1_weights_v, p2_weights_v)   # visualize regression lines on testing data



if __name__ == "__main__":
    main()