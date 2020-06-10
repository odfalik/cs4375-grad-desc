import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def main(datasets):
    regr = linear_model.LinearRegression()

    regr.fit(datasets['training_x_df'], datasets['training_y_df'])

    predictions = regr.predict(datasets['testing_x_df'])

    # print('Weights: \n', regr.coef_[0])
    weights = regr.intercept_ + regr.coef_[0]
    print(f'MSE: %.2f with weights {regr.coef_[0]}' % mean_squared_error(datasets['testing_y_df'], predictions))

    # print('Coefficient of determination: %.2f' % r2_score(datasets['testing_y_df'], predictions))

    # Plot outputs
    # plt.scatter(datasets['training_x_df'], datasets['training_y_df'],  color='black')
    # plt.plot(datasets['testing_x_df'], predictions, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()
    
    # After all iterations
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(datasets['training_x_df'][['weight']], datasets['training_x_df'][['horsepower']], datasets['training_y_df'], cmap='Set1')
    plt.show()
