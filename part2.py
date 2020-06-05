import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def main(datasets):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(datasets['training_x_df'], datasets['training_y_df'])

    # Make predictions using the testing set
    predictions = regr.predict(datasets['testing_x_df'])

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
        % mean_squared_error(datasets['testing_y_df'], predictions))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
        % r2_score(datasets['testing_y_df'], predictions))

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
