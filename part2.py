import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def train(model, datasets):
    model.fit(datasets['training_x_df'], datasets['training_y_df'])
    predictions = model.predict(datasets['training_x_df'])
    mse = mean_squared_error(datasets['training_y_df'], predictions)
    return mse, np.hstack((model.intercept_, np.array(model.coef_[0]))).reshape(-1, 1)


def test(model, datasets):
    predictions = model.predict(datasets['testing_x_df'])
    return mean_squared_error(datasets['testing_y_df'], predictions)