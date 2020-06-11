import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random

class LinRegModel(object):

    def __init__(self, draw_plots):
        self.draw_plots = draw_plots

    def calcMSE(self, err_v):
        mse_m = err_v.transpose().dot(err_v) / (2 * len(err_v))
        return mse_m[0,0]

    def calcErrV(self, x_m, w_v, y_v):
        h_v = x_m.dot(w_v)
        err_v = h_v - y_v
        return err_v

    def getNewWeight(self, old_weight, learning_rate, err_v, xi_v):

        sum = 0
        for j, xi in enumerate(xi_v):           # loop over all measurements assoc w/ specific parameter
            sum += err_v[j].item() * xi         # sum that point's error times the point's dimension corresponding to weight

        gradient = sum / len(xi_v)
        new_weight = old_weight - learning_rate * gradient
        return new_weight


    def train(self, training_x_df, training_y_df, descents=1, learning_rate=1, delta_weight_threshold=.00001):
        self.regressands = list(training_x_df)
        self.regressor = list(training_y_df)[0]
        true_v = training_y_df.to_numpy().reshape(-1, 1)                # vector of true area values
        data_m = training_x_df.to_numpy()                               
        data_m = np.hstack((np.ones((len(data_m),1)), data_m))          # matrix of data points
        training_log = np.zeros((1, data_m.shape[1] + 3))               # initialize log
        
        for descent in range(descents):

            weights_v = (np.random.random_sample((len(data_m[0]), 1)) - (1/2)) / 5  # randomly initialize weights vector
            err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)            # calculate error vector
            # training_log = np.vstack((training_log, np.hstack(( weights_v.T, np.array([self.calcMSE(err_v), 0]).reshape(1,2) ))))

            for step in range(1000000):
                
                old_weights_v = np.array(weights_v, copy=True)
                old_err_v = err_v
                old_MSE = self.calcMSE(err_v)

                new_weights_v = np.zeros((len(weights_v), 1))   # zeroed vector of weights
                for i, xi_v in enumerate(data_m.transpose()):
                    new_weights_v[i] = self.getNewWeight( old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v )
                weights_v = new_weights_v
                err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcMSE(err_v)

                delta_weights_v = np.absolute(old_weights_v - new_weights_v)
                if ((delta_weights_v < delta_weight_threshold).all()):  # end condition
                    iter_MSE = new_MSE
                    break
                elif (new_MSE > old_MSE):                       # if gradient ascent by overstepping
                    learning_rate = learning_rate * 0.7         # lower learning rate
                    weights_v = old_weights_v                   # revert weights
                    err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                else:
                    if (step % 200 == 0):
                        training_log = np.vstack((training_log, np.hstack(( np.array([descent, step, new_MSE]).reshape(1,3), weights_v.T ))))
                    learning_rate = learning_rate * 1.1         # accelerate learning rate
            

        # After all descents
        # training log post processing
        training_log = np.delete(training_log, obj=0, axis=0)   # remove zeros row from training_log
        MSE_col = training_log[:,2]
        min_MSE_index = np.where(MSE_col == np.amin(MSE_col))   # index of row of plot data with minimum MSE
        min_MSE_row = training_log[min_MSE_index]
        min_MSE = min_MSE_row[0,1]
        self.weights_v = min_MSE_row[0,3:]                      # ideal weights selected from descent with lowest MSE

        if (self.draw_plots):
            
            # 3d gradient descent demonstration
            if (len(self.regressands) > 2):     # ensure enough descents and dimensions to plot
                fig1 = plt.figure(1)
                ax = plt.axes(projection='3d')
                weight_indices = (3, 4)         # selects two regressands whose weight to plot
                ax.set_xlabel('Weight ' + self.regressands[weight_indices[0]])
                ax.set_ylabel('Weight ' + self.regressands[weight_indices[1]])
                ax.set_zlabel('MSE')
                ax.scatter3D(
                    xs=training_log[:,weight_indices[0]],   # weight
                    ys=training_log[:,weight_indices[1]],   # weight
                    zs=training_log[:,2],                   # MSE
                    c=training_log[:,0],                    # descent
                    cmap='Set1'
                )

            # plot of MSE, all weights over steps for a single iteration
            if (descents == 1):
                fig2, (ax1, ax2) = plt.subplots(2)
                for idx, regressand in enumerate(self.regressands):
                    color = (random.random(), random.random(), random.random())
                    ax2.plot(training_log[:,1], training_log[:,3+idx], c=color, label=regressand+' weight')
                ax2.legend(loc='upper right')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Weight')
                ax1.set_ylabel('MSE')
                ax1.plot(training_log[:,1], training_log[:,2], 'b-', label='MSE')

            plt.show()

        return min_MSE

    def test(self, testing_x_df, testing_y_df):
        true_v = testing_y_df.to_numpy().reshape(-1, 1)                 # vector of true area values
        data_m = testing_x_df.to_numpy()
        data_m = np.hstack((np.ones((len(data_m),1)), data_m))          # matrix of data points
        predicted_v = data_m.dot(self.weights_v).reshape(-1, 1)
        mse = self.calcMSE(predicted_v - true_v)
        return predicted_v

def train(model, datasets):
    mse = model.train(
        training_x_df=datasets['training_x_df'],
        training_y_df=datasets['training_y_df'],
        descents=1, learning_rate=0.0000000001, delta_weight_threshold=0.0000001  # training hyperparameters
    )
    return mse, model.weights_v

def test(model, datasets):
    predictions_v = model.test(
        testing_x_df=datasets['testing_x_df'],
        testing_y_df=datasets['testing_y_df']
    )
    return model.calcMSE(datasets['testing_y_df'].to_numpy())
