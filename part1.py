import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class LinRegModel(object):

    def __init__(self):
        pass

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


    def train(self, training_x_df, training_y_df, iterations=1, learning_rate=1, delta_weight_threshold=.00001):
        true_v = training_y_df.to_numpy().reshape(-1, 1)                # vector of true area values
        data_m = training_x_df.to_numpy()                               
        data_m = np.hstack((np.ones((len(data_m),1)), data_m))          # matrix of data points

        plot_data = np.zeros((1, data_m.shape[1] + 2))
        
        for iteration in range(iterations):

            weights_v = (np.random.random_sample((len(data_m[0]), 1)) - (1/2)) / 5  # randomly initialize weights vector
            err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)            # calculate error vector
            # plot_data = np.vstack((plot_data, np.hstack(( weights_v.T, np.array([self.calcMSE(err_v), 0]).reshape(1,2) ))))

            for step in range(10000000):

                old_weights_v = np.array(weights_v, copy=True)
                old_MSE = self.calcMSE(err_v)

                new_weights_v = np.zeros((len(weights_v), 1)) # zeroed vector of weights
                for i, xi_v in enumerate(data_m.transpose()):
                    new_weights_v[i] = self.getNewWeight( old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v )
                weights_v = new_weights_v
                err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcMSE(err_v)

                delta_weights_v = np.absolute(old_weights_v - new_weights_v)
                if ((delta_weights_v < delta_weight_threshold).all()):
                    iter_MSE = new_MSE
                    break
                elif (new_MSE > old_MSE):                   # if gradient ascent by overstepping
                    learning_rate = learning_rate * 0.7     # lower learning rate
                    weights_v = old_weights_v               # revert weights
                else:
                    plot_data = np.vstack((plot_data, np.hstack(( np.array([iteration, new_MSE]).reshape(1,2), weights_v.T ))))
                    learning_rate = learning_rate * 1.1     # accelerate learning rate
            

        
        # After all iterations

        plot_data = np.delete(plot_data, obj=0, axis=0)         # remove zeros row from plot_data
        MSE_col = plot_data[:,1]
        min_MSE_index = np.where(MSE_col == np.amin(MSE_col))  # index of row of plot data with minimum MSE
        min_MSE_row = plot_data[min_MSE_index]
        min_MSE = min_MSE_row[0,1]
        self.weights_v = min_MSE_row[0,2:]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            xs=plot_data[:,6],  # weight
            ys=plot_data[:,3],  # weight
            zs=plot_data[:,1],  # MSE
            c=plot_data[:,0],   # iteration
            cmap='Set1'
        )
        plt.show()

    def test(self, testing_x_df, testing_y_df):
        true_v = testing_y_df.to_numpy().reshape(-1, 1)                # vector of true area values
        data_m = testing_x_df.to_numpy()
        data_m = np.hstack((np.ones((len(data_m),1)), data_m))          # matrix of data points
        predicted_v = data_m.dot(self.weights_v).reshape(-1, 1)
        mse = self.calcMSE(predicted_v - true_v)
        return predicted_v