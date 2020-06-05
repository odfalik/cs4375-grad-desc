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


    def train(self, training_x_df, training_y_df, iterations=1, learning_rate=1):
        true_v = training_y_df.to_numpy().reshape(-1, 1)                # vector of true area values
        data_m = training_x_df.to_numpy()                               # matrix of data points
        data_m = np.hstack((np.ones((len(data_m),1)), data_m))

        plot_data = np.zeros((1,5))
        
        for iteration in range(iterations):

            weights_v = (np.random.random_sample((len(data_m[0]), 1)) - (1/2)) / 5  # initialize weights vector to random value [0.0, 1.0)


            err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
            # plot_data = np.vstack((plot_data, np.hstack(( weights_v.T, np.array([self.calcMSE(err_v), 0]).reshape(1,2) ))))


            for step in range(10000000):

                old_weights_v = np.array(weights_v, copy=True)
                old_MSE = self.calcMSE(err_v)
                # if (step == 0): print(f'init weights:{weights_v} MSE:{old_MSE}')
                new_weights_v = np.zeros((len(weights_v), 1)) # empty vector of weights

                for i, xi_v in enumerate(data_m.transpose()):
                    new_weights_v[i] = self.getNewWeight( old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v )
                weights_v = new_weights_v
                err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcMSE(err_v)


                plot_data = np.vstack((plot_data, np.hstack(( weights_v.T, np.array([new_MSE, iteration]).reshape(1,2) ))))

                if (new_MSE < 90):
                    # print(f'END CONDITION with MSE:{new_MSE} after {step} steps. weights_v:{weights_v}')
                    break
                elif (new_MSE > old_MSE):
                    # print(f'Gradient ascent with MSE:{new_MSE} after {step} steps')
                    learning_rate = learning_rate * 0.7
                    weights_v = old_weights_v
                else:
                    learning_rate = learning_rate * 1.1
                    pass
        
        # After all iterations
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plot_data = np.delete(plot_data, obj=0, axis=0)
        ax.scatter3D(plot_data[:,1], plot_data[:,2], plot_data[:,3], c=plot_data[:,4], cmap='Set1')
        plt.show()

    def test(self, df, y_col_name):
        pass

def main(datasets):
    model = LinRegModel()
    model.train(
        training_x_df=datasets['training_x_df'],
        training_y_df=datasets['training_y_df'],
        iterations=15,
        learning_rate=0.000000005
    )