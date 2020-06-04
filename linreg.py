import numpy as np

class LinRegModel(object):

    def __init__(self):
        pass

    def calcMSE(self, err_v):
        return err_v.transpose().dot(err_v) / (2 * len(err_v))

    def getHypothesizedValue(self, weights, point):
        return weights * point

    def calcErr(self, x_m, w_v, y_v):
        h_v = x_m.dot(w_v)
        return h_v - y_v

    def getNewWeight(self, old_weight, learning_rate, err_v, xi_v):

        sum = 0
        for j, xi in enumerate(xi_v):           # loop over all measurements assoc w/ specific parameter
            sum += err_v[j] * xi                # sum that point's error times the point's dimension corresponding to weight

        gradient = sum / len(xi_v)
        return old_weight - learning_rate * gradient


    def train(self, df, y_attr_name, iterations=1, learning_rate=1):
        true_v = df[y_attr_name].to_numpy().reshape(-1, 1)              # vector of true area values
        data_m = df.drop(columns=y_attr_name).to_numpy()                # matrix of data points
        
        # print(data_m)

        for iteration in range(iterations):

            weights_v = np.random.random_sample((len(data_m[0]), 1))            # initialize weights vector to random value [0.0, 1.0)
            print(f'weights_v init: {weights_v}')

            # while(True):    # descent stopping condition
            for step in range(100000): # TEMP TODO switch to legitimate stopping condition
                
                err_v = self.calcErr(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_weights_v = np.empty((len(weights_v), 1)) # empty vector of weights

                for i, xi_v in enumerate(data_m.transpose()):
                    print(f'i:{i}, xi_v:{xi_v}, old_weight:{weights_v[i]}')
                    np.append(
                        new_weights_v,
                        self.getNewWeight(
                            old_weight=weights_v[i],
                            learning_rate=learning_rate,
                            err_v=err_v,
                            xi_v=xi_v
                        )
                    )
                weights_v = new_weights_v

                if (step % 200 == 0):
                    print(f'i:{iteration} weights:{weights_v} MSE:{self.calcMSE(err_v)}')





    def test(self, df, y_col_name):
        pass