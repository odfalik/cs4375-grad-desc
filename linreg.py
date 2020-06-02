import numpy as np

class LinReg(object):

    def __init__(self, sample_df, learning_rate, iterations):
        self.sample_df = sample_df
        self.num_samples = len(sample_df)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def calcMSE(self, err_vector):
        return np.matmul( err_vector.transpose(), err_vector ) / 2 * self.num_samples

    def calcErr(self, x, w, y):
        return np.subtract( np.matmul(x, w), y )

    def getHypothesizedValue(self, weights, point):
        return weights * point

    def train(self):

        true_values = self.sample_df.area.to_numpy()    # vector of true area values
        self.sample_df = self.sample_df.drop(columns='area')      # data points, remove area column from data
        sample_mat = self.sample_df.to_numpy()

        for i in range(self.iterations):
            weights = np.random.random_sample(len(sample_mat[0]))   # initialize weights vector for each iteration


            while(True):   # descent stopping conditions

                err_vector = self.calcErr(sample_mat, weights, true_values)

                for sample in sample_mat:

                    new_weights = np.empty(1)
                    for old_weight in weights:

                        sum_thing = 0       # TODO clean up
                        for j in range(0, np.shape(sample_mat)[1]):  # iterate on each attribute
                            sum_thing += err_vector[j] * sample[j]
                        gradient = sum_thing / self.num_samples

                    new_weight = old_weight - self.learning_rate * gradient
                    weights = np.append(new_weights, new_weight)

                print(self.calcMSE(err_vector))
                # if (calcMSE(err_vector) < )
