import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random, math

class LinRegModel(object):

    def __init__(self, draw_plots):
        self.draw_plots = draw_plots

    def drawPlots(self, training_log, descents):
        
        # 3d gradient descent demonstration
        if (len(self.regressands) > 2):     # ensure enough descents and dimensions to plot
            fig1 = plt.figure(1)
            ax_a1 = plt.axes(projection='3d')
            regressand_indices = (2, 3)     # selects two regressands whose weight to plot
            ax_a1.set_xlabel(self.regressands[regressand_indices[0]] + ' weight')
            ax_a1.set_ylabel(self.regressands[regressand_indices[1]] + ' weight')
            ax_a1.set_zlabel('MSE')

            def update(val):
                ax_a1.clear()
                regressand_indices = (int(slider1.val), int(slider2.val))
                ax_a1.set_xlabel(self.regressands[regressand_indices[0]] + ' weight')
                ax_a1.set_ylabel(self.regressands[regressand_indices[1]] + ' weight')
                ax_a1.set_zlabel('MSE')
                weight_a_v = training_log[:,5+regressand_indices[0]]
                weight_b_v = training_log[:,5+regressand_indices[1]]
                mse_col_v =  training_log[:,2]
                descent_v =  training_log[:,0]
                step_v =     training_log[:,1]
                if (descents == 1):
                    ax_a1.scatter3D(
                        xs=weight_a_v,  # weight
                        ys=weight_b_v,  # weight
                        zs=mse_col_v,   # MSE
                        c=step_v,       # step
                        cmap='plasma'
                    )
                    ax_a1.plot3D( xs=weight_a_v, ys=weight_b_v, zs=mse_col_v ) # connect the dots
                else:
                    ax_a1.scatter3D(
                        xs=weight_a_v,  # weight
                        ys=weight_b_v,  # weight
                        zs=mse_col_v,   # MSE
                        c=descent_v,    # descent or step
                        cmap='Set1'
                    )
                fig1.canvas.draw()

            axcolor = 'lightgoldenrodyellow'
            ax_regressand1 = plt.axes([0.1, 0.25, 0.1, 0.025], facecolor=axcolor)
            ax_regressand2 = plt.axes([0.1, 0.2, 0.1, 0.025], facecolor=axcolor)
            slider1 = Slider(ax_regressand1, 'Regressand 1', 0, len(self.regressands)-1, valinit=regressand_indices[0], valstep=1)
            slider2 = Slider(ax_regressand2, 'Regressand 2', 0, len(self.regressands)-1, valinit=regressand_indices[1], valstep=1)
            update(1)
            slider1.on_changed(update)
            slider2.on_changed(update)

        # plot of MSE, all weights over steps for a single descent
        if (descents == 1):
            fig2, (ax_b1, ax_b2, ax_b3) = plt.subplots(3, constrained_layout=True)
            for idx, regressand in enumerate(self.regressands):
                color = (random.random(), random.random(), random.random())
                ax_b2.plot(training_log[:,1], training_log[:,5+idx], c=color, label=regressand+' weight')
            ax_b2.plot(training_log[:,1], training_log[:,4], c=color, label='bias')   # plot bias
            ax_b2.legend(fontsize='small', bbox_to_anchor=(1.01,1), loc="upper left")
            ax_b2.set_xlabel('Step')
            ax_b2.set_ylabel('Weight')
            ax_b1.set_ylabel('MSE')
            ax_b1.plot(training_log[:,1], training_log[:,2], 'b-', label='MSE')
            ax_b1.set_title('Gradient Descent')
            ax_b3.plot(training_log[:,1], training_log[:,3], 'b-', label='Learning Rate')
            ax_b3.legend(fontsize='small', bbox_to_anchor=(1.01,1), loc="upper left")

        plt.show()

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
        training_log = np.zeros((1, data_m.shape[1] + 4))               # initialize log, later used for graphs and analysis
        best_MSE = math.inf
        
        for descent in range(descents):
            
            weights_v = (np.random.random_sample((len(data_m[0]), 1)) - (1/2)) / 5  # randomly initialize weights vector
            err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)            # calculate error vector
            # training_log = np.vstack((training_log, np.hstack(( weights_v.T, np.array([self.calcMSE(err_v), 0]).reshape(1,2) ))))

            step = 1
            while (step < 50000):
                
                old_weights_v = np.array(weights_v, copy=True)
                old_err_v = err_v
                old_MSE = self.calcMSE(err_v)

                new_weights_v = np.zeros((len(weights_v), 1))   # zeroed vector of weights
                for i, xi_v in enumerate(data_m.transpose()):   # adjust each weight
                    new_weights_v[i] = self.getNewWeight( old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v )
                weights_v = new_weights_v
                err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcMSE(err_v)

                delta_weights_v = np.absolute(old_weights_v - new_weights_v)
                if ((delta_weights_v < delta_weight_threshold).all()):  # end condition
                    iter_MSE = new_MSE
                    break
                elif (new_MSE > old_MSE):                                           # if gradient ascent by overstepping
                    learning_rate = learning_rate * 0.99                            # throttle learning rate
                    weights_v = old_weights_v                                       # revert weights
                    err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)    # revert err_v
                else:
                    if (step % math.ceil(math.log(step+2, 1.01)/2) == 0):             # log more of earlier steps
                        training_log = np.vstack((training_log, np.hstack(( np.array([descent, step, new_MSE, learning_rate]).reshape(1,4), weights_v.T ))))
                    if (step % 1000 == 0):
                        print(f'Descent {descent} \t Step {step} \t MSE {new_MSE}')
                    learning_rate = learning_rate * 1.002                           # accelerate learning rate
                    step += 1

            if (new_MSE < best_MSE):
                best_MSE = new_MSE
                self.weights_v = weights_v  
            

        # After all descents
        # training log post processing
        if (training_log.shape[0] != 1):
            training_log = np.delete(training_log, obj=0, axis=0)   # remove zeros row from training_log

        if (self.draw_plots):
            self.drawPlots(training_log, descents)

        return best_MSE

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
        descents=3, learning_rate=2.1, delta_weight_threshold=0.0001   # training hyperparameters
    )
    return mse, model.weights_v

def test(model, datasets):
    predictions_v = model.test(
        testing_x_df=datasets['testing_x_df'],
        testing_y_df=datasets['testing_y_df']
    )
    return model.calcMSE(predictions_v - datasets['testing_y_df'].to_numpy())
