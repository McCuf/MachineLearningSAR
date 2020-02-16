from StochasticGradientDescentModel import get_Stochastic_Gradient_Descent_Model
from BasinHoppingModel import BasinHoppingDescentModel
import numpy as np
import matplotlib.pyplot as plt
from watsonInterface import watsonInit
from watsonInterface import training_data
from watsonInterface import watson_weights
from matplotlib.animation import FuncAnimation

class SAR:

    def __init__(self):
        watsonInit()
        self.basin_hopping_descent = BasinHoppingDescentModel()
        self.stochastic_gradient_descent = get_Stochastic_Gradient_Descent_Model()
        self.basin_hopping_descent.get_basin_SAR()

    def visualize_DescentModels(self):


        fig,ax = plt.subplots(figsize=(12,12))
        self.stochastic_gradient_descent.visualize_stochastic_descent(ax)
        self.basin_hopping_descent.visualize_cost_descent(ax)
        plt.legend()
        plt.show()

    def visualize_grid(self):
        plt.scatter(np.array(training_data)[:,0],np.array(training_data)[:,1], cmap=watson_weights, vmin=np.amin(watson_weights), vmax=np.amax(watson_weights))
        plt.show()

    def visualize_path_stochastic(self, key='stochastic'):
        flight_path = None
        flight_path_x = None
        flight_path_y = None
        if (key == 'stochastic'):
            flight_path, index = self.stochastic_gradient_descent.evaluate()
            flight_path_x = np.array(flight_path)[:,0]
            flight_path_y = np.array(flight_path)[:,1]
        else:
            index = self.basin_hopping_descent.basin_result.x
            flight_path = [training_data[x] for x in np.argsort(index)]
            flight_path_x = np.array(flight_path)[:, 0]
            flight_path_y = np.array(flight_path)[:, 1]

        for i, val in enumerate(flight_path_x):
            if i > 0 and (i % 4 == 0):
                np.array(list(flight_path_x).insert(i, np.amax(flight_path_x)))
                np.array(list(flight_path_x).insert(i, val))
                np.array(list(flight_path_y).insert(i, 0))
                np.array(list(flight_path_y).insert(i, flight_path_y[i]))
        fig = plt.figure()

        ax = plt.axes(xlim=(np.amin(flight_path_x),np.amax(flight_path_x)),
                      ylim=(np.amin(flight_path_y),np.amax(flight_path_y)))
        line, = ax.plot([], [], lw=3)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            x = flight_path_x[:(i%len(flight_path_x))]
            y = flight_path_y[:(i%len(flight_path_y))]
            line.set_data(x, y)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init,
                     frames=200, interval=20, blit=True)

        anim.save('flight_path.gif', writer='imagemagick')



