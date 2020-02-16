import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from watsonInterface import training_data
from watsonInterface import watson_weights



class BasinHoppingDescentModel:
    def __init__(self):
        self.prev_cost_array = None
        self.basin_result = None

    def get_basin_SAR(self):
        self.prev_cost_array = []
        init_guess = [np.linalg.norm(x) for x in training_data]
        self.basin_result = basinhopping(self.cost_basinhopping, [init_guess])

    def cost_basinhopping(self,input_array):
        kinput_array = np.argsort(input_array)
    # print(np.argsort(array_of_activations[:,0]).reshape(len(array_of_activations)))

        grab_array = [training_data[x] for x in kinput_array]
        watson_array = [watson_weights[x] for x in kinput_array]
        N = len(grab_array)
        i = 0
        j = 0
        n = 4
        position = np.array([1.0, 0, 0])
        dist = 0
        number_of_trips = 0
        total_dist = 0
        while (i < N):
            number_of_trips += 1
            while (j < n and i < N):
                dist += (np.linalg.norm(np.array(grab_array[i]) - np.array(position))) * watson_array[i]
                position = grab_array[i]
                i += 1
                j += 1
                total_dist += dist
            total_dist += np.linalg.norm(position - np.array([1.0,0,0]))

            dist = 0
            j = 0
            position = np.array([1.0, 0, 0])
        self.prev_cost_array.append(total_dist)
        return total_dist

    def visualize_cost_descent(self,  ax):
        if (self.prev_cost_array):
            ax.plot(self.prev_cost_array, label='Basin hopping cost descent')
        return ax


