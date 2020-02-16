import numpy as np
import matplotlib.pyplot as plt
from modelFunctions import sigmoid
from modelFunctions import sigmoid_prime
from watsonInterface import watson_weights
from watsonInterface import training_data

class model_fit:
    def __init__(self, sizes, init_cost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 3) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.prev_cost = init_cost
        self.prev_outlayer = np.random.randn(sizes[0], 3)
        self.cost_array = []
        self.init_cost = init_cost
        self.lowest_cost = 1e10
        self.best_biases = self.biases
        self.best_weights = self.weights

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_datas, epochs, mini_batch_size, eta):

        for j in range(epochs):

            mini_batch = training_datas
            self.update_mini_batch(training_datas, eta)
            self.cost_array.append(self.prev_cost)
            if (self.prev_cost < self.lowest_cost):
                self.lowest_cost = self.prev_cost
                self.best_weights = self.weights
                self.best_biases = self.biases

        print("Training Complete")

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(np.array(b).shape) for b in self.biases]
        nabla_w = [np.zeros(np.array(w).shape) for w in self.weights]
        x = np.array(mini_batch)[:, 0]
        y = np.array(mini_batch)[:, 1]
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [np.array(w) - (eta / len(mini_batch)) * nw \
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [np.array(b) - (eta / len(mini_batch)) * nb \
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(np.array(b).shape) for b in self.biases]
        nabla_w = [np.zeros(np.array(w).shape) for w in self.weights]

        activation = x

        activations = [x]
        zs = []

        for i in range(self.num_layers - 1):
            z = np.dot(np.array(self.weights[i]), activation) + np.array(self.biases[i])
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], x) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self):

        x = np.array(training_data)

        test_results = np.array(self.feedforward(x))
        rescue_plan = np.argsort([np.linalg.norm(x) for x in test_results])
        return [x[i] for i in rescue_plan], rescue_plan

    def cost_derivative(self, output_activations, input_array):



        cost_k = self.cost(output_activations, input_array)
        ret_val = cost_k
        self.prev_outlayer = output_activations
        self.prev_cost = cost_k

        return ret_val

    def visualize_stochastic_descent(self, ax):
        if(self.cost_array):
            ax.plot(self.cost_array, label='Stochasctic Gradient Descent Model')
        return ax


    def cost(self, array_of_activations, input_array):

        self.array_of_mag = np.argsort([np.linalg.norm(x) for x in np.array(array_of_activations)])
        self.grab_array = [input_array[x] for x in self.array_of_mag]
        watson_array = [watson_weights[x] for x in self.array_of_mag]
        N = len(self.grab_array)
        i = 0
        j = 0
        n = 4
        position_home = np.array([np.amax(np.array(self.grab_array)[:,0]), 0, 0])
        dist = 0
        number_of_trips = 0
        total_dist = 0
        while (i < N):
            number_of_trips += 1
            position = position_home
            while (j < n and i < N):
                dist += (np.linalg.norm(np.array(self.grab_array[i]) - np.array(position_home))) * np.float64(watson_array[i])
                position = self.grab_array[i]
                i += 1
                j += 1
                total_dist += dist
            total_dist += np.linalg.norm(position - np.array([1.0,0,0]))

            dist = 0
            j = 0


        return total_dist


def static_cost(array_of_activations, input_array):
    array_of_mag = np.argsort([np.linalg.norm(x) for x in np.array(array_of_activations)])
    grab_array = [input_array[x] for x in array_of_mag]
    watson_array = [watson_weights[x] for x in array_of_mag]
    N = len(grab_array)
    i = 0
    j = 0
    n = 4
    position_home = np.array([np.amax(np.array(grab_array)[:,0]), 0, 0])
    dist = 0
    number_of_trips = 0
    total_dist = 0

    while (i < N):
        number_of_trips += 1
        position = position_home
        while (j < n and i < N):
            dist += (np.linalg.norm(np.array(grab_array[i]) - np.array(position))) * np.float64(watson_array[i])
            position = grab_array[i]
            i += 1
            j += 1
            total_dist += dist
        total_dist += np.linalg.norm(position-position_home)

        dist = 0
        j = 0


    return total_dist
def get_Stochastic_Gradient_Descent_Model():

    io_neurons = len(training_data)
    input_data = [np.array([x,[0,0,0]]) for x in training_data]
    initial_cost = static_cost([np.linalg.norm(x) for x in training_data], training_data)

    model_pipe_1 = model_fit(sizes=[io_neurons, 15, 15, io_neurons], init_cost=initial_cost)
    pipeline_1_eta = 0.1
    model_pipe_1.SGD(training_datas=input_data, epochs=1000, mini_batch_size=io_neurons, eta=pipeline_1_eta)

    pipeline_2_eta = pipeline_1_eta / (initial_cost - model_pipe_1.lowest_cost)
    model_pipe_2 = model_fit(sizes=[io_neurons, 15, 15, io_neurons], init_cost = initial_cost)
    model_pipe_2.SGD(training_datas=input_data, epochs=1000, mini_batch_size=io_neurons, eta=pipeline_2_eta)

    pipe_line_3_eta = pipeline_2_eta / (initial_cost - model_pipe_2.lowest_cost)
    model_pipe_3 = model_fit(sizes=[io_neurons, 15, 15, io_neurons], init_cost = initial_cost)
    model_pipe_3.SGD(training_datas=input_data, epochs=1000, mini_batch_size=io_neurons, eta=pipe_line_3_eta)

    if ((model_pipe_3.lowest_cost < model_pipe_2.lowest_cost) and (model_pipe_3.lowest_cost < model_pipe_1.lowest_cost)):
        print("Returning pipeline three. Final attempt at tuning hyper perameters")
        model_pipe_3.weights = model_pipe_3.best_weights
        model_pipe_3.biases = model_pipe_3.best_biases
        return model_pipe_3
    elif (model_pipe_2.lowest_cost < model_pipe_1.lowest_cost):
        print("Returning pipeline two. Third attempt overfit the model. Returning Second tuning")
        model_pipe_2.weights = model_pipe_2.best_weights
        model_pipe_2.biases = model_pipe_2.best_biases
        return model_pipe_2
    else:
        print("Returning first pipeline")
        model_pipe_1.weights = model_pipe_1.best_weights
        model_pipe_1.biases = model_pipe_1.best_biases
        return model_pipe_1

