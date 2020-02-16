# Machine Learning Search and Rescue

----
## Cuse Hacks 2020 IBM Challenge

---
### Info
This python application determines an optimal path for first responders to take 
when extracting people from areas affected by wild-fires by using machine 
learning to solve a version of the travelling salesman problem. IBM Watson
Machine Learning is used to determine the relative safety of persons in the region
affected and establish a weighting system for those in the most danger

[Search and Rescue flight path](visualization/flight_path.gif)"

The model accounts for limited capacity of the emergency vehciles but does not estimate
a total time to recover all persons in danger.

# Use

---
The first step is to initialize an instance of the model
The SAR class automatically iniitalizes and computes the Watson Factors and obtains
the model training data

initializing an instance of the SAR class will also result in a call to create an
instance of StochasticGradientDescentModel and a BasinHoppingModel.
Both computations can take a few minutes (parallelizing this process/optimizing
is a key next step)

    from SearchAndRescue import SAR
    search_and_rescue_plan = SAR()

From this point you can visualize the flight plan for the stochastic solution
and for the basin hopping solution with the following respective function calls

    search_and_rescue_plan.visualize_path_stochastic(key='stochastic', gif_name='flightplan.gif')
This function call creates a gif animation and saves the file to the string in the function call

---
To visualize the Basin hopping model path:

    search_and_rescue_plan.visualize_path_stochastic(key='other',name='flight_basin.gif')

---
To compute the ordering of the vector positions to visit (Stochastic model):

---
    flight_plan, indicies = search_and_rescue_plan.stochastic_gradient_descent.evaluate()
    
 flight_plan is an ordered array containing vector which represent the coordinate positions of the 
 people being trapped in the area affected by the wild fire
 
 indicies is the ordering of the training_data array which is computed by using 
 a different machine learning model developed with the IBM Watson toolkit
 
 #### Brief aside:
    from watsonInterface import training_data
 is the unordered array over which the travelling emergency vehicle problem is sovled
    from watsonInterface import watson_weights
 are the weights corresponding to the training_data array (a value close to 0 indicates 
 the model will attempt to prioritize this target)
 
 ---
To compute the ordering of the vector positions to visit (Basin hopping model)
be sure to import the numpy library.

    indicies = search_and_rescue_plan.basin_hopping_descent.basin_result.x
    flight_plan = [training_data[x] for x in np.argsort(indicies)]
