import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def print_matrices(self):
        print("Outer layer Weight matrix: ", self.weights1)#final trained Outer layer Weight matrix 
        print("Hidden layer weight matrix: ", self.weights2)#final trained Hidden layer weight matrix

		
if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Generated Weights Before Training: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[30, 40, 50],[40, 50, 20],[50, 20, 15],[20, 15, 60],[15, 60, 70],[60, 70, 50]],dtype=float)

    training_outputs = np.array([20, 15, 60, 70, 50, 40],dtype=float)
	
    def __init__(self, training_inputs, training_ouputs, learning_rate):
        self.input = training_inputs#input
        self.bind_values(training_inputs) #Scale units
		
        #Outer layer Weight matrix(2*3) weight of matrix from input to hidden
        self.weights1 = [[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]]
        #Hidden layer weight matrix(1*2)weight of matrix from hiddent to output
        self.weights2 = [[0.5, 0.1]]
        
        self.target = training_outputs#Expected output
        self.bind_values(training_outputs) #Scale units
        self.learning_rate = learning_rate
		
		
		
    def back_prop(self, input, target):
        #Backward propagate through the network
        target_vector = np.array(target, ndmin=2).T#Value of y ie the expected output
        input = np.array(input, ndmin=2).T#Value of the input
        
        output_vector1 = np.dot(self.weights1, input)
        output_vector_hidden = sigmoid(output_vector1)#Output of the hidden layer

        output_vector2 = np.dot(self.weights2, output_vector_hidden)
        output_vector_network = sigmoid(output_vector2)#Actual Output of the network
        
        output_errors = target_vector - output_vector_network#Error in output ie expected output- actual output
    

        print("Input: \n", input)
        print("Expected: ", target)
        print("Actual: ", output_vector_network)
        print("Error :", output_errors, "\n")

        # update the weights:Adjusting second set (hidden -> output) weights
        tmp = output_errors * output_vector_network *             (1.0 - output_vector_network)

        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)

        self.weights2 += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights2.T, output_errors)

        # update the weights:Adjusting first set (input-> hidden) weights
        tmp = hidden_errors * output_vector_hidden *             (1.0 - output_vector_hidden)
        self.weights1 += self.learning_rate *             np.dot(tmp, input.T)		
		
    #training taking place
    def train(self):#Training the model
        for i in range(len(self.input)):#For loop to loop through inputs
            self.back_prop(self.input[i], self.target[i])# call function for back propagation

   

    #neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    
    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data based on trained weight matrices")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    