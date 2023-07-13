"""
Neural Network Module

Contrains NeuralNetwork, Layer, and ActivationFunction classes
"""

# imports
import numpy as np

# NeuralNetwork class
class NeuralNetwork():

    _counter = 0 # counter

    # initialize
    def __init__(self, name: str=None, layer_list:list["Layer"]=None):
        NeuralNetwork._counter += 1
        # assign name to NN or default to "NNxx" where xx is the counter with leading zeros
        if name:
            self.name = name
        else:
            self.name = f"NN{self._counter:02d}"
        print(f"Neural Network {self.name} initialized")
        self.num_layers = 0
        self.layer_list = []
        if layer_list:
            self._set_layers(layer_list)

    # add a single layer to the NN
    def _add_layers(self, layer_list:list["Layer"]=None):
        if layer_list:
            for k1, temp_layer in enumerate(layer_list):
                self.num_layers += 1
                self.layer_list.append(layer_list[k1])
                self.layer_list[-1].layer_num = self.num_layers-1
                if self.layer_list[-1].layer_num == 0:
                    self.layer_list[-1]._initialize_weights_and_biases()
                else:                    
                    self.layer_list[-1]._initialize_weights_and_biases(self.layer_list[-2])
                print(f"{self.name} added layer {self.layer_list[-1].name} with {self.layer_list[-1].num_neurons} neurons and {self.layer_list[-1].activation_function_type} activation")

    # set layers
    def _set_layers(self, layer_list:list["Layer"]=None):        
        if layer_list:
            self.layer_list = layer_list
            self.num_layers = len(self.layer_list)
            for k1, temp_layer in enumerate(self.layer_list):
                self.layer_list[k1].layer_num = k1
                if k1 == 0:
                    self.layer_list[k1]._initialize_weights_and_biases()
                else:                    
                    self.layer_list[k1]._initialize_weights_and_biases(self.layer_list[k1-1])
                print(f"{self.name} added layer {self.layer_list[k1].name} with {self.layer_list[k1].num_neurons} neurons and {self.layer_list[k1].activation_function_type} activation")

    def _compute_output(self, nn_input):
        if self.layer_list is not None and nn_input is not None:
            for k1, temp_layer in enumerate(self.layer_list):
                if k1 == 0:
                    layer_output = temp_layer._compute_layer_output(nn_input.T)
                elif k1 >= 1:
                    layer_output = temp_layer._compute_layer_output(layer_output)
                # print(f"Layer {k1} output:")
                # print(layer_output)
            return layer_output.T
        else:
            return None
        
    def _train_step(self, xx, yy, alpha):
        # yy_pred = self._compute_output(xx)
        for k1 in np.arange(self.num_layers,0,-1)-1:
            if k1 == self.num_layers-1:
                self.layer_list[k1]._compute_gradients(self, yy, self.layer_list[k1-1], None)
            elif k1 <= self.num_layers-2 and k1 >= 1:
                self.layer_list[k1]._compute_gradients(self, yy, self.layer_list[k1-1], self.layer_list[k1+1])
            elif k1 == 0:
                pass
        for k1 in range(1,self.num_layers):
            self.layer_list[k1]._update_layer(alpha)
        

########
class Layer():

    _counter = 0

    def __init__(self, num_neurons, activation_function_type=None, layer_type: str=None):
        Layer._counter += 1
        self.layer_num = 0
        self.name = f"L{self._counter:02d}"        
        self.num_neurons = num_neurons
        if activation_function_type == None:
            activation_function_type = "linear"
        self.activation_function_type = activation_function_type
        self.activation_function = ActivationFunction(self.activation_function_type)
        self.layer_type = layer_type        
        print(f"Layer {self.name} initialized | Neurons: {self.num_neurons} | Activation: {self.activation_function_type}")


    def _initialize_weights_and_biases(self, previous_layer:"Layer"=None):
        if self.layer_num == 0:
            self.weights = None
            self.biases = None
        else:
            self.weights = np.random.randn(self.num_neurons, previous_layer.num_neurons)
            # print(self.weights)
            self.weights = self.weights/np.max(np.abs(self.weights))
            # print(self.weights)
            self.biases = np.random.randn(self.num_neurons,1)
            # print(self.biases)
            self.biases = self.biases/np.max(np.abs(self.biases))
            # print(self.biases)

    def _compute_layer_output(self, layer_input):
        if self.layer_num == 0:
            self.zz = layer_input
            self.aa = self.zz
        else:
            self.zz = self._compute_zz(layer_input)
            self.aa = self._compute_aa(self.zz)
        return self.aa

    def _compute_zz(self, prev_layer_aa):
        if self.layer_num == 0:
            self.zz = prev_layer_aa
        else:
            self.zz = self.weights@prev_layer_aa + self.biases
        return self.zz

    def _compute_aa(self, curr_layer_zz):
        if self.layer_num == 0:
            self.aa = self.zz
        else:
            self.aa = self.activation_function._compute_activation(curr_layer_zz)
        return self.aa
    
    def _compute_gradients(self, nn, yy, prev_layer:"Layer"=None, next_layer:"Layer"=None):

        yy = yy.T

        # print(self.aa.shape)
        # print(yy.shape)

        m = yy.shape[1]

        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

        self.dC_daL = np.zeros((self.aa.shape[0], yy.shape[1]))
        self.daL_dzL = np.zeros((self.aa.shape[0], yy.shape[1]))

        # print(f"\nComputing gradients for layer {self.layer_num}")
        # print(f"Weights shape: {self.weights.shape}")

        # print(f"aa shape: {self.aa.shape}")
        # print(f"yy shape: {yy.shape}")

        # print(f"dC_daL shape: {self.dC_daL.shape}")
        
        if self.layer_num >= 1:
            
            self.daL_dzL = self.activation_function._compute_derivative(self.zz)

            #
            if self.layer_num == nn.num_layers - 1:
                self.dC_daL = 2*(self.aa - yy)
            elif self.layer_num <= nn.num_layers - 2 and self.layer_num >= 1:
                self.dC_daL = (next_layer.weights.T@next_layer.dC_daL)*self.daL_dzL

            # print(self.dC_daL.shape)
            # print(np.sum(self.dC_daL*self.daL_dzL,1).shape)

            #
            self.grad_weights = 1/m * (self.dC_daL*self.daL_dzL)@(prev_layer.aa.T)
            self.grad_biases = 1/m * np.sum(self.dC_daL*self.daL_dzL,1).reshape(self.biases.shape[0],1)

            
            # one_hot_Y = one_hot(Y)
            # dZ2 = A2 - one_hot_Y
            # dW2 = 1 / m * dZ2.dot(A1.T)
            # db2 = 1 / m * np.sum(dZ2,1).reshape(A2.shape[0],1)
            # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
            # dW1 = 1 / m * dZ1.dot(X.T)
            # db1 = 1 / m * np.sum(dZ1,1).reshape(A1.shape[0],1)
            # return dW1, db1, dW2, db2            
        
        elif self.layer_num == 0:            
            pass
        
        # print(f"Gradient of cost function wrt to layer {self.layer_num} weights:")
        # print(self.grad_weights)
        # print(f"Gradient of cost function wrt to layer {self.layer_num} biases:")
        # print(self.grad_biases)


    def _update_layer(self, alpha):
        self.weights = self.weights - alpha*self.grad_weights
        self.biases = self.biases - alpha*self.grad_biases



########
class ActivationFunction():

    def __init__(self, type: str=None):

        if not type:
            self.type = "linear"
        else:
            self.type = type


    def _compute_activation(self, zz):

        self.aa = np.zeros(zz.shape)
        if self.type == "linear":
            self.aa = zz
        if self.type == "relu":
            self.aa = np.maximum(zz, 0)
        if self.type == "softmax":
            self.aa = np.exp(zz)/np.sum(np.exp(zz),0)
        if self.type == "tanh":
            self.aa = (np.exp(zz) - np.exp(-zz)) / (np.exp(zz) + np.exp(-zz))
        return self.aa

    def _compute_derivative(self, zz):

        self.daL_dzL = np.zeros(zz.shape)
        if self.type == "linear":
            self.daL_dzL = np.ones(zz.shape)
        elif self.type == "relu":
            # daLm_dzLm = np.zeros((self.aa.shape[0], yy.shape[1]))
            self.daL_dzL = np.zeros(zz.shape)
            self.daL_dzL[(zz >= 0)] = 1
        elif self.type == "softmax":
            self.daL_dzL = np.ones(zz.shape)
        elif self.type == "tanh":
            self.daL_dzL = (2 / (np.exp(zz) + np.exp(-zz)))**2
        return self.daL_dzL

