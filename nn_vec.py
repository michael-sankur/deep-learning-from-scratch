#

import numpy as np


# class ActivationFunction():

#     def __init__(self):
#         pass


class NeuralNetwork():

    _counter = 0

    def __init__(self, name: str=None):
        NeuralNetwork._counter += 1
        if name:
            self.name = name
        else:
            self.name = f"NN{self._counter:02d}"
        print(f"Neural Network {self.name} initialized")

    def _set_layers(self, layer_list=None):        
        if layer_list:
            self.layer_list = layer_list
            self.num_layers = len(self.layer_list)
            for k1, temp_layer in enumerate(self.layer_list):
                layer_list[k1].layer_num = k1
                if k1 == 0:
                    layer_list[k1]
                else:                    
                    layer_list[k1]._initialize_weights_and_biases(layer_list[k1-1])

    def _compute_output(self, nn_input):
        if self.layer_list is not None and nn_input is not None:
            for k1, temp_layer in enumerate(self.layer_list):
                if k1 == 0:
                    layer_output = temp_layer._compute_layer_output(nn_input)
                elif k1 >= 1:
                    layer_output = temp_layer._compute_layer_output(layer_output)
                # print(f"Layer {k1} output:")
                # print(layer_output)
            return layer_output        
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
        



class Layer():

    _counter = 0

    def __init__(self, layer_type, num_neurons, activation_function=None):
        Layer._counter += 1
        self.layer_num = 0
        self.layer_type = layer_type
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.name = f"L{self._counter:02d}"
        print(f"Layer {self.name} initialized | Type: {self.layer_type} | Neurons: {self.num_neurons}")


    def _initialize_weights_and_biases(self, previous_layer):
        if self.layer_num == 0:
            pass
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
        self.zz = self.weights@prev_layer_aa + self.biases
        return self.zz

    def _compute_aa(self, curr_layer_zz):

        if self.activation_function == "linear":
            self.aa = curr_layer_zz
        if self.activation_function == "relu":
            self.aa = np.maximum(curr_layer_zz, 0)
        if self.activation_function == "softmax":
            self.aa = np.exp(curr_layer_zz)/np.sum(np.exp(curr_layer_zz),0)
        if self.activation_function == "tanh":
            self.aa = (np.exp(curr_layer_zz) - np.exp(-curr_layer_zz)) / (np.exp(curr_layer_zz) + np.exp(-curr_layer_zz))
        return self.aa    

    def _compute_gradients(self, nn: "NeuralNetwork", yy, prev_layer: "Layer"=None, next_layer: "Layer"=None):

        m = yy.shape[1]

        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

        self.dC_daL = np.zeros((self.aa.shape[0], yy.shape[1]))
        self.daL_dzL = np.zeros((self.aa.shape[0], yy.shape[1]))

        # print(f"\nComputing gradients for layer {self.layer_num}")
        # print(f"Weights shape: {self.weights.shape}")
        
        if self.layer_num >= 1:
            if self.activation_function == "linear":
                daLm_dzLm = np.ones(self.zz.shape)
            elif self.activation_function == "relu":
                daLm_dzLm = np.zeros((self.aa.shape[0], yy.shape[1]))
                daLm_dzLm[(self.zz >= 0)] = 1
                # if self.zz[m,0] < 0: daLm_dzLm = 0
                # if self.zz[m,0] >= 0: daLm_dzLm = 1
            elif self.activation_function == "softmax":
                daLm_dzLm = np.ones(self.zz.shape)
            elif self.activation_function == "tanh":
                daLm_dzLm = (2 / (np.exp(self.zz) + np.exp(-self.zz)))**2
            self.daL_dzL = daLm_dzLm

            #
            if self.layer_num == nn.num_layers - 1:
                self.dC_daL = 2*(self.aa - yy)
            elif self.layer_num <= nn.num_layers - 2 and self.layer_num >= 1:
                self.dC_daL = (next_layer.weights.T@next_layer.dC_daL)*self.daL_dzL

            # print(self.dC_daL.shape)
            # print(self.daL_dzL.shape)
            # print(prev_layer.aa.shape)

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


