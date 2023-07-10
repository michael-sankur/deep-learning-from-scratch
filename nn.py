#

import numpy as np


class ActivationFunction():

    def __init__(self):
        pass


class NeuralNetwork():

    def __init__(self):

        print("Neural Network initialized")

    def _set_layers(self, layer_list=None):

        self.layer_list = layer_list

        if self.layer_list:

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
        
    def _train_step(self, xx, yy):

        yy_pred = self._compute_output(xx)

        for k1 in np.arange(self.num_layers,0,-1)-1:

            if k1 == self.num_layers-1:
                self.layer_list[k1]._compute_gradients(self, yy, self.layer_list[k1-1], None)
            elif k1 <= self.num_layers-2 and k1 >= 1:
                self.layer_list[k1]._compute_gradients(self, yy, self.layer_list[k1-1], self.layer_list[k1+1])
            elif k1 == 0:
                pass

        for k1 in range(1,self.num_layers):

            self.layer_list[k1]._update_layer()
        



class Layer():

    _counter = 0

    def __init__(self, layer_type, num_neurons, activation_function=None):

        Layer._counter += 1

        self.layer_num = 0

        self.layer_type = layer_type

        self.num_neurons = num_neurons

        self.activation_function = activation_function

        self.name = f"L{str(self._counter)}"

        print(f"Layer initialized")
        print(f"Name {self.name} | Type: {self.layer_type} | Neurons: {self.num_neurons}")


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

            # print("\n")

    def _compute_layer_output(self, layer_input):

        if self.layer_num == 0:
            self.zz = layer_input
            self.aa = self.zz
        else:
            self.zz = self._compute_z(layer_input)
            self.aa = self._compute_a(self.zz)
        return self.aa

    def _compute_z(self, prev_layer_acti):

        self.zz = self.weights@prev_layer_acti + self.biases

        return self.zz

    def _compute_a(self, curr_layer_zz):

        if not self.activation_function:

            self.aa = curr_layer_zz

        if self.activation_function == "linear":

            self.aa = curr_layer_zz

        if self.activation_function == "relu":

            self.aa = np.max(curr_layer_zz, 0)

        if self.activation_function == "tanh":

            self.aa = (np.exp(curr_layer_zz) - np.exp(-curr_layer_zz)) / (np.exp(curr_layer_zz) + np.exp(-curr_layer_zz))

        return self.aa
    

    def _compute_gradients(self, nn: "NeuralNetwork", yk, prev_layer: "Layer"=None, next_layer: "Layer"=None):

        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

        self.dC_daL = np.zeros(self.aa.shape)
        self.daL_dzL = np.zeros(self.aa.shape)

        # print(f"\nComputing gradients for layer {self.layer_num}")
        # print(f"Weights shape: {self.weights.shape}")
        
        if self.layer_num >= 1:

            if self.layer_num == nn.num_layers - 1:

                for m in range(0,self.weights.shape[0]):

                    dC_daLm = 2*(yk[m,0] - self.aa[m,0])
                    self.dC_daL[m,0] = dC_daLm

            elif self.layer_num <= nn.num_layers - 2 and self.layer_num >= 1:

                for n in range(0,self.aa.shape[0]):

                    dC_daL1n = 0

                    for m in range(0,next_layer.aa.shape[0]):

                        dC_daL1n += next_layer.dC_daL[m,0]*next_layer.daL_dzL[m,0]*next_layer.weights[m,n]

                    self.dC_daL[n,0] = dC_daL1n

            # print(f"Gradient of cost function wrt to layer {self.layer_num} activations:")
            # print(self.dC_daL)

            for m in range(0,self.weights.shape[0]):

                if self.activation_function == "linear":
                    daLm_dzLm = 1
                elif self.activation_function == "relu":
                    if self.zz[m,0] < 0: daLm_dzLm = 0
                    if self.zz[m,0] >= 0: daLm_dzLm = 1
                elif self.activation_function == "tanh":
                    daLm_dzLm = (2 / (np.exp(self.zz[m,0]) + np.exp(-self.zz[m,0])))**2
                self.daL_dzL[m,0] = daLm_dzLm

                dC_daLm = self.dC_daL[m,0]
                daLm_dzLm = self.daL_dzL[m,0]

                for n in range(0,self.weights.shape[1]):                    

                    dzLm_dwLmn = prev_layer.aa[n,0]

                    self.grad_weights[m,n] = dC_daLm*daLm_dzLm*dzLm_dwLmn

                dC_dbLm = dC_daLm*daLm_dzLm*1
                self.grad_biases[m,0] = dC_dbLm
            
        
        elif self.layer_num == 0:
            
            pass
        
        # print(f"Gradient of cost function wrt to layer {self.layer_num} weights:")
        # print(self.grad_weights)
        # print(f"Gradient of cost function wrt to layer {self.layer_num} biases:")
        # print(self.grad_biases)


    def _update_layer(self):

        self.weights = self.weights - 1e-6*self.grad_weights
        self.biases = self.biases - 1e-6*self.grad_biases


