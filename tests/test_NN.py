
import numpy as np
from neuralnetwork.neural_network_module import NeuralNetwork, Layer, ActivationFunction


def test_NN_initialization():
    nn = NeuralNetwork(name="nn_test")
    assert nn.name == "nn_test"
    assert nn.num_layers == 0

def test_Layer_initialization():
    test_layer = Layer(32, "linear")
    assert test_layer.name == "L01"
    assert test_layer.num_neurons == 32
    assert test_layer.activation_function_type == "linear"
    assert test_layer.activation_function.type == "linear"

def test_NN_set_Layers():
    nn = NeuralNetwork()
    layer_list = []
    layer_list.append(Layer(2, None))
    layer_list.append(Layer(4, "linear"))
    layer_list.append(Layer(6, "relu"))
    layer_list.append(Layer(8, "tanh"))
    layer_list.append(Layer(2, "softmax"))
    nn._set_layers(layer_list)

    assert nn.num_layers == len(layer_list)
    assert nn.layer_list[0].weights == None
    assert nn.layer_list[0].biases == None
    assert nn.layer_list[0].activation_function_type == None
    assert nn.layer_list[0].activation_function.type == "linear"

    for k1 in range(1,nn.num_layers):
        assert nn.layer_list[k1].layer_num == k1
        assert nn.layer_list[k1].weights.shape == (nn.layer_list[k1].num_neurons, nn.layer_list[k1-1].num_neurons)
        assert nn.layer_list[k1].biases.shape == (nn.layer_list[k1].num_neurons, 1)
        assert nn.layer_list[k1].activation_function_type == layer_list[k1].activation_function_type
        assert nn.layer_list[k1].activation_function.type == layer_list[k1].activation_function.type

def test_layer_output():
    # nn = NeuralNetwork()
    # layer_list = []
    # layer_list.append(Layer(4, None))
    # layer_list.append(Layer(4, "linear"))
    # nn._set_layers(layer_list)

    # input = np.ones((1,4))

    # assert (nn.layer_list[0]._compute_zz(input) == input).all()
    # assert (nn.layer_list[0]._compute_aa(input) == input).all()
    # layer_output = nn.layer_list[0].aa


    # assert (nn._compute_output(input) == nn.layer_list[1].weights@input + nn.layer_list[1].biases).all()

    nn = NeuralNetwork()
    layer_list = []
    layer_list.append(Layer(4, None))
    layer_list.append(Layer(4, "linear"))
    layer_list.append(Layer(4, "relu"))
    layer_list.append(Layer(4, "tanh"))
    layer_list.append(Layer(4, "softmax"))
    nn._set_layers(layer_list)

    input = np.ones((1,4))

    assert (nn.layer_list[0]._compute_zz(input.T) == input.T).all()
    assert (nn.layer_list[0]._compute_aa(input.T) == input.T).all()
    layer_output = nn.layer_list[0].aa

    assert (nn.layer_list[1]._compute_zz(layer_output) == nn.layer_list[1].weights@layer_output + nn.layer_list[1].biases).all()
    layer_output = nn.layer_list[1]._compute_zz(layer_output)
    assert (nn.layer_list[1]._compute_aa(layer_output) == layer_output).all()
    layer_output = nn.layer_list[1]._compute_aa(layer_output)

    assert (nn.layer_list[2]._compute_zz(layer_output) == nn.layer_list[2].weights@layer_output + nn.layer_list[2].biases).all()
    layer_output = nn.layer_list[2]._compute_zz(layer_output)
    assert (nn.layer_list[2]._compute_aa(layer_output) == np.maximum(layer_output, 0)).all()
    layer_output = nn.layer_list[2]._compute_aa(layer_output)

    assert (nn.layer_list[3]._compute_zz(layer_output) == nn.layer_list[3].weights@layer_output + nn.layer_list[3].biases).all()
    layer_output = nn.layer_list[3]._compute_zz(layer_output)
    assert (nn.layer_list[3]._compute_aa(layer_output) == (np.exp(layer_output) - np.exp(-layer_output)) / (np.exp(layer_output) + np.exp(-layer_output))).all()
    layer_output = nn.layer_list[3]._compute_aa(layer_output)

    assert np.max(np.abs(np.tanh(layer_output) - (np.exp(layer_output) - np.exp(-layer_output)) / (np.exp(layer_output) + np.exp(-layer_output)))) < 1e-12
           
    assert (nn.layer_list[4]._compute_zz(layer_output) == nn.layer_list[4].weights@layer_output + nn.layer_list[4].biases).all()
    layer_output = nn.layer_list[4]._compute_zz(layer_output)
    assert (nn.layer_list[4]._compute_aa(layer_output) == (np.exp(layer_output)/np.sum(np.exp(layer_output)))).all()
    layer_output = nn.layer_list[4]._compute_aa(layer_output)

    # for k1 in range(1,nn.num_layers):
    #     assert (nn.layer_list[k1]._compute_zz(layer_output) == nn.layer_list[k1].weights@layer_output + nn.layer_list[k1].biases).all()
    #     layer_output = nn.layer_list[k1]._compute_zz(layer_output)
    #     assert (nn.layer_list[k1]._compute_aa(layer_output) == layer_output).all()
    #     layer_output = nn.layer_list[k1]._compute_aa(layer_output)
