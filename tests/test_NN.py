
import numpy as np
from neuralnetwork.nn_vec import NeuralNetwork, Layer


# def test_Document_initialization(client):
#     client.set_collection("test_collection")
#     document = Document(document="Test Document", kind="Test Kind")
#     assert document.document == "Test Document"
#     assert document.kind == "Test Kind"
#     document_from_chroma = Document(id=document.id)
#     assert document_from_chroma.document == "Test Document"
#     assert document_from_chroma.kind == "Test Kind"

def test_NN_initialization():
    nn = NeuralNetwork("nn_test")
    assert nn.name == "nn_test"
    assert nn.num_layers == 0