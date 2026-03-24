import copy
import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._tmp_label_tensor = None

    @property
    def tmp_label_tensor(self):
        return self._tmp_label_tensor

    @tmp_label_tensor.setter
    def tmp_label_tensor(self, value):
        self._tmp_label_tensor = value

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.tmp_label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output = self.loss_layer.forward(input_tensor, label_tensor)
        return output

    def backward(self):
        error_tensor = self.loss_layer.backward(self.tmp_label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(0, iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        input_tensor_copy = np.copy(input_tensor)
        for layer in self.layers:
            input_tensor_copy = layer.forward(input_tensor_copy)
        return input_tensor_copy
