import numpy as np
from Discriminator import Discriminator
from Discriminator import sigmoid

class Generator():
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def predict(self, z):
        """
        Generator predict
        :param z: noise that created randomly
        :return: result of sigmoid function
        """
        prediction = self.weights * z + self.biases
        return sigmoid(prediction)

    def error(self, z, D):
        """

        :param z:
        :param D:
        :return:
        """
        d_pred = D.predict(z)
        g_pred = self.predict(d_pred)
        return -np.log(g_pred)

    def derivatives(self, z, D):
        """
        calculate derivatives of weights and bias
        :param z: input values
        :param D:  Discriminator Class
        :return: derivative of weights and bias
        """
        discriminator_weights = D.weights
        discriminator_bias = D.bias
        d_pred = D.predict(z)
        g_pred = self.predict(d_pred)
        factor = -(1 - d_pred) * discriminator_weights * d_pred * (1 - g_pred)
        derivatives_weights = factor * z
        derivatives_bias = factor
        return derivatives_weights, derivatives_bias

    def update(self, z, D, alpha):
        """
        performing gradient descent
        :param z: input values
        :param D: Discriminator Class
        :param alpha: learning rate parameter
        :return: result of  gradient descent
        """
        error_before = self.error(z, D)
        derivative_weights, derivative_bias = self.derivatives(z, D)
        self.weights -= alpha * derivative_weights
        self.biases -= alpha * derivative_bias
        error_after = self.error(z, D)
        print("Error before:", error_before, " - ", "Error after:", error_after)