import numpy as np

def sigmoid(x):
    """
    result of sigmoid function
    :param x: x value
    :return: sigmoid result. This means y value
    """
    y = 1 / (1 + np.exp(-x))
    return y

def relu(x):
    return np.max(x, 0)


class Discriminator():

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, x):
        """
        firstly multiplication x values and data. Then add bias. Lastly give sigmoid function as input
        Prediction_process
        :return: result of sigmoid value
        """

        x = np.dot(x, self.weights) + self.bias
        return sigmoid(x)

    def error_image(self, image):
        """
        calculate image error -ln(prediction)
        :param image: image values
        :return: error value
        """
        prediction = self.predict(image)

        # if prediction decrease. The error will increase
        error = -np.log(prediction)
        return error

    def derivatives_image(self, image):
        """
        calculate derivative of weights and bias
        :param image: image values
        :return: derivatives of weights and derivatives of bias
        """
        prediction = self.predict(image)
        derivatives_weights = np.dot(-(1 - prediction),image)
        derivatives_bias = -(1 - prediction)
        return derivatives_weights, derivatives_bias

    def update_image(self, alpha, x):
        """
        update image. This mean perform gradient descent
        :param alpha: learning rate parameter
        :param x: image values
        :return: updated self weights and bias
        """
        weights, bias = self.derivatives_image(x)
        self.weights -= alpha*weights
        self.bias -= alpha*bias

    def error_noise(self, noise):
        """
        calculate error of noise
        :param noise: noise image values
        :return: error of noise
        """
        prediction = self.predict(noise)
        # want to be 0
        error = -np.log(1 - prediction)
        return error

    def derivatives_noise(self, noise):
        """
        calculate derivatives of noise weights and bias
        :param noise: noise image values
        :return:derivatives weights and derivatives bias
        """
        prediction = self.predict(noise)
        derivatives_weights = noise * prediction
        derivatives_bias = prediction

        return derivatives_weights, derivatives_bias

    def update_noise(self, alpha, noise):
        """
        update noise. This mean performing gradient descent to noise
        :param alpha: learning rate parameter
        :param noise: noise values
        :return: updated to self weights and self bias
        """
        weights, bias = self.derivatives_noise(noise)
        self.weights -= weights * alpha
        self.bias -= bias * alpha
