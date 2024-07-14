import numpy as np

class Activation:

  @staticmethod
  def relu(x):
    return np.maximum(x, 0)

  @staticmethod
  def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)
  
  @staticmethod
  def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))

  @staticmethod
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  @staticmethod
  def softmax(x):
      expX = np.exp(x - np.max(x, axis=0, keepdims=True))
      return expX / np.sum(expX, axis=0, keepdims=True)

  @staticmethod
  def tanh(x):
    return np.tanh(x)

  @staticmethod
  def linear(x):
    return x
  
  @staticmethod
  def valid_activation_functions():
      return [Activation.relu, Activation.sigmoid, Activation.softmax,
              Activation.tanh, Activation.linear]



