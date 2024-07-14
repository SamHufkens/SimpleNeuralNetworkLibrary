import numpy as np
from Tools.activation import Activation
from Tools.layer import Layer

class NeuralNetwork:

  def __init__(self, n_input, n_output, output_activation_function):
    self.n_input = n_input
    self.n_output = n_output
    self.output_activation_function = output_activation_function
    self.hidden_layers = []
    self.loss_list = []
    self.paramaters = {}
    self.gradients = {}
    self.loss_function = None
    self.learning_rate = None

  """
  Add hidden layer to nn
  """
  def add_hidden_layer(self, n_neurons, activation_function):

    if activation_function not in Activation.valid_activation_functions():
      raise Exception("Invalid activation function")
    
    if (n_neurons <= 0):
      raise Exception("Invalid number of neurons")

    layer = Layer(n_neurons, activation_function)
    self.hidden_layers.append(layer)

  """
  Initialize weights and biases for nn
  """
  def init_paramaters(self):
    
    n_input_current = self.n_input # store initial input size for first layer

    for i, layer in enumerate(self.hidden_layers):

      w = np.random.randn(layer.n_neurons, n_input_current)*0.01
      b = np.zeros((layer.n_neurons, 1))

      self.paramaters[f"w{i+1}"] = w
      self.paramaters[f"b{i+1}"] = b

      n_input_current = layer.n_neurons # Update input size for the next layer

    # Weights and biases for output layer
    output_index = len(self.hidden_layers) + 1
    w_output = np.random.randn(self.n_output, n_input_current)*0.01
    b_output = np.zeros((self.n_output, 1))

    self.paramaters[f"w{output_index}"] = w_output
    self.paramaters[f"b{output_index}"] = b_output

  """"
  Forward all data through the nn
  """
  def forward_propagation(self, x):

    forward_cache = {}

    a = x # Store input as initial activation

    for i in range(len(self.hidden_layers)+1):
        
        w = self.paramaters[f'w{i+1}'] # Weights
         
        b = self.paramaters[f'b{i+1}'] # Biases

        z = np.dot(w, a) + b

        # Check if we're at last layer
        if i < len(self.hidden_layers):
            a = self.hidden_layers[i].activation_function(z)
        else:
            a = self.output_activation_function(z)

        forward_cache[f'z{i+1}'] = z
        forward_cache[f'a{i+1}'] = a

    return forward_cache

  """"
  Calculate loss after one EPOCH
  """
  def calculate_loss(self, y, y_pred):
    return self.loss_function(y, y_pred)

  # Compile the nn
  def compile(self, loss_function, learning_rate=0.001):
    self.init_paramaters()
    self.loss_function = loss_function
    self.learning_rate = learning_rate

  """
  Go through nn in reverse, calculate derivitives
  """
  def back_propagation(self, x, y, forward_cache):

      m = x.shape[1]

      # First, calculations for output layer
      a_output = forward_cache[f'a{len(self.hidden_layers)+1}']
      dz_output = (a_output - y)

      dw_output = (1 / m) * np.dot(dz_output, forward_cache[f'a{len(self.hidden_layers)}'].T)
      db_output = (1/m) * np.sum(dz_output, axis=1, keepdims=True)

      self.gradients[f'dw{len(self.hidden_layers)+1}'] = dw_output
      self.gradients[f'db{len(self.hidden_layers)+1}'] = db_output

      dz_next = dz_output

      # Then, calculations for hidden layers in reverse
      for i in range(len(self.hidden_layers), 0, -1):

        w_current = self.paramaters[f'w{i}']
        a_current = forward_cache[f'a{i}']

        derivative = None
        if self.hidden_layers[i-1].activation_function == Activation.relu:
          derivative = Activation.derivative_relu(a_current)
        elif self.hidden_layers[i-1].activation_function == Activation.tanh:
          derivative = Activation.derivative_tanh(a_current)
        else:
          raise Exception("No derrivative for this activation function")

        dz_current = (1/m)*np.dot(self.paramaters[f'w{i+1}'].T, dz_next)*derivative

        if i > 1:
          dw_current = (1/m)*np.dot(dz_current, forward_cache[f'a{i-1}'].T)
        else:
          dw_current = (1/m)*np.dot(dz_current, x.T)

        db_current = (1/m)*np.sum(dz_current, axis = 1, keepdims = True)

        self.gradients[f'dw{i}'] = dw_current
        self.gradients[f'db{i}'] = db_current

        dz_next = dz_current

  """"
  Update all weights and biases
  """
  def update_parameters(self):

    for i in range(len(self.hidden_layers)+1):

        w = self.paramaters[f'w{i+1}']
        b = self.paramaters[f'b{i+1}']

        dw = self.gradients[f'dw{i+1}']
        db = self.gradients[f'db{i+1}']

        w = w - self.learning_rate * dw
        b = b - self.learning_rate * db

        self.paramaters[f'w{i+1}'] = w
        self.paramaters[f'b{i+1}'] = b

  """
  Training loop
  """
  def train(self, x, y, epochs):

    if epochs <= 0:
      raise Exception("Number of epochs can't be zero or negative")

    for i in range(epochs):

      cache = self.forward_propagation(x)

      loss = self.calculate_loss(y, cache[f'a{len(self.hidden_layers)+1}'])

      self.back_propagation(x, y, cache)

      self.update_parameters()

      if (i % (epochs/10)==0):
        print(f"Loss after epoch {i} is", loss)

      self.loss_list.append(loss)

  """
  Calculate accuracy
  """
  def accuracy(self, inp, labels):
    cache = self.forward_propagation(inp)
    a_out = cache[f'a{len(self.hidden_layers) + 1}']
    predictions = np.argmax(a_out, axis=0)
    labels = np.argmax(labels, axis=0)
    return np.mean(predictions == labels) * 100

