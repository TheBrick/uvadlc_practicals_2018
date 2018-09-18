"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.linear_layers = []
    number_of_inputs = n_inputs
    for hidden_layer_size in n_hidden:
      linear = LinearModule(number_of_inputs, hidden_layer_size)
      self.linear_layers.append(linear)
      number_of_inputs = hidden_layer_size
    linear = LinearModule(number_of_inputs, n_classes)
    self.linear_layers.append(linear)
    self.relu_layer = ReLUModule()
    self.softmax_layer = SoftMaxModule()
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for layer in self.linear_layers[:-1]:
      out = layer.forward(out)
      out = self.relu_layer.forward(out)
    out = self.linear_layers[-1].forward(out)
    out = self.softmax_layer.forward(out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dout = self.softmax_layer.backward(dout)
    dout = self.linear_layers[-1].backward(dout)
    for layer in reversed(self.linear_layers[:-1]):
      dout = self.relu_layer.backward(dout)
      dout = layer.backward(dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return

  def step (self, learning_rate):
    for layer in self.linear_layers:
      layer.params['weight'] -= learning_rate * layer.grads['weight']
      layer.params['bias']   -= learning_rate * layer.grads['bias']
    return
