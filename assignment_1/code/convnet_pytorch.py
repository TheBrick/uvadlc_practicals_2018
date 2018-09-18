"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    layers = []
    layers.append(nn.Conv2d(3, 64, 3, 1, 1))           # conv1
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(3, 2, 1))               # maxpool1
    layers.append(nn.Conv2d(64, 128, 3, 1, 1))         # conv2
    layers.append(nn.BatchNorm2d(128))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(3, 2, 1))               # maxpool2
    layers.append(nn.Conv2d(128, 256, 3, 1, 1))        # conv3_a
    layers.append(nn.Conv2d(256, 256, 3, 1, 1))        # conv3_b
    layers.append(nn.BatchNorm2d(256))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(3, 2, 1))               # maxpool3
    layers.append(nn.Conv2d(256, 512, 3, 1, 1))        # conv4_a
    layers.append(nn.Conv2d(512, 512, 3, 1, 1))        # conv4_b
    layers.append(nn.BatchNorm2d(512))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(3, 2, 1))               # maxpool4
    layers.append(nn.Conv2d(512, 512, 3, 1, 1))        # conv5_a
    layers.append(nn.Conv2d(512, 512, 3, 1, 1))        # conv5_b
    layers.append(nn.BatchNorm2d(512))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(3, 2, 1))               # maxpool5
    layers.append(nn.AvgPool2d(1, 1, 0))               # avgpool

    self.sequential = nn.Sequential(*layers)
    self.linear = nn.Linear(512, n_classes)            # linear
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
    out = self.sequential(x)
    out = self.linear(out.reshape(x.shape[0], -1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
