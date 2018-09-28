"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import torch
import torch.nn as nn
import cifar10_utils
import time

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  b = targets.shape[0]
  p = torch.argmax(predictions, dim=1)
  t = torch.argmax(targets, dim=1)
  accuracy = torch.sum(p == t).to(torch.float32) / b
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = cifar10_utils.get_cifar10(FLAGS.data_dir)
  n_inputs = 3*32*32
  n_classes = 10
  batches_per_epoch = (int) (data['test'].images.shape[0] / FLAGS.batch_size)  # need this for test set
  model = MLP(n_inputs, dnn_hidden_units, n_classes).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = None
  if FLAGS.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
  if FLAGS.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
  if FLAGS.optimizer == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
  max_accuracy = 0.0
  start_time = time.perf_counter()
  for step in range(1, FLAGS.max_steps+1):
    x, y = get_batch(data, 'train', FLAGS.batch_size, device)
    predictions = model.forward(x)
    training_loss = loss_fn(predictions, y.argmax(dim=1))
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()
    if step == 1 or step % FLAGS.eval_freq == 0:
      with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for test_batch in range(batches_per_epoch):
          x, y = get_batch(data, 'test', FLAGS.batch_size, device)
          predictions = model(x)
          test_loss += loss_fn(predictions, y.argmax(dim=1)) / batches_per_epoch
          test_acc += accuracy(predictions, y) / batches_per_epoch
        if test_acc > max_accuracy:
          max_accuracy = test_acc
        print("step %d/%d: training loss: %.3f test loss: %.3f accuracy: %.1f%%"
              % (step, FLAGS.max_steps, training_loss, test_loss, test_acc*100))

  time_taken = time.perf_counter() - start_time
  csv = open("results.csv", "a+")
  csv.write("%s;%s;%f;%f;%f;%d;%d;%d;%f;%.3f\n" % (
            FLAGS.dnn_hidden_units,
            FLAGS.optimizer,
            FLAGS.learning_rate,
            FLAGS.momentum,
            FLAGS.weight_decay,
            FLAGS.batch_size,
            FLAGS.max_steps,
            FLAGS.eval_freq,
            max_accuracy,
            time_taken))
  csv.close()
  print("Done. Scored %.1f%% in %.1f seconds." % (max_accuracy*100, time_taken))

def get_batch(data, type, size, device):
  x, y = data[type].next_batch(size)
  x = torch.tensor(x, dtype=torch.float32, device=device)
  y = torch.tensor(y, dtype=torch.uint8, device=device)
  return x, y
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--optimizer', type = str, default = "SGD",
                      help='\'Adam\' or \'SGD\' or \'RMSprop\'')
  parser.add_argument('--weight_decay', type = float, default = 0.0,
                      help='Weight decay')
  parser.add_argument('--momentum', type = float, default = 0.0,
                      help='Momentum for SGD solver')
  FLAGS, unparsed = parser.parse_known_args()

  main()