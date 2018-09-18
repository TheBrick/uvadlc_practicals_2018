"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
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
  p = np.argmax(predictions, axis=1)
  t = np.argmax(targets, axis=1)
  accuracy = np.sum(p == t) / b
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
  data = cifar10_utils.get_cifar10(FLAGS.data_dir)
  n_inputs = 3*32*32
  n_classes = 10
  model = MLP(n_inputs, dnn_hidden_units, n_classes)
  loss_fn = CrossEntropyModule()
  max_accuracy = 0.0
  start_time = time.perf_counter()
  for step in range(1, FLAGS.max_steps+1):
    x, targets = data['train'].next_batch(FLAGS.batch_size)
    input = x.reshape((FLAGS.batch_size, -1))
    predictions = model.forward(input)
    gradient = loss_fn.backward(predictions, targets)
    model.backward(gradient)
    model.step(FLAGS.learning_rate)
    if step == 1 or step % FLAGS.eval_freq == 0:
      training_loss = loss_fn.forward(predictions, targets)
      test_predictions = model.forward(data['test'].images.reshape(data['test'].num_examples, -1))
      test_loss = loss_fn.forward(test_predictions, data['test'].labels)
      test_acc = accuracy(test_predictions, data['test'].labels)
      if test_acc > max_accuracy:
        max_accuracy = test_acc
      print("step %d/%d: training loss: %.3f test loss: %.3f accuracy: %.1f%%"
              % (step, FLAGS.max_steps, training_loss, test_loss, test_acc*100))

  time_taken = time.perf_counter() - start_time
  print("Done. Scored %.1f%% in %.1f seconds." % (max_accuracy*100, time_taken))
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
  FLAGS, unparsed = parser.parse_known_args()

  main()