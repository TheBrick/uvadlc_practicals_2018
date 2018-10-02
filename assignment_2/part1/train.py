################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # fixme

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length,
                           config.input_dim,
                           config.num_hidden,
                           config.num_classes,
                           config.batch_size,
                           device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length,
                     config.input_dim,
                     config.num_hidden,
                     config.num_classes,
                     config.batch_size,
                     device)

    model = model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if config.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    if config.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Better timing
    t1 = time.time()

    output_file = "results/{}_len{}_{}_batch{}.csv".format(config.model_type, config.input_length, config.optim, config.batch_size)
    f = open(output_file, "w+")
    f.write("step;accuracy\n")
    f.close()

    accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Prepare for torch
        x = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
        y = torch.tensor(batch_targets, dtype=torch.long, device=device)

        # Forward pass
        predictions = model(x)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        if step % config.print_every == 0:
            # Just for time measurement (changed to measure average every time it prints)
            t2 = time.time()
            examples_per_second = (10*config.batch_size)/float(t2-t1)

            accuracy = torch.sum(predictions.argmax(dim=1) == y).to(torch.float32) / len(batch_inputs)
            accuracies.append(accuracy)
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

            f = open(output_file, "a+")
            f.write("%d;%f\n" % (step, accuracy))
            f.close()

            # Only for time measurement of step through network
            t1 = time.time()

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    print(np.percentile(np.array(accuracies), 95))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--optim', type=str, default="Adam", help='Which optimizer to use (RMSprop, Adam)')

    config = parser.parse_args()

    # Train the model

    # lengths = range(6, 15)
    # for length in lengths:
    #     config.input_length = length
    train(config)