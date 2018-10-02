# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from part3.dataset import TextDataset
from part3.model import TextGenerationModel

################################################################################

def sample_greedy(output):
    return output.argmax(dim=2).squeeze().item()

def sample_mixed(output, temperature):
    dist = torch.softmax(output.squeeze()/temperature, dim=0)
    sample = torch.multinomial(dist, 1)
    return sample.item()

def sample_model_randomly (model, length, dataset, device, temperature):
    # Take random character from vocabulary as int
    random_char = torch.randint(dataset.vocab_size, (1, 1), dtype=torch.long, device=device)
    return finish_phrase(model, random_char, length, dataset, device, temperature)

def finish_phrase(model, phrase, length, dataset, device, temperature):
    with torch.no_grad():
        # Convert it to one-hot representation
        phrase_onehot = one_hot(phrase, dataset.vocab_size)
        # Run through model
        out, (h, c) = model(phrase_onehot)
        # Sample from output distribution
        sample = sample_mixed(out[:,-1,:], temperature)

        samples = phrase.view(-1).tolist()
        samples.append(sample)
        for t in range(length - 1):
            # Convert previous sample to one-hot
            input = one_hot(torch.tensor(sample, dtype=torch.long, device=device).view(1,-1), dataset.vocab_size)
            out, (h, c) = model(input, (h, c))
            sample = sample_mixed(out, temperature)
            samples.append(sample)

        text = dataset.convert_to_string(samples)
        return text

def one_hot (batch, vocab_size):
    onehot_size = list(batch.shape)
    onehot_size.append(vocab_size)
    onehots = torch.zeros(onehot_size, device=batch.device)
    onehots.scatter_(2, batch.unsqueeze(-1), 1)
    return onehots

def train(config):

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # No parameter this time? Okay

    # Initializing dataset first because vocab size is needed for LSTM parameters
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size,
                                config.seq_length,
                                dataset.vocab_size,
                                config.lstm_num_hidden,
                                config.lstm_num_layers)
    model = model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Write settings to output so that they show up in log file
    print(config)

    # Better timing
    t1 = time.time()

    # Calculate number of epochs needed to run for --training_steps number of batches.
    epoch_count = (int)(config.train_steps / len(data_loader)) + 1
    for epoch in range(epoch_count):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            total_step_counter = epoch * len(data_loader) + step

            # Prepare for torch
            x = torch.stack(batch_inputs, dim=1).to(device)
            x = one_hot(x, dataset.vocab_size)
            y = torch.stack(batch_targets, dim=1).to(device)

            # Forward pass
            predictions, _ = model(x)
            loss = criterion(predictions.transpose(2,1), y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_step_counter % config.print_every == 0:
                # Just for time measurement (changed to measure average every time it prints)
                t2 = time.time()
                examples_per_second = (config.print_every*config.batch_size)/float(t2-t1)

                accuracy = torch.sum(predictions.argmax(dim=2) == y).to(torch.float32) / (config.batch_size * config.seq_length)
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), total_step_counter,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                ))
                # Only for time measurement of step through network
                t1 = time.time()

            # Generate some sentences by sampling from the model
            if total_step_counter % config.sample_every == 0:
                text = sample_model_randomly(model, config.seq_length, dataset, device, config.temperature)
                print("Text sample (temp=%.1f): \"%s\"" % (config.temperature, text))

                # Only for time measurement of step through network
                t1 = time.time()

            if total_step_counter == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
        print("Finished epoch %d/%d" % (epoch, epoch_count))
    print('Done training.')

    # Save model to file for later exploitation
    torch.save(model, config.txt_file + ".model.pt")

    # Print a bunch of long random samples
    temperatures = [0.01, 0.5, 1.0, 2.0]
    for temperature in temperatures:
        for i in range(5):
            text = sample_model_randomly(model, 1000, dataset, device, temperature)
            print("Text sample (temp=%.1f): \"%s\"" % (temperature, text))


################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Custom parameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
