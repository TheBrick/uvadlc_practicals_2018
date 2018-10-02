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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # Create tensors of right sizes
        # LSTM part
        self.Wfx = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.Wix = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.Wgx = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.Wox = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.Wfh = nn.Parameter(torch.empty(num_hidden, num_hidden))
        self.Wih = nn.Parameter(torch.empty(num_hidden, num_hidden))
        self.Wgh = nn.Parameter(torch.empty(num_hidden, num_hidden))
        self.Woh = nn.Parameter(torch.empty(num_hidden, num_hidden))
        self.bf = nn.Parameter(torch.empty(num_hidden))
        self.bi = nn.Parameter(torch.empty(num_hidden))
        self.bg = nn.Parameter(torch.empty(num_hidden))
        self.bo = nn.Parameter(torch.empty(num_hidden))
        self.c = torch.empty(batch_size, num_hidden, device=device)  # cell state
        self.h = torch.empty(batch_size, num_hidden, device=device)  # output state
        # Linear part
        self.Wph = nn.Parameter(torch.empty(num_hidden, num_classes))
        self.bp = nn.Parameter(torch.empty(num_classes))

        # Initialize weights
        mean = 0.0
        std = 1/seq_length
        nn.init.normal_(self.Wgx, mean=mean, std=std)
        nn.init.normal_(self.Wix, mean=mean, std=std)
        nn.init.normal_(self.Wfx, mean=mean, std=std)
        nn.init.normal_(self.Wox, mean=mean, std=std)
        nn.init.normal_(self.Wgh, mean=mean, std=std)
        nn.init.normal_(self.Wih, mean=mean, std=std)
        nn.init.normal_(self.Wfh, mean=mean, std=std)
        nn.init.normal_(self.Woh, mean=mean, std=std)
        nn.init.normal_(self.Wph, mean=mean, std=std)
        nn.init.constant_(self.bi, 0.0)
        nn.init.constant_(self.bg, 0.0)
        nn.init.constant_(self.bf, 0.0)
        nn.init.constant_(self.bo, 0.0)
        nn.init.constant_(self.bp, 0.0)

        # Administrative stuff
        self.sequence_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim

    def forward(self, x):
        # Reset states
        self.c.detach_()
        self.h.detach_()
        nn.init.constant_(self.c, 0.0)
        nn.init.constant_(self.h, 0.0)

        for t in range(self.sequence_length):
            xt = x[:,t].view(self.batch_size, self.input_dim)
            # The 4 gates
            f = torch.sigmoid(xt @ self.Wfx + self.h @ self.Wfh + self.bf)
            i = torch.sigmoid(xt @ self.Wix + self.h @ self.Wih + self.bi)
            g = torch.tanh   (xt @ self.Wgx + self.h @ self.Wgh + self.bg)
            o = torch.sigmoid(xt @ self.Wox + self.h @ self.Woh + self.bo)
            # New memory state
            self.c = g * i + self.c * f
            # New output
            self.h = torch.tanh(self.c) * o

        # Linear classifier part (not using Softmax because using nn.CrossEntropyLoss)
        p = self.h @ self.Wph + self.bp
        return p
