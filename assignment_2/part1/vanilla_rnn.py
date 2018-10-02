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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()


        # Create tensors of right sizes
        self.U = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.W = nn.Parameter(torch.empty(num_hidden, num_hidden))
        self.V = nn.Parameter(torch.empty(num_hidden, num_classes))
        self.bh = nn.Parameter(torch.empty(num_hidden))
        self.bp = nn.Parameter(torch.empty(num_classes))
        self.h = torch.empty(batch_size, num_hidden, device=device)

        # Initialize values
        mean = 0.0
        std = 0.001
        nn.init.normal_(self.U, mean=mean, std=std)
        nn.init.normal_(self.W, mean=mean, std=std)
        nn.init.normal_(self.V, mean=mean, std=std)
        nn.init.constant_(self.bh, 0.0)
        nn.init.constant_(self.bp, 0.0)

        # Administrative stuff
        self.sequence_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim

    def forward(self, x):
        # Reset hidden state
        self.h.detach_()
        nn.init.constant_(self.h, 0.0)

        for t in range(self.sequence_length):
            xt = x[:,t].view(self.batch_size, self.input_dim)
            self.h = xt @ self.U + self.h @ self.W + self.bh
            self.h = torch.tanh(self.h)

        out = self.h @ self.V + self.bp

        return out