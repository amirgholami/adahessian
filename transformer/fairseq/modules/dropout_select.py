# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class DropoutSelect(nn.Module):
    """docstring for D"""
    def __init__(self, dropout_type, dropout_gama=0.5, inplace=False):
        super().__init__()
        self.dropout_type = dropout_type
        self.dropout_gama = dropout_gama
        self.inplace = inplace
        dropout_alpha = 1.0
        self.dropout_alpha = dropout_alpha
        if 1 == "bernoulli":
            # multiply based
            self.dist = torch.distributions.bernoulli.Bernoulli( torch.tensor([dropout_gama]) )
        else:
            # inject different types of noise with special control of the variance
            if dropout_type == "gamma":
                self.dist = torch.distributions.gamma.Gamma( torch.tensor([dropout_alpha]), torch.tensor([dropout_gama]) ) 
            elif dropout_type == "gumbel":
                self.dist = torch.distributions.gumbel.Gumbel( torch.tensor([0.0]), torch.tensor([dropout_gama]) )
            elif dropout_type == "beta":
                self.dist = torch.distributions.beta.Beta( torch.tensor([dropout_alpha]), torch.tensor([dropout_gama]) ) 
            elif dropout_type == "laplace":
                self.dist = torch.distributions.laplace.Laplace( torch.tensor([0.0]), torch.tensor([dropout_gama]) )
            elif dropout_type == "chi":
                self.dist = torch.distributions.chi2.Chi2( torch.tensor([dropout_gama]) )
            elif dropout_type == "normal":
                self.dist = torch.distributions.normal.Normal( torch.tensor([0.0]), torch.tensor([dropout_gama]) )

    def extra_repr(self):
        return 'dropout_type={dropout_type}, dropout_gama={dropout_gama}, inplace={inplace}'.format(**self.__dict__)

    def forward(self, x, p, training=True):
        if training is False:
            return x
        if self.dropout_type == "none":
            return F.dropout( x, p=p, training=True, inplace=self.inplace )
        elif self.dropout_type == "bernoulli":
            # multiply based
            noise = self.dist.expand( x.shape ).sample().to(x.device)
            scale = p / self.dropout_gama
            x = x * noise * scale
        else:
            noise = self.dist.expand( x.shape ).sample().to(x.device)
            # inject different types of noise with special control of the variance
            if self.dropout_type == "gamma":
                scale = (p - self.dropout_alpha * self.dropout_gama) * ( self.dropout_alpha ** -0.5 )
            elif self.dropout_type == "gumbel":
                scale = (6 ** 0.5) * (p -  0.5772 *  self.dropout_gama) / np.pi
            elif self.dropout_type == "beta":
                scale = (self.dropout_alpha + self.dropout_gama) * \
                        (( ( self.dropout_alpha + self.dropout_gama + 1) / self.dropout_alpha ) ** 0.5) * \
                        ( p - self.dropout_alpha / (self.dropout_alpha + self.dropout_gama) )
            elif self.dropout_type == "chi":
                scale = ( p - self.dropout_gama ) / (2 ** 0.5)
            elif self.dropout_type == "normal":
                scale = p
            x = x + noise * scale
        return x
