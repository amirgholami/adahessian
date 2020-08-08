#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami, Sheng Shen
# All rights reserved.
# This file is part of AdaHessian library.
#
# AdaHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AdaHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with adahessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
import numpy as np


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)

        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, gradsH):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = self.param_groups[0]['params']

        v = [torch.randint_like(p, high=2, device='cuda') for p in params]
        for v_i in v:
            v_i[v_i == 0] = -1
        hvs = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                tmp_output = torch.abs(hv * vi)
                hutchinson_trace.append(tmp_output) # Hessian diagonal block size is 1 here.
            elif len(param_size) == 4:  # Conv kernel
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2, 3], keepdim=True)) / vi[0, 1].numel() # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                hutchinson_trace.append(tmp_output)
        
        return hutchinson_trace

    def step(self, gradsH, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # get the Hessian diagonal
        hut_trace = self.get_trace(gradsH)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = deepcopy(gradsH[i].data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    1 - beta2, hut_trace[i], hut_trace[i])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])

                # make update
                p.data = p.data - \
                    group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss
