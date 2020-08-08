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
import types

import torch
import torch.optim
import torch.distributed as dist

from copy import deepcopy
import numpy as np

from . import FairseqOptimizer, register_optimizer


@register_optimizer('adahessian')
class FairseqAdahess(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Adahess(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--block-length', default=1, type=int,
                            help='We use this number for length of the hessian average block')
        parser.add_argument('--hessian-power', type=float, default=1, metavar='H',
                            help='Hessian power')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'block_length': self.args.block_length,
            'single_gpu': self.args.single_gpu,
            'hessian_power': self.args.hessian_power
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class Adahess(torch.optim.Optimizer):
    """Implements AdamHess algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, block_length=1, hessian_power=1, single_gpu=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adahess, self).__init__(params, defaults)

        self.block_length = block_length
        self.single_gpu = single_gpu
        self.hessian_power = hessian_power

    def get_trace(self, gradsH):
        """
        compute the Hessian vector product with v, at the current gradient point.
        or compute the gradient of <gradsH,v>.
        :param v: a list of torch tensors
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = self.param_groups[0]['params']
        params = list(filter(lambda x: x.requires_grad, params) )

        v = [torch.randint_like(p, high = 2) for p in params]

        # this is for distributed setting
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        for v_i in v:
            v_i[v_i < 0.5] = -1
            v_i[v_i >= 0.5] = 1

        hvs = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True,  retain_graph=True)

        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()
            if len(param_size) <= 1: # for Bias and LN 
                tmp_output = torch.abs( hv * vi)  + 0.
                hutchinson_trace.append( tmp_output )
            elif len(param_size) == 2: # Matrix
                tmp_output1 = torch.abs((hv * vi + 0.)).view(-1, self.block_length) # faltten to the N times self.block_length
                tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
                tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
                hutchinson_trace.append(tmp_output3)
        

        # this is for distributed setting
        if not self.single_gpu:
            for output1 in output:
                dist.all_reduce(output1)
        
        return hutchinson_trace

    def step(self, gradsH=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        
        hut_trace = self.get_trace(gradsH)
        

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                # grad = p.grad.data.float()
                grad = deepcopy(gradsH[i].data.float())

                if grad.is_sparse:
                    raise RuntimeError('AdaHessian does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_hessian_diag_sq'] = state['exp_hessian_diag_sq'].type_as(p_data_fp32)
                    

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(1 - beta2, hut_trace[i] , hut_trace[i])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                
                if self.hessian_power < 1:
                    denom = ((exp_hessian_diag_sq.sqrt() / math.sqrt(bias_correction2)) ** self.hessian_power).add_(group['eps'])
                else:
                    denom = (exp_hessian_diag_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # do weight decay
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
