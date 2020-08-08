# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import math
# import types

# import torch
# import torch.optim
# import torch.distributed as dist

# from copy import deepcopy
# import numpy as np

# from . import FairseqOptimizer, register_optimizer


# @register_optimizer('adamhess')
# class FairseqAdamhess(FairseqOptimizer):
#     """Adam optimizer for fairseq.

#     Important note: this optimizer corresponds to the "AdamW" variant of
#     Adam in its weight decay behavior. As such, it is most closely
#     analogous to torch.optim.AdamW from PyTorch.
#     """

#     def __init__(self, args, params):
#         super().__init__(args)
#         # if torch.cuda.is_available():
#             # try:
#             #     from apex.optimizers import FusedAdam as _FusedAdam  # noqa
#             #     self._optimizer = FusedAdam(params, **self.optimizer_config)
#             # except ImportError:
#         self._optimizer = Adamhess(params, **self.optimizer_config)
#         # print('do I used adamhess here?')
#         # else:
#             # self._optimizer = Adamhess(params, **self.optimizer_config)

#     @staticmethod
#     def add_args(parser):
#         """Add optimizer-specific arguments to the parser."""
#         # fmt: off
#         parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
#                             help='betas for Adam optimizer')
#         parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
#                             help='epsilon for Adam optimizer')
#         parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
#                             help='weight decay')
#         parser.add_argument('--adamhess-version', default='v2', type=str,
#                             help='adamhess version')
#         parser.add_argument('--block-length', default=1, type=int,
#                             help='if adamhess-version is blockwise, we need this number for length of the block')
#         parser.add_argument('--delay-steps', default=1, type=int,
#                             help='if adamhess-version is blockwise, we need this number for length of the block')
#         # fmt: on

#     @property
#     def optimizer_config(self):
#         """
#         Return a kwarg dictionary that will be used to override optimizer
#         args stored in checkpoints. This allows us to load a checkpoint and
#         resume training using a different set of optimizer args, e.g., with a
#         different learning rate.
#         """
#         return {
#             'lr': self.args.lr[0],
#             'betas': eval(self.args.adam_betas),
#             'eps': self.args.adam_eps,
#             'weight_decay': self.args.weight_decay,
#             'adamhess_version': self.args.adamhess_version,
#             'block_length': self.args.block_length,
#             'single_gpu': self.args.single_gpu,
#             'delay_steps': self.args.delay_steps
#         }

#     def average_params(self):
#         """Reduce Params is only used during BMUF distributed training."""
#         state_dict = self.optimizer.state_dict()
#         total_gpus = float(dist.get_world_size())

#         for _, value in state_dict["state"].items():
#             value["exp_avg"] /= total_gpus
#             value["exp_avg_sq"] /= total_gpus
#             dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
#             dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


# class Adamhess(torch.optim.Optimizer):
#     """Implements AdamHess algorithm.
#     """

#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False, adamhess_version='v2', block_length=1, delay_steps=1, single_gpu=False):
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad)
#         super(Adamhess, self).__init__(params, defaults)

#         self.version = adamhess_version
#         self.block_length = block_length
#         self.single_gpu = single_gpu
#         # print(betas, weight_decay)
#         print(f'Adamhess Version is: {self.version} {self.block_length}')
#         self.hut_steps = 1
#         self.num_step = 0 # used for record v*Hv
#         self.delay_steps = delay_steps
#     @property
#     def supports_memory_efficient_fp16(self):
#         return True


#     def get_trace(self, gradsH):
#         """
#         compute the Hessian vector product with v, at the current gradient point.
#         or compute the gradient of <gradsH,v>.
#         :param v: a list of torch tensors
#         :param gradsH: a list of torch variables
#         :return: a list of torch tensors
#         """

#         params = self.param_groups[0]['params']
#         params = list(filter(lambda x: x.requires_grad, params) )
#         # here we rm the embedding layer from the second backprop
#         # params = params[1:] #


#         # print(params)
#         # asdf
#         # w = self.param_groups[0]['weight_decay']
#         w = 0.
#         # output = [torch.zeros_like(p, device = 'cuda') for p in params]
#         output = [0. for p in params]
#         for i in range(self.hut_steps):
#             v = [torch.randint_like(p, high = 2, device = 'cuda') for p in params]

#             # this is for distributed setting
#             if not self.single_gpu:
#                 for v1 in v:
#                     dist.all_reduce(v1)
#             for v_i in v:
#                 v_i[v_i < 0.5] = -1
#                 v_i[v_i >= 0.5] = 1
#             hvs = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True,  retain_graph=True)
#             if self.version == 'layerwise': # layer-wise
#                 hutchinson_trace = [torch.abs(torch.sum(hv*vi)/vi.numel())+w for hv, vi in zip(hvs, v)]
#             elif self.version == 'elementwise': # single prameter
#                 hutchinson_trace = [torch.abs(hv*vi)+w for hv, vi in zip(hvs, v)]
#             elif self.version == 'depthwise': # depth-wise
#                 hutchinson_trace = []
#                 for hv, vi in zip(hvs, v):
#                     param_size = hv.size()
#                     if len(param_size) <= 1: # for bias and BN params
#                         tmp_output = torch.abs( hv * vi)  + w
#                         hutchinson_trace.append( tmp_output )
#                     elif len(param_size) == 2: #Conv kernel
#                         tmp_output = torch.abs( torch.sum(hv * vi, dim=[1], keepdim=True) ) / vi[0].numel() + w
#                         hutchinson_trace.append( tmp_output )
#             elif self.version == 'blockwise':
#                 hutchinson_trace = []
#                 for hv, vi in zip(hvs, v):
#                     param_size = hv.size()
#                     if len(param_size) <= 1: # for bias and BN params
#                         tmp_output = torch.abs( hv * vi)  + w
#                         hutchinson_trace.append( tmp_output )
#                     elif len(param_size) == 2: # matrix
#                         tmp_output1 = torch.abs((hv * vi + w)).view(-1, self.block_length) # faltten to the N times self.block_length
#                         tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
#                         tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
#                         hutchinson_trace.append(tmp_output3)
#             else:
#                 raise(f"{self.version} is not supported yet!")
#             output = [a + b/float(self.hut_steps) for a, b in zip(output, hutchinson_trace)]

#         # this is for distributed setting
#         if not self.single_gpu:
#             for output1 in output:
#                 dist.all_reduce(output1)
        
#         # record all results:
#         # np_list = []
#         # np_list.append(deepcopy(output[1]).cpu().numpy().flatten())
#         # np_list.append(deepcopy(output[100]).cpu().numpy().flatten())
#         # np_list.append(deepcopy(output[199]).cpu().numpy().flatten())
#         # np_list.append(deepcopy(output[256]).cpu().numpy().flatten())
#         # for i, output1 in enumerate(output):
#             # print(i, ' ** ', output1.size())
#         # np.save(f"iwslt14_de_en_result_new/adamhess_record_vhv/iter_{self.num_step}", np_list)
#         # self.num_step += 1
#         # print(hutchinson_trace)
#         return output

#     def step(self, gradsH=None, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         # if gradsH == None:
#             # raise("Error: For AdamHess, you must pass gradsH for Hessian computation")
#         # if self.num_step % self.delay_steps == 0 or self.num_step <= 8000:
#         hut_trace = self.get_trace(gradsH) 

#         # here assume we have two machines, one do gradient computation, and another do Hessian computation, we do one step delay since the Hessian computation is based on the gradient. 
#         # if self.num_step == 0:
#         #     hut_trace = self.get_trace(gradsH)
#         #     self.hut_trace_prev = deepcopy(hut_trace)
#         # else:
#         #     hut_trace = deepcopy(self.hut_trace_prev)
#         #     hut_trace_prev = self.get_trace(gradsH)

#         self.num_step += 1

#         # hut_trace = self.get_trace(gradsH)

#         for group in self.param_groups:
#             for i, p in enumerate(group['params']):
#                 if p.grad is None:
#                     continue
#                 # grad = p.grad.data.float()
#                 grad = deepcopy(gradsH[i].data.float())

#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                 amsgrad = group['amsgrad']

#                 p_data_fp32 = p.data.float()

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p_data_fp32)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
#                     if amsgrad:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
#                 else:
#                     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
#                     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
#                     if amsgrad:
#                         state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 if amsgrad:
#                     max_exp_avg_sq = state['max_exp_avg_sq']
#                 beta1, beta2 = group['betas']

#                 state['step'] += 1

#                 # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, hut_trace[i] , hut_trace[i])

#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']

#                 if amsgrad:
#                     # Maintains the maximum of all 2nd moment running avg. till now
#                     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                     # Use the max. for normalizing running avg. of gradient
#                     # denom = max_exp_avg_sq.sqrt().add_(group['eps'])
#                     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#                 else:
#                     # denom = exp_avg_sq.sqrt().add_(group['eps'])
#                     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

#                 # bias_correction1 = 1 - beta1 ** state['step']
#                 # bias_correction2 = 1 - beta2 ** state['step']
#                 step_size = group['lr'] / bias_correction1

#                 # original ADAMW in Fairseq is wrong.
#                 if group['weight_decay'] != 0:
#                     p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

#                 p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

#                 p.data.copy_(p_data_fp32)

#         return loss
