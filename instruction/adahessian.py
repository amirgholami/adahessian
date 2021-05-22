import math
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
import numpy as np


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
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
        spatial_average_block_size (int, optional): Spatial average block size for 1d tensors (default: (-1, -1, -1, -1) ). 
        Here for now, we only write down the tensor size from 1D to 4D. For higher dimension tensors (e.g., 5D), you can incorporate 
        the code by yourself. 
            -1 for 1D: no spatial average 
            -1 for 2D: use the entire row as the spatial average
            -1 for 3D (we assume it is a 1D Conv, you can customize it): use the channel (last dimension) of 1D Conv as spatial average
            -1 for 4D (we assume it is a 2D Conv, you can customize it): use the channel (last two dimension) of 2D Conv as spatial average
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, spatial_average_block_size=(-1, -1, -1, -1), single_gpu=True):
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
        self.single_gpu = single_gpu 
        super(Adahessian, self).__init__(params, defaults)

        self.spatial_average_block_size = spatial_average_block_size

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        v = [ 2 * torch.randint_like(p, high=2, device='cuda') - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.] = -1.
                v_i[v_i >= 0.] = 1.


        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        bs_1D, bs_2D, bs_3D, bs_4D = self.spatial_average_block_size

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()

            hv_abs = hv.abs()

            if len(param_size) <= 1:  
                # For 1D tensor, e.g.,, bias, BatchNorm, LayerNorm etc.
                # Usually, you do not need to set spatial aveging for it, i.e., Hessian diagonal block size is 1 here.
                
                if bs_1D == -1:
                    hutchinson_trace.append(hv_abs)
                else:
                    tmp_output1 = hv_abs.view(-1, bs_1D) # faltten to the N times bs_1D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_1D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 2: 
                # For 2D tensor, e.g., the matrix in the fully-connected layer.
                # This is a normal case for MLP, Transformer models. 
                # Usually, a spatial averaging needs to be used here to get the best result.

                if bs_2D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[1], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_2D) # faltten to the N times bs_2D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_2D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 3:
                # For 3D tensor, e.g., the 1D Conv layer.
                # This layer is usually used for Char-LM.

                if bs_3D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[2], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_3D) # faltten to the N times bs_3D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_3D).view(param_size)
                    hutchinson_trace.append(tmp_output3)


            elif len(param_size) == 4:  
                # For 4D tensor, e.g, the 2D Conv layer
                # This layer is usually used for CV tasks.

                if bs_4D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[2, 3], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_4D) # faltten to the N times bs_4D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_4D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            else:
                raise RuntimeError(f'You need to write your customized function to support this shape: {param_size}')

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())

        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal
        hut_traces = self.get_trace(params, grads)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = deepcopy(grads[i].data)
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
                    1 - beta2, hut_traces[i], hut_traces[i])

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
