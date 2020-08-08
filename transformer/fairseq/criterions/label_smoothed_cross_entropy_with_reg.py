# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch 
from fairseq import utils

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from . import register_criterion


@register_criterion('label_smoothed_cross_entropy_with_reg')
class LabelSmoothedCrossEntropyCriterionWithReg(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.reg_lambda_hidden = args.reg_lambda_hidden
        self.reg_lambda_div = args.reg_lambda_div
        self.reg_lambda_consis = args.reg_lambda_consis
        self.reg_lambda_decov = args.reg_lambda_decov
        self.reg_lambda_pre = args.reg_lambda_pre
        self.reg_lambda_fract = args.reg_lambda_fract
        self.hook_flag = False

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        super(LabelSmoothedCrossEntropyCriterionWithReg,
              LabelSmoothedCrossEntropyCriterionWithReg).add_args(parser)
        parser.add_argument('--reg-lambda-hidden', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        parser.add_argument('--reg-lambda-div', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        parser.add_argument('--reg-lambda-consis', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        parser.add_argument('--reg-lambda-fract', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        parser.add_argument('--reg-lambda-decov', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        parser.add_argument('--reg-lambda-pre', default=0.0, type=float, metavar='D',
                            help='weight for the regularization loss')
        # div_loss scale: 0.3, consis_loss scale: 0.3, reg_loss scale: 1.3
        # pre_loss scale: 0.1
        # original_loss scale: 9.3

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        self.decov_hooks = []
        self.pre_hooks = []
        def hook_fn(module, input, output):
            # if isinstance(output, tuple):
            #     tmp_tensor = output[0]
            # else:
            #     tmp_tensor = output
            if self.reg_lambda_decov != 0.0:
                decov_tmp = input[0]#.contiguous().view( -1, input[0].shape[-1] )
                decov_tmp = decov_tmp - decov_tmp.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
                decov_tmp = decov_tmp.mean(dim=0)
                # decov_tmp = torch.bmm(decov_tmp, decov_tmp.transpose(1, 2)) / decov_tmp.shape[1]
                # decov_tmp = 0.5 * ( decov_tmp.norm(dim=(1,2)) - ( torch.diag(decov_tmp[0]).unsqueeze(0) * decov_tmp ).norm(dim=(1,2)) )
                # decov_tmp = decov_tmp.norm() / decov_tmp.shape[0]
                # decov_tmp = input[0].contiguous().view( -1, input[0].shape[-1] )
                # decov_tmp = decov_tmp - decov_tmp.mean(dim=0)
                decov_tmp = torch.mm(decov_tmp, decov_tmp.transpose(1, 0)) / decov_tmp.shape[0]
                decov_tmp = 0.5 * ( decov_tmp.norm() - ( torch.diag(decov_tmp).unsqueeze(0) * decov_tmp ).norm() )
                # decov_tmp = torch.abs(decov_tmp).mean()
                self.decov_hooks.append( torch.abs(decov_tmp) )
                del decov_tmp

            if self.reg_lambda_pre != 0.0:
                pre_tmp = torch.abs( input[0] ).sum(dim=-1).mean()
                self.pre_hooks.append( pre_tmp )
                del pre_tmp
            # torch.cuda.empty_cache()
            
        if not self.hook_flag:
            for layer in model.encoder.layers:
                # before the norlization module
                layer.self_attn_layer_norm.register_forward_hook(hook_fn)            
                layer.final_layer_norm.register_forward_hook(hook_fn)
            
            for layer in model.decoder.layers:
                layer.final_layer_norm.register_forward_hook(hook_fn)        
                layer.self_attn_layer_norm.register_forward_hook(hook_fn)
                layer.encoder_attn_layer_norm.register_forward_hook(hook_fn)           
            self.hook_flag = True

        net_output = model(**sample['net_input'], return_all_hiddens=True)

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if self.reg_lambda_decov != 0.0:
            decov_loss = 0.0
            for idx, decov_inp in enumerate(self.decov_hooks):
                decov_loss += decov_inp / len( self.decov_hooks )
            decov_loss = decov_loss * sample['ntokens']

        if self.reg_lambda_pre != 0.0:
            pre_loss = 0.0
            for idx, pre_inp in enumerate(self.pre_hooks):
                pre_loss += pre_inp / len( self.pre_hooks ) / (len( model.encoder.layers ) + len(model.decoder.layers) )
            pre_loss = pre_loss * sample['target'].size(0)

        del self.decov_hooks, self.pre_hooks
        torch.cuda.empty_cache()

        def check_mask_expert(norm):
            if norm is None:
                return None
            return norm.__dict__.get("mask_expert", None)


        mask_experts_enc = [ check_mask_expert( model.encoder.layer_norm ) ]
        for layer in model.encoder.layers:
            mask_experts_enc.append( check_mask_expert( layer.final_layer_norm ) )
            mask_experts_enc.append( check_mask_expert( layer.self_attn_layer_norm ) )

        mask_experts_dec = [ check_mask_expert( model.decoder.layer_norm ) ]
        for layer in model.decoder.layers:
            mask_experts_dec.append( check_mask_expert( layer.final_layer_norm ) )
            mask_experts_dec.append( check_mask_expert( layer.self_attn_layer_norm ) )
            mask_experts_dec.append( check_mask_expert( layer.encoder_attn_layer_norm ) )

        mask_experts_enc = list(filter(lambda x: x is not None, mask_experts_enc))
        mask_experts_dec = list(filter(lambda x: x is not None, mask_experts_dec))

        mask_experts_enc_flag, mask_experts_dec_flag = len(mask_experts_enc), len(mask_experts_dec)
        if mask_experts_enc:
            mask_experts_enc = torch.stack( mask_experts_enc )
        if mask_experts_dec:
            mask_experts_dec = torch.stack( mask_experts_dec )

        div_loss, consis_loss = None, None
        if mask_experts_enc_flag:
            div_loss = ( mask_experts_enc.std(dim=1) / mask_experts_enc.mean(dim=1) ).norm(dim=-1).mean(dim=-1)
            consis_loss = ( mask_experts_enc.std(dim=0) / mask_experts_enc.mean(dim=0) ).norm(dim=-1).mean(dim=-1)

        if mask_experts_dec_flag:

            if div_loss is None:
                div_loss = ( mask_experts_dec.std(dim=1) / mask_experts_dec.mean(dim=1) ).norm(dim=-1).mean(dim=-1)
                consis_loss += ( mask_experts_dec.std(dim=0) / mask_experts_dec.mean(dim=0) ).norm(dim=-1).mean(dim=-1)
            else:
                div_loss += ( mask_experts_dec.std(dim=1) / mask_experts_dec.mean(dim=1) ).norm(dim=-1)
                consis_loss += ( mask_experts_dec.std(dim=0) / mask_experts_dec.mean(dim=0) ).norm(dim=-1).mean(dim=-1)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        reg_loss = 0.0
        logging_output = dict()
        dec_len, enc_len = len(net_output[1]['inner_states']), len(net_output[1]['encoder_states'])
        for inner_enc in net_output[1]['encoder_states'][2:]:
            reg_loss += (inner_enc.norm(dim=-1) - net_output[1]['encoder_states'][1].norm(dim=-1)).abs().sum(dim=1).mean(dim=0) / (enc_len-1)

        for inner_dec in net_output[1]['inner_states'][2:]:
            reg_loss += (inner_dec.norm(dim=-1) - net_output[1]['inner_states'][1].norm(dim=-1)).abs().sum(dim=1).mean(dim=0) / (dec_len-1)

        if self.reg_lambda_decov != 0.0:
            logging_output['reg_loss_decov'] = utils.item(decov_loss.data)
            loss += self.reg_lambda_decov * decov_loss

        if self.reg_lambda_pre != 0.0:
            logging_output['reg_loss_pre'] = utils.item(pre_loss.data)
            loss += self.reg_lambda_pre * pre_loss

        # Compute alignment loss only for training set and non dummy batches.
        if reg_loss != 0.0 and self.reg_lambda_hidden != 0.0:
            logging_output['reg_loss_hidden'] = utils.item(reg_loss.data)
            loss += self.reg_lambda_hidden * reg_loss

        if div_loss is not None and self.reg_lambda_div != 0.0:
            div_loss = div_loss * sample_size
            logging_output['reg_loss_div'] = utils.item(div_loss.data)
            loss += self.reg_lambda_div * div_loss

        if consis_loss is not None and self.reg_lambda_consis != 0.0:
            consis_loss = consis_loss * sample_size
            logging_output['reg_loss_consis'] = utils.item(consis_loss.data)
            loss += self.reg_lambda_consis * consis_loss

        logging_output.update({
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        })

        return loss, sample_size, logging_output



    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'reg_loss_hidden': sum(log.get('reg_loss_hidden', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reg_loss_div': sum(log.get('reg_loss_div', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reg_loss_consis': sum(log.get('reg_loss_consis', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reg_loss_decov': sum(log.get('reg_loss_decov', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reg_loss_pre': sum(log.get('reg_loss_pre', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reg_loss_fract': sum(log.get('reg_loss_fract', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
