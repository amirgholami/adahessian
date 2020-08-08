# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import scipy
import numpy as np

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
            'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.args.regression_target:
            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )
        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if not self.args.regression_target:
            preds = logits.max(dim=1)[1]
            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )
            tp = ((targets == 1) * (preds == 1)).sum().item()
            fp  = ((targets == 0) * (preds == 1)).sum().item()
            fn  = ((targets == 1) * (preds == 0)).sum().item()
            tn  = ((targets == 0) * (preds == 0)).sum().item()
            logging_output.update(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn
            )

        else:
            logging_output.update(
                    preds=preds,
                    targets=targets)
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        tp = sum(log.get('tp', 0) for log in logging_outputs)
        fp = sum(log.get('fp', 0) for log in logging_outputs)
        fn  = sum(log.get('fn', 0) for log in logging_outputs)
        tn  = sum(log.get('tn', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'tp' : tp,
            'fp' : fp,
            'fn' : fn,
            'tn' : tn
        }


        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences)
            '''
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            agg_output.update(f1=2*precision*recall/(precision+recall))
            agg_output.update(mcc=(tp *tn -fp *fn)/np.sqrt((tp+fp)* (fp+fn) * (tn+fp) *(tn+fn)))
            '''

        if 'preds' in logging_outputs[0].keys():
            preds = np.concatenate(log.get('preds', np.array([])) for log in logging_outputs)
            targets = np.concatenate(log.get('targets', np.array([])) for log in logging_outputs)
            '''
            agg_output.update(pearson=scipy.stats.pearsonr(preds, targets)[0])
            agg_output.update(spearmon=scipy.stats.spearmonr(preds, targets)[0])
            '''

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
