The AdaHessian optimizer code is implemented [here](https://github.com/amirgholami/adahessian/blob/master/transformer/fairseq/optim/adahessian.py). The rest of the code for NLP is based on [fairseq repo](https://github.com/pytorch/fairseq) (v0.8.0). Please follow [this link](https://fairseq.readthedocs.io/) for a detailed documentation about the original code base, and [this link](https://github.com/pytorch/fairseq/tree/v0.8.0/examples/translation) for some examples of training baseline Transformer models for machine translation with fairseq.

We also provide [pre-trained models](#pre-trained-models) on IWSLT'14 German to English translation datasets that were used in our paper.

# Requirements and Installation
Please first install Pytorch and other dependencies:
```
conda env create -f environment.yml
```
Afterwards, install Fairseq:
```
python setup.py build develop
```

# Reproducing Paper Results
First, please refer to this [page](https://github.com/pytorch/fairseq/tree/master/examples/translation) to prepare the IWSLT'14 German to English translation datasets (or directly download the preprocessed IWSLT dataset from [here](https://drive.google.com/file/d/1fBG7DmbH0luD8EKqjviG5Equgkaxv3vv/view?usp=sharing)).

You can also run the following scripts to reproduce the results reported in Table 3 of the [paper](https://arxiv.org/pdf/2006.00719.pdf):
```
bash config/{adam, adahessian}.sh
```

# Pre-trained Models
We also provide the pre-trained model on IWSLT'14 German to English translation for reproducing the results. 

Description | Dataset | Model | Test set(s)
---|---|---|---
Transformer `small` | [IWSLT14 German-English](https://drive.google.com/file/d/1fBG7DmbH0luD8EKqjviG5Equgkaxv3vv/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/file/d/1cs34wY3NhFq1J_bGdTsMGsjGFMtVsbAs/view?usp=sharing) | IWSLT14 test set (shared vocab): <br> [download (.tbz2)](https://drive.google.com/open?id=1Vza4Yh7ev1336fWpgxGalkSLhb5dHxBa)

Example uasge: 
```
# Put the downloaded model at /adahessian/transformer/adahessian_pretrained_model.pt
# Then use the prepared script to get the result
bash config/adahessian_pretrained_model.sh
# you should get the result under pretrained_result/trans/res.txt
| Generate test with beam=5: BLEU4 = 35.90, 69.5/44.2/30.1/21.0 (BP=0.963, ratio=0.964, syslen=126373, reflen=131156)
```
