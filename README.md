## Introduction
![Block](imgs/diagonal_illustration.png)

AdaHessian is a second order based optimizer for the neural network training based on PyTorch. The library supports the training of convolutional neural networks ([image_classification](https://github.com/amirgholami/adahessian/tree/master/image_classification)) and transformer-based models ([transformer](https://github.com/amirgholami/adahessian/tree/master/transformer)). 

Please see [this paper](https://arxiv.org/pdf/2006.00719.pdf) for more details on the AdaHessian algorithm.


## Usage
Please first clone the AdaHessian library to your local system:
```
git clone https://github.com/amirgholami/adahessian.git
```

After cloning, please enter either image_classification or transformer folder for further information.

## Citation
AdaHessian has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

```text
@article{yao2020adahessian,
  title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning},
  author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Keutzer, Kurt and Mahoney, Michael W},
  journal={arXiv preprint arXiv:2006.00719},
  year={2020}
}
```