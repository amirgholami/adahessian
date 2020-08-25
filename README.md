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

## For different kernel size (e.g, matrix, Conv1D, Conv2D, etc)
We found out it would be helpful to add instruction about how to adopt AdaHessian for your own models and problems. Hence, we add a prototype version of AdaHessian as well as some useful comments in the instruction folder. 

## External implementations and discussions
We thank a lot for all the individuals and groups who have implemented their own AdaHessian versions and discussed AdaHessian. We include the following links in case you are interested to learn more about AdaHessian.

Description | Link | New Features
---|---|---
Reddit Discussion | [Link](https://www.reddit.com/r/MachineLearning/comments/i76wxd/adahessian_an_adaptive_second_orderoptimizer_for/) | --
Fast.ai Discussion | [Link](https://forums.fast.ai/t/adahessian/76214/15) | -- 
Best-Deep-Learning-Optimizers Code| [Link](https://github.com/lessw2020/Best-Deep-Learning-Optimizers/tree/master/adahessian) | --
ada-hessian Code | [Link](https://github.com/davda54/ada-hessian) | Support Delayed Hessian Update
AdaHessian Analysis | [Link](https://github.com/githubharald/analyze_ada_hessian) | Analyze AdaHessian on a 2D example

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
