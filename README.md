## Introduction
![Block](imgs/diagonal_illustration.png)

AdaHessian is a second order based optimizer for the neural network training based on PyTorch. The library supports the training of convolutional neural networks for now and will support transformer-based models soon. 

Please see [this paper](https://arxiv.org/pdf/2006.00719.pdf) for more details on the AdaHessian algorithm.


## Usage
Please first clone the AdaHessian library to your local system:
```
git clone https://github.com/amirgholami/adahessian.git
```

After cloning, please use Anaconda to install all the dependencies:
```
conda env create -f environment.yml
```

Then, please activate your conda environment:
```
conda activate adahessian
```

After all the previous steps, let's train the model! 
```
export CUDA_VISIBLE_DEVICES=0; python main.py [--batch-size] [--test-batch-size] [--epochs] [--lr] [--lr-decay] [--lr-decay-epoch] [--seed] [--weight-decay] [--depth] [--optimizer]

optional arguments:
--batch-size                training batch size (default: 256)
--test-batch-size           testing batch size (default:256)
--epochs                    total number of training epochs (default: 160)
--lr                        initial learning rate (default: 0.15)
--lr-decay                  learning rate decay ratio (default: 0.1)
--lr-decay-epoch            epoch for the learning rate decaying (default: 80, 120)
--seed                      used to reproduce the results (default: 1)
--weight-decay              weight decay value (default: 5e-4)
--depth                     depth of resnet (default: 20)
--optimizer                 optimizer used for training (default: adahessian)
```

Note that, to reproduce our results in the paper, please use the following arguments for different optimizer:
```
SGD:
--lr 0.1
--optimizer sgd

ADAM:
--lr 0.001
--optimizer adam

ADAMW:
--lr 0.01
--optimizer adamw

AdaHessian:
--lr 0.15
--optimizer adahessian
```

You can also run the following scripts to reproduce the results reported in Table 2 of the [paper](https://arxiv.org/pdf/2006.00719.pdf):
```
bash config/resnet20_cifar10/{sgd, adam, adamw, adahessian}.sh

bash config/resnet32_cifar10/{sgd, adam, adamw, adahessian}.sh
```
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



