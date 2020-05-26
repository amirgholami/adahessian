## Introduction
AdaHessian is a second order based optimizer for the neural network training based on PyTorch. The library currently supports the training of convolutional neural networks for now and will support transformer based model soon. 


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

## Citation
AdaHessian has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

* Z. Yao, A. Gholami, S. Shen, K Keutzer, M. Mahoney. ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning, under review.


