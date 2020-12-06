# AdaHessian in TensorFlow 2.0 on CIFAR-10
The code is modified from [this repo](https://github.com/gahaalt/ResNets-in-tensorflow2). Our adahessian optimizer code is [here](https://github.com/amirgholami/adahessian/blob/master/adahessian_tf/adahessian.py).

## Usage
Please first install TensorFlow 2.0 and other dependencies:
```
conda env create -f environment.yml
```
Then, please activate your conda environment:
```
conda activate adahessian_tf
```

After all the previous steps, let's train the model! 
```
export CUDA_VISIBLE_DEVICES=0; python run_experiments.py
```
Note that you can train the model with SGD/Adam by modifying the experiments.yaml file. 

We also include the training log file in logs for resnet20 for your reference.