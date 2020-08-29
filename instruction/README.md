![Block](../imgs/block_hessian_conv_matrix.png)


Here we include instrutions on how to use the block averaging used in
AdaHessian for different kernel sizes (e.g. matrix, 1D Conv, 2D Conv, etc). The
instruction is written based on the adahessian.py file in this directory. In particular, in Lines
87 -- 144
[here](https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py)
we show how to use block averaging for various different kernels. For 1D, 3D, and 4D tensors, we give
two choices to set the spatial averaging. Instructions are included in the comments. However, we wanted to emphasize that for the second approach
described in the code, one spatial averaging size (i.e. self.block_length) maybe not suitable for all different kernels. For example,
self.block_length = 5 cannot be used for 1D (with size 20), 3D (with size
20x20x5) and 4D (with size 256x256x3x3) kernels since 256x256x3x3 cannot be
divided by 5. Hence, if you want to use the second way, you may want to use
different self.block_length for different kernels. 
