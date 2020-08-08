# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from .norms.mask_layernorm import LayerNorm
from torch.nn.utils import spectral_norm

def NormSelect(norm_type, embed_dim, head_num=None):
    if norm_type == "layer":
        return LayerNorm(embed_dim)
