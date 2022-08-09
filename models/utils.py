import math

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = mge.tensor(1 - drop_prob, dtype=x.dtype)
    size = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + mge.random.normal(mean=0, std=1, size=size)
    random_tensor = F.floor(random_tensor)  # binarize
    print(random_tensor)
    output = x / keep_prob * random_tensor
    return output

class DropPath(M.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerNorm(M.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = mge.Parameter(F.ones((normalized_shape)))
        self.bias = mge.Parameter(F.zeros((normalized_shape)))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.nn.layer_norm(x, self.normalized_shape, True, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdims=True)
            s = F.pow((x - u), 2).mean(1, keepdims=True)
            x = (x - u) / F.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    x = mge.tensor(np.random.randn(2, 3, 224, 224))
    out = drop_path(x, 0.3, True, True)
    print(out.shape)
    a = out[0][0]
    print(F.sum(a==0))
