import time
from functools import partial

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

from models.convnext import Block, ConvNeXt, LayerNorm
from models.torch_models import Block as TorchBlock
from models.torch_models import ConvNeXt as TorchConvNeXt
from models.torch_models import LayerNorm as TorchLayerNorm

GLOBAL_RTOL = 1e-3
BATCH_SIZE = 8
DTYPE_MAPPER = {
    # 'float16': (np.float16, torch.float16),
    'float32': (np.float32, torch.float32),
    # 'float64': (np.float64, torch.float64),
}

KWARDS_MAPPER = {
    "Block": [
        {"dim": 1024, "drop_path": 0.2, "layer_scale_init_value":1e-6}
    ],
    "ConvNeXt": [
        {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]}, # tiny
        {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]}, # small
        {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]}, # base
        {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]}, # large
    ],
    "LayerNorm": [
        {"normalized_shape": 512, "eps": 1e-6, "data_format": "channels_first"},
        {"normalized_shape": 512, "eps": 1e-6, "data_format": "channels_last"},
    ]
}


CLASS_MAPPER = {
    "Block": (Block, TorchBlock),
    "ConvNeXt": (ConvNeXt, TorchConvNeXt),
    "LayerNorm": (LayerNorm, TorchLayerNorm),
}


def generate_inputs(shape, dtype='float32'):
    inp = np.random.randn(*shape)
    types = DTYPE_MAPPER[dtype]
    mge_inp = mge.tensor(inp, dtype=types[0])
    torch_inp = torch.tensor(inp, dtype=types[1])
    return mge_inp, torch_inp


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module

def convert_state_dict(torch_module, torch_dict):
    mge_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_module, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k and not is_in_string(['norm', 'pwconv'], k):
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        mge_dict[k] = data

    return mge_dict


def is_in_string(targets: list, s: str):
    return any(t in s for t in targets)


def convert_dtype(m):
    pass


def test_func(mge_tensor, torch_tensor):
    mge_out = mge_tensor.numpy()
    if torch.cuda.is_available():
        torch_out = torch_tensor.detach().cpu().numpy()
    else:
        torch_out = torch_tensor.detach().numpy()
    result = np.isclose(mge_out, torch_out, rtol=GLOBAL_RTOL)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def get_channels(kwards):
    for n in ['dim', 'normalized_shape', 'in_channels', 'num_channels', 'normalized_shape', 'num_features']:
        if n in kwards:
            ch = kwards[n]
            if isinstance(ch, list):
                return ch
            return [ch]
    else:
        if 'dims' in kwards:
            return [3]
        return list(np.random.randint(1, 2048, size=[1]))


def main():
    print(f"Begin test with rtol = {GLOBAL_RTOL}, batch size ={BATCH_SIZE}")
    print()
    unalign_list = []
    for k, (mge_class, torch_class) in CLASS_MAPPER.items():
        kwards = KWARDS_MAPPER.get(k, [{}])
        print(f"Begin test {k}:")
        for kw in kwards:
            print(f"\t with kwards {kw}:")
            mge_module = mge_class(**kw)
            mge_module.eval()
            torch_module = torch_class(**kw)
            torch_module.eval()
            channels = get_channels(kw)
            # for sp_dim in [64, 224, 512, 1024]:
            for sp_dim in [32]:
                input_shape = (BATCH_SIZE, *channels, sp_dim, sp_dim)
                for dtype in DTYPE_MAPPER.keys():
                    mge_inp, torch_inp = generate_inputs(input_shape, dtype)
                    if "LayerNorm" in k and kw["data_format"] == "channels_last":
                        mge_inp = mge_inp.transpose(0, 2, 3, 1)
                        torch_inp = torch_inp.permute(0, 2, 3, 1)
                    print(f"\t\t with shape {mge_inp.shape}:")
                    print(f"\t\t\t with dtype {dtype}:")
                    torch_dict = torch_module.state_dict()
                    mge_dict = convert_state_dict(torch_module, torch_dict)
                    mge_module.load_state_dict(mge_dict)

                    st = time.time()
                    mge_out = mge_module(mge_inp)
                    mge_time = time.time() - st

                    st = time.time()
                    torch_out = torch_module(torch_inp)
                    torch_time = time.time() - st

                    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
                    if not allclose:
                        unalign_list.append(k)
                    print(f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    print(f"\t\t\t\ttime used: megengine: {mge_time : .4f}s, torch: {torch_time : .4f}s")
    print(f"Test down, unaligned module: {list(set(unalign_list))}")


if __name__ == "__main__":
    a = M.Conv2d(1, 1, 1)
    main()
