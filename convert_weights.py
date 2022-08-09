import argparse
import os

import megengine
import numpy as np
import torch

from models.convnext import *
from models.torch_models import model_urls


def main(torch_name):
    url = model_urls[torch_name]
    torch_state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)['model']
    new_dict = {}
    model = convnext_tiny()
    s = model.state_dict()

    for k, v in torch_state_dict.items():
        data = v.numpy()
        if len(data.shape) != len(s[k].shape) and not 'dwconv.weight' in k:
                data = data.reshape(1, -1, 1, 1)
        if 'dwconv.weight' in k:
            data = np.expand_dims(data, 1)
        new_dict[k] = data

    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, "./pretrained/convnext_tiny_1k_224_ema.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="which model to convert from torch to megengine",
    )
    args = parser.parse_args()
    main(args.model)
