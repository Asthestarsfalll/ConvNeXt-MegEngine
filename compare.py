import time

import megengine as mge
import numpy as np
import torch

from models.convnext import convnext_tiny
from models.torch_models import convnext_tiny as torch_model_builder

mge_model = convnext_tiny(True)
torch_model = torch_model_builder(pretrained=True)
torch_time = meg_time = 0.0

def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)

for i in range(15):
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)
    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = softmax(mge_out)
    torch_out = softmax(torch_out)
    print(f"abs error: {np.mean(np.abs(torch_out - mge_out))}")
    print(f"numpy allclose result: {np.allclose(torch_out, mge_out, rtol=1e-3)}")

print(f"meg time: {meg_time}, torch time: {torch_time}")

