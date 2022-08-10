# ConvNeXt-MegEngine

The MegEngine Implementation of ConvNeXt.

## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/

```bash
python convert_weights.py -m convnext_tiny_1k
```

Test on ImageNet (Not sure whether it will work)

```bash
python test.py -h
```

```bash
python test.py --arch convnext_tiny_1k --ckpt path/to/ckpt --ngpus 1 --workers 8 --print-freq 20 --val-batch-size 16
```

Import from megengine.hub:

Way 1:

```python
from  megengine import hub
modelhub = hub.import_module(repo_info='asthestarsfalll/convnext-megengine', git_host='github.com')

# load ConvNeXt model and custom on you own
convnext = modelhub.ConvNeXt(num_classes=10, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])

# load pretrained model 
pretrained_model = modelhub.convnext_tiny(pretrained=True) 
```

Way 2:

```python
from  megengine import hub

# load ConvNeXt model and custom on you own
model_name = 'ConvNeXt'
kwards = dict(
    num_classes=10, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]
)
model = hub.load(
    repo_info='asthestarsfalll/convnext-megengine', entry=model_name, git_host='github.com', **kwards)

# load pretrained model 
model_name = 'convnext_tiny'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/convnext-megengine', entry=model_name, git_host='github.com', pretrained=True)
```

Currently only support convnext_tiny, you can run convert_weights.py to convert other models.
For example:

```bash
  python convert_weights.py -m convnext_small_1k
```

Then load state dict manually.

```python
model = modelhub.convnext_small()
model.load_state_dict(mge.load('./pretrained/convnext_small_1k.pkl'))
# or
model_name = 'convnext_small'
model = hub.load(
    repo_info='asthestarsfalll/convnext-megengine', entry=model_name, git_host='github.com')
model.load_state_dict(mge.load('./pretrained/convnext_small_1k.pkl'))
```

## TODO

- [ ] add object detection codes
- [ ] add semantic segmentation codes
- [ ] add train codes

## Reference

[The official implementation of ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
