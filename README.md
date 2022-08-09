# ConvNeXt-MegEngine

The MegEngine Implementation of ConvNeXt.

## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

Convert trained weights from torch to megengine

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



## TODO

- [ ] add object detection codes
- [ ] add semantic segmentation codes
- [ ] add train codes

## Reference

[The official implementation of ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
