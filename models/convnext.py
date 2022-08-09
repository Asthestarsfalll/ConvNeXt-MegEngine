import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np

from .utils import DropPath, LayerNorm


class Block(M.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = M.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = M.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = M.GELU()
        self.pwconv2 = M.Linear(4 * dim, dim)
        self.gamma = mge.Parameter(
            layer_scale_init_value * F.ones((dim))) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else M.Identity()

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = inp + self.drop_path(x)
        return x


class ConvNeXt(M.Module):

    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = []# stem and 3 intermediate downsampling conv layers
        stem = M.Sequential(
            M.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = M.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    M.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = []
        dp_rates=[x.item() for x in F.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = M.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = M.LayerNorm(dims[-1], eps=1e-6)
        self.head = M.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight *= head_init_scale
        self.head.bias *= (head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (M.Conv2d, M.Linear)):
            M.init.xavier_normal_(m.weight)
            M.init.zeros_(m.bias)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/39/files/a9171fa3-eb68-4a74-8d1c-047e37e201ef"
)
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     128, 256, 512, 1024], **kwargs)
    return model


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     192, 384, 768, 1536], **kwargs)
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[
                     256, 512, 1024, 2048], **kwargs)
    return model


if __name__ == '__main__':
    model = convnext_tiny()
    inp = mge.random.normal(size=(2, 3, 224, 224))
    out = model(inp)
    print(out.shape)
