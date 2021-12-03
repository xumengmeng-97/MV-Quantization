import torch
import torch.nn as nn
import torch.nn.functional as F
from zoo.mobilenet_v1.mobilenet_utils import *

from collections import namedtuple
import functools
from collections import OrderedDict

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't'])  # t is the expension factor

V1_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
    Conv(kernel=1, stride=1, depth=1280),
]

# Conv2d = Conv2dTF
# Conv2d = nn.Conv2d

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            # nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.Conv2d(inp, oup, kernel, stride, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        x_shape_b, x_shape_c, x_shape_x, x_shape_y = x.shape
        x_padding = torch.zeros([x_shape_b, x_shape_c, x_shape_x+1, x_shape_y+1]).cuda()
        x_padding[:, :, 0:x_shape_x, 0:x_shape_y] = x[:, :, :, :]
        # if self.conv[0].stride[0] == 1:
        #     F.pad(x, [1, 1, 1, 1])
        # elif self.conv[0].stride[0] == 2:
        #     F.pad(x, [0, 1, 0, 1])
        return self.conv(x_padding)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True)
            ),
            # pw
            nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        )
        self.depth = oup
        self.stride = stride

    def forward(self, x):
        # # if self.conv[0][0].stride[0] == 1:
        # if self.stride == 1:
        #     F.pad(x, [1, 1, 1, 1])
        # # elif self.conv[0][0].stride[0] == 2:
        # if self.stride == 1:
        #     F.pad(x, [0, 1, 0, 1])
        return self.conv(x)

class _conv_dw_s(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw_s, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True)
            ),
            # pw
            nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        )
        self.depth = oup
    def forward(self, x):
        x_shape_b, x_shape_c, x_shape_x, x_shape_y = x.shape
        x_padding = torch.zeros([x_shape_b, x_shape_c, x_shape_x+1, x_shape_y+1]).cuda()
        x_padding[:, :, 0:x_shape_x, 0:x_shape_y] = x[:, :, :, :]
        return self.conv(x_padding)


def mobilenet_base(conv_defs=V1_CONV_DEFS, depth=lambda x: x, in_channels=3):
    layers = []
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.kernel, conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            if conv_def.stride == 1:
                layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            elif conv_def.stride == 2:
                layers += [_conv_dw_s(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
    return layers, in_channels


class MobileNet(nn.Module):
    def __init__(self, version='1', depth_multiplier=1.0, min_depth=8, num_classes=1001, dropout=0.2):
        super(MobileNet, self).__init__()
        self.dropout = dropout
        conv_defs = V1_CONV_DEFS if version == '1' else V2_CONV_DEFS

        if version == '1':
            depth = lambda d: max(int(d * depth_multiplier), min_depth)
            self.features, out_channels = mobilenet_base(conv_defs=conv_defs, depth=depth)

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

        for m in self.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eps = 0.001
                m.momentum = 0.003

    def forward(self, x):
        x = self.features(x)
        x = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x = F.dropout(x, self.dropout, self.training)
        x = self.classifier(x)
        x = x.squeeze(3).squeeze(2)
        return x


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


MobileNet_v1 = wrapped_partial(MobileNet, version='1')

mobilenet_v1 = wrapped_partial(MobileNet, version='1', depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(MobileNet, version='1', depth_multiplier=0.25)

def gpu_model_cpu(model):
    for _item in model.keys():
        if _item.startswith('features.'):
            model[_item[len('features.'):]] = model.pop(_item)
    return model


if __name__ == "__main__":
    m_net = MobileNet_v1()
    for item in m_net.state_dict().keys():
        print(item, m_net.state_dict()[item].shape)
    m_net.load_state_dict(torch.load('./mobilenet_v1_1.0_224.pth'))

    print(m_net)
