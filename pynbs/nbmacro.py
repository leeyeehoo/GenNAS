import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F

candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS = OrderedDict()
OPS['id'] = lambda inp, oup, stride: Identity(inp=inp, oup=oup, stride=stride)
OPS['ir_3x3_t3'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS['ir_5x5_t6'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=6, stride=stride, k=5)


class Identity(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Identity, self).__init__()
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.conv = nn.Sequential(
                # dw            
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)



class Network(nn.Module):
    def __init__(self, arch, last_channels, num_classes=10, stages=[2, 3, 3], init_channels=32):
        super(Network, self).__init__()
        assert len(arch) == sum(stages)

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        arch_ = arch.copy()
        features = []
        channels = init_channels
        for indx,stage in enumerate(stages):
            for idx in range(stage):
                op_func = OPS[candidate_OP[arch_.pop(0)]]
                if idx == 0 and indx > 0:
                    # stride = 2 
                    features.append(op_func(channels, channels*2, 2))
                    channels *= 2
                else:
                    features.append(op_func(channels, channels, 1))
                
        self.features = nn.ModuleList(features)
        self.out0 = nn.Sequential(nn.BatchNorm2d(channels//4),nn.Conv2d(channels//4,last_channels[0],1))
        self.out1 = nn.Sequential(nn.BatchNorm2d(channels//2),nn.Conv2d(channels//2,last_channels[1],1))
        self.out2 = nn.Sequential(nn.BatchNorm2d(1280),nn.Conv2d(1280,last_channels[2],1))
        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1))
#             nn.BatchNorm2d(1280))
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.classifier = nn.Linear(1280, num_classes)


    def forward(self, x):
        x = self.stem(x)
        for indx, feature in enumerate(self.features):
            if indx ==2:
                o0 = self.out0(F.interpolate(x,(8,8)))
            if indx ==5:
                o1 = self.out1(F.interpolate(x,(8,8)))
            x = feature(x)
        o2 = self.out2(F.interpolate(self.out(x),(8,8)))
        return [o0,o1,o2]
