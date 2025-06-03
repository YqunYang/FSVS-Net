import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.utils import model_zoo


def load_weights_sequential(target, source_state, extra_chan=1):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c, extra_chan, w, h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_chan=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1 + extra_chan, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


def resnet18(pretrained=True, extra_chan=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_chan)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']), extra_chan)
    return model


def resnet50(pretrained=True, extra_chan=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_chan)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']), extra_chan)
    return model


class LFM(nn.Module):
    def __init__(self, num_channels):
        super(LFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()

        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)  # 1, 1, H, W

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        y = y * (1 - high_filter)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y


class HFM(nn.Module):
    def __init__(self, num_channels):
        super(HFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()

        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)  # 1, 1, H, W

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        y = y * high_filter

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
