import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import math
from torch.utils import model_zoo
import torch.nn.functional as F
from networks.FSVS_loss import LossComputer
from networks.FSVS_loss import SIEComputer
from utils.SIE_modules import LFM
from utils.SIE_modules import HFM
from utils.IB_module import VibModel
import numpy as np
from dataloaders.data import convert_mask
from dataloaders.data import convert_new_mask
from dataloaders.data import convert_one_hot
import torch.optim as optim
import os


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


def Soft_aggregation(ps, max_obj):
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit


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


def resnet18(weights=True, extra_chan=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_chan)
    if weights:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']), extra_chan)
    return model


def resnet50(weights=True, extra_chan=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_chan)
    if weights:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']), extra_chan)
    return model


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


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=False, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=False, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024


        return f16, f8, f4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.SIE_pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.pred = nn.Conv2d(256, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)


        self.boundary_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.mask_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.boundary_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.boundary_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.fusion_1 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.fusion_2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.predict_boundary_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.predict_boundary_2 = nn.Conv2d(256, 1, 3, 1, 1)
        self.predict_mask = nn.Conv2d(256, 256, 3, 1, 1)

        self.hfm = HFM(256)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)


        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2 * ab - a_sq) / math.sqrt(CK)

        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mo, affinity)
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)


class FSVS(nn.Module):
    def __init__(self, single_object=True, phase='new_train', lr=None, local_rank=0):
        super().__init__()

        self.single_object = single_object
        self.local_rank = local_rank


        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

            # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.phase = phase

        self.loss_computer = SIEComputer()
        self.max_obj = 1


        resnet = models.resnet50(weights=True)
        self.conv1 = nn.Conv2d(1026, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.res5 = resnet.layer4  # 1/32, 2048
        self.avgpool = nn.AvgPool2d(16, 16)
        self.fc = resnet.fc
        self.fc2 = nn.Linear(1000, 1)

        self.scaler = torch.cuda.amp.GradScaler()
        self.lfm = LFM(1024)
        self.ib = VibModel(b_threshold=0.9)


    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def encode_key(self, frame, deep_mask=None):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        f16 = self.lfm(f16)
        if self.phase == "new_train":
            f16, kl, ds_loss = self.ib(f16, deep_mask, phase=self.phase)
        else:
            f16, kl= self.ib(f16, phase=self.phase)
        k16 = self.key_proj(f16)

        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        if self.phase == "new_train":
            return k16, f16_thin, f16, f8, f4, kl, ds_loss
        else:
            return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2)  # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)

        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = F.softmax(logits, dim=1)[:, 1]

        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:, 0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:, 1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = Soft_aggregation(ps=prob, max_obj=self.max_obj)
        ps = prob
        prob = torch.softmax(logits, dim=1)


        return logits, prob


    def save_checkpoint(self, epoch, path, name):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint_path = os.path.join(path, name + 'checkpoint.pth')
        checkpoint = {
            'epoch': epoch,
            'network': self.FSVS.module.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        epoch = checkpoint['epoch']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']

        map_location = 'cuda:%d' % self.local_rank
        self.FSVS.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)

        print('Model loaded.')

        return epoch

    def forward(self, data, frame, mask=None, keys=None, num_objects=None, criterion=None, max_obj=None,
                test_T=0, prev_value=None, prev_k16=None, pred=None, info=None, frame_certainty_map=None, epoch=None):
        if self.phase == 'test':
            _, T, _, _, _ = frame.shape
            k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(frame)
            ref_value = []
            prev_value = []
            for t in range(0, T):
                if t == 0:
                    this_mask_SIE = convert_one_hot(mask[:, t], 1)

                    ref_value = self.encode_value(frame[:, t], kf16[:, t], mask[:, t], 1)
                elif t == 1:

                    this_logits, this_mask = self.segment(k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                                                          k16[:, :, t - 1:t], ref_value)
                    # ————————————————————————————————————————————————————————————————————————————————

                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

                else:

                    this_logits, this_mask = self.segment(
                        k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                        k16[:, :, t - 2:t], values)

                    ref_value = prev_value
                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

            return pred

        elif self.phase == 'eval':
            _, T, _, _, _ = frame.shape
            k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(frame)
            ref_value = []
            prev_value = []
            frame_certainty_map = {}
            for t in range(0, T):
                if t == 0:
                    ref_value = self.encode_value(frame[:, t], kf16[:, t], mask[:, t])
                elif t == 1:
                    this_logits, this_mask = self.segment(k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                                                          k16[:, :, t - 1:t], ref_value)
                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

                    uncertain = torch.max(this_mask, dim=1)[0].unsqueeze(dim=0)
                    uncertain = F.interpolate(uncertain.float(), size=[32, 32], mode='bilinear')
                    out = torch.argmax(this_mask, dim=1, keepdim=True).float()
                    out = F.interpolate(out.float(), size=[32, 32], mode='bilinear')
                    # print('kf16, out, uncertain size:', kf16[:, t].size(), out.size(), uncertain.size())
                    input_qua = torch.cat((kf16[:, t], out, uncertain), dim=1)
                    r4 = self.conv1(input_qua)
                    r5 = self.res5(r4)
                    r5 = self.avgpool(r5)
                    r5 = r5.view(-1, 2048)
                    output = F.relu(self.fc(r5))
                    output_quality = F.sigmoid(self.fc2(output))

                    cer_flag = this_mask[0][1].clone().detach()
                    cer_flag[cer_flag <= 0.95] = 0
                    cer_flag[cer_flag > 0.95] = 1
                    numbers_cer = cer_flag.sum(dim=[0, 1])
                    if numbers_cer > 100:
                        frame_certainty_map[info['frame_name'][t]] = output_quality.squeeze().item()
                    else:
                        frame_certainty_map[info['frame_name'][t]] = 1.0

                else:
                    this_logits, this_mask = self.segment(
                        k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                        k16[:, :, t - 2:t], values)
                    ref_value = prev_value
                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

                    uncertain = torch.max(this_mask, dim=1)[0].unsqueeze(dim=0)
                    uncertain = F.interpolate(uncertain.float(), size=[32, 32], mode='bilinear')
                    out = torch.argmax(this_mask, dim=1, keepdim=True).float()
                    out = F.interpolate(out.float(), size=[32, 32], mode='bilinear')
                    input_qua = torch.cat((kf16[:, t], out, uncertain), dim=1)
                    r4 = self.conv1(input_qua)
                    r5 = self.res5(r4)
                    r5 = self.avgpool(r5)
                    r5 = r5.view(-1, 2048)
                    output = F.relu(self.fc(r5))
                    ouput_quality = F.sigmoid(self.fc2(output))

                    cer_flag = this_mask[0][1].clone().detach()
                    cer_flag[cer_flag <= 0.95] = 0
                    cer_flag[cer_flag > 0.95] = 1
                    numbers_cer = cer_flag.sum(dim=[0, 1])
                    if numbers_cer > 100:
                        frame_certainty_map[info['frame_name'][t]] = output_quality.squeeze().item()
                    else:
                        frame_certainty_map[info['frame_name'][t]] = 1.0

            return pred, frame_certainty_map

        elif self.phase == 'train':
            FSVSout = {}
            self.max_obj = mask.shape[2] - 1

            N, T, C, H, W = frame.size()
            for idx in range(N):
                k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(frame)

                if self.single_object:
                    ref_v = self.encode_value(frame[:, 0], kf16[:, 0], mask[:, 0])

                    # Segment frame 1 with frame 0
                    prev_logits, prev_mask = self.segment(k16[:, :, 1], kf16_thin[:, 1], kf8[:, 1], kf4[:, 1],
                                                          k16[:, :, 0:1], ref_v)
                    prev_v = self.encode_value(frame[:, 1], kf16[:, 1], prev_mask)
                    values = torch.cat([ref_v, prev_v], 2)
                    del ref_v

                    # Segment frame 2 with frame 0 and 1
                    this_logits, this_mask = self.segment(
                        k16[:, :, 2], kf16_thin[:, 2], kf8[:, 2], kf4[:, 2],
                        k16[:, :, 0:2], values)

                    FSVSout['mask_1'] = prev_mask
                    FSVSout['mask_2'] = this_mask
                    FSVSout['logits_1'] = prev_logits
                    FSVSout['logits_2'] = this_logits


                    ious = []
                    uncertainty = []
            tmp_out = []
            batch_out = []
            r16s = []
            for idx in range(N):
                for t in range(1, T):
                    gt = mask[idx, t:t + 1]
                    pred = FSVSout['mask_%d' % t]
                    No = num_objects[idx].item()
                    uncertain = torch.max(pred, dim=1)[0]
                    uncertainty.append(uncertain)
                    iou_loss = 1 - criterion(pred, gt, No)
                    ious.append(iou_loss)
                    tmp_out.append(FSVSout['mask_%d' % t])
                    r16s.append(kf16[:, t])
            r16s = torch.cat(r16s, dim=0)
            uncertainty = torch.stack(uncertainty, dim=0)
            uncertainty = F.interpolate(uncertainty.float(), size=[32, 32], mode='bilinear')
            batch_out.append(torch.cat(tmp_out, dim=0))
            batch_out = torch.stack(batch_out, dim=0)
            out = torch.argmax(batch_out, dim=2, keepdim=True)
            out = out.view(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
            out = F.interpolate(out.float(), size=[32, 32], mode="bilinear")
            input_qua = torch.cat((r16s, out, uncertainty), dim=1)
            r4 = self.conv1(input_qua)
            r5 = self.res5(r4)
            r5 = self.avgpool(r5)
            r5 = r5.view(-1, 2048)
            output = F.relu(self.fc(r5))
            output_quality = F.sigmoid(self.fc2(output))

            return batch_out, output_quality, ious

        elif self.phase == 'new_train':
            data = {}
            N, T, _, _, _ = frame.shape
            # k16, kf16_thin, kf16, kf8, kf4, kl, dsloss = self.encode_key(frame, deep_mask)
            k16, kf16_thin, kf16, kf8, kf4, kl, dsloss = self.encode_key(frame, mask)
            ref_value = []
            prev_value = []
            batch_out = []
            kf16s = []
            tmp_out = []
            # frame_certainty_map = {}
            for t in range(0, T):
                if t == 0:
                    this_mask_SIE = convert_one_hot(mask[:, t], 1)
                    ref_value = self.encode_value(frame[:, t], kf16[:, t], mask[:, t], 1)
                elif t == 1:

                    this_logits, this_mask = self.segment(k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                                                          k16[:, :, t - 1:t], ref_value)

                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)


                    kf16s.append(kf16[:, t])
                    out = this_mask
                    tmp_out.append(out)

                    current_mask = convert_one_hot(mask[:, t], 1)


                else:
                    this_logits, this_mask = self.segment(
                        k16[:, :, t], kf16_thin[:, t], kf8[:, t], kf4[:, t],
                        k16[:, :, t - 2:t], values)
                    ref_value = prev_value
                    prev_value = self.encode_value(frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)

                    kf16s.append(kf16[:, t])
                    out = this_mask
                    tmp_out.append(out)

            batch_out.append(torch.cat(tmp_out, dim=0))
            batch_out = torch.stack(batch_out, dim=0)

            kf16s = torch.cat(kf16s, dim=0)
            ious = []
            uncertainty = []
            for idx in range(N):
                for t in range(1, T):
                    gt = mask[idx, t:t + 1]
                    pred = batch_out[idx, t - 1:t]
                    No = num_objects[idx].item()
                    uncertain = torch.max(pred, dim=1)[0]
                    uncertainty.append(uncertain)
                    iou_loss = 1 - criterion(pred, gt, No)
                    ious.append(iou_loss)
            uncertainty = torch.stack(uncertainty, dim=0)
            uncertainty = F.interpolate(uncertainty.float(), size=[32, 32], mode='bilinear')
            out = torch.argmax(batch_out, dim=2, keepdim=True)
            out = out.view(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
            out = F.interpolate(out.float(), size=[32, 32], mode="bilinear")
            input_qua = torch.cat((kf16s, out, uncertainty), dim=1)
            r4 = self.conv1(input_qua)
            r5 = self.res5(r4)
            r5 = self.avgpool(r5)
            r5 = r5.view(-1, 2048)
            output = F.relu(self.fc(r5))
            output_quality = F.sigmoid(self.fc2(output))


            return batch_out, output_quality, ious, kl, dsloss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    deep_size = 8
    x = torch.Tensor(4, 3, 1, image_size, image_size)
    deep_mask = torch.Tensor(4, 3, 1, deep_size, deep_size)
    x.to('cuda')
    deep_mask.to('cuda')
    print("x size: {}".format(x.size()))
