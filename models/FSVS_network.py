import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FSVS_modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4


        self.SIE_pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)


        self.pred = nn.Conv2d(256, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)


        # boundary
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

        x_b = self.boundary_1(x)
        x_b = self.hfm(x_b)
        x_m = self.mask_1(x)
        x_m2b = F.relu(self.fusion_1(x_m))
        x_b = x_b + x_m2b
        x_b = self.boundary_2(x_b)
        x_b = F.relu(self.boundary_3(x_b))
        x_b2m = F.relu(self.fusion_2(x_b))
        x_m = x_m + x_b2m
        x_m = self.predict_mask(x_m)
        x_SIE_m = self.SIE_pred(F.relu(x_m))
        x_m = self.pred(F.relu(x_m))

        x_b = F.relu(self.predict_boundary_1(x_b))
        x_b = self.predict_boundary_2(x_b)

        x_SIE_m = F.interpolate(x_SIE_m, scale_factor=4, mode='bilinear', align_corners=False)
        x_m = F.interpolate(x_m, scale_factor=4, mode='bilinear', align_corners=False)
        x_b = F.interpolate(x_b, scale_factor=4, mode='bilinear', align_corners=False)
        return x_m, x_b, x_SIE_m



class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


def Soft_aggregation(ps, max_obj):
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit


class FSVS(nn.Module):
    def __init__(self, single_object):
        super().__init__()
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
        self.lfm = LFM(1024)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def encode_key(self, frame):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        f16 = self.lfm(f16)
        k16 = self.key_proj(f16)

        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

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
            # mr = self.memory.readout(affinity, mv16, qv16)

            # original and new
            logits, boundary, logits_SIE = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob_SIE = torch.sigmoid(logits_SIE)

            # new
            # logits, boundary = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)

            # original
            prob = F.softmax(logits, dim=1)[:, 1]

            # new
            # prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:, 0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:, 1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        # new
        logits_SIE = self.aggregate(prob_SIE)
        prob_SIE = F.softmax(logits_SIE, dim=1)[:, 1:]

        logits = Soft_aggregation(ps=prob, max_obj=self.max_obj)
        ps = prob
        prob = torch.softmax(logits, dim=1)
        return logits, prob, boundary, logits_SIE, prob_SIE


    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError
