import torch.nn.functional as F
from collections import defaultdict
import torch
import torch.nn as nn
from dataloaders.data import convert_one_hot


def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union


def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou



def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img, pad):
    if pad[2] + pad[3] > 0:
        img = img[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        img = img[:, :, :, pad[0]:-pad[1]]
    return img


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i'] + 1) / (values['hide_iou/u'] + 1)


def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i'] + 1) / (values['hide_iou/sec_u'] + 1)


iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]



class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20, end_warm=70, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()


        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))

        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class SIEComputer:
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def compute(self, frame, mask, data, epoch):
        losses = defaultdict(int)

        b, s, _, _, _ = mask.shape
        selector = torch.FloatTensor([1, 0])

        for i in range(1, s):
            for j in range(b):
                loss1, p = self.bce(data['logits_%d' % i][j:j + 1, :2],
                                    convert_one_hot(mask[j:j + 1, i], 1)[0, :, :, :], epoch)

                # Boundary Loss
                boundary_logits = data['boundary_%d' % i][j:j + 1]
                laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32,
                                                device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
                boundary_targets = F.conv2d(convert_one_hot(mask[j:j + 1, i], 1).float(), laplacian_kernel, padding=1)
                boundary_targets = boundary_targets.clamp(min=0)
                boundary_targets[boundary_targets > 0.1] = 1
                boundary_targets[boundary_targets <= 0.1] = 0
                if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
                    boundary_targets = F.interpolate(
                        boundary_targets, boundary_logits.shape[2:], mode='bilinear')

                bce_loss = F.binary_cross_entropy_with_logits(boundary_logits[:, 0:1], boundary_targets)
                dice_loss = dice_loss_func(torch.sigmoid(boundary_logits[:, 0:1]), boundary_targets)
                loss2 = bce_loss + dice_loss


            losses['p'] += p / b / (s - 1)

            losses['loss_%d' % i] = losses['loss_%d' % i] + 0.05 * loss2 / b

        losses['total_loss'] += losses['loss_%d' % i]

        new_total_i, new_total_u = compute_tensor_iu(data['mask_%d' % i] > 0.5, convert_one_hot(mask[:, i], 1) > 0.5)
        losses['hide_iou/i'] += new_total_i
        losses['hide_iou/u'] += new_total_u

        return losses


class LossComputer:
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def compute(self, frame, mask, out, epoch):
        losses = defaultdict(int)

        b, s, _, _, _ = mask.shape
        for i in range(1, s):
            for j in range(b):
                loss, p = self.bce(out['logits_%d' % i][j, :2], mask[j, i])

                losses['loss_%d' % i] += loss / b
                losses['p'] += p / b / (s - 1)

            losses['total_loss'] += losses['loss_%d' % i]

            new_total_i, new_total_u = compute_tensor_iu(out['mask_%d' % i] > 0.5, mask[:, i] > 0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u


        return losses
