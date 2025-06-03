import numpy as np
import math

import torch
import os
import shutil
import cv2
import random
from PIL import Image
from dataloaders.data import ROOT
from options import OPTION as opt


def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + '-' + str(epoch) + '.pth')
    torch.save(state, filepath, _use_new_zipfile_serialization=False)
    print('==> save model at {}'.format(filepath))
    if is_best:
        cpy_file = os.path.join(checkpoint, filename + '_model_best.pth')
        shutil.copyfile(filepath, cpy_file)
        print('==> save best model at {}'.format(cpy_file))


def save_best_checkpoint(state, epoch, is_best, checkpoint='checkpoint', round_name='0', filename='checkpoint'):
    # filepath = os.path.join(checkpoint, filename + '-'+ str(epoch) + '.pth')
    # torch.save(state, filepath, _use_new_zipfile_serialization=False)
    # print('==> save model at {}'.format(filepath))
    if is_best:
        # best_file = os.path.join(checkpoint, round_name, filename + '_model_best.pth')
        best_file = os.path.join(checkpoint, round_name)
        if not os.path.exists(best_file):
            os.makedirs(best_file)
        best_addr = os.path.join(best_file, filename + '_best' + '.pth')

        torch.save(state, best_addr, _use_new_zipfile_serialization=False)
        print('==> save best model at {}'.format(best_file))
        # shutil.copyfile(filepath, cpy_file)
        # print('==> save best model at {}'.format(cpy_file))


def save_newest_checkpoint(state, epoch, checkpoint='checkpoint', round_name='0', filename='checkpoint'):
    # filepath = os.path.join(checkpoint, filename + '-'+ str(epoch) + '.pth')
    # torch.save(state, filepath, _use_new_zipfile_serialization=False)
    # print('==> save model at {}'.format(filepath))

    # best_file = os.path.join(checkpoint, round_name, filename + '_model_best.pth')
    best_file = os.path.join(checkpoint, round_name)
    if not os.path.exists(best_file):
        os.makedirs(best_file)
    best_addr = os.path.join(best_file, filename + '_newest' + '.pth')

    torch.save(state, best_addr, _use_new_zipfile_serialization=False)
    print('==> save newest model at {}'.format(best_file))
    # shutil.copyfile(filepath, cpy_file)
    # print('==> save best model at {}'.format(cpy_file))


def write_mask(mask, info, opt, directory='results'):
    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    directory = os.path.join(ROOT, directory)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    if 'frame' not in info:
        min_t = 0
        step = 1
    else:
        min_t = min(info['frame'])
        step = 5

    for t in range(mask.shape[0]):
        # print('mask[0]')
        # print(mask[t].shape)
        # print(mask[t])
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]

        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # print(rescale_mask.argmax(axis=2))

        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        if len(rescale_mask[rescale_mask > 0]) < 10:
            rescale_mask[rescale_mask != 0] = 0
        output_name = '{}'.format(name + '-' + info['frame_name'][t])
        # output_name = '{}'.format(info['frame_name'][t])
        if opt.save_indexed_format:
            # print('resize')
            # print(rescale_mask.max())
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')
        else:
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max() + 1):
                seg[rescale_mask == k, :] = info['palette'][(k * 3):(k + 1) * 3]
            inp_img = cv2.imread(
                os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)


def write_mask2(mask, info, names, opt, directory='results'):
    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    directory = os.path.join(ROOT, directory)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    if 'frame' not in info:
        min_t = 0
        step = 1
    else:
        min_t = min(info['frame'])
        step = 5

    for t in range(mask.shape[0]):
        # print('mask[0]')
        # print(mask[t].shape)
        # print(mask[t])
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]

        m = m.transpose((1, 2, 0))
        # print('m')
        # print(m.shape)
        # print(m)
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # print('rescale1')
        # print(rescale_mask.max())
        # print(rescale_mask.shape)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        # print('rescale2')
        # print(rescale_mask.max())
        if len(rescale_mask[rescale_mask > 0]) < 10:
            rescale_mask[rescale_mask != 0] = 0
        output_name = '{}'.format(name + '-' + names[t])
        if opt.save_indexed_format:
            # print('resize')
            # print(rescale_mask.max())
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            maskk = np.array(Image.open(os.path.join(video, output_name))).astype(np.float32)
            if len(maskk[maskk == 1]) > 0 or t == 0:
                im.save(os.path.join(video, output_name), format='PNG')
        else:
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max() + 1):
                seg[rescale_mask == k, :] = info['palette'][(k * 3):(k + 1) * 3]
            inp_img = cv2.imread(
                os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)


def mask_iou(pred, target):
    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum(inter / union) / N

    return iou


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in opt.milestone:
        opt.learning_rate *= opt.gamma
        for pm in optimizer.param_groups:
            pm['lr'] *= opt.learning_rate


def pointwise_dist(points1, points2):
    # compute the point-to-point distance matrix

    N, d = points1.shape
    M, _ = points2.shape

    p1_norm = torch.sum(points1 ** 2, dim=1, keepdim=True).expand(N, M)
    p2_norm = torch.sum(points2 ** 2, dim=1).unsqueeze(0).expand(N, M)
    cross = torch.matmul(points1, points2.permute(1, 0))

    dist = p1_norm - 2 * cross + p2_norm

    return dist


def furthest_point_sampling(points, npoints):
    """
    points: [N x d] torch.Tensor
    npoints: int

    """

    old = 0
    output_idx = []
    output = []
    dist = pointwise_dist(points, points)
    fdist, fidx = torch.sort(dist, dim=1, descending=True)

    for i in range(npoints):
        fp = 0
        while fp < points.shape[0] and fidx[old, fp] in output_idx:
            fp += 1

        old = fidx[old, fp]
        output_idx.append(old)
        output.append(points[old])

    return torch.stack(output, dim=0)


def counting_dice():
    root_dir = 'data/'
    pred_dir = root_dir + opt.output_dir + '/' + opt.project_name + '/'
    gt_dir = root_dir + opt.project_name + '/' + opt.mask_folder + '/'

    total_dice = 0
    total_length = 0

    for case in os.listdir(pred_dir):

        case_dice = 0
        # case_length = 0

        for image in os.listdir(os.path.join(pred_dir, case)):
            pred_image_tensor = torch.tensor(
                np.array(Image.open(os.path.join(pred_dir, case, image)), dtype=np.float32))

            gt_path = os.path.join(gt_dir, case, image.split('-')[-1])
            if not os.path.exists(gt_path):
                continue
            gt_image_tensor = torch.tensor(
                np.array(Image.open(os.path.join(gt_dir, case, image.split('-')[-1])), dtype=np.float32))

            pred_flat = pred_image_tensor.view(-1)
            gt_flat = gt_image_tensor.view(-1)

            # Compute intersection and union
            intersection = (pred_flat * gt_flat).sum()
            union = pred_flat.sum() + gt_flat.sum()

            # Compute Dice coefficient
            epsilon = 1e-6
            dice = (2 * intersection + epsilon) / (union + epsilon)

            case_dice += dice.item()
            total_dice += dice.item()

        length = len(os.listdir(os.path.join(pred_dir, case)))
        case_dice = case_dice / len(os.listdir(os.path.join(pred_dir, case)))
        # print(case, ' dice = ', case_dice)
        case_length = len(os.listdir(os.path.join(pred_dir, case)))
        total_length += case_length
        # total_dice += case_dice

    total_dice = total_dice / total_length
    print('total dice = ', total_dice)

    return total_dice
