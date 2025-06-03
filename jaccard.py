# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
# -----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np
import glob
import os
from PIL import Image
from options import OPTION as opt


def db_eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
            np.sum((annotation | segmentation), dtype=np.float32)


def eval_jaccard():
    root_dir = 'data/'
    pred_dir = root_dir + opt.output_dir + '/' + opt.project_name + '/'
    gt_dir = root_dir + opt.project_name + '/' + opt.mask_folder + '/'

    filelist = os.listdir(pred_dir)
    jaccards = np.zeros(len(filelist))
    prefix_dict = {}

    for i in range(0, len(filelist)):

        item = os.path.join(pred_dir, filelist[i])
        preds = os.listdir(item)
        _mask = []
        _gt = []
        jaccard_sum = 0
        jaccard_count = 0
        for ii in range(len(preds)):
            filename = os.path.join(item, preds[ii])
            mask = np.array(Image.open(filename)).astype(np.float32)
            last_dash_index = preds[ii].rfind('-')
            gt = np.array(Image.open(os.path.join(gt_dir, filelist[i], preds[ii][last_dash_index + 1:]))).astype(
                np.float32)
            _mask.append(mask)
            _gt.append(gt)
            jaccards[i] = db_eval_iou(np.stack(_gt, 0), np.stack(_mask, 0))
            jaccard = db_eval_iou(gt, mask)
            jaccard_sum += jaccard
            jaccard_count += 1

        prefix = filelist[i].split('-')[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = {'jaccard_sum': 0, 'jaccard_count': 0}

        prefix_dict[prefix]['jaccard_sum'] += jaccard_sum
        prefix_dict[prefix]['jaccard_count'] += jaccard_count

    for prefix, data in prefix_dict.items():
        jaccard_avg = data['jaccard_sum'] / data['jaccard_count']
        print(f"前缀为{prefix}的数据平均Jaccard值为: {jaccard_avg}")

    J = jaccards.mean()
    print("jaccard:", J)
    return J


if __name__ == '__main__':
    eval_jaccard()
