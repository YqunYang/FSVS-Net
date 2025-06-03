import torch
import os
import math
import numpy as np
import torch

import json
import random
import pickle

from PIL import Image
from torch.utils.data import Dataset
from options import OPTION as opt
import copy

DATA_CONTAINER = {}
ROOT = 'data'
MAX_TRAINING_OBJ = 1
MAX_TRAINING_SKIP = 5


def multibatch_collate_fn(batch):
    # 8.5修改 加入向上向下推理
    # min_time = min([sample[0].shape[0] for sample in batch])
    # frames = torch.stack([sample[0] for sample in batch])
    # masks = torch.stack([sample[1] for sample in batch])
    #
    # objs = [torch.LongTensor([sample[2]]) for sample in batch]
    # objs = torch.cat(objs, dim=0)
    #
    # try:
    #     info = [sample[3] for sample in batch]
    # except IndexError as ie:
    #     info = None
    #
    # return frames, masks, objs, info
    # -------------------------------

    # middle to both side
    # min_time = min([sample[0].shape[0] for sample in batch])
    # frames_midup = torch.stack([sample[0] for sample in batch])
    # masks_midup = torch.stack([sample[1] for sample in batch])
    # frames_middown = torch.stack([sample[2] for sample in batch])
    # masks_middown = torch.stack([sample[3] for sample in batch])

    # bi-direction
    min_time = min([sample[0].shape[0] for sample in batch])
    frames_topdown = torch.stack([sample[0] for sample in batch])
    masks_topdown = torch.stack([sample[1] for sample in batch])
    frames_bottomup = torch.stack([sample[2] for sample in batch])
    masks_bottomup = torch.stack([sample[3] for sample in batch])
    #

    objs = [torch.LongTensor([sample[4]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[5] for sample in batch]
    except IndexError as ie:
        info = None

    # return frames_midup, masks_midup, frames_middown, masks_middown, objs, info
    return frames_topdown, masks_topdown,frames_bottomup, masks_bottomup, objs, info


def convert_mask(mask, max_obj):
    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj + 1):
        oh.append(mask == k)

    oh = np.stack(oh, axis=2)

    return oh


def convert_mask_new(mask, max_obj):
    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj + 1):
        oh.append(mask == k)

    oh_cpu_list = [tensor.cpu() for tensor in oh]
    oh_np_list = [tensor.numpy() for tensor in oh_cpu_list]
    result = np.stack(oh_np_list, axis=1)

    return result


def convert_one_hot(oh, max_obj):
    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj + 1):
        mask[oh[:, :, k] == 1] = k

    return mask


def convert_one_hot_new(oh, max_obj):
    # for i in range(oh.shape[0]):  # Loop over samples
    #     for j in range(oh.shape[1]):  # Loop over channels
    #         mask = np.zeros_like(oh[i, j, 0], dtype=np.uint8)
    #         for k in range(max_obj + 1):  # Loop over classes
    #             mask[oh[i, j, k] == 1] = k
    #
    #         masks[i, j, 0] = mask
    masks = torch.argmax(oh, dim=2)
    masks = masks[:, :, np.newaxis]

    return masks


class BaseData(Dataset):

    def increase_max_skip(self):
        pass

    def set_max_skip(self):
        pass


class CVCVOS(BaseData):

    def __init__(self, train=True, sampled_frames=3,
                 transform=None, max_skip=2, increment=1, samples_per_video=12):
        data_dir = os.path.join(ROOT, opt.project_name)

        split = 'train' if train else 'test'

        self.root = data_dir
        self.part_frames = []
        self.none_frames = []
        with open(os.path.join(data_dir, opt.txt_path, split + '.txt'), "r") as lines:
            videos = []
            for line in lines:
                _video = line.rstrip('\n')
                videos.append(_video)
                path = os.path.join(self.root, opt.mask_folder, line.rstrip('\n'))
                _mask_dir = os.path.join(self.root, opt.mask_folder)  # each object each color
                frames = os.listdir(path)

                part_frames = []
                none_frames = []
                for i in range(0, len(frames)):
                    _mask = os.path.join(_mask_dir, line.rstrip('\n'), frames[i])
                    assert os.path.isfile(_mask)
                    maskk = np.array(Image.open(_mask)).astype(np.float32)
                    if len(maskk[maskk == 1]) > 0:
                        part_frames.append(frames[i])
                    else:
                        none_frames.append(frames[i])
                self.part_frames.append(part_frames)
                self.none_frames.append(none_frames)
        self.videos = videos

        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames

        self.length = len(self.videos) * samples_per_video
        self.max_obj = 1

        self.transform = transform
        self.train = train
        self.max_skip = max_skip
        self.increment = increment

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]
        # path = os.path.join(self.root, 'Task10_mask', vid)
        # _mask_dir = os.path.join(self.root, 'Task10_mask')
        # _image_dir = os.path.join(self.root, 'Task10_origin')
        path = os.path.join(self.root, opt.mask_folder, vid)
        _mask_dir = os.path.join(self.root, opt.mask_folder)
        _image_dir = os.path.join(self.root, opt.trainset_image_folder)
        frames = os.listdir(path)
        part_frames = self.part_frames[(idx // self.samples_per_video)]
        none_frames = self.none_frames[(idx // self.samples_per_video)]
        whole_frames = frames
        whole_frames.sort(key=lambda x: int(x.split('.')[0]))
        frames = part_frames
        frames.sort(key=lambda x: int(x.split('.')[0]))
        nframes = len(frames)
        num_obj = 0
        info = {'name': vid}
        while num_obj == 0:
            while num_obj == 0:
                if self.train:
                    last_sample = -1
                    sample_frame = []
                    nsamples = min(self.sampled_frames, nframes)

                    # for i in range(nsamples):___________________________原数据加载方法：先加载参考帧，然后加载参考帧左侧全部，然后加载参考帧右侧全部
                    #     if i == 0:
                    #         last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]
                    #     else:
                    #         # print('last_sample + 1', last_sample + 1,
                    #         #       'min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1)',
                    #         #       min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1))
                    #         last_sample = random.sample(
                    #             range(last_sample + 1,
                    #                   min(last_sample + self.max_skip + 2, nframes - nsamples + i + 1)), 1)[0]
                    #     sample_frame.append(frames[last_sample])

                    for i in range(nsamples):  # ____________________________新数据加载方法：先加载参考帧，然后顺序加载参考帧后面的帧
                        if i == 0:
                            last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]
                            sample_frame.append(frames[last_sample])
                        else:
                            sample_frame.append(frames[last_sample + i])

                    # none_index = random.sample(range(0, len(none_frames)), 1)[0]
                    # sample_frame.append(none_frames[none_index])

                    # sample_frame = whole_frames  # _____________________________新新数据加载方法：直接在测试集上进行训练
                else:
                    sample_frame = whole_frames  # 选出参考数据
                    nsamples = len(sample_frame)
                if self.train:
                    ref_index = 0
                else:
                    ran = random.sample(range(-2, 3), 1)[0]
                    # ref_index = whole_frames.index(frames[int(len(frames) / 2 + ran)])
                    ref_index = 0  # 针对测试集图像数量过少进行更改，后续要改回来
                # if self.train:
                #     t = None
                #     for ii in range(0, int(ref_index / 2) + 1):
                #         t = sample_frame[ii]
                #         sample_frame[ii] = sample_frame[ref_index - ii]
                #         sample_frame[ref_index - ii] = t  # 位置调整：将参考数据放入第一帧，后续交互分布
                # else:
                #     sample_frames = []
                #     sample_frames.append(sample_frame[ref_index])  # 先取到参考帧
                #
                #     # for ii in range(1, max(ref_index + 1, len(sample_frame) - ref_index)):  # 将每一帧取到序列里面
                #     #     if ref_index - ii >= 0:
                #     #         sample_frames.append(sample_frame[ref_index - ii])
                #     #     if ref_index + ii < len(sample_frame):
                #     #         sample_frames.append(sample_frame[ref_index + ii])
                #
                #     for ii in range(1, ref_index + 1):
                #         if ref_index - ii >= 0:
                #             sample_frames.append(sample_frame[ref_index - ii])
                #
                #     # 将参考帧右侧的帧添加进去
                #     for ii in range(1, len(sample_frame) - ref_index):
                #         if ref_index + ii < len(sample_frame):
                #             sample_frames.append(sample_frame[ref_index + ii])
                #
                #     sample_frame = sample_frames  # 更改添加方式，这样跳帧不连续，现在尝试全部添加左边的，然后判断右边有没有，再添加右侧的帧
                frame = [np.array(Image.open(os.path.join(_image_dir, vid, name))) for name in
                         sample_frame]  # frame = [np.array(Image.open(os.path.join(_image_dir, vid, name)).convert('RGB')) for name in sample_frame]
                # frame = [frame_iter[:, :, np.newaxis] for frame_iter in frame]
                info['frame_name'] = [name for name in sample_frame]
                mask = []
                for name in sample_frame:
                    _m = np.array(Image.open(os.path.join(_mask_dir, vid, name)))
                    _m[_m != 1] = 0
                    # _m[_m!=2]=0
                    # _m[_m==2]=1
                    mask.append(_m)
                num_obj = int(mask[0].max())

            # STCN中不使用独热编码，mask只有一个维度，但是train和test的transfrom中都使用独热编码的mask，因此不在这里修改
            mask = [convert_mask(msk, self.max_obj) for msk in mask]

            info['palette'] = [0, 0, 0, 255, 255, 255]
            info['size'] = frame[0].shape[:2]
            if self.transform is None:
                raise RuntimeError('Lack of proper transformation')
            frame, mask = self.transform(frame, mask, False)

            if self.train:
                num_obj = 0
                for i in range(1, MAX_TRAINING_OBJ + 1):
                    if torch.sum(mask[0, i]) > 0:
                        num_obj += 1
                    else:
                        break

            # return frame, mask, num_obj, info

            # bi-direction

            if nsamples < 3:
                frame_topdown = frame
                mask_topdown = mask
                frame_bottomup = frame
                mask_bottomup = mask

            else:
                frame_topdown = copy.deepcopy(frame)
                mask_topdown = copy.deepcopy(mask)
                frame_bottomup = copy.deepcopy(frame)
                mask_bottomup = copy.deepcopy(mask)

            for ii in range(0, nsamples // 2):
                t_i = copy.deepcopy(frame_bottomup[ii])
                t_m = copy.deepcopy(mask_bottomup[ii])
                frame_bottomup[ii] = frame_bottomup[nsamples - 1 - ii]
                frame_bottomup[nsamples - 1 - ii] = t_i
                mask_bottomup[ii] = mask_bottomup[nsamples - 1 - ii]
                mask_bottomup[nsamples - 1 - ii] = t_m

        return frame_topdown, mask_topdown, frame_bottomup, mask_bottomup, num_obj, info

        #     # middle to both side
        #     frame_midup = copy.deepcopy(frame)
        #     mask_midup = copy.deepcopy(mask)
        #     frame_middown = copy.deepcopy(frame)
        #     mask_middown = copy.deepcopy(mask)
        #
        #     frame_midup = frame_midup[0:6]
        #     mask_midup = mask_midup[0:6]
        #     frame_middown = frame_middown[5:nsamples]
        #     mask_middown = mask_middown[5:nsamples]
        #
        #     for ii in range(0, 3):
        #         t_i = copy.deepcopy(frame_midup[5 - ii])
        #         t_m = copy.deepcopy(mask_midup[5 - ii])
        #         frame_midup[5 - ii] = frame_midup[ii]
        #         mask_midup[5 - ii] = mask_midup[ii]
        #         frame_midup[ii] = t_i
        #         mask_midup[ii] = t_m
        #
        # return frame_midup, mask_midup, frame_middown, mask_middown, num_obj, info

    def __len__(self):

        return self.length


DATA_CONTAINER[opt.project_name] = CVCVOS
