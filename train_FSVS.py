from dataloaders.data_thinv import ROOT, DATA_CONTAINER, multibatch_collate_fn, convert_one_hot_new, convert_mask_new, \
    convert_one_hot
from dataloaders.transform import TrainTransform, TestTransform
from utils.logger import Logger, AverageMeter
from utils.loss import *
from utils.utility import write_mask, save_checkpoint, adjust_learning_rate, save_best_checkpoint, \
    save_newest_checkpoint, counting_dice
from models.models_FSVS import FSVS
from jaccard import eval_jaccard

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as distributed
import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict
from options import OPTION as opt


from torch.nn.parallel import DistributedDataParallel as DDP
from utils.IB_module import VibModel

MAX_FLT = 1e6
np.set_printoptions(threshold=1e6)


def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id to train the network, split with comma')
    parser.add_argument("--local_rank", default=-1)
    parser.add_argument("--world_size", default=1)
    return parser.parse_args()


def main():
    start_epoch = 0
    random.seed(0)

    args = parse_args()
    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id)

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    # Data
    print('==> Preparing dataset')

    input_dim = opt.input_size

    train_transformer = TrainTransform(size=input_dim)
    test_transformer = TestTransform(size=input_dim)

    try:
        if isinstance(opt.trainset, list):
            datalist = []
            for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                ds = DATA_CONTAINER[dataset](
                    train=True,
                    sampled_frames=opt.sampled_frames,
                    transform=train_transformer,
                    max_skip=max_skip,
                    samples_per_video=opt.samples_per_video
                )
                datalist += [ds] * freq

            trainset = data.ConcatDataset(datalist)

        else:
            max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
            trainset = DATA_CONTAINER[opt.trainset](
                train=True,
                sampled_frames=opt.sampled_frames,
                transform=train_transformer,
                max_skip=max_skip,
                samples_per_video=opt.samples_per_video
            )
    except KeyError as ke:
        print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
        print(list(DATA_CONTAINER.keys()))
        exit()

    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
    )

    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn, drop_last=True,
                                  prefetch_factor=4, pin_memory=True)

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn, prefetch_factor=4, pin_memory=True)
    # Model
    print("==> creating model")

    # net = STM(opt.keydim, opt.valdim, 'train',
    #           mode=opt.mode, iou_threshold=opt.iou_threshold)

    # local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(args.local_rank)

    net = FSVS(single_object=True, phase='train', lr=opt.learning_rate, local_rank=args.local_rank)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.eval()

    net = nn.DataParallel(net).cuda()
    # net = net.to(device)
    # net = DDP(net,
    #             device_ids=[args.local_rank],
    #             output_device=args.local_rank).to(device)

    # set training parameters
    for p in net.parameters():
        p.requires_grad = True

    criterion = None
    celoss = binary_entropy_loss

    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj: celoss(pred, target, obj) + lovasz_softmax(pred, target)
    elif opt.loss == 'cross_1':
        criterion = cross_entropy_loss_1
    elif opt.loss == 'focal and iou':
        criterion = lambda pred, target, obj: focal_loss(pred, target, obj) + mask_iou(pred, target, obj)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    optimizer = None

    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                              momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                               betas=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.solver == 'GSFM':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                               betas=opt.momentum, weight_decay=opt.weight_decay)

    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    # Resume
    title = 'STM'
    minloss = float('inf')
    # max_jaccard = 0.0
    max_dice = 0.0

    opt.checkpoint = osp.join(osp.join(opt.checkpoint, opt.round_name))
    if not osp.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        # max_jaccard = checkpoint['max_jaccard']

        # max_dice = checkpoint['max_dice']

        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        skips = checkpoint['max_skip']

        # net.load_model(opt.resume)

        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                trainloader.dataset.set_max_skip(skips)
        except:
            print('[Warning] Initializing max skip fail')

        logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=True)
    else:
        if opt.initial:
            print('==> Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial)
            if isinstance(weight, OrderedDict):
                net.module.load_param(weight)
            else:
                net.module.load_param(weight['state_dict'])

        logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=False)
        start_epoch = 0

    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    # Train and val
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)

    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        net.module.phase = 'new_train'
        train_loss = train(trainloader,
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=True,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold,
                           lr=opt.learning_rate)

        if (epoch + 1) % opt.epoch_per_test == 0:
            net.module.phase = 'test'
            test_loss = test(testloader,
                             model=net.module,
                             criterion=criterion,
                             epoch=epoch,
                             use_cuda=True)

        # append logger file
        logger.log(epoch + 1, opt.learning_rate, train_loss)

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()

                # save model
        # max_jaccard = 0.0
        if (epoch + 1) % opt.epoch_per_test == 0:
            D = counting_dice()
            is_best = D > max_dice
            max_dice = max(D, max_dice)
            # J = eval_jaccard()
            # is_best = J > max_jaccard
            # max_jaccard = max(J, max_jaccard)

        # is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
            if isinstance(trainloader.dataset, data.ConcatDataset) \
            else trainloader.dataset.max_skip

        # 最佳以及最新权重保存，调试期间只保存最佳权重
        # 程序因为网络问题中途中断，改为保存最新和最佳权重

        if (epoch + 1) % opt.epoch_per_save == 0 and is_best:
            save_best_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': train_loss,
                'minloss': minloss,
                # 'max_jaccard': max_jaccard,
                'max_dice': max_dice,
                'optimizer': optimizer.state_dict(),
                'max_skip': skips,
            }, epoch + 1, is_best, checkpoint=opt.checkpoint, round_name=opt.round_name, filename=opt.mode)

        save_newest_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'loss': train_loss,
            'minloss': minloss,
            # 'max_jaccard': max_jaccard,
            'max_dice': max_dice,
            'optimizer': optimizer.state_dict(),
            'max_skip': skips,
        }, epoch + 1, checkpoint=opt.checkpoint, round_name=opt.round_name, filename=opt.mode)

    logger.close()

    print('minimum loss:')
    print(minloss)
    print('max Dice: ')
    print(max_dice)


# bi-direction
def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold, lr=None):
    # switch to train mode
    loss_func = nn.MSELoss()
    data_time = AverageMeter()
    loss = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for batch_idx, data in enumerate(trainloader):
        # frames, masks, objs, infos = data  # 8.2修改，增加从上至下推理以及从下至上推理
        frames_topdown, masks_topdown, frames_bottomup, masks_bottomup, objs, infos = data



        FSVS_masks_topdown = convert_one_hot_new(masks_topdown, 1)
        FSVS_masks_bottomup = convert_one_hot_new(masks_bottomup, 1)


        data_time.update(time.time() - end)

        if use_cuda:
            frames_topdown = frames_topdown.cuda()
            masks_topdown = masks_topdown.cuda()
            frames_bottomup = frames_bottomup.cuda()
            masks_bottomup = masks_bottomup.cuda()
            FSVS_masks_topdown = FSVS_masks_topdown.cuda()
            FSVS_masks_bottomup = FSVS_masks_bottomup.cuda()


            objs = objs.cuda()



        objs[objs == 0] = 1

        N, T, C, H, W = frames_topdown.size()
        max_obj = masks_topdown.shape[2] - 1


        total_loss = 0.0


        out_td, quality_td, ious_td, kl_td, dsl_td = model(data=data, frame=frames_topdown, mask=masks_topdown,
                                                           num_objects=objs, criterion=mask_iou_loss)
        out_bu, quality_bu, ious_bu, kl_bu, dsl_bu = model(data=data, frame=frames_bottomup, mask=masks_bottomup,
                                                           num_objects=objs, criterion=mask_iou_loss)


        for idx in range(N):
            for t in range(0, T):
                if t == 0:
                    gt = masks_topdown[idx, t:t + 1]
                    pred = out_bu[idx, T - 2:T - 1]
                    pred = pred.cuda()
                    No = objs[idx].item()
                    total_loss = total_loss + criterion(pred, gt, No)

                elif t == 11:
                    gt = masks_topdown[idx, t:t + 1]
                    pred = out_td[idx, T - 2:T - 1]
                    pred = pred.cuda()
                    No = objs[idx].item()
                    total_loss = total_loss + criterion(pred, gt, No)



                else:
                    continue
                total_loss = total_loss + criterion(pred, gt, No)


        total_loss += 0.5 * (0.1*kl_td.item() + dsl_td.item() + 0.1*kl_bu.item() + dsl_bu.item())


        total_loss = total_loss / (N * 2)

        if total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)
        # # compute gradient and do SGD step (divided by accumulated steps)
        total_loss /= iter_size


        total_loss.backward()

        if (batch_idx + 1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()
        time
        # measure elapsed
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.sum,
            loss=total_loss
        )
        bar.next()
    bar.finish()

    return loss.avg


# bi_direction
def test(testloader, model, criterion, epoch, use_cuda):
    data_time = AverageMeter()

    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames_topdown, masks_topdown, frames_bottomup, masks_bottomup, objs, infos = data

            if use_cuda:
                frames_topdown = frames_topdown.cuda()
                frames_bottomup = frames_bottomup.cuda()
                masks_topdown = masks_topdown.cuda()
                masks_bottomup = masks_bottomup.cuda()

            num_objects = objs[0]
            info = infos[0]
            max_obj = masks_topdown.shape[2] - 1
            # compute output
            t1 = time.time()

            N, T, _, H, W = frames_topdown.shape
            # mask = masks_topdown[0]
            # pred = [mask[0:1]]
            pred_td = [masks_topdown[0][0:1]]
            pred_bu = [masks_bottomup[0][0:1]]
            keys = []
            vals = []

            pred_td = model(data=data, frame=frames_topdown, mask=masks_topdown, pred=pred_td, num_objects=num_objects,
                            max_obj=max_obj)
            pred_bu = model(data=data, frame=frames_bottomup, mask=masks_bottomup, pred=pred_bu,
                            num_objects=num_objects, max_obj=max_obj)

            # pred = torch.cat(pred, dim=0)
            # pred = pred.detach().cpu().numpy()

            pred_td = torch.cat(pred_td, dim=0)
            pred_bu = torch.cat(pred_bu, dim=0)
            pred_td = pred_td.detach().cpu().numpy()
            pred_bu = pred_bu.detach().cpu().numpy()

            pred = np.zeros((pred_td.shape[0] - 1, *pred_td.shape[1:]))
            for t in range(T - 1):
                if t == 0:
                    # whole[t, :, :, :] = ROI_to_whole(pred_td[t, :, :, :], top=123, left=120, ori_size=512)
                    pred[t, :, :, :] = pred_td[t, :, :, :]
                else:
                    pred[t, :, :, :] = pred_td[t, :, :, :] * (T - 2 - t) / (T - 3) + pred_bu[T - t - 1, :, :, :] * (
                            t - 1) / (T - 3)
                    # whole[t, :, :, :] = ROI_to_whole(pred_td[t, :, :, :], top=123, left=120, ori_size=512)
                    # pred[t, :, :, :] = pred_td[t, :, :, :]

            write_mask(pred, info, opt, directory=opt.output_dir)

            toc = time.time() - t1

            data_time.update(toc, 1)

            # plot progress
            bar.suffix = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return



if __name__ == '__main__':
    main()
