"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.FSVS_network import FSVS
from networks.STCN_loss import LossComputer
from dataloaders.data import convert_one_hot


class FSVSModel:
    def __init__(self, single_object=True, save_path=None, lr=0, phase='new_train', local_rank=0):
        # self.para = para
        self.single_object = single_object
        self.local_rank = local_rank

        self.FSVS = nn.parallel.DistributedDataParallel(
            FSVS(self.single_object).cuda(),
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)


        self.save_path = save_path
        self.loss_computer = LossComputer()

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.FSVS.parameters()), lr=lr, weight_decay=1e-7)
        self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000


        self.phase = phase

    def do_pass(self, frame, mask=None, criterion=None, pred=None, info=None, epoch=None):
        if self.phase == 'test':
            _, T, _, _, _ = frame.shape
            k16, kf16_thin, kf16, kf8, kf4 = self.FSVS('encode_key', frame)
            ref_value = []
            prev_value = []
            for t in range(0, T):
                if t == 0:
                    ref_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], mask[:, t])
                elif t == 1:
                    this_logits, this_mask, this_boundary = self.FSVS('segment', k16[:, :, t], kf16_thin[:, t],
                                                                      kf8[:, t],
                                                                      kf4[:, t],
                                                                      k16[:, :, t - 1:t], ref_value)
                    prev_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

                else:
                    this_logits, this_mask, this_boundary = self.FSVS('segment',
                                                                      k16[:, :, t], kf16_thin[:, t], kf8[:, t],
                                                                      kf4[:, t],
                                                                      k16[:, :, t - 2:t], values)
                    ref_value = prev_value
                    prev_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], this_mask)
                    values = torch.cat([ref_value, prev_value], 2)
                    pred.append(this_mask)

            return pred

        elif self.phase == 'new_train':
            data = {}
            N, T, _, _, _ = frame.shape
            k16, kf16_thin, kf16, kf8, kf4 = self.FSVS('encode_key', frame)
            ref_value = []
            prev_value = []
            batch_out = []
            kf16s = []
            tmp_out = []
            # frame_certainty_map = {}
            for t in range(0, T):
                if t == 0:
                    this_mask_SIE = convert_one_hot(mask[:, t], 1)
                    ref_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], this_mask_SIE)
                elif t == 1:
                    this_logits, this_mask, this_bondary, this_logits_SIE, this_mask_SIE = self.FSVS('segment',
                                                                                                       k16[:, :, t],
                                                                                                       kf16_thin[:,
                                                                                                       t], kf8[:, t],
                                                                                                       kf4[:, t],
                                                                                                       k16[:, :,
                                                                                                       t - 1:t],
                                                                                                       ref_value)


                    data['mask_%d' % t] = this_mask_SIE
                    data['logits_%d' % t] = this_logits_SIE
                    data['boundary_%d' % t] = this_bondary

                    prev_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], this_mask_SIE)
                    values = torch.cat([ref_value, prev_value], 2)
                    # pred.append(this_mask)

                    kf16s.append(kf16[:, t])
                    out = this_mask_SIE
                    tmp_out.append(out)

                else:
                    this_logits, this_mask, this_bondary, this_logits_SIE, this_mask_SIE = self.FSVS('segment',
                                                                                                       k16[:, :, t],
                                                                                                       kf16_thin[:, t],
                                                                                                       kf8[:, t],
                                                                                                       kf4[:, t],
                                                                                                       k16[:, :,
                                                                                                       t - 2:t], values)
                    ref_value = prev_value
                    prev_value = self.FSVS('encode_value', frame[:, t], kf16[:, t], this_mask_SIE)
                    values = torch.cat([ref_value, prev_value], 2)


                    kf16s.append(kf16[:, t])
                    out = this_mask_SIE
                    tmp_out.append(out)

                    data['mask_%d' % t] = this_mask_SIE
                    data['logits_%d' % t] = this_logits_SIE
                    data['boundary_%d' % t] = this_bondary

            batch_out.append(torch.cat(tmp_out, dim=0))
            batch_out = torch.stack(batch_out, dim=0)

            losses = self.loss_computer.compute(frame, mask, data, epoch)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.scaler.step(self.op)
            print('total_loss: ', losses['total_loss'])

            return batch_out, losses['total_loss']

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

    def train(self):
        self.phase = 'new_train'
        self._is_train = True
        self._do_log = True
        self.FSVS.eval()
        return self

    def val(self):
        self.phase = 'test'
        self._is_train = False
        self._do_log = True
        self.FSVS.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.FSVS.eval()
        return self
