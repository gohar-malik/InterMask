import time
import torch
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from sparsemax import Sparsemax

import numpy as np
from collections import defaultdict
from collections import OrderedDict
import os
from os.path import join as pjoin

from data.utils import MotionNormalizerTorch, face_joint_indx, fid_l, fid_r
from data.quaternion import *
from utils.utils import print_current_loss
from eval import evaluation_during_training
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.normalizer = MotionNormalizerTorch(self.device)
        self.InteractionLoss = torch.nn.SmoothL1Loss(reduction='none')
        self.softmax = Sparsemax(dim=-1)

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    
    
    def forward(self, batch_data):
        
        if self.opt.dataset_name == "interhuman":
            name, conds, motion1, motion2, m_lens = batch_data
        elif self.opt.dataset_name == "interx":
            _, _, conds, _, motions, m_lens, _ = batch_data
            motion1, motion2 = motions.split(6, dim=-1)

        motion1 = motion1.detach().float().to(self.device)
        motion2 = motion2.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        # print(f"Motions from dataset: {motion1.shape}, {motion2.shape}")
        # print(f"Motion lenghts: {m_lens}")
        
        code_idx1, _ = self.vq_model.encode(motion1)
        code_idx2, _ = self.vq_model.encode(motion2)
        code_idx = torch.cat([code_idx1, code_idx2], dim=1)
        # print(f"Code Index: {code_idx1.shape}, {code_idx2.shape}, {code_idx.shape}")
        
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        m_lens = m_lens // 4
        # print(f"Motion Lengths: {m_lens}")

        _loss, _acc, _, _, _ = self.t2m_transformer(code_idx[..., 0], conds, m_lens)
        return _loss, _acc
        
       

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None ,
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_') for k in missing_keys])
        
        self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer
        try:
            self.scheduler.load_state_dict({key: checkpoint['scheduler'][key] for key in ["last_epoch", "_step_count"]}) # Scheduler
        except:
            print('Resume wo scheduler')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, test_loader, eval_wrapper):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        # for name, p in self.t2m_transformer.named_parameters():
        #     print(name)
        
        total_iters = self.opt.max_epoch * len(train_loader)
        self.opt.milestones = [int(total_iters * 0.5), int(total_iters * 0.7), int(total_iters * 0.85)]
        self.opt.warm_up_iter = len(train_loader) // 4
        self.opt.log_every = len(train_loader) // 10
        self.opt.save_latest = len(train_loader) // 2
        
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        print(f'Milestones: {self.opt.milestones}')
        print('Warm Up Iterations: %04d, Log Every: %04d, Save Latest: %04d' % (self.opt.warm_up_iter, self.opt.log_every, self.opt.save_latest))
        
        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            it = it // self.opt.log_every * self.opt.log_every
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        logs = defaultdict(def_value, OrderedDict())

        max_acc = -np.inf
        min_loss = np.inf
        min_fid = np.inf
        max_top1 = -np.inf

        if self.opt.do_eval:
            eval_file = pjoin(self.opt.eval_dir, 'evaluation_training.log')

        while epoch < self.opt.max_epoch:
            epoch += 1
            self.t2m_transformer.train()
            self.vq_model.eval()

            if epoch > 200:
                self.opt.eval_every_e = 4
            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)


                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > max_acc:
                print(f"Improved accuracy from {max_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'best_acc.tar'), epoch, it)
                max_acc = np.mean(val_acc)
            
            if np.mean(val_loss) < min_loss:
                print(f"Improved Loss from {min_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_loss = np.mean(val_loss)

            if self.opt.do_eval:
                if epoch % self.opt.eval_every_e == 0:
                    self.vq_model.eval()
                    self.t2m_transformer.eval()
                    fid, mat, top1 = evaluation_during_training(self.opt, self.vq_model, test_loader, 
                                                                eval_wrapper, epoch, eval_file, trans=self.t2m_transformer)
                    self.logger.add_scalar('Test/FID', fid, epoch)
                    self.logger.add_scalar('Test/Matching', mat, epoch)
                    self.logger.add_scalar('Test/Top1', top1, epoch)
                    if fid < min_fid:
                        min_fid = fid
                        self.save(pjoin(self.opt.model_dir, 'best_fid.tar'), epoch, it)
                        print('Best FID Model So Far!~')
                    if top1 > max_top1:
                        max_top1 = top1
                        self.save(pjoin(self.opt.model_dir, 'best_top1.tar'), epoch, it)
                        print('Best Top1 Model So Far!~')
                
            print('\n')