import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE
from options.trans_option import TrainTransOptions

from utils.get_opt import get_opt
from utils.utils import fixseed


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    if vq_opt.dataset_name == "interhuman":
        vq_opt.dim_joint = 12
    if vq_opt.dataset_name == "interx":
        vq_opt.dim_joint = 6
   
    vq_model = RVQVAE(vq_opt,
                        vq_opt.dim_joint,
                        vq_opt.nb_code,
                        vq_opt.code_dim,
                        vq_opt.output_emb_width,
                        vq_opt.down_t,
                        vq_opt.stride_t,
                        vq_opt.width,
                        vq_opt.depth,
                        vq_opt.dilation_growth_rate,
                        vq_opt.vq_act,
                        vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'best_fid.tar'),  map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('decoder.conv') or k.startswith('decoder.resnets')for k in missing_keys])
    print(f'Loading VQ Model {opt.vq_name}, epoch {ckpt["ep"]}')
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainTransOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')


    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    vq_model, vq_opt = load_vq_model()

    if opt.dataset_name == "interhuman":
        opt.data_root = 'data/InterHuman'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.test_batch_size = 96
        fps = 30

        from data.interhuman import InterHumanDataset
        from models.evaluator.evaluator import EvaluatorModelWrapper

        opt.mode = "train"
        train_dataset = InterHumanDataset(opt)
        opt.mode = "val"
        val_dataset = InterHumanDataset(opt)

        if opt.do_eval:
            opt.mode = "val"
            test_dataset = InterHumanDataset(opt)
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, drop_last=True, num_workers=0, shuffle=False)

            evalmodel_cfg = get_opt("checkpoints/eval_model/eval_model.yaml", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, opt.device)
        else:
            test_loader = None
            eval_wrapper = None
    
    elif opt.dataset_name == "interx":
        opt.data_root = 'data/InterX'
        opt.motion_dir = pjoin(opt.data_root, 'motions')
        opt.text_dir = pjoin(opt.data_root, 'texts_processed')

        opt.motion_rep = "smpl"
        opt.joints_num = 55 if opt.motion_rep == "global" else 56
        opt.max_motion_length = 150
        opt.max_text_len = 35
        opt.unit_length = 4
        
        opt.test_batch_size = 32
        vq_opt.dim_joint = 6
        fps = 30

        from data.interx import Text2MotionDatasetHHI, Text2MotionDatasetV2HHI, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'glove'), 'hhi_vab')
        train_dataset = Text2MotionDatasetV2HHI(opt, 
                                           pjoin(opt.data_root, 'splits/train.txt'),
                                           w_vectorizer, 
                                           pjoin(opt.motion_dir, 'train.h5'))
        val_dataset = Text2MotionDatasetV2HHI(opt, 
                                         pjoin(opt.data_root, 'splits/val.txt'),
                                         w_vectorizer, 
                                         pjoin(opt.motion_dir, 'val.h5'))
        
        if opt.do_eval:
            test_dataset = Text2MotionDatasetV2HHI(opt, 
                                                pjoin(opt.data_root, 'splits/val.txt'), 
                                                w_vectorizer, 
                                                pjoin(opt.motion_dir, 'val.h5'))
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, 
                                    num_workers=4, drop_last=True, collate_fn=collate_fn, shuffle=True)
            
            wrapper_opt = get_opt("checkpoints/hhi/Comp_v6_KLD01/opt.txt", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        else:
            test_loader = None
            eval_wrapper = None
    
    else:
        raise KeyError('Dataset Does not Exists')

    clip_version = 'ViT-L/14@336px'
    opt.num_tokens = vq_opt.nb_code
    mask_transformer = MaskTransformer(code_dim=vq_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      clip_dim=768,#512,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=opt)

    pc_transformer = sum(param.numel() for param in mask_transformer.parameters_wo_clip())
    print('Total parameters of the Masked Transformer=: {:.2f}M'.format(pc_transformer / 1000_000))

    

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=False, pin_memory=True)
    

    opt.save_vis = False
    opt.gen_react = False

    trainer = MaskTransformerTrainer(opt, mask_transformer, vq_model)

    trainer.train(train_loader, val_loader, test_loader, eval_wrapper=eval_wrapper)