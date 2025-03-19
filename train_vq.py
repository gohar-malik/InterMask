import os
from os.path import join as pjoin
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse


from utils.get_opt import get_opt

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "interhuman":
        opt.data_root = 'data/InterHuman'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.test_batch_size = 96
        fps = 30
        
        # lazy import
        from data.interhuman import InterHumanMotion, InterHumanDataset
        from models.evaluator.evaluator import EvaluatorModelWrapper

        opt.mode = "train"
        train_dataset = InterHumanMotion(opt)
        opt.mode = "val"
        val_dataset = InterHumanMotion(opt)

        if opt.do_eval:
            opt.mode = "val"
            test_dataset = InterHumanDataset(opt)
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, 
                                    drop_last=True, num_workers=0, shuffle=False)

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
        dim_pose = 12 if opt.motion_rep == "global" else 6
        fps = 30

        from data.interx import MotionDatasetV2HHI, Text2MotionDatasetV2HHI, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        train_dataset = MotionDatasetV2HHI(opt, 
                                           pjoin(opt.data_root, 'splits/train.txt'), 
                                           pjoin(opt.motion_dir, 'train.h5'))
        val_dataset = MotionDatasetV2HHI(opt, 
                                         pjoin(opt.data_root, 'splits/val.txt'), 
                                         pjoin(opt.motion_dir, 'val.h5'))
        
        if opt.do_eval:
            test_dataset = Text2MotionDatasetV2HHI(opt, 
                                                pjoin(opt.data_root, 'splits/val.txt'), 
                                                WordVectorizer(pjoin(opt.data_root, 'glove'), 'hhi_vab'), 
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
    
    
    net = RVQVAE(opt,
                opt.dim_joint,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    pc_vq_enc = sum(param.numel() for param in net.encoder.parameters())
    pc_vq_dec = sum(param.numel() for param in net.decoder.parameters())
    print(net)
    print('Total parameters of VQVAE: {}M'.format(pc_vq/1000_000))
    print('Total parameters of encoder: {}M'.format(pc_vq_enc/1000_000))
    print('Total parameters of decoder: {}M'.format(pc_vq_dec/1000_000))
    print('Total parameters of all models: {}M'.format((pc_vq_enc+pc_vq_dec)/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)
    

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=False, pin_memory=True)

    opt.save_vis = False
    trainer.train(train_loader, val_loader, test_loader=test_loader, eval_wrapper=eval_wrapper)