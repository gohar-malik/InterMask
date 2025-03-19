import numpy as np
import torch
import os
import time
from os.path import join as pjoin
from tqdm import tqdm

from options.eval_option import arg_parse
from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer

from utils.get_opt import get_opt
from utils.utils import fixseed
from data.utils import MotionNormalizer
from utils.plot_script import preprocess_plot_motion

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch), map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('decoder.conv') or k.startswith('decoder.resnets')for k in missing_keys])
    
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

def load_trans_model(model_opt, which_model):
    # clip_version = 'ViT-B/32'
    clip_version = 'ViT-L/14@336px'
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=768,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    assert all([k.startswith('clip_') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def gen_motions(opt, file, texts, net, trans=None, motion_len=90):
    normalizer = MotionNormalizer()
    net = net.to(opt.device)
    net.eval()
    trans = trans.to(opt.device)
    trans.eval()
    
    num_samples = 1
    motion_lens = torch.tensor([motion_len]*num_samples)
    ids_length = (motion_lens.detach().long().to(opt.device)//4)
    file = f"infer_{file.split('.')[0]}"
    
    
    for cond_scale in opt.cond_scales:
        for time_steps in opt.time_steps:
            for topkr in opt.topkr:
                time_taken = []
                for i, text in tqdm(enumerate(texts)):
                    with torch.no_grad():
                        text = [text] * num_samples
                        
                        tick = time.time()
                        motion_ids = trans.generate(text, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        motion_ids1, motion_ids2 = motion_ids[:, :motion_ids.shape[1]//2], motion_ids[:, motion_ids.shape[1]//2:]
                        
                        motion1_output = net.forward_decoder(motion_ids1.unsqueeze_(-1).to(opt.device))
                        motion2_output = net.forward_decoder(motion_ids2.unsqueeze_(-1).to(opt.device))
                        time_taken.append(time.time() - tick)

                        motion_output = torch.cat([motion1_output, motion2_output], dim=-1)
                        
                        if opt.dataset_name == "interhuman":
                            motions_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], 2, -1)
                            motions_output = normalizer.backward(motions_output.cpu().detach().numpy())

                            for motion_i in range(motions_output.shape[0]):
                                gen_file_name = f"{file}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_{motion_i:02d}"  
                                preprocess_plot_motion(motions_output[motion_i],
                                                    text[0], 
                                                    opt.vis_dir,
                                                    opt.npy_dir,
                                                    gen_file_name,
                                                    foot_ik=True)
                        elif opt.dataset_name == 'interx':
                            motion_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], -1)

                            for motion_i in range(motion_output.shape[0]):
                                gen_file_name = f"{file}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_{motion_i:02d}.npy"  
                                np.save(pjoin(opt.vis_dir, gen_file_name), motion_output[motion_i].cpu().detach().numpy())
                print(f"Avg Time taken: {np.mean(time_taken)} secs")

if __name__ == '__main__':
    opt = arg_parse()
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    if opt.use_trans:
        trans_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        main_opt = get_opt(trans_opt_path, opt.device)
        fixseed(main_opt.seed)

        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, main_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, opt.device)
        
        main_opt.num_tokens = vq_opt.nb_code
        main_opt.code_dim = vq_opt.code_dim
    else:
        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        main_opt = get_opt(vq_opt_path, opt.device)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    if opt.dataset_name == "interhuman":
        main_dir = 'animation_infer'
        opt.vis_dir = pjoin(opt.save_root, main_dir, 'keypoint_mp4')
        opt.npy_dir = pjoin(opt.save_root, main_dir, 'keypoint_npy')
        os.makedirs(opt.vis_dir, exist_ok=True)
        os.makedirs(opt.npy_dir, exist_ok=True)

    elif opt.dataset_name == "interx":
        opt.vis_dir = pjoin(opt.save_root, 'animation_infer', 'smpl_npy')
        os.makedirs(opt.vis_dir, exist_ok=True)
    
    if main_opt.dataset_name == "interhuman":
        main_opt.data_root = 'data/InterHuman'
        main_opt.joints_num = 22
        dim_pose = 12
        fps = 30
    
    elif main_opt.dataset_name == "interx":
        main_opt.data_root = 'data/InterX'
        opt.data_root = main_opt.data_root
        main_opt.motion_dir = pjoin(main_opt.data_root, 'motions')
        main_opt.text_dir = pjoin(main_opt.data_root, 'texts_processed')
        main_opt.joints_num = 56
        dim_pose = 6
        fps = 30
        main_opt.max_motion_length = 150
        main_opt.max_text_len = 35
        main_opt.unit_length = 4
    
    else:
        raise KeyError('Dataset Does not Exists')
    
    with open("./prompts.txt") as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]
    print(texts)
    
    
    for file in os.listdir(opt.model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        
        print(f"\n\nLoading model epoch: {file}")
        trans = load_trans_model(main_opt, file)
        net, _ = load_vq_model(vq_opt, "best_fid.tar")
        
        gen_motions(opt, file, texts, net, trans)
