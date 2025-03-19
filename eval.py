import numpy as np
import torch
import os
from os.path import join as pjoin
from tqdm import tqdm
from datetime import datetime

from options.eval_option import arg_parse
from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer

from utils.metrics import *
from utils.get_opt import get_opt
from utils.utils import fixseed
from collections import OrderedDict

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings, emb_scale)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings, emb_scale)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times, emb_scale, divide_by)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                if len(batch) == 5:
                    batch[2] = batch[2][0]
                    batch[3] = batch[3][0]
                    batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times, emb_scale, divide_by)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            if replication > 0:
                opt.save_vis = False
            motion_loaders['ground truth'].dataset.normalize = True
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                print(f'Generating motions from {motion_loader_name}')
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                if mm_motion_loader is not None:
                    mm_motion_loaders[motion_loader_name] = mm_motion_loader
            motion_loaders['ground truth'].dataset.normalize = False

            print(f'\n==================== Replication {replication} ====================')
            print(f'\n==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            if mm_motion_loaders:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!\n')
            print(f'!!! DONE !!!\n', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            if mm_motion_loaders:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

def evaluation_during_training(opt, net, test_loader, eval_wrapper_passed, epoch, file, trans=None):
    mm_num_samples = 0 #100
    mm_num_repeats = 30
    time_steps = 20
    cond_scale = 2
    topkr = 0.9

    global eval_wrapper, emb_scale, divide_by
    eval_wrapper = eval_wrapper_passed

    test_loader.dataset.normalize = True
    if opt.dataset_name == "interhuman":
        from models.evaluator.evaluator import get_motion_loader
        emb_scale = 6
        divide_by = 2
    elif opt.dataset_name == "interx":
        from models.evaluator.evaluator_interx import get_motion_loader
        emb_scale = 1
        divide_by = 1

    opt.gen_react = False
    gen_motion_loader, _ = get_motion_loader(
                            opt.test_batch_size,
                            net,
                            trans,
                            test_loader.dataset,
                            opt.device,
                            mm_num_samples,
                            mm_num_repeats,
                            None,
                            opt,
                            time_steps,
                            cond_scale,
                            topkr
                            )
    
    test_loader.dataset.normalize = False
    eval_motion_loaders = {'gt': test_loader,
                           'gen': gen_motion_loader}
    
    with open(file, 'a') as f:
        print(f'==================== Epoch {epoch} ====================')
        print(f'\n==================== Epoch {epoch} ====================', file=f, flush=True)

        mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_motion_loaders, f)

        fid_score_dict = evaluate_fid(test_loader, acti_dict, f)

    return fid_score_dict['gen'], mat_score_dict['gen'], R_precision_dict['gen'][0]
    


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
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
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
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

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
        opt.mm_num_samples = 0
        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        main_opt = get_opt(vq_opt_path, opt.device)
    
    mm_num_samples = opt.mm_num_samples
    mm_num_repeats = opt.mm_num_repeats
    mm_num_times = opt.mm_num_times
    diversity_times = opt.diversity_times
    replication_times = opt.replication_times

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.vis_dir = pjoin(opt.save_root, 'animation')
    
    if opt.dataset_name == "interhuman":
        opt.npy_dir = pjoin(opt.vis_dir, 'keypoint_npy')
        opt.vis_dir = pjoin(opt.vis_dir, 'keypoint_mp4')
        os.makedirs(opt.npy_dir, exist_ok=True)
    elif opt.dataset_name == "interx":
        opt.vis_dir = pjoin(opt.vis_dir, 'smpl_npy')
    
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.vis_dir, exist_ok=True)
    react_name = "react_" if opt.gen_react else ""     

    if main_opt.dataset_name == "interhuman":
        main_opt.data_root = 'data/InterHuman'
        main_opt.joints_num = 22
        dim_pose = 12
        fps = 30
        opt.batch_size = 96
        main_opt.mode = "test"
        emb_scale = 6
        divide_by = 2
        
        from models.evaluator.evaluator import EvaluatorModelWrapper, get_dataset_motion_loader, get_motion_loader
        evalmodel_cfg = get_opt("checkpoints/eval_model/eval_model.yaml", opt.device, complete=False)
        eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, opt.device)

    elif main_opt.dataset_name == "interx":
        main_opt.data_root = 'data/InterX'
        opt.data_root = main_opt.data_root
        main_opt.motion_dir = pjoin(main_opt.data_root, 'motions')
        main_opt.text_dir = pjoin(main_opt.data_root, 'texts_processed')
        main_opt.motion_rep = "smpl"
        main_opt.joints_num = 55 if main_opt.motion_rep == "global" else 56 
        dim_pose = 12 if main_opt.motion_rep == "global" else 6
        fps = 30
        opt.batch_size = 32
        main_opt.max_motion_length = 150
        main_opt.max_text_len = 35
        main_opt.unit_length = 4
        emb_scale = 1
        divide_by = 1

        from models.evaluator.evaluator_interx import EvaluatorModelWrapper, get_dataset_motion_loader, get_motion_loader
        wrapper_opt = get_opt("checkpoints/hhi/Comp_v6_KLD01/opt.txt", opt.device, complete=False)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        raise KeyError('Dataset Does not Exists')
    
    
    data_cfg = main_opt
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, opt.batch_size)

    def make_callable(net, file, trans=None):
        return lambda: get_motion_loader(
                                        opt.batch_size,
                                        net,
                                        trans,
                                        gt_dataset,
                                        opt.device,
                                        mm_num_samples,
                                        mm_num_repeats,
                                        file,
                                        opt,
                                        time_step,
                                        cond_scale,
                                        topkr
                                        )
    
    if opt.use_trans:
        for cond_scale in opt.cond_scales:
            for time_step in opt.time_steps:
                for topkr in opt.topkr:
                    eval_motion_loaders = {}
                    for file in os.listdir(opt.model_dir):
                        if opt.which_epoch != "all" and opt.which_epoch not in file:
                            continue
                        
                        print(f"\n\nLoading model epoch: {file}")
                        trans = load_trans_model(main_opt, file)
                        net, ep = load_vq_model(vq_opt, "best_fid.tar")
                        
                        file = react_name + file
                        eval_motion_loaders[file] = make_callable(net, file, trans)
                    
                    which_epoch = opt.which_epoch
                    
                    log_file_name = f'evaluation_{which_epoch}_ts{time_step}_cs{cond_scale}_topkr{topkr}.log'
                    log_file_name = react_name + log_file_name
                    log_file = pjoin(opt.eval_dir, log_file_name)
                    evaluation(log_file)
    else:
        eval_motion_loaders = {}
        for file in os.listdir(opt.model_dir):
            if opt.which_epoch != "all" and opt.which_epoch not in file:
                continue
            cond_scale, time_step, topkr = None, None, None
            
            print(f"\n\nLoading model epoch: {file}")
            net, ep = load_vq_model(main_opt, file)
            eval_motion_loaders[file] = make_callable(net, file)
            
        log_file = pjoin(opt.eval_dir, f'evaluation_{opt.which_epoch}.log')
        evaluation(log_file)
