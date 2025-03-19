from os.path import join as pjoin
import torch
from torch.utils.data import Dataset, DataLoader
from data.interhuman import InterHumanDataset
from data.utils import MotionNormalizer
from utils.plot_script import preprocess_plot_motion
# from models import *
import copy
import random
import time
import numpy as np
from models.evaluator.evaluator_models import InterCLIP
from tqdm import tqdm

class EvaluationDataset(Dataset):

    def __init__(self, model, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1
                    # text = list(text) * mm_num_repeats
                    # motion_lens = torch.tensor([motion_lens[0].item()] * mm_num_repeats)
                
                if trans is None:
                    motion1_output, _, _ = self.model(motion1[:, :motion_lens[0].item()].float().to(device))
                    motion2_output, _, _ = self.model(motion2[:, :motion_lens[0].item()].float().to(device))
                
                else:
                    ids_length = (motion_lens.detach().long().to(device)//4)

                    for num_repeat in range(num_repeats):
                        if opt.gen_react:
                            code_idx1, _ = self.model.encode(motion1[:, :motion_lens[0].item()].float().to(device))
                            motion_ids = trans.generate_reaction(text, code_idx1[..., 0], ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        else:
                            motion_ids = trans.generate(text, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        
                        motion_ids1, motion_ids2 = motion_ids[:, :motion_ids.shape[1]//2], motion_ids[:, motion_ids.shape[1]//2:]
                        
                       
                        motion1_output_one = self.model.forward_decoder(motion_ids1.unsqueeze_(-1).to(device))
                        motion2_output_one = self.model.forward_decoder(motion_ids2.unsqueeze_(-1).to(device))
                        
                        if num_repeat == 0:
                            motion1_output = motion1_output_one
                            motion2_output = motion2_output_one
                        else:
                            motion1_output = torch.cat((motion1_output, motion1_output_one), dim=0)
                            motion2_output = torch.cat((motion2_output, motion2_output_one), dim=0)


                    
                gen_motion_len = motion1_output.shape[1]

                padding_len = motion1.shape[1] - motion1_output.shape[1]
                B, D = motion1_output.shape[0], motion1_output.shape[2]
                padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                motion1_output = torch.concat((motion1_output, padding_zeros), dim=1)
                motion2_output = torch.concat((motion2_output, padding_zeros), dim=1)
                
                if opt.gen_react:
                    batch.update({"output": torch.cat([motion1.to(device), motion2_output], dim=-1)})
                else:
                    batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())
                # motions_output = motions_output.cpu().detach().numpy()

                if trans is None:
                    save_vis_n = 10
                else:
                    save_vis_n = 20

                if i  < save_vis_n and opt.save_vis:
                    motions_input = torch.cat([motion1, motion2], dim=-1)[0]
                    motions_input = motions_input.reshape(motions_input.shape[0], 2, -1)
                    motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())
                    # motions_input = motions_input.cpu().detach().numpy()
                    
                    preprocess_plot_motion(motions_input[:motion_lens[0].item(), :, :],  text[0],
                                           opt.vis_dir, opt.npy_dir,
                                           f"{file.split('.')[0]}_{i:02d}_gt", foot_ik=False)
                    if trans is None:
                        gen_file_name = f"{file.split('.')[0]}_{i:02d}_gen"
                    else:
                        gen_file_name = f"{file.split('.')[0]}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_gen"
                        
                    preprocess_plot_motion(motions_output[0][:gen_motion_len, :, :], text[0],
                                           opt.vis_dir, opt.npy_dir,
                                           gen_file_name, foot_ik=True,)
                
                # if i >= save_vis_n and opt.save_vis:
                #     exit()

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length


                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': gen_motion_len,
                            'text': text[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': gen_motion_len,
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 'interhuman':
        print('Loading dataset %s ...' % opt.dataset_name)

        dataset = InterHumanDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset




def get_motion_loader(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    # Currently the configurations of two datasets are almost the same
    start = time.time()
    dataset = EvaluationDataset(model, trans, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, file=file, opt=opt, time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader




def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin('checkpoints/eval_model/interclip.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
