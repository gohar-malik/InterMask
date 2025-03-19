import os
from os.path import join as pjoin
import h5py
import numpy as np
from tqdm import tqdm
import torch

from data.body_model.body_model import BodyModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bm = BodyModel(bm_fname="data/body_model/smplx/SMPLX_NEUTRAL.npz", num_betas=10).to(device)

motion_root = "data/InterX/motions/"
for split in ["train", "val", "test"]:
    motion_src = pjoin(motion_root, f"{split}.h5")
    motion_dst = pjoin(motion_root, f"{split}_global.h5")
    motion_dst_file = h5py.File(motion_dst, "w")

    with h5py.File(motion_src, "r") as mf:
        keys = list(mf.keys())

        for key in tqdm(keys, desc=f"Processing {split}"):
            motion = mf[key][:].astype('float32')
            T, J, _ = motion.shape
            motion  = torch.tensor(motion, dtype=torch.float32).to(device)
            motion1, motion2 = motion[:,:,:3].clone(), motion[:,:,3:].clone()
            joints_seq_1 = torch.zeros((T, J-1, 3)).to(device)
            joints_seq_2 = torch.zeros((T, J-1, 3)).to(device)
            
            for t in range(T):
                pose1 = motion1[[t]]
                hand_pose1 = torch.cat([pose1[:, 25:40, :], pose1[:, 40:55, :]], dim=1)
                bm_output1 = bm(root_orient=pose1[:, 0, :],
                                pose_body=pose1[:, 1:22, :].reshape(1, -1),
                                pose_hand = hand_pose1.reshape(1, -1))
                joints_seq_1[t,:,:] = bm_output1.Jtr + pose1[:,-1,:]
            
                pose2 = motion2[[t]]
                hand_pose2 = torch.cat([pose2[:, 25:40, :], pose2[:, 40:55, :]], dim=1)
                bm_output2 = bm(root_orient=pose2[:, 0, :],
                                pose_body=pose2[:, 1:22, :].reshape(1, -1),
                                pose_hand = hand_pose2.reshape(1, -1))
                joints_seq_2[t,:,:] = bm_output2.Jtr + pose2[:,-1,:]
            
            # from utils.plot_script import plot_3d_motion_2views, plot_3d_motion
            # from utils.paramUtil import t2m_kinematic_chain, hhi_kinematic_chain, hhi_left_hand_chain, hhi_right_hand_chain
            # kinematic_chain =  hhi_kinematic_chain + hhi_left_hand_chain + hhi_right_hand_chain

            # joints_seq = [joints_seq_1.detach().cpu().numpy(), joints_seq_2.detach().cpu().numpy()]
            # plot_3d_motion("interx_smpl_to_jonts_sample.mp4", kinematic_chain, joints_seq, "text", fps=30)

            vel1 = joints_seq_1[1:] - joints_seq_1[:-1]
            vel1 = torch.cat([vel1, torch.zeros(1, J-1, 3).to(device)], axis=0)
            vel2 = joints_seq_2[1:] - joints_seq_2[:-1]
            vel2 = torch.cat([vel2, torch.zeros(1, J-1, 3).to(device)], axis=0)

            motion1_final = torch.cat([joints_seq_1, vel1, motion1[:,:55,:]], axis=2)
            motion2_final = torch.cat([joints_seq_2, vel2, motion2[:,:55,:]], axis=2)
            motion_final = torch.cat([motion1_final, motion2_final], axis=2).detach().cpu().numpy()
            # print(motion_final.shape)
            motion_dst_file.create_dataset(key, data=motion_final, dtype='f4')
            # exit()
