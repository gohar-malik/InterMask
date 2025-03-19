import numpy as np
import torch
from data.body_model.body_model import BodyModel
import data.rotation_conversions as geometry

class InterxKinematics():
    def __init__(self):
        self.bm = BodyModel(bm_fname="data/body_model/smplx/SMPLX_NEUTRAL.npz", num_betas=10)
        self.bm.eval()
    
    def rot6d_to_axisangle(self, motionrot6d):
        # Root Translation and Velocity       
        root = motionrot6d[:, :, -1, :]
        root_trans = root[:, :, :3]
        root_vel = root[:, :, 3:]

        # Whole body pose
        pose = motionrot6d[:, :, :-1, :]
        pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))

        # Root Orientation
        root_orient = pose[:, :, 0, :]
        body_pose = pose[:, :, 1:, :]

        return root_trans, root_vel, root_orient, body_pose
    
    def forward(self, motions):
        """
        Args:
            motions: torch.Tensor of shape (B, T, 56, dim) 
            T=64 for vqvae
            dim=6 for 6d rots
        """
        B, T, J, dim = motions.shape
        self.bm.to(motions.device)
        
        root_trans, root_vel, root_orient, body_pose = self.rot6d_to_axisangle(motions)
        motions_pos = []
        #torch.zeros(B, T, J-1, 3, requires_grad=True).to(motions.device)
        for b in range(B):
            bm_output = self.bm(root_orient = root_orient[b],
                                    pose_body= body_pose[b, :, 0:21, :].reshape(T, -1),
                                    pose_hand = body_pose[b, :, 24:, :].reshape(T, -1))
            motions_pos.append(bm_output.Jtr + root_trans[b].unsqueeze(-2))
            # motions_pos[b] = bm_output.Jtr + root_trans[b].unsqueeze(-2)
        motions_pos = torch.stack(motions_pos, dim=0)
        return motions_pos

class InterxNormalizerTorch():
    def __init__(self):
        mean = np.load("data/interx_mean.npy")
        std = np.load("data/interx_std.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x
    



    