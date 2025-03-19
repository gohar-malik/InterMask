import torch
from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
from data.quaternion import *
from data.utils import *
from data.interx_utils import *

class Geometric_Losses:
    def __init__(self, recons_loss, joints_num, dataset_name, device):
        
        if recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss()
        elif recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss()
        
        self.joints_num = joints_num
        self.fids = [*fid_l, *fid_r]
        self.dataset_name = dataset_name
        if self.dataset_name == 'interhuman':
            self.normalizer = MotionNormalizerTorch(device)
        elif self.dataset_name == 'interx':
            self.normalizer = InterxNormalizerTorch()
            self.kinematics = InterxKinematics()

    def calc_foot_contact(self, motion, pred_motion):
            if self.dataset_name == 'interhuman':
                B, T, _ = motion.shape
                motion = motion[..., :self.joints_num * 3]
                motion = motion.reshape(B, T, self.joints_num, 3)
                
                pred_motion = pred_motion[..., :self.joints_num * 3]
                pred_motion = pred_motion.reshape(B, T, self.joints_num, 3)
            
            
            feet_vel = motion[:, 1:, self.fids, :] - motion[:, :-1, self.fids,:]
            pred_feet_vel = pred_motion[:, 1:, self.fids, :] - pred_motion[:, :-1, self.fids,:]
            feet_h = motion[:, :-1, self.fids, 1]
            pred_feet_h = pred_motion[:, :-1, self.fids, 1]
            # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

            ## Calculate contacts
            thres = 0.001
            velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
                [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

            feet_x = (feet_vel[..., 0]) ** 2
            feet_y = (feet_vel[..., 1]) ** 2
            feet_z = (feet_vel[..., 2]) ** 2
            contact = ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)
            
            fc_loss = self.l1_criterion(pred_feet_vel[contact], torch.zeros_like(pred_feet_vel)[contact])
            if torch.isnan(fc_loss):
                fc_loss = torch.tensor(0).to(motion.device)
                if contact.sum() != 0:
                    print("FC nan but contact not 0")
            return fc_loss
                                 
    def calc_bone_lengths(self, motion):
        if self.dataset_name == 'interhuman':
            motion_pos = motion[..., :self.joints_num*3]
            motion_pos = motion_pos.reshape(motion_pos.shape[0], motion_pos.shape[1], self.joints_num, 3)
        elif self.dataset_name == 'interx':
            motion_pos = motion
        bones = []
        for chain in kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                bone = (motion_pos[..., chain[i], :] - motion_pos[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B,T,P,1]
                bones.append(bone)

        return torch.cat(bones, dim=-1)
    
    def calc_loss_geo(self, pred_rot, gt_rot, eps=1e-7):
        if self.dataset_name == "interhuman":
            pred_rot = pred_rot.reshape(pred_rot.shape[0], pred_rot.shape[1], -1, 6)
            gt_rot = gt_rot.reshape(gt_rot.shape[0], gt_rot.shape[1], -1, 6)


        pred_m = cont6d_to_matrix(pred_rot).reshape(-1,3,3)
        gt_m = cont6d_to_matrix(gt_rot).reshape(-1,3,3)

        m = torch.bmm(gt_m, pred_m.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+eps, 1-eps))

        return torch.mean(theta)
    
    def forward(self, motions, pred_motion):
        if self.dataset_name == 'interhuman':
            loss_rec = self.l1_criterion(pred_motion[..., :-4], motions[..., :-4])
            
            loss_explicit = self.l1_criterion(pred_motion[:, :, :self.joints_num*3],
                                            motions[:, :, :self.joints_num*3])
            
            loss_vel = self.l1_criterion(pred_motion[:, 1:, :self.joints_num*3] - pred_motion[:, :-1, :self.joints_num*3],
                                        motions[:, 1:, :self.joints_num*3] - motions[:, :-1, :self.joints_num*3])
            
            loss_bn = self.l1_criterion(self.calc_bone_lengths(pred_motion), self.calc_bone_lengths(motions))

            loss_geo = self.calc_loss_geo(pred_motion[..., self.joints_num*6: self.joints_num*6 + (self.joints_num-1)*6],
                                        motions[..., self.joints_num*6: self.joints_num*6 + (self.joints_num-1)*6])
            
            loss_fc = self.calc_foot_contact(self.normalizer.backward(motions), self.normalizer.backward(pred_motion))
            
            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, None, None
        elif self.dataset_name == 'interx':
            loss_rec = self.l1_criterion(pred_motion, motions)

            pred_motions_pos = self.kinematics.forward(pred_motion)
            motions_pos = self.kinematics.forward(motions)

            loss_explicit = self.l1_criterion(pred_motions_pos, motions_pos)

            loss_vel = self.l1_criterion(pred_motions_pos[:,1:,:,:] - pred_motions_pos[:,:-1,:,:],
                                            motions_pos[:,1:,:,:] - motions_pos[:,:-1,:,:])
            
            loss_bn = self.l1_criterion(self.calc_bone_lengths(pred_motions_pos[:,:,:22,:]), self.calc_bone_lengths(motions_pos[:,:,:22,:]))

            loss_geo = self.calc_loss_geo(pred_motion[:,:,:-1,:], motions[:,:,:-1,:])

            loss_fc = self.calc_foot_contact(motions_pos, pred_motions_pos)


            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, motions_pos, pred_motions_pos
    

class Inter_Losses:
    def __init__(self, recons_loss, joints_num, dataset_name, device):
        self.dataset_name = dataset_name
        if recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss('none')
        elif recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss(reduction='none')
        
        self.joints_num = joints_num
        if self.dataset_name == 'interhuman':
            self.normalizer = MotionNormalizerTorch(device)
        elif self.dataset_name == 'interx':
            self.normalizer = InterxNormalizerTorch()
            self.kinematics = InterxKinematics()
    
    def calc_dm_loss(self, motion_joints, pred_motion_joints, thresh_pred=1, thresh_tgt=0.1):

        pred_motion_joints1 = pred_motion_joints[..., 0:1, :, :].reshape(-1, self.joints_num, 3)
        pred_motion_joints2 = pred_motion_joints[..., 1:2, :, :].reshape(-1, self.joints_num, 3)
        motion_joints1 = motion_joints[..., 0:1, :, :].reshape(-1, self.joints_num, 3)
        motion_joints2 = motion_joints[..., 1:2, :, :].reshape(-1, self.joints_num, 3)
        
        pred_distance_matrix = torch.cdist(pred_motion_joints1.contiguous(), pred_motion_joints2)
        tgt_distance_matrix = torch.cdist(motion_joints1.contiguous(), motion_joints2)
        
        pred_distance_matrix = pred_distance_matrix.reshape(pred_distance_matrix.shape[0], -1).reshape(self.B, self.T, self.joints_num*self.joints_num) # B*T, njoints=22, 22 -> B, T, 484
        tgt_distance_matrix = tgt_distance_matrix.reshape(pred_distance_matrix.shape[0], -1).reshape(self.B, self.T, self.joints_num*self.joints_num)
        
        dm_mask = (pred_distance_matrix < thresh_pred).float()
        dm_tgt_mask = (tgt_distance_matrix < thresh_tgt).float()
        
        dm_loss = (self.l1_criterion(pred_distance_matrix, tgt_distance_matrix) * dm_mask).sum() / (dm_mask.sum() + 1.e-7)
        dm_tgt_loss = (self.l1_criterion(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix)) * dm_tgt_mask).sum()/ (dm_tgt_mask.sum() + 1.e-7)
        
        return dm_loss + dm_tgt_loss
    
    def calc_ro_loss(self, motion_joints, pred_motion_joints):

        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = pred_motion_joints[..., r_hip, :] - pred_motion_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = motion_joints[..., r_hip, :] - motion_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])

        ro_loss = self.l1_criterion(pred_relative_rot[..., [0, 2]],
                                    tgt_relative_rot[..., [0, 2]]).mean()

        return ro_loss
    
    def forward(self, motion1, motion2, pred_motion1, pred_motion2):
        B, T = motion1.shape[:2]
        self.B = B
        self.T = T
        
        if self.dataset_name == 'interhuman':
            motions = torch.cat([motion1.unsqueeze(-2), motion2.unsqueeze(-2)], dim=-2)
            motions = self.normalizer.backward(motions)
            
            pred_motion = torch.cat([pred_motion1.unsqueeze(-2), pred_motion2.unsqueeze(-2)], dim=-2)
            pred_motion = self.normalizer.backward(pred_motion)
            
            pred_motion_joints = pred_motion[..., :self.joints_num * 3].reshape(B, T, -1, self.joints_num, 3)
            motion_joints = motions[..., :self.joints_num * 3].reshape(B, T, -1, self.joints_num, 3)
        elif self.dataset_name == 'interx':
            motion_joints = torch.cat([motion1.unsqueeze(2), motion2.unsqueeze(2)], dim=2)
            pred_motion_joints = torch.cat([pred_motion1.unsqueeze(2), pred_motion2.unsqueeze(2)], dim=2)
        
        ro_loss = self.calc_ro_loss(motion_joints, pred_motion_joints)
        dm_loss = self.calc_dm_loss(motion_joints, pred_motion_joints)
        
        return dm_loss, ro_loss