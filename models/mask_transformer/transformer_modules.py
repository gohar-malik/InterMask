import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import numpy as np
import copy

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
    
class AdaLNModulation(nn.Module):
    def __init__(self, d_model, nchunks=6):
        super(AdaLNModulation, self).__init__()
        self.nchunks = nchunks

        self.model = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, nchunks*d_model, bias = True),
        )
    
    def forward(self, cond):
        return self.model(cond).chunk(self.nchunks, dim=1)

class SpaTempAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout,
                 spa_dim=5):
        super(SpaTempAttnLayer, self).__init__()

        self.spa_dim = spa_dim

        self.spa_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.spatial_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.spa_dropout = nn.Dropout(dropout)

        self.temp_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temp_dropout = nn.Dropout(dropout)

    def forward(self, src, 
                shift_spa, scale_spa, gate_spa, 
                shift_temp, scale_temp, gate_temp, 
                src_key_padding_mask=None, src2=None):
        
        # Reshape input for spatial and temporal attention
        spa_src, temp_src, spa_src2, temp_src2, pad, \
            shift_spa, scale_spa, gate_spa, shift_temp, scale_temp, gate_temp = self.create_spa_temp_attn_inputs(
                                                                                    src, src2, src_key_padding_mask,
                                                                                    shift_spa, scale_spa, gate_spa,
                                                                                    shift_temp, scale_temp, gate_temp
                                                                                    )
        
        # Spatial multihead self-attention block
        spa_src_mod = modulate(self.spa_norm(spa_src), shift_spa, scale_spa)
        if src2 is None:
            spa_src = spa_src + gate_spa.unsqueeze(0) * self.spa_dropout(self.spatial_attention(spa_src_mod, spa_src_mod, spa_src_mod,
                                                                                                key_padding_mask=None,
                                                                                                need_weights=False)[0])
        else:
            spa_src = spa_src + gate_spa.unsqueeze(0) * self.spa_dropout(self.spatial_attention(spa_src_mod, spa_src2, spa_src2,
                                                                                                key_padding_mask=None,
                                                                                                need_weights=False)[0])
        

        # Temporal multihead self-attention block
        temp_src_mod = modulate(self.temp_norm(temp_src), shift_temp, scale_temp)
        if src2 is None:
            temp_src = temp_src + gate_temp.unsqueeze(0) * self.temp_dropout(self.temporal_attention(temp_src_mod, temp_src_mod, temp_src_mod,
                                                                                                    key_padding_mask=pad,
                                                                                                    need_weights=False)[0])
        else:
            temp_src = temp_src + gate_temp.unsqueeze(0) * self.temp_dropout(self.temporal_attention(temp_src_mod, temp_src2, temp_src2,
                                                                                                    key_padding_mask=pad,
                                                                                                    need_weights=False)[0])

        spa_src, temp_src = self.reshape_spa_temp_attn_outputs(spa_src, temp_src) 
        spa_temp_src = spa_src + temp_src

        return spa_temp_src
    
    def create_spa_temp_attn_inputs(self, src, src2, padding_mask, 
                                    shift_spa, scale_spa, gate_spa,
                                    shift_temp, scale_temp, gate_temp):
        n_tokens, B, d = src.shape
        self.B, self.d = B, d

        src_wo_text_3d = src.reshape(self.spa_dim, -1, B, d)
        spa_src = src_wo_text_3d.reshape(self.spa_dim, -1, d)
        temp_src = src_wo_text_3d.permute(1,0,2,3)
        temp_src = temp_src.reshape(-1, self.spa_dim*B, d)
        
        if src2 is not None:
            src2_wo_text_3d = src2.reshape(self.spa_dim, -1, B, d)
            spa_src2 = src2_wo_text_3d.reshape(self.spa_dim, -1, d)
            temp_src2 = src2_wo_text_3d.permute(1,0,2,3)
            temp_src2 = temp_src2.reshape(-1, self.spa_dim*B, d)
        else:
            spa_src2 = None
            temp_src2 = None
            
        shift_spa = shift_spa.repeat(spa_src.shape[1]//B, 1)
        scale_spa = scale_spa.repeat(spa_src.shape[1]//B, 1)
        gate_spa = gate_spa.repeat(spa_src.shape[1]//B, 1)


        # temp_src = src_wo_text_3d.permute(1,0,2,3)
        # temp_src = temp_src.reshape(-1, self.spa_dim*B, d)
        
        # temp_src = src_wo_text_3d.permute(1,0,2,3)
        # temp_src = temp_src.reshape(-1, self.spa_dim*B, d)
        shift_temp = shift_temp.repeat(self.spa_dim, 1)
        scale_temp = scale_temp.repeat(self.spa_dim, 1)
        gate_temp = gate_temp.repeat(self.spa_dim, 1)

        pad = padding_mask.permute(1,0)
        pad = pad.reshape(self.spa_dim, -1, B)
        pad = pad.permute(1,0,2)
        pad = pad.reshape(-1, self.spa_dim*B)
        pad = pad.permute(1,0)

        return spa_src, temp_src, spa_src2, temp_src2, pad, shift_spa, scale_spa, gate_spa, shift_temp, scale_temp, gate_temp
    
    def reshape_spa_temp_attn_outputs(self, spa_out, temp_out):
        temp_out = temp_out.reshape(-1, self.spa_dim, self.B, self.d)
        temp_out = temp_out.permute(1,0,2,3)
        temp_out = temp_out.reshape(-1, self.B, self.d)

        spa_out = spa_out.reshape(self.spa_dim, -1, self.B,  self.d)
        spa_out = spa_out.reshape(-1, self.B, self.d)

        return spa_out, temp_out
    
class LocalInteractionAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, spa_dim=5, window_factor=3):
        super(LocalInteractionAttnLayer, self).__init__()

        self.inter_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.inter_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.interaction_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inter_dropout = nn.Dropout(dropout)

        self.spa_dim = spa_dim
        self.window_factor = window_factor
    
    def create_local_attn_mask(self, src):
        timesteps = src.shape[0] // self.spa_dim
        window = timesteps // self.window_factor
        
        ones = torch.ones((timesteps,timesteps), dtype=bool, device=src.device)
        upper_tri = torch.triu(ones, diagonal = window//2+1)
        lower_tri = torch.tril(ones, diagonal = -window//2)
        mask = (upper_tri + lower_tri)

        mask = mask.repeat(self.spa_dim, self.spa_dim)

        return mask

    def forward(self, src, src2, shift, scale, shift2, scale2, gate, src_key_padding_mask=None):
        # local_attn_mask = self.create_local_attn_mask(src)
        
        src_mod = modulate(self.inter_norm(src), shift, scale)
        src_mod2 = modulate(self.inter_norm2(src2), shift2, scale2)
        src = src + gate.unsqueeze(0) * self.inter_dropout(self.interaction_attention(src_mod, src_mod2, src_mod2,
                                                                                attn_mask = None,
                                                                                key_padding_mask=src_key_padding_mask,
                                                                                need_weights=False)[0])

        return src
    
class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(SelfAttnLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, shift, scale, gate, src_key_padding_mask=None):
        
        src_mod = modulate(self.norm(src), shift, scale)
        src = src + gate.unsqueeze(0) * self.dropout(self.attention(src_mod, src_mod, src_mod,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=False)[0])

        return src

class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FFN, self).__init__()

        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        
        
        self.model = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, src, shift, scale, gate):
        src = src + gate.unsqueeze(0) * self.ffn_dropout(self.model(modulate(self.ffn_norm(src), shift, scale)))
        return src
    
class InterMTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, nbp):
        super(InterMTransformerBlock, self).__init__()
        self.spa_dim = nbp
        
        self.adaLN_mod_combined = AdaLNModulation(d_model, 6)
        self.self_attn = SelfAttnLayer(d_model, nhead, dropout)
        self.ffn_combined = FFN(d_model, dim_feedforward, dropout)

        self.adaLN_mod_split = AdaLNModulation(d_model, 14)
        self.spa_temp_attn = SpaTempAttnLayer(d_model, nhead, dropout, spa_dim=self.spa_dim)
        # self.spa_temp_attn_cross = SpaTempAttnLayer(d_model, nhead, dropout, spa_dim=self.spa_dim)
        self.local_inter_attn = LocalInteractionAttnLayer(d_model, nhead, dropout, spa_dim=self.spa_dim)
        self.ffn_split = FFN(d_model, dim_feedforward, dropout)

        
        
        

    def forward(self, src, cond, src_key_padding_mask=None):
        N, B, d = src.shape
        
        # AdaLN modulation
        shift_self, scale_self, gate_self, \
            shift_ffn_c, scale_ffn_c, gate_ffn_c = self.adaLN_mod_combined(cond)

        # Self-Attention
        src = self.self_attn(src, 
                            shift_self, scale_self, gate_self,
                            src_key_padding_mask=src_key_padding_mask)

        # FFN
        src = self.ffn_combined(src, shift_ffn_c, scale_ffn_c, gate_ffn_c)

        ################### Split ###################
        src1, sep, src2  = src.split([N//2, 1, N//2])
        pad,_,_ = src_key_padding_mask.split([N//2, 1, N//2], dim=-1)

        # AdaLN modulation
        # shift_cross_spa, scale_cross_spa, gate_cross_spa, shift_cross_temp, scale_cross_temp, gate_cross_temp, \
        shift_spa, scale_spa, gate_spa, shift_temp, scale_temp, gate_temp, \
            shift_cross, scale_cross, shift_cross2, scale_cross2, gate_cross, \
                shift_ffn_s, scale_ffn_s, gate_ffn_s = self.adaLN_mod_split(cond) 

        # Spatial-Temporal Attention
        # src1 = self.spa_temp_attn(src1, 
        src1_spa_temp = self.spa_temp_attn(src1, 
                                    shift_spa, scale_spa, gate_spa, 
                                    shift_temp, scale_temp, gate_temp, 
                                    src_key_padding_mask=pad)
        # src2 = self.spa_temp_attn(src2,
        src2_spa_temp = self.spa_temp_attn(src2,
                                    shift_spa, scale_spa, gate_spa, 
                                    shift_temp, scale_temp, gate_temp, 
                                    src_key_padding_mask=pad)

        # Spatial-Temporal Attention Cross
        # src1_cross = self.spa_temp_attn_cross(src1_spa_temp,
        #                             shift_cross_spa, scale_cross_spa, gate_cross_spa,
        #                             shift_cross_temp, scale_cross_temp, gate_cross_temp,
        #                             src_key_padding_mask=pad, src2=src2)
        # src2_cross = self.spa_temp_attn_cross(src2_spa_temp,
        #                             shift_cross_spa, scale_cross_spa, gate_cross_spa,
        #                             shift_cross_temp, scale_cross_temp, gate_cross_temp,
        #                             src_key_padding_mask=pad, src2=src1)

        # Local Interaction Attention
        src1_cross = self.local_inter_attn(src1_spa_temp, src2,
                                    shift_cross, scale_cross, shift_cross2, scale_cross2, gate_cross,
                                    src_key_padding_mask=pad)
        # src2 = self.local_inter_attn(src2, src1,
        src2_cross = self.local_inter_attn(src2_spa_temp, src1,
                                    shift_cross, scale_cross, shift_cross2, scale_cross2, gate_cross,
                                    src_key_padding_mask=pad)

        # FFN
        src1 = self.ffn_split(src1_cross, shift_ffn_s, scale_ffn_s, gate_ffn_s)
        src2 = self.ffn_split(src2_cross, shift_ffn_s, scale_ffn_s, gate_ffn_s)

        ################# Concat #################
        src = torch.cat([src1, sep, src2], dim=0)

        return src
    

    
class InterMTransformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, nbp):
        super(InterMTransformer, self).__init__()

        block = InterMTransformerBlock(d_model=d_model,
                                nhead=nhead,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout,
                                nbp=nbp)
             
        module_list = []
        
        for _ in range(num_layers):
            module_list.append(copy.deepcopy(block))

        self.blocks =  ModuleList(module_list)
        
    def forward(self, src, cond, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape (751, B, d_model)
            src_key_padding_mask: Tensor, shape (B, 751) 

        Additional token is for the condition
        """

        for block in self.blocks:
            src = block(src, cond, src_key_padding_mask=src_key_padding_mask)

        src = torch.cat([src[:src.shape[0]//2, : ,:],
                        src[(src.shape[0]//2)+1:, : ,:]])
                
        return src