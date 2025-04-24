import random

import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
from data.utils import fid_l, fid_r
from utils.paramUtil import t2m_edge_list

class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_dim=12,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=2,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.joints_num = args.joints_num
        self.dataset_name = args.dataset_name

        if self.dataset_name == "interhuman":
            filter_s = None
            stride_s = None
            spatial_upsample = (2.2, 2)
            gcn=True

        elif self.dataset_name == "interx":
            filter_s = 6
            stride_s = 3
            spatial_upsample = (3.5, 3.3)
            gcn=False

        self.encoder = Encoder(args, input_dim, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, 
                               filter_s=filter_s, stride_s=stride_s, gcn=gcn)
        self.decoder = Decoder(args, input_dim, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, 
                               spatial_upsample=spatial_upsample, gcn=gcn)
        
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0
        }
        
        self.quantizer = ResidualVQ(**rvqvae_config)
    
    def preprocess(self, x):
        if self.dataset_name == "interhuman":
            pos = x[..., :self.joints_num*3].reshape([x.shape[0], x.shape[1], self.joints_num, 3])
            vel = x[..., self.joints_num*3 : self.joints_num*3*2].reshape([x.shape[0], x.shape[1], self.joints_num, 3])
            
            rot = x[..., self.joints_num*3*2 : self.joints_num*3*2 + (self.joints_num-1)*6].reshape([x.shape[0], x.shape[1], self.joints_num-1, 6])
            rot = torch.cat([torch.zeros(rot.shape[0], rot.shape[1], 1, 6).to(x.device), rot], dim=2)
            
            joints = torch.cat([pos, vel, rot], dim=-1)
        else:
            joints = x
        joints = joints.permute(0, 3, 2, 1).float() # B, D=12, J=22, T 
 
        return joints

    def postprocess(self, x):
        x = x.permute(0, 3, 2, 1).float()

        if self.dataset_name == "interhuman":
            pos = x[:,:,:,:3].reshape([x.shape[0], x.shape[1], -1])
            vel = x[:,:,:,3:6].reshape([x.shape[0], x.shape[1], -1])
            rot = x[:,:,1:,6:6+6].reshape([x.shape[0], x.shape[1], -1])
            fc = torch.zeros((x.shape[0], x.shape[1], 4)).to(x.device)
            
            x = torch.cat([pos, vel, rot, fc], dim=-1)
        return x

    def encode(self, x):
        # N, T, _, _ = x.shape

        x_in = self.preprocess(x) # B, D=12, J=22, T || B, J=22xD=12, T
        x_encoder = self.encoder(x_in) # B, D=512, 5, T/2 || B, J=7xD=512, T//4
        
        
        encoder_shape = x_encoder.shape
        x_encoder = x_encoder if len(encoder_shape) == 3 else x_encoder.reshape(encoder_shape[0], encoder_shape[1], -1)
        
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True) # B,375,1; 1,B,512,375
        return code_idx, all_codes

    def forward(self, x, verbose=False):
        
        # Encode
        x_in = self.preprocess(x) # B, D=12, J=22, T || B, J=22xD=12, T
        if verbose: print(f'preprocess: {x_in.shape}')
        
        x_encoder = self.encoder(x_in) # B, D=512, 5, T/2 || B, J=7xD=512, T//4
        if verbose: print(f'encoder: {x_encoder.shape}')
        
        ## quantization
        encoder_shape = x_encoder.shape
        x_encoder = x_encoder if len(encoder_shape) == 3 else x_encoder.reshape(encoder_shape[0], encoder_shape[1], -1)
        if verbose: print(f'reshape: {x_encoder.shape}')
            
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        if verbose: print(f'quantizer: {x_quantized.shape}')
        x_quantized = x_quantized.reshape(encoder_shape)

        
        ## decoder
        x_out = self.decoder(x_quantized) # B, D=12, J=22, T || B, J=22xD=12, T
        if verbose: print(f'decoder: {x_out.shape}')
        
        x_out = self.postprocess(x_out) # B,T,D=262
        if verbose: print(f'postprocess: {x_out.shape}')
        # if verbose: exit()
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x, soft_lookup=False):
        
        
        if not soft_lookup:
            x_d = self.quantizer.get_codes_from_indices(x)
            # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        else:
            x_d = self.quantizer.get_soft_codes_from_probs(x)
        x = x_d.sum(dim=0).permute(0, 2, 1) # B,T,D=512 -> B,D,T
        
        x = x.reshape(x.shape[0], x.shape[1], 5, x.shape[2]//5) # B,D,T -> B,D,5,T/5
        
        # decoder
        x_out = self.decoder(x)
        x_out = self.postprocess(x_out)
        return x_out