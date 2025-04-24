import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
from models.vq.resnet import Resnet
from utils.paramUtil import t2m_edge_indices as edge_indices


class Encoder(nn.Module):
    def __init__(self,
                 opt,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        conv_layer = nn.Conv2d
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        self.gcn_layer1 = SAGEConv(input_emb_width, width, project=True)
        self.gcn_act1 = nn.ReLU()
        self.gcn_layer2 = SAGEConv(width, width, project=True)
        self.gcn_act2 = nn.ReLU()

        # blocks.append(conv_layer(input_emb_width, width, 3, 1, 1))
        # blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                conv_layer(input_dim, width, filter_t, stride_t, pad_t),
                Resnet(conv_dim, width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(conv_layer(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, edge_indices=edge_indices):
        B, D, J, T = x.shape
        x = x.permute(0,3,2,1)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.reshape(-1, x.shape[2])
        edge_indices = edge_indices.to(x.device)
        x = self.gcn_act1(self.gcn_layer1(x, edge_indices))
        x = self.gcn_act2(self.gcn_layer2(x, edge_indices))
        x = x.reshape(-1, J, x.shape[1])
        x = x.reshape(B, -1, x.shape[1], x.shape[2])
        x = x.permute(0,3,2,1)

        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 opt,
                 conv_dim=1,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        # blocks = []
        self.vq_dec_inter = opt.vq_dec_inter
        
        if conv_dim == 1:
            conv_layer = nn.Conv1d
        elif conv_dim == 2:
            conv_layer = nn.Conv2d
        else:
            raise ValueError("conv_dim should be 1 or 2")
        
        self.conv_pre = conv_layer(output_emb_width, width, 3, 1, 1)
        self.relu_pre = nn.ReLU()
        
        spatial_upsample = (2.2, 2)
        temporal_upsample = (2, 2)
        self.resnets= []
        for i in range(down_t):
            out_dim = width
            if conv_dim == 1:
                scale_factor = temporal_upsample[i]
            elif conv_dim == 2:
                scale_factor = (spatial_upsample[i], temporal_upsample[i])
            
            resnet = nn.Sequential(
                Resnet(conv_dim, width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                conv_layer(width, out_dim, 3, 1, 1)
            )
            self.resnets.append(resnet)
        self.resnets = nn.ModuleList(self.resnets)
        
        self.conv_post = conv_layer(width, width, 3, 1, 1)
        self.relu_post = nn.ReLU()
        # self.conv_final = conv_layer(width, input_emb_width, 3, 1, 1)
        
        self.model = nn.Sequential(*[*[self.conv_pre, self.relu_pre], 
                                     *self.resnets, 
                                     *[self.conv_post, self.relu_post]])

        self.gcn_layer1 = SAGEConv(width, width, project=True)
        self.gcn_act = nn.ReLU()
        self.gcn_layer2 = SAGEConv(width, input_emb_width, project=True)
        
        if self.vq_dec_inter:
            self.mixed_attn1 = MixedAttention(width, 4, 0.1)
            self.mixed_attn2 = MixedAttention(width, 4, 0.1)
            self.mixed_attn3 = MixedAttention(width, 4, 0.1)
           

    def sage_forward(self, x, edge_indices=edge_indices):
        B, D, J, T = x.shape
        x = x.permute(0,3,2,1)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.reshape(-1, x.shape[2])
        edge_indices = edge_indices.to(x.device)
        x = self.gcn_layer1(x, edge_indices)
        x = self.gcn_layer2(self.gcn_act(x), edge_indices)
        x = x.reshape(-1, J, x.shape[1])
        x = x.reshape(B, -1, x.shape[1], x.shape[2])
        x = x.permute(0,3,2,1)
        
        return x
        
    def forward(self, x, x2=None):
        if self.vq_dec_inter and x2 is not None:
            return self.forward_interaction(x,x2)
        
        x = self.model(x)
        x = self.sage_forward(x)
        
        return x
    
    def forward_interaction(self, x1, x2):
        x1 = self.conv_pre(x1)
        x1 = self.relu_pre(x1)
        x2 = self.conv_pre(x2)
        x2 = self.relu_pre(x2)
        
        x1_attn = self.mixed_attn1(x1, x2)
        x2_attn = self.mixed_attn1(x2, x1)
        
        x1 = self.resnets[0](x1_attn)
        x2 = self.resnets[0](x2_attn)
        x1_attn = self.mixed_attn2(x1, x2)
        x2_attn = self.mixed_attn2(x2, x1)
        
        x1 = self.resnets[1](x1_attn)
        x2 = self.resnets[1](x2_attn)
        x1_attn = self.mixed_attn3(x1, x2)
        x2_attn = self.mixed_attn3(x2, x1)
        
        x1 = self.conv_post(x1_attn)
        x1 = self.relu_post(x1)
        x2 = self.conv_post(x2_attn)
        x2 = self.relu_post(x2)
        
        x1 = self.sage_forward(x1)
        x2 = self.sage_forward(x2)  
        return x1, x2
    

class MixedAttention(nn.Module):

    def __init__(self, latent_dim,
                       num_heads,
                       dropout):
        super().__init__()
        self.num_heads = num_heads

        ## Attention
        self.self_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.cross_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_self = nn.Linear(latent_dim, latent_dim)
        self.value_self = nn.Linear(latent_dim, latent_dim)
        self.key_cross = nn.Linear(latent_dim, latent_dim)
        self.value_cross = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        ## FFN
        # self.ffn_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        # self.ffn = nn.Sequential(
        #     nn.Linear(latent_dim, dim_feedforward),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, latent_dim),
        # )
        
        # self.ffn_dropout = nn.Dropout(dropout)
        
        self.zero_conv = nn.Conv2d(latent_dim, latent_dim, 1, 1, 0)
        self.zero_conv.weight.data.zero_()
        self.zero_conv.bias.data.zero_()
    
    def forward(self, x, x2):
        """
        x: B, D, J, T
        """
        x_orig = x
        orig_shape = x.shape
        
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # B,T,D
        x2 = x2.reshape(x2.shape[0], x2.shape[1], -1).permute(0, 2, 1) # B,T,D
        
        B, T, D = x.shape
        N = T*2
        H = self.num_heads
        
        # B, T, D
        query = self.query(self.self_norm(x)).view(B, T, H, -1)
        # B, N, D
        key = torch.cat((
            self.key_cross(self.cross_norm(x2)),
            self.key_self(self.self_norm(x))
        ), dim=1).view(B, N, H, -1)
        
        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        attention = F.softmax(attention, dim=2)
        
        value = torch.cat((
            self.value_cross(self.cross_norm(x2)),
            self.value_self(self.self_norm(x)),
        ), dim=1).view(B, N, H, -1)
        
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        
        # y = y + self.ffn_dropout(self.ffn(self.ffn_norm(y))) 
        
        y = y.permute(0, 2, 1).reshape(orig_shape)
        y = x_orig + self.zero_conv(y)
        return y
    

if __name__ == "__main__":
    import time
    opt = {"vq_dec_inter": True} 
    opt = type("opt", (object,), opt)()
    dec = Decoder(opt,2, 12, 512, 2, 2, 512, 2,3, activation='relu', norm=None)
    input = torch.randn((4, 512, 5, 16))
    
    start  = time.time()
    # output = dec(input)
    # print(f"Output: {output.shape}")
    
    input2 = torch.randn((4, 512, 5, 16))
    output, output2 = dec(input, x2=input2)
    print(f"Output: {output.shape, output2.shape}")
    
    print(f'Latency: {time.time() - start} sec')
    
    params = sum(param.numel() for param in dec.parameters())
    print('Total parameters: {}M'.format(params/1000_000))