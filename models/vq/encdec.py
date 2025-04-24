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
                 norm=None,
                 filter_s=None,
                 stride_s=None,
                 gcn=False):
        super().__init__()

        conv_layer = nn.Conv2d
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        filter_s = filter_t if filter_s is None else filter_s
        stride_s = stride_t if stride_s is None else stride_s

        self.gcn = gcn
        if self.gcn:
            self.gcn_layer1 = SAGEConv(input_emb_width, width, project=True)
            self.gcn_act1 = nn.ReLU()
            self.gcn_layer2 = SAGEConv(width, width, project=True)
            self.gcn_act2 = nn.ReLU()
        else:
            blocks.append(conv_layer(input_emb_width, width, 3, 1, 1))
            blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            resnet = nn.Sequential(
                conv_layer(input_dim, width, (filter_s,filter_t), (stride_s,stride_t), pad_t),
                Resnet(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(resnet)
        blocks.append(conv_layer(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def gcn_forward(self, x, edge_indices=edge_indices):
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
        return x
    
    def forward(self, x):
        if self.gcn:
            x = self.gcn_forward(x)
        
        return self.model(x)


class Decoder(nn.Module):
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
                 norm=None,
                 spatial_upsample=None,
                 gcn=False):
        super().__init__()
        
        conv_layer = nn.Conv2d

        blocks = []
        
        blocks.append(conv_layer(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        temporal_upsample = (2, 2)
        spatial_upsample = temporal_upsample if spatial_upsample is None else spatial_upsample
        
        for i in range(down_t):
            out_dim = width
            scale_factor = (spatial_upsample[i], temporal_upsample[i])
            
            resnet = nn.Sequential(
                Resnet(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                conv_layer(width, out_dim, 3, 1, 1)
            )
            blocks.append(resnet)

        blocks.append(conv_layer(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        self.gcn = gcn
        if not self.gcn:
            blocks.append(conv_layer(width, input_emb_width, 3, 1, 1))
        else:
            self.gcn_layer1 = SAGEConv(width, width, project=True)
            self.gcn_act = nn.ReLU()
            self.gcn_layer2 = SAGEConv(width, input_emb_width, project=True)

        self.model = nn.Sequential(*blocks)
            
           
    def gcn_forward(self, x, edge_indices=edge_indices):
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
    
    def forward(self, x):
        x = self.model(x)
        
        if self.gcn:
            x = self.gcn_forward(x)
        
        return x
    