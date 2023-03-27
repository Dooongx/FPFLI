# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g. 
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret


class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size, 
                                            bias=True,padding=(kernel_size//2)),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(n_feats, n_feats, kernel_size, 
                                            bias=True,padding=(kernel_size//2)))
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        
        return res
    
class EDSR_backbone(nn.Module):
    def __init__(self, n_resblocks=16,n_feats=128,kernel_size = 3,n_colors=1):
        super(EDSR_backbone, self).__init__()
        
        self.n_resblocks = n_resblocks 
        self.n_feats = n_feats 
        self.kernel_size  = kernel_size 
        self.n_colors = n_colors
        # define head module
        self.head = nn.Conv2d(self.n_colors, self.n_feats, self.kernel_size, 
                                            bias=True,padding=(self.kernel_size//2))
        # define body module
        body = []
        for _ in range(self.n_resblocks):
            body.append(ResBlock(self.n_feats,self.kernel_size))
        
        body.append(nn.Conv2d(self.n_feats, self.n_feats, self.kernel_size, 
                                            bias=True,padding=(self.kernel_size//2)))
        self.body = nn.Sequential(*body)
       

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return res

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding= 3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning
    
class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2))


        self.output = nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.output(x)
        return x    

    
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
    
class NIII(nn.Module):

    def __init__(self, feat_dim=128, guide_dim=128, mlp_dim=[1024,512,256,128]):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        self.int_encoder = RDN(num_channels=1, num_features = self.guide_dim, growth_rate = self.guide_dim, num_blocks = 4, num_layers = 4)
        self.tau_encoder = RDN(num_channels=1, num_features = self.guide_dim, growth_rate = self.guide_dim, num_blocks = 4, num_layers = 4)

        imnet_in_dim = self.feat_dim + self.guide_dim * 2 + 2
        
        self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=self.mlp_dim)
        
    def query(self, feat, coord, hr_guide, lr_guide):
        
        b, c, h, w = feat.shape 
        B, N, _ = coord.shape
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        rx = 1 / h
        ry = 1 / w
        preds = []
        k = 0
       
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += (vx) * rx 
                coord_[:, :, 1] += (vy) * ry
                k += 1
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) 
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) 
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) 
                q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)
                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1) 
                preds.append(pred)

        preds = torch.stack(preds, dim=-1) 
        weight = F.softmax(preds[:,:,1,:], dim=-1)
        ret = (preds[:,:,0,:] * weight).sum(-1, keepdim=True)

        return ret

    def forward(self, data):
        HR_Int, LR_tau, coord, res, LR_Int = data['hr_int'], data['lr_tau'], data['hr_coord'], data['lr_pixel'], data['lr_int']
        hr_guide = self.int_encoder(HR_Int) 
        lr_guide = self.int_encoder(LR_Int)
        feat = self.tau_encoder(LR_tau)
        res = res + self.query(feat, coord, hr_guide, lr_guide)
        res = F.relu(res)
        return res    
