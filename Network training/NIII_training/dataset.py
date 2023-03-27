# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import tqdm


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

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel

def DownSizeImg_2D(Img_HR,k):
    # The size of Img_HR is [1,H,W]
    Img_HR = Img_HR.unsqueeze(0)
    f = torch.ones([1,1,k,k]).type(torch.float)
    Img_LR = F.conv2d(Img_HR, f, stride=[k,k])
    return Img_LR.squeeze()

class Mydataset(Dataset):
    def __init__(self, data_root= '', scale = 8, pre_upsample = False):
        super().__init__()
        
        imgset = []
        for (root, _, names) in os.walk(data_root):
            for name in names:
                fullpath = os.path.join(root,name)
                imgset.append(fullpath)
        
        self.imgset = imgset
        self.scale = scale
        self.pre_upsample = pre_upsample
      
        
    def __len__(self):
        return len(self.imgset)
    

    def __getitem__(self, idx):
        imgs = np.load(self.imgset[idx]) 
        HR_tau = imgs['GT_tau']
        HR_Int = imgs['Int_HR']
        LR_tau = imgs['tau_LR']
        
        h, w = HR_Int.shape
        # normalize
        HR_tau_max = HR_tau.max()+1e-8
        LR_tau_max = LR_tau.max()+1e-8
        HR_tau = HR_tau  / HR_tau_max 
        LR_tau = LR_tau / LR_tau_max
        
        HR_Int_max = HR_Int.max()+1e-8
        HR_Int = HR_Int / HR_Int_max
        
  
        # follow DKN, use bicubic upsampling of PIL
        LR_tau_upscale = np.array(Image.fromarray(LR_tau).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            LR_tau = LR_tau_upscale
            
        # to tensor
        HR_Int = torch.from_numpy(HR_Int).unsqueeze(0).float()
        LR_Int = DownSizeImg_2D(HR_Int,self.scale).unsqueeze(0)
        HR_tau = torch.from_numpy(HR_tau).unsqueeze(0).float()
        LR_tau = torch.from_numpy(LR_tau).unsqueeze(0).float()
        LR_tau_upscale = torch.from_numpy(LR_tau_upscale).unsqueeze(0).float()

        HR_Int = HR_Int.contiguous()
        LR_Int = LR_Int.contiguous()
        HR_tau = HR_tau.contiguous()
        LR_tau = LR_tau.contiguous()
        LR_tau_upscale = LR_tau_upscale.contiguous()

        # to pixel
        
            
        hr_coord, hr_pixel = to_pixel_samples(HR_tau)

        lr_pixel = LR_tau_upscale.view(-1, 1)

        
    
        return {
            'hr_int': HR_Int,
            'lr_int': LR_Int,
            'lr_tau': LR_tau,
            'lr_pixel': lr_pixel,
            'hr_pixel': hr_pixel,
            'hr_coord': hr_coord,
            'hr_tau': HR_tau,
            'hr_tau_max': HR_tau_max,
            'lr_tau_max': LR_tau_max,
            'idx': idx,
        }   

        


