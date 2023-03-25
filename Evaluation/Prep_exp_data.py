# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:31:21 2021

@author: syb18175
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


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

def downSizeImg_2D(Img_HR,k):
    # The size of Img_HR is [1,H,W]
    Img_HR = Img_HR.unsqueeze(0)
    f = torch.ones([1,1,k,k]).type(torch.float)
    Img_LR = F.conv2d(Img_HR, f, stride=[k,k])
    return Img_LR.squeeze()

class prep_exp_data():
    def __init__(self, HR_Int, LR_tau, scale = 8):
        
        
        self.HR_Int = HR_Int
        self.LR_tau = LR_tau
        self.scale = scale
        
    def prep_data(self):
        
        h, w = self.HR_Int.shape
        # normalize
        LR_tau_max = self.LR_tau.max()+1e-8
        LR_tau = self.LR_tau / LR_tau_max
        HR_Int_max = self.HR_Int.max()+1e-8
        HR_Int = self.HR_Int / HR_Int_max
        LR_tau_upscale = np.array(Image.fromarray(LR_tau).resize((w, h), Image.BICUBIC))
        # to tensor
        HR_Int = torch.from_numpy(HR_Int).unsqueeze(0).float()
        LR_Int = downSizeImg_2D(HR_Int,self.scale).unsqueeze(0)
        LR_tau = torch.from_numpy(LR_tau).unsqueeze(0).float()
        LR_tau_upscale = torch.from_numpy(LR_tau_upscale).unsqueeze(0).float()
        HR_Int = HR_Int.contiguous()
        LR_Int = LR_Int.contiguous()
        LR_tau = LR_tau.contiguous()
        LR_tau_upscale = LR_tau_upscale.contiguous()
        
        hr_coord, _ = to_pixel_samples(HR_Int)
        lr_pixel = LR_tau_upscale.view(-1, 1)

                 
        return {
            'hr_int': HR_Int.unsqueeze(0),
            'lr_int': LR_Int.unsqueeze(0),
            'lr_tau': LR_tau.unsqueeze(0),
            'lr_pixel': lr_pixel.unsqueeze(0),
            'hr_coord': hr_coord.unsqueeze(0),
            'lr_tau_max': LR_tau_max,
        }   




