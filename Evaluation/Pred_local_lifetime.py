# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from tau_net import tauNet
from utils import timer


model = tauNet()


class Pred_local_lifetime():
    def __init__(self, img, irf, scaling_ratio = 4, h = 0.039,thres = 50, file_path = r'./model_parameters/LLE_parameter.pth'):
        self.img = torch.tensor(img)
        self.scaling_ratio = scaling_ratio
        self.multiplier = h*100
        self.thres = thres
        self.irf = irf
        parameters = torch.load(file_path)
        model.load_state_dict(parameters)
        model.eval()
    
    def downSizeImg(self):
        self.img = self.img.unsqueeze(0).unsqueeze(0)
        f = torch.ones([1, 1, self.scaling_ratio, self.scaling_ratio, 1]).type(torch.float)
        lr_img = F.conv3d(self.img, f, stride=[self.scaling_ratio, self.scaling_ratio, 1])
        return lr_img.squeeze(0).squeeze(0)
    
    def eval_local_tau(self):
        lr_img = self.downSizeImg()
        Size_x, Size_y, Length = lr_img.shape[0], lr_img.shape[1], lr_img.shape[2]
        lr_img = lr_img.reshape(Size_x*Size_y, Length)

        ytem = np.zeros(lr_img.shape)
        irf_np = self.irf
        
        for i in range(lr_img.shape[0]):
            ytem[i, :] = lr_img[i, :]
            ytem[i, :] = ytem[i, :].reshape(1, 1, ytem.shape[1])

            ytem[i, :] = ytem[i, :]/(ytem[i, :].max()+1e-8)

        ytem = torch.from_numpy(ytem)
        irf = torch.from_numpy(irf_np)
        irf = torch.broadcast_to(irf, lr_img.shape)
        irf = irf.unsqueeze(1)
        ytem = ytem.unsqueeze(1)

        tau = model(ytem.float(),irf.float()).detach().numpy()
        
        tau = tau*self.multiplier
        
        for i in range(lr_img.shape[0]):
            if lr_img[i, :].sum() <= self.thres:
                tau[i] = 0

        tau = tau.reshape([Size_x, Size_y])
        return tau
        
@timer
def LLE(hist2d_mask,irf,r = 8, bin_w = 0.039, p_path = r'./model_parameters/LLE_parameter.pth'):
    
    lr_tau = Pred_local_lifetime(hist2d_mask,irf,scaling_ratio = r, h = bin_w, file_path = p_path).eval_local_tau()
    return lr_tau        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    

