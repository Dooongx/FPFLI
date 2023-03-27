# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""


import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
class convlayer(nn.Module):
    def __init__(self,dim, kernel_size):
        super().__init__()
        self.res = nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(dim)
                )))
        self.ponitconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(dim)
    def forward(self,x):
        y = self.res(x)
        y = self.ponitconv(y)
        y = self.activate(y)
        y = self.bn(y)
        
        return y
        
class ConvMixer_tau(nn.Module):
    def __init__(self,dim = 16, depth = 8, kernel_size = 9, patch_size = 8):
        super().__init__()
        self.embd_decay = nn.Conv1d(1,dim,kernel_size=patch_size,stride=patch_size) 
        self.embd_irf = nn.Conv1d(1,dim,kernel_size=patch_size,stride=patch_size) 
        # self.activate = nn.GELU()
        # self.bn = nn.BatchNorm1d(dim*2)
        self.conv_blocks= nn.ModuleList()
        for _ in range(depth):
            self.conv_blocks.append(convlayer(dim*2, kernel_size))
        self.flatten = nn.Flatten()
        self.ponitconv1 = nn.Conv1d(int(dim*2*256/patch_size), dim, kernel_size=1)
        self.ponitconv2 = nn.Conv1d(dim, int(dim//2), kernel_size=1)
        self.ponitconv3 = nn.Conv1d(int(dim//2), 1, kernel_size=1)
        
        
        nn.Linear(int(dim*2*256/patch_size),1)
        
    def forward(self,decay,irf):
        decay = self.embd_decay(decay)
        irf = self.embd_irf(irf)
        y = torch.cat([decay, irf], dim=1)
        # y = self.activate(y)
        # y = self.bn(y)
        for layer in self.conv_blocks:
            y = layer(y)
        b,c,l = y.shape
        y = y.view(b,c*l,1)
        y =  self.ponitconv1(y)
        y =  self.ponitconv2(y)
        y =  self.ponitconv3(y)
        out = y.view(-1)
        
        return out
        
        
    
# =============================================================================
# #TEST
# from torchsummary import summary
# model = ConvMixer_tau(16,8,9,8)
# summary(model, [(1, 256),(1,256)],device='cpu')
# =============================================================================






