# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
    




def visualize_2d(x, batched=False, renormalize=False):
    # x: [B, 3, H, W] or [B, 1, H, W] or [B, H, W]
    

    if batched:
        x = x[0]
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    if len(x.shape) == 3:
        if x.shape[0] == 3:
            x = x.transpose(1, 2, 0) # to channel last
        elif x.shape[0] == 1:
            x = x[0] # to grey
        
    print(f'[VISUALIZER] {x.shape}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    if len(x.shape) == 3:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.matshow(x)
    plt.show()
    
class RMSEMeter:
    def __init__(self, report_per_image = False):
        self.report_per_image = report_per_image
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, data, preds, truths, eval=False):
        preds, truths = self.prepare_inputs(preds, truths) # [B, 1, H, W]

        if eval:
            B, C, H, W = data['image'].shape
            preds = preds.reshape(B, 1, H, W)
            truths = truths.reshape(B, 1, H, W)

            # clip borders (reference: https://github.com/cvlab-yonsei/dkn/issues/1)
            preds = preds[:, :, 6:-6, 6:-6]
            truths = truths[:, :, 6:-6, 6:-6]
            
        # rmse
        rmse = np.sqrt(np.mean(np.power(preds - truths, 2)))
        
        # to report per-image rmse 
        if self.report_per_image:
            print('rmse = ', rmse)

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "rmse"), self.measure(), global_step)

    def report(self):
        return f'RMSE = {self.measure():.6f}'