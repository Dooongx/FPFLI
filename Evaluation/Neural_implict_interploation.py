# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import torch
import numpy as np
from NIII_model import NIII
from utils import timer


class neural_implict_interploation():
    def __init__(self, data,ratio):
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NIII().to(self.device)
        self.checkpoint = rf'./model_parameters/NIII_parameter_L{ratio}.pth.tar'
        self.checkpoint_dict = torch.load(self.checkpoint, map_location= self.device)
        self.model.load_state_dict(self.checkpoint_dict['model'])
        self.model.eval()
            
    
    def prepare_data(self, data):
            if isinstance(data, list):
                for i, v in enumerate(data):
                    if isinstance(v, np.ndarray):
                        data[i] = torch.from_numpy(v).to(self.device)
                    if torch.is_tensor(v):
                        data[i] = v.to(self.device)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v).to(self.device)
                    if torch.is_tensor(v):
                        data[k] = v.to(self.device)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(self.device)
            else: # is_tensor
                data = data.to(self.device)

            return data    
        
    def inference(self):
        
        with torch.no_grad():
      
            data = self.prepare_data(self.data)
            
            B, C, H, W = data['hr_int'].shape
            preds = self.model(data)
            preds = preds * data['lr_tau_max']
            preds = preds.reshape(B, 1, H, W)
            preds = preds.detach().cpu().numpy() # [B, 1, H, W]
            
        return preds
            
@timer
def NIII_process(nii_data,ratio):
    result = neural_implict_interploation(nii_data, ratio).inference()
    return result    