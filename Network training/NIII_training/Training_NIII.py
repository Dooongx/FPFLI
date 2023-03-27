# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""
import sys, os
import glob
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
from dataset import Mydataset
from NIII_model import NIII
from utils import RMSEMeter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import scipy.io as io
import warnings
warnings.filterwarnings('ignore')

#Hyperparameter setting

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='FPFLI_MaxNorm')
parser.add_argument('--mixed_precision', type=bool, default=True, help = 'use mixed precision to train the network')
parser.add_argument('--lr',  default=0.4e-4, type=float, help='learning rate')
parser.add_argument('--lr_step',  default=40, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=0.8, type=float, help='learning rate decay gamma')
parser.add_argument('--local_size', default=8, type=int, help='local size')
parser.add_argument('--num_worker', type=int, default=8, help = 'num of workers')
parser.add_argument('--best_mode', type=str, default='min')
parser.add_argument('--use_loss_as_metric', type=bool, default=False, help='use loss as the first metirc')
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--max_keep_ckpt', type=int, default=1, help = 'max num of saved ckpts in disk')
parser.add_argument('--start_epoch', type=int, default=0, help = 'start epoch')
parser.add_argument('--n_epochs', type=int, default=201, help = 'num of train epoch')
parser.add_argument('--load_trained_parameters', type=bool, default=True, help = 'load trained parameters')
parser.add_argument('--test_model', type=bool, default=True, help = 'test only')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

#workspace setting
workspace=f'TrainingResult_Localarea_s{args.local_size}' # workspace to save logs & ckpts
os.makedirs(workspace, exist_ok=True)        
log_path = os.path.join(workspace, f"log_{args.name}.txt")
log_ptr = open(log_path, "a+")
ckpt_path = os.path.join(workspace, 'checkpoints')
best_path = f"{ckpt_path}/{args.name}.pth.tar"
os.makedirs(ckpt_path, exist_ok=True)

#Prepare dataset
dataset = Mydataset
train_dataset = dataset(data_root = rf'E:\training_dataset_local_lifetime_s{args.local_size}', scale = args.local_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=True, num_workers= args.num_worker)

test_dataset = dataset(data_root = rf'E:\testing_dataset_local_lifetime_s{args.local_size}', scale =  args.local_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False, num_workers= args.num_worker)

# define model
model = NIII().to(device)

# define loss
criterion = nn.L1Loss().to(device)
test_criterion = nn.MSELoss().to(device)

# define optimizer
optimizer = optim.Adam(model.parameters(), lr= args.lr) #use large eps to prevent NaN loss
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= args.lr_step, gamma= args.lr_gamma)
scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

metrics = [RMSEMeter()]
#%%
class Trainer():
    def __init__(self,
                 args,
                 workspace,
                 log_path,
                 log_ptr,
                 ckpt_path,
                 best_path,
                 train_loader,
                 test_loader,
                 model,
                 criterion,
                 test_criterion,
                 optimizer,
                 scheduler,
                 metrics
                 ):
        self.args = args
        self.device =device
        self.workspace = workspace
        self.log_path = log_path
        self.log_ptr = log_ptr
        self.ckpt_path = ckpt_path
        self.best_path = best_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.test_criterion = test_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.global_step = 0
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.stats = {
                "loss": [],
                "valid_loss": [],
                "results": [], # metrics[0], or valid_loss
                "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
                "best_result": None,
                }
        
               
    def log(self,*info):
        print(*info)
        if self.log_path:
            print(*info, file=self.log_ptr) 

    def prepare_data(self,data):
            if isinstance(data, list):
                for i, v in enumerate(data):
                    if isinstance(v, np.ndarray):
                        data[i] = torch.from_numpy(v).to(device)
                    if torch.is_tensor(v):
                        data[i] = v.to(device)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v).to(device)
                    if torch.is_tensor(v):
                        data[k] = v.to(device)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(device)
            else: # is_tensor
                data = data.to(device)

            return data   
   
    def train_one_epoch(self, epoch, train_loader, metrics = [],writer = None):
        self.log(f"==> Start Training Epoch {epoch}, lr={optimizer.param_groups[0]['lr']} ...")
        pbar = tqdm.tqdm(total=len(train_loader) * train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        total_loss = []
        for metric in metrics:
            metric.clear()
        for data in train_loader:
            self.global_step += 1
            data = self.prepare_data(data)
            optimizer.zero_grad()
            gt = data['hr_pixel']
            
            if self.args.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(data)
                    loss = criterion(pred, gt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                pred = self.model(data)
                loss = criterion(pred, gt)
                loss.backward()
                optimizer.step()
                
            if ~np.isnan(loss.item()):
                total_loss.append(loss.item())
                for metric in metrics:
                    metric.update(data, pred, gt)
                
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), self.global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], self.global_step)
                        
            pbar.set_description(f'loss={total_loss[-1]:.4f}')
            pbar.update(train_loader.batch_size )
    
        average_loss = np.mean(total_loss)
        self.stats["loss"].append(average_loss)
    
        pbar.close()
        for metric in metrics:
            self.log(metric.report())
        if writer is not None:
            metric.write(writer, epoch, prefix="train")
        metric.clear()
        self.log(f"==> Finished Epoch {epoch}, average_loss={average_loss:.4f}")


    def evaluate_one_epoch(self, epoch, train_loader, metrics = [],writer = None):
        self.log(f"++> Evaluate at epoch {epoch} ...")
        pbar = tqdm.tqdm(total=len(train_loader) * train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        total_loss = []
        for metric in metrics:
            metric.clear()
        self.model.eval()
    
    
        with torch.no_grad():
            for data in train_loader:
                data = self.prepare_data(data)
                pred = self.model(data)
                gt = data['hr_pixel']
                loss = criterion(pred, gt)
                
                if ~np.isnan(loss.item()):
                    total_loss.append(loss.item())
                    for metric in metrics:
                        metric.update(data, pred, gt)
                
                pbar.set_description(f'loss={total_loss[-1]:.4f}')
                pbar.update(train_loader.batch_size)
    
        average_loss = np.mean(total_loss)
        self.stats["valid_loss"].append(average_loss)
        pbar.close()
        if not self.args.use_loss_as_metric and len(self.metrics) > 0:
            result = metrics[0].measure()
            self.stats["results"].append(result if self.args.best_mode == 'min' else - result) # if max mode, use -result
        else:
            self.stats["results"].append(average_loss) # if no metric, choose best by min loss
        for metric in metrics:
            self.log(metric.report())
        if writer is not None:
            metric.write(writer, epoch, prefix="evaluation")
        metric.clear()
        self.log(f"++> Evaluate epoch {epoch} Finished, average_loss={average_loss:.4f}")

    def train(self):
        self.log(f"[INFO] Training model: {self.args.name} | Training time: {self.time_stamp} | Device: {self.device} | Save path: {self.workspace}")
        self.log(f"[INFO] #parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}")
        
        
        if self.args.load_trained_parameters:
            checkpoint = f'./TrainingResult_downsize_s{self.args.local_size}/checkpoints/{self.args.name}.pth.tar'
            checkpoint_dict = torch.load(checkpoint, map_location= self.device)
            model.load_state_dict(checkpoint_dict['model'])
            self.log(f"[INFO] Model parameters are loaded from  {checkpoint}")
        elif self.args.start_epoch >0:
            self.load_checkpoint()
                
        else:
            self.log("[INFO] Model randomly initialized ...")
        
        writer = SummaryWriter(os.path.join(self.workspace, "run", self.args.name))
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            
            self.train_one_epoch(epoch,self.train_loader,metrics =self.metrics , writer = writer)
            if self.workspace is not None:
                self.save_checkpoint(epoch, full=True, best=False)
            if epoch % self.args.eval_interval == 0:
                self.evaluate_one_epoch(epoch, self.train_loader,metrics =self.metrics, writer = writer)
                self.save_checkpoint(epoch, full=False, best=True)
            self.scheduler.step()
            
        self.log("==> Finished Training ")    
        # self.log_ptr.close()
            
    def test(self, save_path=None):
            if save_path is None:
                save_path = os.path.join(self.workspace, 'testing_results', f'{self.args.name}_{self.args.local_size}')
            os.makedirs(save_path, exist_ok=True)
            
            self.log(f"==> Start Test, save results to {save_path}")


            pbar = tqdm.tqdm(total=len(self.test_loader) * self.test_loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            total_loss = []
            model.eval()
            with torch.no_grad():
                for data in self.test_loader:
                    
                    data = self.prepare_data(data)
                    GT_taus = data['hr_tau']
                    B, C, H, W = data['hr_int'].shape
                    preds = model(data)
                    preds = preds * data['lr_tau_max'].type_as(preds) 
                    GT_taus = GT_taus* data['hr_tau_max'].type_as(preds) 
                    preds = preds.reshape(B, 1, H, W)
                    loss = test_criterion(preds, GT_taus)
                    if ~np.isnan(loss.item()):
                        total_loss.append(loss.item())
                    
                    preds = preds.detach().cpu().numpy() # [B, 1, H, W]
                    GT_taus = GT_taus.detach().cpu().numpy()
                    
                    for b in range(preds.shape[0]):
                        idx = data['idx'][b]
                        if not isinstance(idx, str):
                            idx = str(idx.item())
                        pred = preds[b][0]
                        GT_tau = GT_taus[b][0]
                        pred_result = {'img':pred}
                        io.savemat(os.path.join(save_path, f'{idx}_pred.mat'),pred_result)
                        plt.imsave(os.path.join(save_path, f'{idx}_GT.png'), GT_tau, cmap='inferno',vmin = 0.5, vmax = 4.5)
                        plt.imsave(os.path.join(save_path, f'{idx}.png'), pred, cmap='inferno',vmin = 0.5, vmax = 4.5)

                    pbar.update(self.test_loader.batch_size)
            average_loss = np.mean(total_loss)
            self.log("==> Finished Testing ")
            self.log(f"average mse is: {average_loss}")
            self.log_ptr.close()



    def save_checkpoint(self, epoch,full=False, best=False):

            state = {
                'epoch': epoch,
                'stats': self.stats,
                'model': model.state_dict(),
            }

            if full:
                state['optimizer'] = optimizer.state_dict()
                state['lr_scheduler'] = scheduler.state_dict()
            
            if not best:

                file_path = f"{self.ckpt_path}/{self.args.name}_ep{epoch:04d}.pth.tar"

                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.args.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

                torch.save(state, file_path)

            else:    
                if len(self.stats["results"]) > 0:
                    if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                        self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                        self.stats["best_result"] = self.stats["results"][-1]
                        torch.save(state, best_path)
                else:
                    self.log("[INFO] no evaluated results found, skip saving best checkpoint.")


    def load_checkpoint(self, checkpoint=None):
           if checkpoint is None:
               checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{self.args.name}_ep*.pth.tar'))
               if checkpoint_list:
                   checkpoint = checkpoint_list[-1]
               else:
                   self.log("[INFO] No checkpoint found, model randomly initialized.")
                   return

           checkpoint_dict = torch.load(checkpoint, map_location= device)
           
           if 'model' not in checkpoint_dict:
               model.load_state_dict(checkpoint_dict)
               return

           model.load_state_dict(checkpoint_dict['model'])
           
           self.stats = checkpoint_dict['stats']
           self.start_epoch = checkpoint_dict['epoch']
           
           if optimizer and  'optimizer' in checkpoint_dict:
               try:
                   optimizer.load_state_dict(checkpoint_dict['optimizer'])
                   self.log("[INFO] loaded optimizer.")
               except:
                   self.log("[WARN] Failed to load optimizer. Skipped.")
           
           if scheduler and 'lr_scheduler' in checkpoint_dict:
               try:
                   scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                   self.log("[INFO] loaded scheduler.")
               except:
                   self.log("[WARN] Failed to load scheduler. Skipped.")
                   


if __name__ == '__main__':
    trainer =  Trainer(args,
                       workspace,
                       log_path,
                       log_ptr,
                       ckpt_path,
                       best_path,
                       train_loader,
                       test_loader,
                       model,
                       criterion,
                       test_criterion,
                       optimizer,
                       scheduler,
                       metrics)
    if args.test_model is not True:
        trainer.train()
    
    checkpoint_dict = f'./{workspace}/checkpoints/{args.name}.pth.tar'
    trainer.load_checkpoint(checkpoint=checkpoint_dict)
    trainer.test(save_path=r'E:\test_data_saved_results')









    
 
    

    

    


