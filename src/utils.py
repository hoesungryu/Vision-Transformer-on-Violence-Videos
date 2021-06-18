import re 
import os
import numpy as np
import glob
import json
from pathlib import Path
import torch 
from torch.utils.data import Dataset, DataLoader



#------------------------------# 
# ResNet(2+1)D
#------------------------------# 
class ViolenceDataset(Dataset):
    def __init__(self, root_npy, config ):
        self.root_npy = root_npy
        self.cfg = config
        self.npy_list = sorted(glob.glob(self.root_npy + '/*.npy'))
        
    def __len__(self):
        return len(self.npy_list)   

    def __getitem__(self, index): 
        npy = self.npy_list[index] # Select npy
        y  = self.get_Y(npy)
        X = self.get_X(npy)     #input npy
        X = np.moveaxis(X, 0, 1)
        y = torch.Tensor([y])
        return X, y
        
    def uniform_sampling(self, video):
        target_frames = self.cfg.DATASET.target_frames
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad>0:
            for i in range(-num_pad,0):
                try: 
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    def get_X(self,npy):
        tmp_np = np.load(npy, mmap_mode='r')
        tmp_np = np.float32(tmp_np)
        tmp_np = self.uniform_sampling(tmp_np)
        return tmp_np
    
    def get_Y(self,npy):
        if bool(re.search('NV_*',npy)) == True:
            y = 0
        else: 
            y = 1
        return y

# def get_loader(dataset, cfg, shuffle=True):
#     return DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle)

# def get_valid_loader(dataset, full_batch,shuffle=True):
#     return DataLoader(dataset, batch_size=full_batch ,shuffle=shuffle)