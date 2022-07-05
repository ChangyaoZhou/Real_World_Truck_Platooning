import torch
import torchvision
from PIL import Image 
import numpy as np
import cv2
 
class MyDataset(torch.utils.data.Dataset):  
    def __init__(self, states, labels): 
        self.states = states
        self.labels = labels
        
        '''
        self.root = root
        self.dict_dataset = dict_dataset
        self.transform = transform
        self.target_transform = target_transform
        imgs = []
        for key,values in self.dict_dataset.items():
            cam = key[0]
            start_frame = key[1]
            end_frame = key[2]
            img1 = cv2.imread(root + 'CameraRGB%01d/image_%05d.png' % (cam,start_frame))
            img2 = cv2.imread(root + 'CameraRGB%01d/image_%05d.png' % (cam,end_frame))
            img1_scaled = cv2.resize(img1,(227,227)) 
            img2_scaled = cv2.resize(img2,(227,227)) 
            img1_cat = img1_scaled.transpose(2,0,1)[None,:,:,:]
            img2_cat = img2_scaled.transpose(2,0,1)[None,:,:,:]
            two_img = np.concatenate((img1_cat, img2_cat), axis = 0)
            label_scaled = cv2.resize(values,(227,227))
            label = label_scaled.transpose(2,0,1) * 1000
            imgs.append((two_img,label)) 
        self.imgs = imgs'''
 
    def __getitem__(self, index):    
        return self.states[index],self.labels[index] 
 
    def __len__(self):  
        return self.states.shape[0]
 

 