import torch
import torchvision
from PIL import Image 
import numpy as np
import cv2
 
class MyDataset_MLP(torch.utils.data.Dataset):  
    def __init__(self, states, labels): 
        self.states = states
        self.labels = labels 
 
    def __getitem__(self, index):    
        return self.states[index],self.labels[index] * 10 
    def __len__(self):  
        return self.states.shape[0]

    
class MyDataset_FCNN(torch.utils.data.Dataset):  
    def __init__(self, root, txtname, transform=None, size=None, target_transform=None): 
        self.root = root
        self.txtname = txtname
        fh = open(root + self.txtname, 'r')  
        input_list = [] 
        label_list = np.empty([0, 2], dtype=float)       
        for line in fh:                
            line = line.rstrip()       
            words = line.split()    
            input_list.append([words[0],float(words[1]), float(words[2])]) 
            label_list = np.append(label_list, np.array([[float(words[3]), float(words[4])]]), axis=0) 
            
        self.input_list = input_list
        self.label_list = label_list
        self.transform = transform
        self.size = size
        self.target_transform = target_transform
 
    def __getitem__(self, index):   
        img_path = self.input_list[index][0] 
        img = cv2.imread(self.root+img_path)
        
        if self.transform is not None:
            img = self.transform(img, self.size, interpolation = cv2.INTER_AREA)
        
        inputs = [img, self.input_list[index][1], self.input_list[index][2]]
 
        return inputs, self.label_list[index] 
 
    def __len__(self):  
        return len(self.label_list)
     
    
class MyDataset_CNN(torch.utils.data.Dataset):  
    def __init__(self,root, txtname, transform=None, size=None, target_transform=None): 
        self.root = root
        self.txtname = txtname
        fh = open(root + self.txtname, 'r')  
        input_list = [] 
        label_list = np.empty([0, 3], dtype=float)       
        for line in fh:                
            line = line.rstrip()        
            words = line.split()     
            input_list.append([words[0], float(words[1]), float(words[2])]) 
            label_list = np.append(label_list, np.array([[float(words[3]), float(words[4]), float(words[5])]]), axis=0) 
            
        self.input_list = input_list
        self.label_list = label_list
        self.transform = transform
        self.size = size
        self.target_transform = target_transform
 
    def __getitem__(self, index):   
        img_path = self.input_list[index][0] 
        img = cv2.imread(self.root+img_path)  
        img = cv2.resize(img, self.size, interpolation = cv2.INTER_AREA)
        if self.transform is not None:
            img = self.transform(img/255)  
            
        inputs = [img, self.input_list[index][1], self.input_list[index][2]]
        return inputs, self.label_list[index] 
 
    def __len__(self):  
        return len(self.label_list)
    

 

 
