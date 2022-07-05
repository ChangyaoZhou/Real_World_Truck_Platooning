import torch
import torchvision
from PIL import Image 
import numpy as np
import cv2
 
class MyDataset1(torch.utils.data.Dataset):  
    def __init__(self, states, labels): 
        self.states = states
        self.labels = labels
   
 
    def __getitem__(self, index):    
        return self.states[index],self.labels[index]
 
 
    def __len__(self):  
        return self.states.shape[0]


    
class MyDataset2(torch.utils.data.Dataset):  
    def __init__(self,root, txtname, transform=None, size=None, target_transform=None): 
        self.root = root
        self.txtname = txtname
        fh = open(root + self.txtname, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        input_list = [] 
        label_list = np.empty([0, 2], dtype=float)       
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，默认为空格 
            words = line.split()   #通过指定分隔符对字符串进行切片 
            input_list.append([words[0],float(words[4]), float(words[8])]) 
            label_list = np.append(label_list, np.array([[float(words[9]), float(words[10])]]), axis=0)
            #label_list.append((float(words[3]), float(words[4])))
            
        self.input_list = input_list
        self.label_list = label_list
        self.transform = transform
        self.size = size
        self.target_transform = target_transform
 
    def __getitem__(self, index):   
        img_path = self.input_list[index][0] 
        img = cv2.imread(self.root+img_path)
        #print(img_path)
        if self.transform is not None:
            img = self.transform(img, self.size, interpolation = cv2.INTER_AREA)
        
        inputs = [img, self.input_list[index][1], self.input_list[index][2]]
 
        return inputs, self.label_list[index] 
 
    def __len__(self):  
        return len(self.label_list)
    
    #train_data = MyDataset(root = './town04_image_data/', txtname = 'offdata_image.txt')

    

 

 