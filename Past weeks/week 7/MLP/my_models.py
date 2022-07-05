import torch
import torch.nn as nn 
import torchvision.models as models
 
class MyModel1(nn.Module):
    # model for relative pose 
    # input five input [delta_x, delta_y, delta_yaw, v_ref, v_ego]  
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 5, out_features = self.neurons[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[0]), 
            nn.Linear(in_features = self.neurons[0], out_features = self.neurons[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[1]), 
            nn.Linear(in_features = self.neurons[1], out_features = self.neurons[2]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[2]), 
            nn.Linear(in_features = self.neurons[2], out_features = self.neurons[3]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[3]), 
            nn.Linear(in_features = self.neurons[3], out_features = self.neurons[4]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[4]), 
            nn.Linear(in_features = self.neurons[4], out_features = 2), 
        ) 
          
    def forward(self, x):  
        return self.predictor(x)
    
    
    
 
class MyModel_2yaw(nn.Module):
    # model for input [delta_x, delta_y, ref_yaw, ego_yaw, v_ref, v_ego] 
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 6, out_features = self.neurons[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[0]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[0], out_features = self.neurons[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[1]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[1], out_features = self.neurons[2]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[2]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[2], out_features = self.neurons[3]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[3]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[3], out_features = self.neurons[4]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[4]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[4], out_features = 2), 
        ) 
          
    def forward(self, x):  
        return self.predictor(x)
    
    
 
    
    
    
    
    
    
