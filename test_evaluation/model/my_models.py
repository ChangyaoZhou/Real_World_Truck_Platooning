import torch
import torch.nn as nn 
import torchvision.models as models
 
class MyModel_MLP_raw(nn.Module): 
    # input 8 inputs [ref_x, ref_y, ref_yaw, ref_v, ego_x, ego_y, ego_yaw, ego_v] 
    # architecture: neurons = [256, 1024, 2048, 1024, 256]
    # train result: mynet_MLP_raw.pth
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 8, out_features = self.neurons[0]),
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
    
class MyModel_MLP_transform(nn.Module):
    # input 5 inputs [delta_x, delta_y, delta_yaw, v_ref, v_ego] 
    # architecture: neurons = [256, 1024, 256]
    # train result: mynet_MLP_transform.pth
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 5, out_features = self.neurons[0]),
            nn.BatchNorm1d(self.neurons[0]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[0], out_features = self.neurons[1]),
            nn.BatchNorm1d(self.neurons[1]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[1], out_features = self.neurons[2]),
            nn.BatchNorm1d(self.neurons[2]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[2], out_features = 2),  
        )  
          
    def forward(self, x):  
        return self.predictor(x)
    
    
class MyModel_FCNN_endtoend(nn.Module):
    # input: image, v_ref, v_ego
    # output: throttle, steering angle
    # train result: mynet_FCNN_endtoend.pth
    def __init__(self): 
        super().__init__()     
        self.features = models.alexnet(pretrained=True).features  #output: N*256*8*11 
        self.predict = nn.Sequential(
            nn.Flatten(),    
            nn.Linear(258*3*5, 512),
            nn.ReLU(), 
            nn.Linear(512, 2)
        ) 

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 150, 200).to(torch.float32) # input:Nx3x150x200 
        features = self.features(img) 
        ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32)  
        ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32)  
        features = torch.cat((features,ref_v, ego_v),1)    
        preds = self.predict(features)  
        return preds
    

    
class MyModel_CNN1(nn.Module):
    # input: image, v_ref, v_ego
    # output: relative transformation, delta_x, delta_y, delta_yaw
    # train result: mynet_CNN_ontraj.pth, mynet_CNN_twosteps.pth
    def __init__(self):
        super().__init__()
        
        self.features = models.alexnet(pretrained=True).features
        self.predict = nn.Sequential( 
            nn.Linear(258, 64),
            nn.ReLU(), 
            nn.Linear(64, 3),
        )
        self.pool = nn.MaxPool2d((2,2)) 
        
    def forward(self, x): 
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)    
        features = self.pool(features) 
        features = features.view(-1, 256)  
        new_features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) 
        preds = self.predict(new_features) 
        return preds
    
    
class MyModel_CNN2(nn.Module):
    # input: image, v_ref, v_ego
    # output: relative transformation, delta_x, delta_y, delta_yaw
    # train result: mynet_CNN_depth.pth, mynet_CNN_stereo.pth
    def __init__(self): 
        super().__init__()        
        self.features = models.alexnet(pretrained=True).features   
        self.extra = nn.MaxPool2d((2,2))
        self.predict = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(258, 64),
            nn.ReLU(), 
            nn.Linear(64, 3),
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)   
        features = self.extra(features) 
        features = features.view(-1, 256)
        features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) 
        preds = self.predict(features)
        return preds
        
    
 