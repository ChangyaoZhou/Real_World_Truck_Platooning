import torch
import torch.nn as nn 
import torchvision.models as models
 
class MyModel1(nn.Module): 
    # input eight input [ref_x, ref_y, ref_yaw, ref_v, ego_x, ego_y, ego_yaw, ego_v] 
    # architecture: neurons = [256, 1024, 2048, 1024, 256]
    # train result: mynet_MLP_states.pth
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
    
    
    
class MyModel2(nn.Module): 
    # input five input [delta_x, delta_y, delta_yaw, v_ref, v_ego] relative transformation
    # architecture: neurons = [128, 256, 256, 256, 128]
    # train result: mynet_relative_pose.pth 
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
    
class MyModel_2yaw(nn.Module):
    # input five input [delta_x, delta_y, yaw_ref, yaw_ego, v_ref, v_ego]  direct substration
    # architecture: neurons = [256, 1024, 2048, 1024, 256]
    # train result: mynet_2yaw.pth
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 6, out_features = self.neurons[0]),
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
    
class MyModel_CNNalex(nn.Module):
    # input: image, v_ref, v_ego
    # output: throttle, steering angle
    # train result: mynet_pretrained_alexnet.pth
    def __init__(self): 
        super().__init__()    
        #self.expands = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.features = models.alexnet(pretrained=True).features  #output: N*256*8*11
        #for param in self.features.parameters():
            #param.requires_grad = False
        self.predict = nn.Sequential(
            nn.Flatten(),    #keep the first dim N, the others will be flattened
            nn.Linear(258*3*5, 512),
            nn.ReLU(), 
            nn.Linear(512, 2)
        ) 

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 150, 200).to(torch.float32) # input:Nx3x300x400 
        features = self.features(img)
        #print(features.shape)
        ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32) #shape: N*1*8*11
        ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32) #shape: N*1*8*11 
        features = torch.cat((features,ref_v, ego_v),1)   #shape: N*32
        preds = self.predict(features) # output:Nx2
        return preds

class MyModel_CNNsimple(nn.Module):
    def __init__(self, channels, kernel_size, neurons, stride = 2):
        super().__init__()
        # input: image, v_ref, v_ego
        # output: throttle, steering angle
        # train result: mynet_cnn_simple.pth
        # architecture:
        #channel = [30, 30, 30, 3]
        #kernel = 5
        #neuron = [64, 32]
        #stride = 2 
        self.channels = channels
        self.kernel_size = kernel_size
        self.neurons = neurons
        self.stride = stride
        
        self.features = nn.Sequential( 
            nn.Conv2d(in_channels = 3, out_channels = self.channels[0], kernel_size = self.kernel_size, stride = self.stride),
            nn.MaxPool2d((2,2)),
            nn.ReLU(), 
            nn.Conv2d(in_channels = self.channels[0], out_channels = self.channels[1], kernel_size = self.kernel_size, stride = self.stride),
            nn.MaxPool2d((2,2)),
            nn.ReLU(), 
            nn.Conv2d(in_channels = self.channels[1], out_channels = self.channels[2], kernel_size = self.kernel_size, stride = self.stride),
            nn.MaxPool2d((2,2)), 
        )
        
        self.predictor = nn.Sequential( 
            nn.Linear(self.channels[2]+2, self.neurons[0]),
            nn.ReLU(), 
            nn.Linear(self.neurons[0], self.neurons[1]),
            nn.ReLU(), 
            nn.Linear(self.neurons[1], 2), 
        )
    def forward(self, x):
        img, ref_v, ego_v = x 
        N = img.shape[0]
        img = img.view(-1, 3, 128, 128).to(torch.float32) # input:Nx3x200x150
        features = self.features(img)  #output: N*30*1*1
        #print(features.shape)
        features = features.view(-1, self.channels[2]) # Nx64  
        new_features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) # Nx66  
        preds = self.predictor(new_features) # output:Nx2
        return preds
        
class MyModel_CNNalex_3output(nn.Module):
    # input: image, v_ref, v_ego
    # output: relative transformation, delta_x, delta_y, delta_yaw
    # train result: mynet_cnn_img1.pth
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
        features = self.features(img)  #output: N*3*1*1
        #print(features.shape)
        features = self.pool(features)
        #print(features.shape) 
        features = features.view(-1, 256) # Nx3
        new_features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) 
        preds = self.predict(new_features) 
        return preds
    
class MyModel_CNNalex_5output(nn.Module): 
    # input: image, v_ref, v_ego
    # output: delta_x, delta_y, delta_yaw(relative transformation), throttle, steering angle 
    # train result: mynet_alexnet_5output.pth
    def __init__(self): 
        super().__init__()        
        self.features = models.alexnet(pretrained=True).features   #output: N*256*3*5
        self.extra = nn.MaxPool2d((2,2))
        self.predict = nn.Sequential(
            nn.Flatten(),                  #keep the first dim N, the others will be flattened
            nn.Linear(258, 64),
            nn.ReLU(), 
            nn.Linear(64, 5),
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)  #output: N*256*3*5 
        features = self.extra(features) 
        features = features.view(-1, 256)
        features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) 
        preds = self.predict(features)
        return preds
      
    
    
    
    
    
    
    
    
