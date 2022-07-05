import torch
import torch.nn as nn 
import torchvision.models as models
 
class MyModel_CNNalex(nn.Module): 
    # CNN with pretrained Alexnet
    def __init__(self): 
        super().__init__()    
        #self.expands = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.features = models.alexnet(pretrained=True).features  #output: N*256*8*11 
        self.predict = nn.Sequential(
            nn.Flatten(),    #keep the first dim N, the others will be flattened
            nn.Linear(258*3*5, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(256), 
            #nn.Linear(256, 64),
            #nn.ReLU(),
            nn.Linear(512, 2)
        ) 

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 150, 200).to(torch.float32) # input:Nx3x300x400
        #features = torch.squeeze(self.features(img)).view(-1, 60)  #output: N*30
        #expands = self.expands(img)
        features = self.features(img)
        #print(features.shape)
        ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32) #shape: N*1*8*11
        ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 3, 5).to(torch.float32) #shape: N*1*8*11
        #ref_v = ref_v.view(-1, 1).to(torch.float32) #shape: N*1
        #ego_v = ego_v.view(-1, 1).to(torch.float32) #shape: N*1
        features = torch.cat((features,ref_v, ego_v),1)   #shape: N*32
        preds = self.predict(features) # output:Nx2
        return preds
    
    
class MyModel_CNN(nn.Module):
    # CNN with paper architecture 
    # configuration: 
        #channel = [30, 30, 30, 3]
        #kernel = 5
        #neuron = [64, 32]
        #stride = 2 
    def __init__(self, channels, kernel_size, neurons, stride = 2):
        super().__init__()
        
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
            #nn.ReLU(),
            
            #nn.Conv2d(in_channels = self.channels[2], out_channels = self.channels[3], kernel_size = self.kernel_size, stride = self.stride)
        )
        
        self.predictor = nn.Sequential( 
            nn.Linear(self.channels[2]+2, self.neurons[0]),
            nn.ReLU(),
            #nn.BatchNorm1d(self.neurons[0]), 
            nn.Linear(self.neurons[0], self.neurons[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(self.neurons[1]), 
            nn.Linear(self.neurons[1], 2),
            #nn.ReLU(),
            #nn.BatchNorm1d(self.neurons[2]),
            #nn.Linear(self.neurons[2], 2),
            #nn.ReLU(),
            #nn.BatchNorm1d(self.neurons[3]),
            #nn.Linear(self.neurons[3], 2) 
        )
    def forward(self, x):
        img, ref_v, ego_v = x 
        N = img.shape[0]
        img = img.view(-1, 3, 128, 128).to(torch.float32) # input:Nx3x128x128
        features = self.features(img)  #output: N*30*1*1
        #print(features.shape)
        features = features.view(-1, self.channels[2]) # Nx64  
        new_features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) # Nx66  
        preds = self.predictor(new_features) # output:Nx2
        return preds
    
    
    
    
    
    
    
