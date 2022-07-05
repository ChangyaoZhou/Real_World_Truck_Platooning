import torch
import torch.nn as nn 
import torchvision.models as models
 
class MyModel1(nn.Module): 
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 5, out_features = self.neurons[0]),
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
    
    
    
class MyModel2(nn.Module):  
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 5, out_features = self.neurons[0]),
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
            nn.Linear(in_features = self.neurons[3], out_features = 2)
            )          
        #self.predictor.cuda()
          
    def forward(self, x):  
        return self.predictor(x)  
    
    
class MyModel_CNN(nn.Module): 
    def __init__(self): 
        super().__init__()        
        self.features = models.alexnet(pretrained=True).features  #output: N*256*8*11
        #for param in self.features.parameters():
            #param.requires_grad = False
        self.predict = nn.Sequential(
            nn.Flatten(),    #keep the first dim N, the others will be flattened
            nn.Linear(258*8*11, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 300, 400).to(torch.float32) # input:Nx3x300x400
        features = self.features(img)  #output: N*256*8*11
        ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 8,11).to(torch.float32) #shape: N*1*8*11
        ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 8,11).to(torch.float32) #shape: N*1*8*11
        features = torch.cat((features,ref_v, ego_v),1)   #shape: N*258*8*11
        preds = self.predict(features) # output:Nx2
        return preds
    
    
    
    
    
    
    
    