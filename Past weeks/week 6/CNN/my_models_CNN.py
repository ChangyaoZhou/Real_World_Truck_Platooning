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
            nn.Linear(in_features = self.neurons[2], out_features = 2),

        ) 
        '''
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[3]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[3], out_features = self.neurons[4]),
            nn.ReLU(),
            nn.BatchNorm1d(self.neurons[4]),
            #nn.Dropout(),
            nn.Linear(in_features = self.neurons[4], out_features = 2), 
        '''
          
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
        #self.features = models.alexnet(pretrained=True).features  #output: N*256*8*11
        #for param in self.features.parameters():
            #param.requires_grad = False
            
        self.features = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=3, stride=2, padding=1),   #75*100
            #nn.BatchNorm2d(30),            
            nn.MaxPool2d(kernel_size=2, stride=2),                  #37*50
            nn.ReLU(),
            nn.Conv2d(30, 60, kernel_size=3, stride=2, padding=1),  #18*25
            #nn.BatchNorm2d(60),            
            nn.MaxPool2d(2, 2),                                     #9*12
            nn.ReLU(),
            nn.Conv2d(60, 60, kernel_size=3, stride=2, padding=1),  #4*6
            #nn.BatchNorm2d(30),
            nn.MaxPool2d(2, 2),                                     #2*3
        
        )
        self.predict = nn.Sequential(
            nn.Flatten(),    #keep the first dim N, the others will be flattened
            nn.Linear(62*2*3, 32),
            nn.ReLU(),
            #nn.BatchNorm1d(256), 
            #nn.Linear(256, 64),
            #nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 150, 200).to(torch.float32) # input:Nx3x300x400
        features = self.features(img)  #output: N*30*2*3
        #print(features.shape)
        ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 2, 3).to(torch.float32) #shape: N*1*2*3
        ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 2, 3).to(torch.float32) #shape: N*1*2*3
        features = torch.cat((features,ref_v, ego_v),1)   #shape: N*32*9*12
        preds = self.predict(features) # output:Nx2
        return preds
    
    
    
    
    
    
    
    