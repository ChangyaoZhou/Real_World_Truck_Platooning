import torch
import torch.nn as nn 
import torchvision.models as models
 

class MyModel_CNN(nn.Module): 
    def __init__(self): 
        super().__init__()        
        self.features = models.alexnet(pretrained=True).features   #output: N*256*3*5
        self.extra = nn.MaxPool2d((2,2))
        self.predict = nn.Sequential(
            nn.Flatten(),                  #keep the first dim N, the others will be flattened
            nn.Linear(258, 64),
            nn.ReLU(),
            #nn.Linear(512, 256),
            #nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)  #output: N*256*3*5
        #print(features.shape)
        features = self.extra(features)
        #ref_v = ref_v.view(-1, 1, 1, 1).repeat(1, 1, 3,3).to(torch.float32) #shape: N*1*3*5
        #ego_v = ego_v.view(-1, 1, 1, 1).repeat(1, 1, 3,3).to(torch.float32) #shape: N*1*3*5
        features = features.view(-1, 256)
        features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32)
        #features = torch.cat((features,ref_v, ego_v),1)   #shape: N*258*3*5
        preds = self.predict(features)
        return preds