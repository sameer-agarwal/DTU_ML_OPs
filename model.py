from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.Cnn1 = nn.Sequential(nn.Conv2d(1,1,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                                    nn.Conv2d(1,1,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        
        self.Linear1 = nn.Sequential(nn.Linear(784,64),nn.ReLU(inplace=True),nn.Linear(64,10),nn.LogSoftmax(dim=1))
    
    def forward(self,x):

        x = self.Cnn1(x)
        x = x.view(x.size(0),-1)
        x = self.Linear1(x)
        return x