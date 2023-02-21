import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





class MLP(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)
        # self.fc2 = nn.Linear(hidden, hidden, bias=True)


    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x



class MLP_VEC(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP_VEC,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)



    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        
        return x






class LINEAR(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super(LINEAR,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)


    def forward(self,x):

        x = self.fc1(x)
        
        return x