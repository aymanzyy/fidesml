#from mpi4py import MPI
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.optim as optim

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.pyplot import figure
NUMPOINTS = 7500
NUMCHANNELS = 5
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1
MODELPATH = "dist_autoenc.pth"
LBMDATAPATH = ""

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(NUMCHANNELS, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*NUMCHANNELS)

    def encoder(self, x): 
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Adds a batch dimension -> shape becomes [1, 7500, 5]
            
        # 1. Now safely swap dimensions to [1, 5, 7500]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, NUMCHANNELS)
    
    def forward(self, x):
        x = self.encoder(x)
        #print("Encoded shape")
        #print(x.shape)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    main()