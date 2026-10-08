from mpi4py import MPI
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
NUMCHANNELS = 4
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1
MODELPATH = "dist_autoenc.pth"
LBMDATAPATH = ""


class PointCloudAE(nn.Module):
    def __init__(self, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        
        # Encoder remains dynamic due to 1D convolutions and max-pooling
        self.conv1 = torch.nn.Conv1d(NUMCHANNELS, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        # Decoder now uses 1D Convolutions to map back to NUMCHANNELS per point
        self.dec1 = nn.Linear(self.latent_size, 256)
        self.dec2 = nn.Linear(256, 256)
        
        # Fully dynamic projection layer instead of a massive flat linear layer
        self.dec_point = torch.nn.Conv1d(256, NUMCHANNELS, 1)

    def encoder(self, x): 
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Shape: [Batch, Points, Channels]
            
        x = x.permute(0, 2, 1)  # Shape: [Batch, Channels, Points]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # This global pooling layer collapses the "Points" dimension entirely
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size) # Shape: [Batch, LatentSize]
        return x
    
    def decoder(self, x, num_points):
        # x shape: [Batch, LatentSize]
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x)) # Shape: [Batch, 256]
        
        # Repeat the latent vector for every point in the current target size
        # Shape becomes: [Batch, 256, num_points]
        x = x.unsqueeze(-1).repeat(1, 1, num_points)
        
        # Map 256 features back to NUMCHANNELS per point
        x = self.dec_point(x) # Shape: [Batch, NUMCHANNELS, num_points]
        
        return x.permute(0, 2, 1) # Shape: [Batch, num_points, NUMCHANNELS]
    
    def forward(self, x):
        # Capture the dynamic number of points from the input tensor
        num_points = x.shape[1] if x.dim() == 3 else x.shape[0]
        
        latent = self.encoder(x)
        out = self.decoder(latent, num_points)
        return out
