import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class AtariNetwork(nn.Module):
    def __init__(self, lr, nActions, inputDims, name, checkpointDir):
        super(AtariNetwork, self).__init__()
        self.checkpointDir = checkpointDir
        self.checkpointFile = os.path.join(self.checkpointDir, name)
        
        self.convLayer1 = nn.Conv2d(inputDims[0], 32, 8, stride = 4)
        self.convLayer2 = nn.Conv2d(          32, 64, 4, stride = 2)
        self.convLayer3 = nn.Conv2d(          64, 64, 3, stride = 1)
        
        fcInputDims = self.calculateConvOutputDims(inputDims)
        
        self.layer1 = nn.Linear(fcInputDims,      512)
        self.layer2 = nn.Linear(        512, nActions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        
    def calculateConvOutputDims(self, inputDims):
        dims = T.zeros(1, *inputDims)
        dims = self.convLayer1(dims)
        dims = self.convLayer2(dims)
        dims = self.convLayer3(dims)
        return int(np.prod(dims.size()))
    
    def forward(self, state):
        conv1 = F.relu(self.convLayer1(state))
        conv2 = F.relu(self.convLayer2(conv1))
        conv3 = F.relu(self.convLayer3(conv2)) #BS x nFilters x H x W
        convReshaped = conv3.view(conv3.size()[0], -1)
        
        flat1 = F.relu(self.fc1(convReshaped))
        qValues = self.fc2(flat1)
        return qValues

    
    def saveCheckpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpointFile)
        print("...       SAVED       ...")
        
    def loadCheckpoint(self):
        print("... loading checkpoint ...")
        T.load_state_dict(T.load(self.checkpointFile))
        print("...       LOADED       ...")
        
    