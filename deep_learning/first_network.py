import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class FirstNetwork(nn.Module):
    def __init__(self, learningRate, nActions, inputDims):
        super(FirstNetwork, self).__init__()
        
        self.fcLayer1 = nn.Linear(*inputDims, 128)     # parameter layer -> hidden layer
        self.fcLayer2 = nn.Linear(128, nActions)        # hidden    layer -> action layer
        
        ''' f(state) = Q(state, *)
            that is, the NN is a functional on the statespace to the actionspace
            the state is the input data, the output data is essentially a vector
            of the form (Q(state, a))_{a in A}
        '''
        
        self.optimizer = optim.Adam(self.parameters(), lr = learningRate)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        layer1 = F.relu(self.fcLayer1(state))
        actionLayer =   self.fcLayer2(layer1)
        return actionLayer