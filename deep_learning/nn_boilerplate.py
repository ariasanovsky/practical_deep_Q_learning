import torch.nn as nn               # layers
import torch.nn.functional as F     # activation functions
import torch.optim as optim         # optimizers
import torch as T                   # everything else


# linear classifer class
class LinearClassifier(nn.Module):  # inherits some useful functions
    def __init__(self, learningRate, nClasses, inputDims):
        super(LinearClassifier, self).__init__()    # call inherited constructor

        # first fully connected layer
        self.fc1 = nn.Linear(*inputDims, 128)       # the * unpacks the iterable of dimensions

        # next layer
        self.fc2 = nn.Linear(128, 256)

        # final layer to classifier
        self.fc3 = nn.Linear(256, nClasses)

        # optimizer of choice & loss function
        self.optimizer = optim.Adam(self.parameters(), lr = learningRate)
        self.loss = nn.CrossEntropyLoss()           # nn.MSELoss() is also valid

        # interfacing with GPU devices or CPU if unavailable
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    ''' pytorch handles backpropogation
        we still need to do forward
    '''
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))          # composing layer with act. fun.
        layer2 = F.sigmoid(self.fc2(layer1))        # passing 1st layer as data, through act. fun.
        layer3 =           self.fc3(layer2)         # no activation, just the layer, handled by loss fun.
        
        return layer3
    
    ''' passes in set of correct labels on data
        updates model to match to it better
    '''
    def updateModel(self, data, labels):
        # pytorch keeps gradients since they are good to inspect
        # we zero it manually
        self.optimizer.zero_grad()
        
        # convert data & labels to tensors!
        # forward function can't handle numpy arrays
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)
        
        ''' tensor : preserves original datatype, lets us preserve memory carefully
            Tensor : will convert datatypes, may consume more memory, possible headaches
        ''' 

        predictions = self.forward(data)        # measure model results...
        cost = self.loss(predictions, labels)   # against ground truth
        cost.backward()                         # propogate measurement through network
        self.optimizer.step()                   # adjusts model accordingly
        
        