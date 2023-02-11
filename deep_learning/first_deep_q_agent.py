import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from first_network import FirstNetwork

class Agent():
    def __init__(self, nActions, inputDims,
        learningRate = 0.001, gamma = 0.99, 
        eps0 = 1.0, epsMin = 0.01, epsDecrement = 1e-5):
            
            ''' capturing '''
            self.learningRate, self.gamma \
            =    learningRate,      gamma
            self.eps, self.epsMin, self.epsDecrement \
            =   eps0,      epsMin,      epsDecrement
            self.nActions, self.inputDims \
            =    nActions,      inputDims

            ''' more initialization '''
            self.actionSpace = tuple(range(nActions))
            self.Q = FirstNetwork(self.learningRate, self.nActions, self.inputDims)
            self.rng = np.random
    
    def chooseAction(self, state):
        if self.rng.random() < self.eps:
            action = np.random.choice(self.actionSpace)
        else:
            state = T.tensor(state, dtype = T.float).to(self.Q.device)
            qValues = self.Q.forward(state)
            action = T.argmax(qValues).item()
        return action

    def decrementEpsilon(self) -> None:
        self.eps = self.eps - self.epsDecrement \
                        if self.eps > self.epsMin \
                   else self.epsMin
    
    def updateQ(self, state, action, reward, state_) -> None:
        ''' initialization'''
        self.Q.optimizer.zero_grad()
        state  = T.tensor(state, dtype = T.float).to(self.Q.device)
        state_ = T.tensor(state_, dtype = T.float).to(self.Q.device)
        action = T.tensor(action).to(self.Q.device)
        reward = T.tensor(reward).to(self.Q.device)
        
        ''' getting predicted Q for '''
        qPredictions  = self.Q.forward(state)
        qPredictions_ = self.Q.forward(state_)
        
        qPrediction = qPredictions[action]
        qNext = qPredictions_.max()
        
        ''' updating Q(state, action)'''
        qTarget = reward + self.gamma * qNext
        loss = self.Q.loss(qTarget, qPrediction).to(self.Q.device)
        
        ''' backprop and step'''
        loss.backward()
        self.Q.optimizer.step()
        
        ''' update eps '''
        self.decrementEpsilon()
