import numpy as np
import torch as T
from atari_network import AtariNetwork
from ..course_utils.replay_memory import ReplayBuffer

class QParameters:
    def __init__(self, gamma, lr, epsMin = 0.01, epsDec = 5e-7):
        self.gamma, self.lr, self.epsMin, self.epsDec\
           = gamma,      lr,      epsMin,      epsDec
        self.eps = 1.0
    
    def decrement(self):
        self.eps = max(self.eps - self.epsDec, self.epsMin))

class AtariAgent:
    def __init__(self, qParams: QParameters, nActions, inputDims,
                 capacity, batchSize, replace = 1000,
                 algo = None, envName = None, checkpointDir = 'tmp/dqn'):
        self.qParameters, self.nActions, self.inputDims\
            =    qParams,      nActions,      inputDims
        self.actions = tuple(i for i in range(nActions))
        self.learnStepCount = 0
        
        self.buffer = ReplayBuffer(capacity, inputDims)
        self.batchSize = batchSize
        self.replace = replace
        
        self.algo, self.envName, self.checkpointDir\
           = algo,      envName,      checkpointDir
        
        self.qCurrent = AtariNetwork(
            self.qParameters.lr, self.nActions, self.inputDims, 
            self.envName + "_" + self.algo + "_q_current", 
            self.checkpointDir)
        
        self.qTarget = AtariNetwork(
            self.qParameters.lr, self.nActions, self.inputDims, 
            self.envName + "_" + self.algo + "_q_target", 
            self.checkpointDir)
        
        self.rand = np.random
        
    def chooseAction(self, state):
        if self.rand.random() < self.qParameters.eps:
            action = self.rand.choice(self.actions)
        else:
            state = T.tensor((state,), dtype = T.float)\
                .to(self.qTarget.device)
            qValues = self.qTarget.forward(state)
            action = T.argmax(qValues).item()
        return action

    