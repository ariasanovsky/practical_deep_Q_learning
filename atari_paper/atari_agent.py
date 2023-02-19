import numpy as np
import torch as T
from atari_network import AtariNetwork

import sys
sys.path.append("..")
from course_utils.replay_memory import ReplayBuffer

class QParameters:
    def __init__(self, gamma, lr, epsMin = 0.01, epsDec = 5e-6):
        self.gamma, self.lr, self.epsMin, self.epsDec\
           = gamma,      lr,      epsMin,      epsDec
        self.eps = 1.0
    
    def decrement(self):
        self.eps = max(self.eps - self.epsDec, self.epsMin)

class AtariAgent:
    def __init__(self, qParams: QParameters, nActions, inputDims,
                 capacity, batchSize, replace = 1000,
                 algo = None, envName = None, checkpointDir = 'tmp\\dqn'):
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

    def tensorConvert(self, values):
        return T.tensor(values).to(self.qTarget.device)
    
    def sampleMemory(self):
        states, actions, rewards, states_, dones\
            = self.buffer.sampleBuffer(self.batchSize)
        
        states  = self.tensorConvert(states)
        actions = self.tensorConvert(actions)
        rewards = self.tensorConvert(rewards)
        states_ = self.tensorConvert(states_)
        dones   = self.tensorConvert(dones)
        return states, actions, rewards, states_, dones
        # I don't like the code duplication, probably can do with iterators
    
    def updateTarget(self):
        if self.learnStepCount % self.replace == 0:
            self.qTarget.load_state_dict(self.qCurrent.state_dict())
    
    def saveModels(self):
        self.qCurrent.saveCheckpoint()
        self.qTarget.saveCheckpoint()
    
    def loadModels(self):
        self.qCurrent.loadCheckpoint()
        self.qTarget.loadCheckpoint()
    
    def learn(self):
        if self.buffer.nMemories < self.batchSize:
            return
        
        self.qCurrent.optimizer.zero_grad()
        self.updateTarget()
        states, actions, rewards, states_, dones\
            = self.sampleMemory()
        # think about the array dims!
        batchIndices = np.arange(self.batchSize)
        
        qPredictions = self.qCurrent.forward(states)[batchIndices, actions]
        qPredictions_ = self.qTarget.forward(states_).max(dim = 1)[0]
        
        for i in range(len(qPredictions_)):
            if dones[i]:
                qPredictions_[i] = 0.0
        
        qUpdates = rewards + self.qParameters.gamma * qPredictions_
        loss = self.qCurrent.loss(qUpdates, qPredictions)
        loss.backward()
        self.qCurrent.optimizer.step()
        
        self.learnStepCount += 1
        self.qParameters.decrement()
