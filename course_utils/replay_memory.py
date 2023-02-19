import numpy as np

class ReplayBuffer():
    def __init__(self, capacity, inputShape, nActions):
        self.capacity = capacity
        self.nMemories = 0
            
        self.states = np.zeros_like(
            (self.capacity, *inputShape), dtype = np.float32)
        self.states_ = np.zeros_like(
            (self.capacity, *inputShape), dtype = np.float32)
        
        self.actions = np.zeros_like(
            self.capacity, dtype = np.int64)
        self.rewards = np.zeros_like(
            self.capacity, dtype = np.float32)
        self.dones = np.zeros_like(
            self.capacity, dtype = np.uint8)
    
    def storeTransition(self, state, action, reward, state_, done):
        i = self.nMemories % self.capacity
        self.states[i], self.actions[i], self.states_[i], self.rewards[i], self.dones[i]\
            = state,          action,          state_,          reward,          done
        self.nMemories += 1
    
    def sampleBuffer(self, batchSize):
        imax = min(self.capacity, self.nMemories)
        b = np.random.choice(imax, batchSize, replace = False)
        
        \
               states,         actions,         states_,         rewards,         dones\
        = self.states[b], self.actions[b], self.states_[b], self.rewards[b], self.dones[b]
        
        return states, actions, states_, rewards, dones
