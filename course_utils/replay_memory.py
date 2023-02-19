import numpy as np

class ReplayBuffer():
    def __init__(self, capacity, inputDims):
        self.capacity = capacity
        self.nMemories = 0
            
        self.states = np.zeros(
            (self.capacity, *inputDims), dtype = np.float32)
        self.states_ = np.zeros(
            (self.capacity, *inputDims), dtype = np.float32)
        
        self.actions = np.zeros(
            self.capacity, dtype = np.int64)
        self.rewards = np.zeros(
            self.capacity, dtype = np.float32)
        self.dones = np.zeros(
            self.capacity, dtype = np.uint8)
    
    def storeTransition(self, state, action, reward, state_, done):
        i = self.nMemories % self.capacity
        
        #print("REPLAY MEM PY L23 shape of state, state_", state.shape, state_.shape)
            
        self.states[i]  = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.states_[i] = state_       
        self.dones[i]   = done
        self.nMemories += 1
    
    def sampleBuffer(self, batchSize):
        imax = min(self.capacity, self.nMemories)
        b = np.random.choice(imax, batchSize, replace = False)
        
        \
               states,         actions,         rewards,         states_,         dones\
        = self.states[b], self.actions[b], self.rewards[b], self.states_[b], self.dones[b]
        
        return states, actions, rewards, states_, dones

