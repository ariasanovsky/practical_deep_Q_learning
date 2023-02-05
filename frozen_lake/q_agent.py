import numpy as np

class Agent():
    def __init__(self, 
        learningRate, gamma, 
        eps0, epsMin, epsDecrement, 
        nActions, nStates, 
    ):
        
        # capture hyperparameters
        self.learningRate = learningRate
        self.gamma = gamma
        self.nActions = nActions
        self.nStates = nStates
        self.eps = eps0
        self.epsMin = epsMin
        self.epsDecrement = epsDecrement

        # initialize Q table (dictionary)
        self.Q = {}
        self.init_Q()

    # initializer for Q table
    def init_Q(self):
        for state in range(self.nStates):
            for action in range(self.nActions):
                self.Q[(state, action)] = 0.0
    
    def qValuesFromState(self, state):
        return np.array([self.Q[(state, action)] \
                for action in range(self.nActions)])

    def chooseAction(self, state):

        # uniformly random action per eps?
        if np.random.random() < self.eps:
            action = np.random.choice(range(self.nActions))
        
        # else, take an optimal action
        else:
            qValues = self.qValuesFromState(state)
            action = np.argmax(qValues) # note: not a random argmax
        
        return action
    
    def decrementEpsilon(self):
        self.eps = self.eps*self.epsDecrement if self.eps > self.epsMin \
                   else self.epsMin
    
    def updateQ(self, state, action, reward, state_):
        qValues = self.qValuesFromState(state_)
        aMax = np.argmax(qValues)

        self.Q[(state, action)] += self.learningRate * \
            (reward 
            + self.gamma * (
                self.Q[(state_, aMax)]
                - self.Q[(state, action)]
            ))
        
        self.decrementEpsilon()