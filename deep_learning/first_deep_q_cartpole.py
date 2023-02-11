import gym
from first_deep_q_agent import Agent

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    nGames =   10000
    chunkSize = 1000
    blockSize =  100
    
    
    winPcts = []
    scores = []
    
    
    agent = Agent(nActions = env.action_space.n, inputDims = env.observation_space.shape)
    
    for i in range(nGames):
        done = False
        state = env.reset()
        score = 0
        
        while not done:
            action = Agent.chooseAction(state)
            state_, reward, done, info = env.step(action)
            Agent.updateQ(state, action, reward, state_)
            score += reward

        # log results
        if i % blockSize == 0:
            winPct = np.mean(scores[-blockSize:])
            winPcts.append(winPct)
        
        if i % chunkSize == 0:
            print(
                "episode", i, 
                "winPct %.2f" % winPct, 
                "eps %.2f"    % agent.eps)

        plt.plot(winPcts)
        plt.show()
