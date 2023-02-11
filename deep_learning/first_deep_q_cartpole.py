import gym
from first_deep_q_agent import Agent

import numpy as np

''' funny python problem -- I run this program in its current dir
    it can't see the parent directory unless I run from there
    adding the parent directory to path ensure I can see it
''' 
import sys
sys.path.append("..")
from course_utils.plots import plotLearningCurve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    nGames =   10000
    chunkSize = 1000
    blockSize =  100
    
    
    winPcts = []
    scores = []
    epsHistory = []
    
    
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
            state = state_
        
        # log results
        scores.append(score)
        epsHistory.append(agent.eps)
        
        if i % blockSize == 0:
            winPct = np.mean(scores[-blockSize:])
            winPcts.append(winPct)
        
        if i % chunkSize == 0:
            print(
                "episode", i, 
                "winPct %.2f" % winPct, 
                "eps %.2f"    % agent.eps)

        fname = 'cartpole_naive_dqn.png'
        xs = range(1, nGames + 1)
        plotLearningCurve(xs, scores, epsHistory, fname)