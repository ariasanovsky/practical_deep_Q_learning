import gym
import matplotlib.pyplot as plt
import numpy as np

from q_agent import Agent

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4")
    agent = Agent(
        learningRate = 0.001,
        gamma        = 0.9, 
        eps0         = 1.0, 
        epsMin       = 0.01, 
        epsDecrement = 0.99999995,
        nActions     = 4, 
        nStates      = 16)

    scores = []
    winPcts = []
    
    nGames = 5000000
    for i in range(nGames):
        done = False
        observation = env.reset() # did the return change in a revision?
        score = 0

        # take actions until done, updating Q and score as well
        while not done:
            action = agent.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            agent.updateQ(observation, action, reward, observation_)
            score += reward
            observation = observation_
        
        # record results
        scores.append(score)
        if i % 1000 == 0:
            winPct = np.mean(scores[-1000:])
            winPcts.append(winPct)
            
            # show more details
            if i % 10000 == 0:
                print(
                    "episode", i, 
                    "winPct %.2f" % winPct, 
                    "eps %.2f"    % agent.eps, 
                )
    plt.plot(winPcts)
    plt.show()
