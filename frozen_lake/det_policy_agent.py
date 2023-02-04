import gym
import numpy as np
import matplotlib.pyplot as plt

'''
the frozen lake
    S * * *     S : starting position
    * X * X     * : valid position (may cause slipping)
    * * * X     X : hole, ends game
    X * * G     G : goal, also ends game

actions
    L, D, R, U <--> 
    0, 1, 2, 3
'''

# assuming 15 states (keys) read left-to-right, top-to-bottom
# values are the actions from the states
# deterministic policy
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

# create environment
env = gym.make("FrozenLake-v1")

# number of episodes and logging values
n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    # initializing episode
    done = False
    obs = env.reset()
    score = 0

    while not done:
        # follow policy until actions taken
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        score += reward
    
    # log results
    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show() 