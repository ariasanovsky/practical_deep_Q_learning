# imports 
import gym
import numpy as np
import matplotlib.pyplot as plt

# set up the frozen lake environment
env = gym.make("FrozenLake-v1")

n_games = 1000 #really n_episodes
win_pct = []
scores = []

for i in range(n_games):
    # initialization
    done = False
    obs = env.reset()
    score = 0

    while not done:
        # acquire and take action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward

    # log results
    scores.append(score)

    # log average of every 10 results 
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

# plot overall win percentage
plt.plot(win_pct)
plt.show()