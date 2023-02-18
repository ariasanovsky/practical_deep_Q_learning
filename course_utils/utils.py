import collections
import cv2
import gym
import gym

import numpy as np
import matplotlib.pyplot as plt

def plotLearningCurve(xs, scores, epss, fname):
    fig = plt.figure()
    ax  = fig.add_subplot(111, label = "1")
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    
    ax.plot(xs, epss, color = "C0")
    ax.set_xlabel("Training Steps", color = "C0")
    ax.set_ylabel(       "Epsilon", color = "C0")
    ax.tick_params(axis = "x", colors = "C0")
    ax.tick_params(axis = "y", colors = "C0")
    
    N = len(scores)
    runningAvg = np.empty(N)
    for t in range(N):
        runningAvg[t] = np.mean(\
            scores[max(0, t - 100) : t+1]
        )
    
    ax2.scatter(xs, runningAvg, color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color = "C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis = "y", color = "C1")
    
    plt.savefig(fname)

class ProcessingParameters:
    def __init__(self, nRepeats = 4, noOps = 0, 
                 clipRewards = False, fireFirst = False):
        self.nRepeats, self.noOps, self.clipRewards, self.fireFirst\
            =nRepeats,      noOps,      clipRewards,      fireFirst
        

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, params: ProcessingParameters = ProcessingParameters()):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.params = params
        self.shape = env.observation_space.low.shape
        self.frameBuffer = np.zeros_like((2, self.shape))
        
    def step(self, action):
        totalReward = 0.0
        done = False
        for i in range(self.params.nRepeats):
            obs, reward, done, info = self.env.step(action)
            if self.params.clipRewards:
                np.clip((reward,), -1, 1)[0]
            totalReward += reward
            
            parity = i % 2
            self.frameBuffer[parity] = obs
            if done:
                break
        
        maxFrame = np.maximum(self.frameBuffer[0], self.frameBuffer[1])
        return maxFrame, totalReward, done, info

    def reset(self):
        obs = self.env.reset()
        
        if noOps > 0:
            noOps = self.rand.randint(self.params.noOps)  + 1
            for _ in range(noOps):
                _, _, done, _ = self.env.step(0)
                if done:
                    self.env.reset()
        
        if self.params.fireFirst:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)
        
        self.frameBuffer = np.zeros_like((2, self.shape))
        self.frameBuffer[0] = obs
        
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape) -> None:
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = self.shape,
            dtype = np.float32)
    
    def observation(self, observation):
        #grayscale
        newFrame = cv2.cvtColor(observation, cv2.COLOR_GRB2GRAY)
        resizedScreen = cv2.resize(
            newFrame,
            self.shape[1:],
            interpolation = cv2.INTER_AREA)
        observation_ = np.array(resizedScreen, dtype = np.uint8).reshape(self.shape)
        observation_ /= 255.0
        return observation_

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, nRepeats) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat( nRepeats, axis = 0),
            env.observation_space.high.repeat(nRepeats, axis = 0),
            dtype = np.float32)
        self.recentFrames = collections.deque(maxlen = nRepeats)
    
    def reset(self):
        self.recentFrames.clear()
        observation = self.env.reset()
        
        # populating recent frame history
        for _ in range(self.recentFrames.maxlen):
            self.recentFrames.append(observation)
        
        return np.array(self.recentFrames).reshape(self.observation_space.low.shape)
    
    def observation(self, observation):
        self.recentFrames.append(observation)
        return np.array(self.recentFrames).reshape(self.observation_space.low.shape)

def processEnv(envName, shape, params: ProcessingParameters = ProcessingParameters()):
    env = gym.make(envName)
    env = RepeatActionAndMaxFrame(env, params)
    env =         PreprocessFrame(env, shape)
    return            StackFrames(env, params.nRepeats)
