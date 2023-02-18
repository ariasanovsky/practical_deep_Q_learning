import numpy as np
import gym

''' wrapped env where an action in the old env repeats, 
    thus observation space is scaled '''
class EnvironmentRepeater(gym.env.Wrapper):
    def __init__(self, env, nRepeats):
        self.env = env
        self.nRepeats = nRepeats
        self.obsBuffer = np.zeros((self.nRepeats, *self.env.observation_space.shape))
    
    def step(self, action):
        totalReward = 0
        done = False
        # repeat the action n times
        for i in range(self.nRepeats):
            state_, reward, done, info = self.env.step(action)
            totalReward += reward
            self.obsBuffer[i] = state_
            if done:
                break
        maxFrame = np.empty(self.env.observation_space.shape)
        return maxFrame, totalReward, done, info

    def reset(self):
        self.env.reset()
        self.obsBuffer = np.zeros((self.nRepeats, *self.env.observation_space.shape))
    
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, newShape):
        #set shape by swapping channels axis
        #set observation space to new shape using gym.spaces.Box (0 to 1.0)
        return
        
    #def observation(self, observation: _Operation) -> _Operation:
    #    return super().observation(observation)
    def observation(self, observation):
        return
        #convert to grayscale
        #resize observation to new shape
        #convert obs to numpy array
        #move channel axis from pos 2 to 0
        #/=255
        #return obs
        return
    
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stackSize):
        #init new obs space 
        return

    def reset(self):
        return

    def observation(self, observation):
        return
    
def make_env(envName, newShape, stackSize):
    return