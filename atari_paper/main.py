import sys
sys.path.append("..")

import numpy as np
from atari_agent import AtariAgent, QParameters

from course_utils.wrappers import processEnv
from course_utils.utils import plotLearningCurve

if __name__ == "__main__":
    gameName = "PongNoFrameskip-v4"
    env = processEnv(gameName)
    bestScore = -np.inf
    loadCheckpoint = False
    
    nGames = 500
    qparams = QParameters(gamma = 0.99, lr = 0.0001, epsDec = 5e-6)
    agent = AtariAgent(
        qParams = qparams,
        nActions = env.action_space.n,
        inputDims = (env.observation_space.shape),
        capacity = 50000, batchSize = 32, replace = 1000,
        algo = "", envName = gameName, checkpointDir = 'tmp\\dqn')
    if loadCheckpoint:
        agent.loadModels()
    
    figureName = agent.algo + "_" + agent.envName\
        + "_lr" + str(agent.qParameters.lr) + "_nGames" + str(nGames)
    figureFileName = "plots\\" + figureName + ".png"
    
    nSteps = 0
    scores, epsHistory, stepCounts = [], [], []
    
    for i in range(nGames):
        done = False
        score = 0
        state = env.reset()
        
        while not done:
            action = agent.chooseAction(state)
            state_, reward, done, info = env.step(action)
            score += reward
            
            if not loadCheckpoint:
                agent.buffer.storeTransition(
                    state, action, reward, state_, int((done)))
                agent.learn()
            
            state = state_
            nSteps += 1
        scores.append(score)
        stepCounts.append(nSteps)
        epsHistory.append(agent.qParameters.eps)
        
        meanScore = np.mean(scores[-100:])
        if meanScore > bestScore:
            if not loadCheckpoint:
                agent.saveModels()
            bestScore = meanScore
        
        print(
            "episode", i,
            "score", score,
            "mean score %.1f best score %.1f eps %.2f"
            % (meanScore, bestScore, agent.qParameters.eps),
            "steps", nSteps
        )
        
    plotLearningCurve(stepCounts, scores, epsHistory, figureFileName)