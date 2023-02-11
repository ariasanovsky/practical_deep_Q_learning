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