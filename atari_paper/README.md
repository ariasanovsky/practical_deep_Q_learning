# Atari paper

Here are some questions to consider when reading the DeepMind paper [*Human-level control through deep reinforcement learning*](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

## Overall take

I'm surprised how simple the ideas in the paper are.  There's no standout brilliant idea.  It's a lot of cleverness piled on incrementally, likely through discussion in a lab for extended periods with smart people combing the literature and meditating and of course benchmarking.  Smaller labs simply don't have the resources to overcome these same obstacles all at once.  

I'm impressed at how elegantly it can be explained.  The heuristics employed aren't localized to a single problem instance and are made in a way which covers a wider scope.

## What algorithm?

1. Deep Q Learning with a CNN on the front to process a screen image.  
2. Replay memory keeps a set of prior observed transitions (SARS') to avoid biasing to the states seen by the current policy.
3. States are actually a history of the last few states and actions.
4. Keep a separate a-v function for targeting that updates slower (no hitting a moving target, stabilization).
5. Make the reward function more naive ($-1$ bad, $0$ neutral, $+1$ good).
6. Clip the error on the Q learnning update step.  Very Lipschitz, it's clever.
7. Inherently off-policy.

## What data structures?

1. $2\times$ CNNs for the a-v function and target, $Q,\hat{Q}$
2. $1\times$ bounded capacity replay memory collection $D$ (any collection, really)
3. $1\times$ squishing function to collapse full history to recent history
4. $1\times$ squishing/convolution functions to hanndle the screen image
5. $1\times$ random number generator
6. $1\times$ widget for performing stochastic gradient descent w.r.t. $D$ and forward propogating updates to $Q$

## What model architecture?

In math notation, $\text{NN}:(\text{history})\to (\text{action-values})$, i.e., the history functional $\phi\mapsto Q(\phi, \cdot)_{a\in A}$.

0. **CNN input**: processed recent pixel pictures and recent actions
    - $84\times 84\times 44 \mapsto_{8\times 8; 4} \cdots \mapsto_{4\times 4; 2} \cdots \mapsto_{3\times 3; 1}512\text{ (and some recitification)}$
1. **CNN output/NN input**: convoluted pictures
2. **NN output**: $4$ to $18$ neurons corresponding to actions, recording the corresponding action-value

## What hyperparameters?

Other NN parameters:

1. agent history length $=4$: number of recent frames in the history input to the CNN

Q parameters:

1. discount factor $\alpha = 0.99$: bias for recent or latent rewards
2. greedy terms: $\varepsilon_0 = 1.0$, $\varepsilon_\infty = 0.1$, final exploration frame $10^{6}$
3. learning rate $\gamma = 0.00025 = 2.5\times 10^{-4}$: used by RMSProp since this is deep Q learning
4. gradient momentum $=0.95$: used by RMSProp, idk
5. squared gradient (denominator) momentum $=0.95$: also used by RMSProp, idk
6. min squared gradient $=0.01$: constant added to denominator of RMPSProp update, idk

SGD parameters:

1. minibatch size $=32$: how many samples from the replay memory to use
2. replay memory size $=10^{6}$: how many frames to look back when sampling
3. target network update frequency $=10^{4}$: how many $Q$ updates needed to prompt a $\hat{Q}$ update
4. action repeat $=4$: how many frames to repeat an action
5. update frequency $=4$: how many action selections between SGD updates to $Q$

Misc parameters:

1. replay start size $=5\cdot 10^{4}$: how many frames to run a uniformly random policy before learning (also to populate the replay memory)
2. no-op max $=30$: maximum number of do-nothing actions performed at the start of an episode

## What results to expect?

We expect $Q$ to be much more stable (measured by calculating updated values over the replay memory) with a general improvement in average score over time.  Also, I lovee these $t$-SNE graphs.  I would love to learn how to make those.

## Implementating the paper

We'll give it a try implementing the full problem.  For the image:

1. convert RGB to grayscale (3 channels to 1)
2. downscale picture to $84\times 84$
3. take max of previous 2 grayscale frames (emulation flickering)
4. repeat each action 4 times
5. swap channels to first position (transposing the image array)
6. stack 4 frames
7. scale inputs to fit in $[0,1]$

We'll use [gym wrappers](https://github.com/openai/gym/tree/master/gym/wrappers) for best practice with tweaking the environment.  We will:

1. override the initializer and have to call the `super` class initializer
2. override the `step` and `reset` functions
3. access the original environment with `self.env`
4. use `gym.ObservationWrapper` in addition to `gym.Wrapper`
