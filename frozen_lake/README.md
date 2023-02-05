# frozen lake AI

## uniformly random AI

Following the course, I begin with a naive AI that navigates the state space with uniformly random actions.  

## Markov decision processes & policies

Next, we explore Markov decision processes.  From a state, an action may nondeterministically transition to a new state (and receive a reward).  For all valid pairs of state and action,

```math
    \sum_{s', r}\mathbb{P}[s', r\; | \; a, s] = 1.  
```

Furthermore the expected reward from a state given an action is

```math
\begin{align*}
    r(s, a) 
    &:= \mathbb{E}[R_t \; | \; S_{t-1} = s\text{ and } A_{t-1} = a]
    \\
    &= \sum_{r\in R}\sum_{s'\in S}\mathbb{P}[s', r\; | \; s, a].  
\end{align*}
```

Total return up to some final time is defined by

```math
    G_t = R_{t+1} + \cdots + R_T.  
```

And the terminal state reached in an episode yields no future returns, i.e.,

```math
G_T = 0.
```

Not all tasks are episodic, however!  Here, rewards may tend to infinity.  We can resolve this by discounting rewards with a new hyperparameter:

```math
    G_t = \sum_{k=0}^\infty \gamma^k\cdot R_{t+k+1}
```

Several hypotheses are assumed here, that's fine.  The convenient recurrence (actually the definition) is

```math
    G_t = R_{t+1} + \gamma G_{t+1}.  
```

Lots of ugly notation is suppressed, which is great.  

## Value functions

The agent's policy maps states to action, probabilistically.

```math
\pi(s, a) = \text{ probability of selecting action }a\text{ from state }s
```

Each state has a value defined by

```math
\begin{align*}
    q_\pi(s, a)
    &:=\mathbb{E}_\pi\left[
        G_t \; | \; S_t = s\text{ and }A_t = a
    \right] 
    \\
    &= \mathbb{E}_\pi\left[
        \sum_{k=0}^\infty \gamma^k\cdot R_{t+k+1} \; | \; S_t = s\text{ and }A_t = a
    \right]
\end{align*}
```

and each state, action pair has a value

```math
\begin{align*}
    v_\pi(s) 
    &:= \mathbb{E}_\pi[G_t \; | \; S_t = s]
    \\
    &= \mathbb{E}_\pi\left[
        \sum_{k=0}^\infty\gamma^k\cdot R_{t+k+1} \; | \; S_t = s
    \right].  
\end{align*}
```

Crucially, we have the relation

```math
    v_\pi(s)
    = \sum_a q_\pi(s, a)
    \cdot \mathbb{P}_\pi[A_t = a\; | \; S_t = s].  
```

With a large state & action space, it's impossible to find these probabilities by brute force.  In practice, we parametrize and monte carlo to estimate them.  
Unpacking terms gives us the Bellman Equation, which we state explicitly here:

```math
    v_\pi(s) = \sum_a\pi(a,s)
    \cdot\sum_{s',r}\mathbb{P}[s',r\; | \; s,a]\cdot\left(r + \gamma v_\pi(s')\right)
```

This equation is the heart of RL.  It lets us recursively monte carlo the value function w.r.t. a policy.  
Policies are paratially ordered entrywise over the state space, and maximal policies are optimal policies.
Optimal policy equations:

```math
\begin{align*}
v_*(s)
&=
    \max_a q_{\pi_*}(s,a)
\\
&=
    \max_a
    \mathbb{E}_{\pi_*}\left[
        G_t\; | \; S_t = s\text{ and }A_t = a
    \right]
\\
&=
    \mathbb{E}_{\pi_*}\left[
        R_{t+1} + \gamma\cdot G_{t+1}\; | \; S_t = s\text{ and }A_t = a
    \right]
\\
&=
    \mathbb{E}_{\pi_*}\left[
        R_{t+1} + \gamma\cdot v_*(S_{t+1})
        \; | \; S_t = s\text{ and }A_t = a
    \right]
\\
&=
\max_a
\sum_{s', r}
    \mathbb{P}\left[
        s', r 
        \; | \;
        s, a
    \right]
    \cdot\left(
        r + \gamma\cdot v_*(s')
    \right)
\end{align*}
```

Similarly,

```math
\begin{align*}
q_*(s, a)
&=
\max_a
\sum_{s', r}
    \mathbb{P}\left[
        s', r 
        \; | \;
        s, a
    \right]
    \cdot\left(
        r + \gamma\cdot \max_{a'} q_*(s', a')
    \right).  
\end{align*}
```

## Implementation heuristics

Explore exploit dilemma is how much to explore the space to monte carlo it vs greedily selecting what is believed to be optimal.
E.g., when exploring a maze, estimate a value of zero for each state but penalize each step with no reward to counteract.
This reduces to a BSF.

Epsilon-greedy gradually cools down a parameter that makes the agent sometimes take a random actionn.
We must keep the parameter positive to appropriately monte carlo the space.

## Temporal difference learning

Update value function at each step using finite differences scaled to step size, which often shrinks over time.  How often to update depends on the algorithm of choice.  Monte carlo updates per episode.
Q learning updates at each time step and is thus `online`.  

With another hyperparameter, the Q learning algorithm updates with:

```math
v_(s_t) 
= v(s_t) 
+ \alpha
\cdot\left(
    R_{t+1}
    +\gamma
    \cdot v(s_{t+1} - v(s_t))
\right).  

```

Under appropriate assumptions, the value function converges to the correct one.
Backtracking to the action-value function:

```math
Q(s_t, a_t)
=
    Q(s_t, a_t)
    + \alpha\cdot\left(
        R_{t+1} + \gamma\cdot\max_{a} Q(S_{t+1}, a)
        - Q(s_t, a_t)
    \right)
```

Typo?

## Tabular estimates in the frozen lake

Since the state space is so small, we can just epsilon-greedy it.
Side note that this is off-policy learning since we use the old policy to get the value function of the updated policy.  SARSA on-policy learning seems harder to implement.  Also if a policy doesn't change much between states, off-policy learning sounds fine in a big state space.

## Recap of Q learning

1. Initialize an initial guess for the value function with zero value on terminal states.
1. Initiialize some hyperparameters that control Q learning.

A step of an episode looks like...

1. Choose an action with epsilon-greedy.
1. Get the new state and reward.
1. Update estimate for action-value function.
