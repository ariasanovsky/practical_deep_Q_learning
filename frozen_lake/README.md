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
\Pi(s, a) = \text{ probability of selecting action }a\text{ from state }s
```

Each state has a value defined by

```math
\begin{align*}
    q_\Pi(s, a)
    &:=\mathbb{E}_\Pi\left[
        G_t \; | \; S_t = s\text{ and }A_t = a
    \right] 
    \\
    &= \mathbb{E}_\Pi\left[
        \sum_{k=0}^\infty \gamma^k\cdot R_{t+k+1} \; | \; S_t = s\text{ and }A_t = a
    \right]
\end{align*}
```

and each state, action pair has a value

```math
\begin{align*}
    v_\Pi(s) 
    &:= \mathbb{E}_\Pi[G_t \; | \; S_t = s]
    \\
    &= \mathbb{E}_\Pi\left[
        \sum_{k=0}^\infty\gamma^k\cdot R_{t+k+1} \; | \; S_t = s
    \right].  
\end{align*}
```

Crucially, we have the relation

```math
    v_\Pi(s)
    = \sum_a q_\Pi(s, a)
    \cdot \mathbb{P}_\Pi[A_t = a\; | \; S_t = s].  
```
