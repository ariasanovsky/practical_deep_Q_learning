Following the course, I begin with a naive AI that navigates the state space with uniformly random actions.  

Next, we explore Markov decision processes.  From a state, an action may nondeterministically transition to a new state (and receive a reward).  For all valid pairs of state and action, 
```math
    \sum_{s', r}\mathbb{P}[s', r\; | \; a, s] = 1.  
```
Furthermore the expected reward from a state given an action is 
```math
    r(s, a) := \mathbb{E}[R_t \; | \; S_{t-1} = s\text{ and } A_{t-1} = a]
    = \sum_{r\in R}\sum_{s'\in S}\mathbb{P}[s', r\; | \; s, a].  
```

Total return up to some final time is defined by 
```math
    G_t = R_{t+1} + \cdots + R_T.  
```

And the terminal state reached in an episode yields no future returns (```math G_T = 0```).  
