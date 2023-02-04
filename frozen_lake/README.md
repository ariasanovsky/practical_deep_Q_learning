Following the course, I begin with a naive AI that navigates the state space with uniformly random actions.  

Next, we explore Markov decision processes.  From a state, an action may nondeterministically transition to a new state (and receive a reward).  

```math
\text{For all valid pairs }(a, s)\text{ of action and state}, 
    \sum_{s', r}\mathbb{P}[(s', r) | (a, s)] = 1.  
```