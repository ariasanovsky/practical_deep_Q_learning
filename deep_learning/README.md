# Deep learning

## Boilerplate code

In this section, we begin some boilerplate code that familiarizes us with neural networks as implemented in pytorch.
Neural nets model real-valued functions with layers, starting with the argments of the function, layered with intermediate variables connected by RELU or other activations.
We fit functions to data by following gradient descent or some other protocol to minimize loss.  It is easy to see that NNs have very simple derivatives which are all handled on the backend.
{Back/Forward}propopogation is a fancy way to say we are passing values {down/up} the neural network.

## First Deep-Q example

We separately build a network class and an agent class.  We will use MSELoss, 2 linear layers, and Relu activation.

## Results

The results are quite poor!  Possible reasons for failure:

1. The NN updates from each example instead of learning from several at once.  In short, we're fitting a function to only one data point as opposed to fitting an ensemble of representative points.
2. For the number of parameters, we sample very few points.
3. It explores very little too quickly.
4. The same network is used to choose the next action and to evaluate the best action from a state.  
5. The Q-learning algorithm biases towards the best actions from each state in its learning.
