# Deep-q-network
A deep-q-network(A neural network for reinforcement learning) implemented using pyTorch. Built following the official pytorch tutorial, for openAI's MountainCar-v0 environment.  

Reinforcement learning is probably our greatest invention. It's amazing to see such simple algorithms that have the power to learn correlations between an agent's actions and rewards, which can simply be adapted a wide variety of scenarios.

This code uses OpenAI's gym environment for a simulated agent's enviroment. To modify this code for a different OpenAI environment, follow the next three simple steps:-

* Modify input_proc.py to get your agent's position
* In dqn.py, modify the convolution network to have the correct number of actions
* Change the environment defined in pydqn.py

and that's it! 


# References

* PyTorch tutorial - https://github.com/pytorch/tutorials
* OpenAI gym - https://github.com/openai/gym
* Q-learning - https://www-s.acm.illinois.edu/sigart/docs/QLearning.pdf
* Playing Atari with Deep Reinforcement Learning - Google Deepmind - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* Deep Reinforcement Learning: Pong from Pixels - Andrej Karpathy - https://karpathy.github.io/2016/05/31/rl/
