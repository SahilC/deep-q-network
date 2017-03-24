import gym
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as T



from dqn import dqn
from dqn import select_action
from dqn import plot_durations
from input_proc import get_screen
from train import train

# Initialize OpenAI's gym cartpole environment
env = gym.make('CartPole-v0')
env = env.unwrapped


env.reset()

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


model = dqn()
optimizer =  optim.RMSprop(model.parameters())

model.type(dtype)

# named tuple to store transitions

train(env,model,optimizer)



