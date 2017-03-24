import gym
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable
from replay_mem import *
from dqn import dqn
from dqn import select_action
from dqn import plot_duration
from replay_mem import replayMemory

# Initialize OpenAI's gym cartpole environment
env = gym.make('CartPole-v0')
env = env.unwrapped

# Variable initialization 
last_sync = 0
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

model = dqn()
optimizer =  optim.RMSProp(model.parameters())

model.type(dtype)
steps_done = 0
episode_durations = []

# named tuple to store transitions
Transitions = namedTuple('Transition',('state','action','next_state','reward'))
mem = replayMemory(10000)





