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

env = gym.make('CartPole-v0')
env = env.unwrapped

