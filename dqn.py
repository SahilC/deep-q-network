import math
import random
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

# Class defining the Convnet architecture
class dqn(nn.Module):
    def __init__(self):
        super(dqn,self).__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(448,2)

    # Feedforward definition with ReLu activations
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0),-1))


EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 200
steps_done = 0

def select_action(state,model):
    global steps_done
    sample = random.random()
    
    # Threshold defines with which probability should we move left/right
    eps_threshold = EPS_END + (EPS_START - EPS_END)* math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state.type(dtype), volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(2)]])

def plot_durations(episode_durations):
    plt.figure(1)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    #display.clear_output(wait=True)
    #display.display(plt.gcf())

