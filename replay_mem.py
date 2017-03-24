import random
from collections import namedtuple

# Class representing a circular buffer which stores states and rewards
class replayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0
        self.Transitions = namedtuple('Transition',('state','action','next_state','reward'))

    # Push - Method that overwrites   
    def push(self, *args):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transitions(*args)
        self.position = (self.position + 1) % self.capacity

    # Randomly samples the transitions
    def sample(self, batch_size):
        return  random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)
