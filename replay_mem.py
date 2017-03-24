import random

class replayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position += 1

    def sample(self, batch_size):
        return  random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)
