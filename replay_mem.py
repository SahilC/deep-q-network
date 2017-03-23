Transitions = namedTuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
