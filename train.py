last_sync = 0
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

model = dqn()
mem = ReplayMemory(10000)
optimizer =  optim.RMSProp(model.parameters())

model.type(dtype)
steps_done = 0
episode_durations = []

def opt_model():
	global last_sync
	if len(memory) < BATCH_SIZE:
		return

	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions)) 