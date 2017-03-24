import torch 

from itertools import count
from dqn import select_action
from dqn import plot_durations
from input_proc import get_screen
from torch.autograd import Variable 
from replay_mem import replayMemory
import torch.nn.functional as F



# Variable initialization 
last_sync = 0
BATCH_SIZE = 128
GAMMA = 0.999
episode_durations = []
mem = replayMemory(10000)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def opt_model(model, optimizer):

	# Return if incorrect batch size
	if len(mem) < BATCH_SIZE:
		return

	# randomly sample a batch of transition states
	transitions = mem.sample(BATCH_SIZE)
	batch = mem.Transitions(*zip(*transitions)) 

	##
	## Builds the computation graph for Q-learning
	##


	non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

	non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)


	state_batch = Variable(torch.cat(batch.state))
	action_batch = Variable(torch.cat(batch.action))
	reward_batch = Variable(torch.cat(batch.reward))
	next_state_values = Variable(torch.zeros(BATCH_SIZE))

	# Compute for each action, predict the probabilities of each current state
	state_action_values = model(state_batch).gather(1,action_batch).cpu()

	# Temporary volatile variable
	non_final_next_states = Variable(non_final_next_states_t,volatile=True)

	# have your model predict the maximum value of your next state
	next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].cpu()

	next_state_values.volitile = False

	# Compute the discounted reward for the next states
	expected_state_action_values = (next_state_values*GAMMA) + reward_batch

	# This function implements huber loss from pytorch
	# Compute the loss such that the difference between predicted probs and expected is minimized
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

	optimizer.zero_grad()
	loss.backward()

	#Clamps the gradients to (-1,1) in-place 
	for p in model.parameters():
		p.grad.data.clamp_(-1,1)

	#Update variables from back-prop
	optimizer.step()

def train(env,model,optimizer):
	for i in count(1):
		env.reset()
		# Build state of current as difference of two frames
		last_screen = get_screen(env)
		current_screen = get_screen(env)
		state = (current_screen - last_screen)
		print("Iteration:",i)

		# Try till the game fails
		for t in count():

			# Select an action at random from memory
			action = select_action(state,model)

			# See effect of action on environment
			_, reward, done, _ = env.step(action[0,0])
			reward = torch.Tensor([reward])

			# If the game ends, then exit loop, else update state
			if not done:
				last_screen = current_screen
				current_screen = get_screen(env)
				next_state = current_screen - last_screen
			else:
				next_state = None

			# Push new state into memory
			mem.push(state,action,next_state,reward)

			state = next_state

			# Update model parameters with new state
			opt_model(model, optimizer)

			if done:
				episode_durations.append(t+1)
				plot_durations(episode_durations)
				break