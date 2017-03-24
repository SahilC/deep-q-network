from input_proc import get_screen

def opt_model():
	global last_sync
	if len(memory) < BATCH_SIZE:
		return

	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions)) 

	##
	## Builds the computation graph for Q-learning
	##

	non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

	non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)

	non_final_next_states = Variable(non_final_next_states_t,volitile=True)
	state_batch = Variable(torch.cat(batch.state))
	action_batch = Variable(torch.cat(batch.action))
	reward_batch = Variable(torch.cat(batch.reward))

	state_action_values = model(state_batch).gather(1,action_batch).cpu()

	next_state_values = Variable(torch.zeros(BATCH_SIZE))
	next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].cpu()
	next_state_values.volitile = False

	expected_state_action_values = (next_state_values*GAMMA) + reward_batch

	# This function implements huber loss from pytorch
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

	optimizer.zero_grad()
	loss.backward()

	#Clamps the gradients to (-1,1) in-place 
	for p in model.parameters():
		p.grad.data.clamp_(-1,1)

	#Update variables from back-prop
	optimizer.step()

def train():
	env.reset()
	last_screen = get_screen()
	current_screen = get_screen()
	state = (current_screen - last_screen)

	for t in count():
		action = select_action(state)
		_, reward, done, _ = env.step(action[0,0])
		reward = torch.Tensor([reward])

		if not done:
			last_screen = current_screen
			current_screen = get_screen()
			next_state = current_screen - last_screen
		else:
			next_state = None

		mem.push(state,action,next_state,reward)


		state = next_state

		opt_model()

		if done:
			episode_durations.append(t+1)
			plot_durations()
			break