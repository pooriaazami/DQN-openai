import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Env import Environment 
from Memory import Memory
from Policy import PolicyNetwork

env = Environment()

NUM_ACTIONS = env.num_actions
BATCH_SIZE = 32
MEMORY_SIZE = 1_000_000
EPSILON = 1.
GAMMA = .99
TAU = 0.005
LR = 1e-4

NUM_EPISODES = 1_000_000
PLOT = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

memory = Memory(MEMORY_SIZE)

policy_net = PolicyNetwork(NUM_ACTIONS).to(DEVICE)
target_net = PolicyNetwork(NUM_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

criterion = nn.MSELoss()
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

steps_done = 0

def select_action(state):
    global steps_done, EPSILON

    p = random.random()
    if p <= EPSILON:
        action = random.randrange(0, NUM_ACTIONS)
    else:
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=DEVICE, dtype=torch.float32)
            action = policy_net(state).max(1).indices.view(1, 1).cpu().squeeze().numpy()

    EPSILON = max(.1, 1 - steps_done / 10**6 * .9)

    steps_done += 1
    return action

def optimize_model():
    global memory, DEVICE, BATCH_SIZE, GAMMA, criterion

    batch = memory.sample(BATCH_SIZE)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_state = torch.tensor(np.array([s for s in batch.next_state if s is not None]), device=DEVICE, dtype=torch.float32)

    state_batch = torch.tensor(np.array(batch.current_state), device=DEVICE, dtype=torch.float32)
    action_batch = torch.tensor(np.array(batch.action), device=DEVICE, dtype=torch.long)
    reward_batch = torch.tensor(np.array(batch.reward), device=DEVICE, dtype=torch.float32)

    state_action_values = policy_net(state_batch).gather(dim=1, index=action_batch.unsqueeze(-1))
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_state).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
def sync_models_weights():
    global target_net, policy_net, TAU

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

def train_model(num_episodes):
    global env, memory, steps_done, PLOT

    reward_log = []
    if PLOT:
        _, game_state_axs = plt.subplots(1, 4)
        _, report_axs = plt.subplots(1, 1)

    for i in range(num_episodes):
        sum_rewards = 0
        state = env.reset()
        while True:
            action = select_action(state)
            new_state, reward, ended = env.step(action)
            sum_rewards += reward

            if PLOT:
                for i in range(4):
                    game_state_axs[i].imshow(new_state[i, :, :], cmap='gray')
            
                plt.pause(.01)

            memory.push(state, reward, action, new_state if not ended else None)
            state = new_state

            if steps_done > BATCH_SIZE:
                optimize_model()
                sync_models_weights()

            if ended:
                reward_log.append(sum_rewards)
                print(sum_rewards)
                if PLOT:
                    report_axs.plot(np.array([reward_log]))
                break

        if (i + 1) % 50 == 0:
            print(f'{i + 1} steps are done') 
            torch.save(policy_net.state_dict(), f'{i + 1}.model')               
                
        if PLOT:
            plt.show()

    torch.save(policy_net.state_dict(), 'final.model')

if __name__ == '__main__':
    train_model(NUM_EPISODES)