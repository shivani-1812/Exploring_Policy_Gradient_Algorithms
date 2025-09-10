import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch

rows, cols = 5, 5
gamma = 0.925    
food_reward = 10
monster_penalty = -8
num_actions = 4
terminal_state = (4, 4)
monster_states = [(0, 3), (4, 1)]
forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
starting_states = [
    (i, j)
    for i in range(rows)
    for j in range(cols)
    if (i, j) not in forbidden_furniture and (i, j) != terminal_state
]
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
episode_rewards = [] 

theta = torch.randn((rows * cols, num_actions), requires_grad=True)  
w = torch.zeros(rows * cols, requires_grad=True)   
optimizer_theta = torch.optim.Adam([theta], lr=1e-3)
optimizer_w = torch.optim.Adam([w], lr=1e-3)

def get_next_state(current_state, intended_action):
    next_action = [
        intended_action,
        (-intended_action[1], intended_action[0]),  
        (intended_action[1], -intended_action[0]), 
        (0, 0),   
    ]
    action_taken = random.choices(next_action, weights=[0.7, 0.12, 0.12, 0.06], k=1)[0]
    nsx = current_state[0] + action_taken[0]
    nsy = current_state[1] + action_taken[1]
    if 0 <= nsx < rows and 0 <= nsy < cols and (nsx, nsy) not in forbidden_furniture:
        return (nsx, nsy)
    return current_state

def get_reward(next_state):
    if next_state in monster_states:
        return monster_penalty
    elif next_state == terminal_state:
        return food_reward
    else:
        return -0.05

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))   
    return exp_x / torch.sum(exp_x)
 
def state_to_index(state):
    return state[0] * cols + state[1]

def reinforce_baseline():
    total_timesteps = 0
    timestep=[]
    rewards = []
    for itr in range(50000):
        episode = []
        current_state = random.choice(starting_states)
        total_reward = 0
        while current_state != terminal_state:
            state_index = state_to_index(current_state)
            logits = theta[state_index]
            probs = softmax(logits)
            action = random.choices(range(num_actions), weights=probs.detach().numpy())[0]
            next_state = get_next_state(current_state, actions[action])
            reward = get_reward(next_state)
            episode.append((current_state, action, reward))
            total_reward += reward
            current_state = next_state
            total_timesteps+=1
        timestep.append(total_timesteps)
        episode_rewards.append(total_reward)
        G = []
        g = 0
        for _, _, reward in reversed(episode):
            g = reward + gamma * g
            G.insert(0, g)
        G = torch.tensor(G, dtype=torch.float32)
        
        optimizer_theta.zero_grad()
        optimizer_w.zero_grad()

        for t, (state, action, reward) in enumerate(episode):
            G_t = G[t]
            state_index = state_to_index(state)
            v_w_s = w[state_index]
            advantage = G_t - v_w_s

            logits = theta[state_index]
            probs = softmax(logits)
            log_prob = torch.log(probs[action])
    
            policy_loss = -log_prob * advantage.detach()   
            policy_loss.backward(retain_graph=True)
    
            baseline_loss = (v_w_s - G_t).pow(2)
            baseline_loss.backward(retain_graph=True)

        optimizer_theta.step()
        optimizer_w.step()

        if itr % 100 == 0:
            print(f"Iteration {itr}, Total Reward: {total_reward:.2f}") 
            rewards.append(total_reward)

            
    print()
    print()
    print("Estimated Value function")
    for i in range(rows):
        for j in range(cols):
            st_i = state_to_index((i,j))
            print(format(w[st_i].item(),".4f"), end=" ")
        print()

    plt.figure(figsize=(10, 6))
    plt.plot(timestep, range(50000), color="blue")
    plt.xlabel("No. of timesteps")
    plt.ylabel("Episode number")
    plt.title("Learning Curve")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Return", color ='black')

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Reward obtained every 100th episode")
    plt.legend()
    plt.grid()
    plt.show()

reinforce_baseline()