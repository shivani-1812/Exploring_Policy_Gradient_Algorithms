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

def one_step_actor_critic(num_episodes=20000):
    rewards_per_episode = []
    timesteps = 0

    for episode in range(num_episodes):
        current_state = random.choice(starting_states)
        I = 1   
        total_reward = 0

        while current_state != terminal_state:
            state_index = state_to_index(current_state)
            logits = theta[state_index]
            probs = softmax(logits)
            action = random.choices(range(num_actions), weights=probs.detach().numpy())[0]

            next_state = get_next_state(current_state, actions[action])
            reward = get_reward(next_state)
            total_reward += reward
            timesteps += 1
 
            next_state_index = state_to_index(next_state)
            v_s = w[state_index]
            v_s_next = w[next_state_index] if next_state != terminal_state else 0
            advantage = reward + gamma * v_s_next - v_s 
            optimizer_w.zero_grad()
            value_loss = -advantage * v_s  
            value_loss.backward(retain_graph=True)
            optimizer_w.step() 
            optimizer_theta.zero_grad()
            log_prob = torch.log(probs[action])
            policy_loss = -log_prob * advantage * I  
            policy_loss.backward(retain_graph=True)
            optimizer_theta.step()
 
            I *= gamma
            current_state = next_state
        rewards_per_episode.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")


    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode, label="Total Return every Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title("One-Step Actor-Critic Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # Print estimated value function
    print("\nEstimated Value Function:")
    for i in range(rows):
        for j in range(cols):
            print(f"{w[state_to_index((i, j))].item():.2f}", end=" ")
        print()

# Run the algorithm
one_step_actor_critic()
