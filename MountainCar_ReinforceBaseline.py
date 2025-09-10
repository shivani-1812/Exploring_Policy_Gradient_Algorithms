import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

action_space = [0, 1, 2]
force = 0.001
gravity = 0.0025
state_dim = 2
action_dim = 3
num_episodes = 5000
class policy_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_probabilities = nn.Linear(128, action_dim)

    def forward(self, state, temperature=1.0):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.action_probabilities(x)
        action_probs = torch.softmax(logits / temperature, dim=-1)  
        return action_probs

class value_network(nn.Module):
    def __init__(self, state_dim):
        super(value_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

def reset(): 
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0.0
    return [position, velocity]

def step(state, action):   
    position, velocity = state
    velocity += (action - 1) * force - np.cos(3 * position) * gravity
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)
    if position == -1.2 and velocity < 0:
        velocity = 0
    return [position, velocity]

def get_reward(state):        
    if state[0] >= 0.5:
        #print("COMPLETE")
        return 0
    return -1

'''def get_reward(state):
    if state[0] >= 0.5: 
        print("COMPLETe")
        return 2.0
    else:
        return (state[0]+1.2)/1.8 - 1.0'''

def compute_returns(rewards, gamma=0.99): 
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def train_step(states, actions, returns, policy_net, value_net, policy_optimizer, value_optimizer):
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    values = value_net(states_tensor).squeeze()
    advantages = returns_tensor - values.detach()

    value_loss = nn.MSELoss()(values, returns_tensor)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    action_probs = policy_net(states_tensor)
    action_log_probs = torch.log(action_probs.gather(1, actions_tensor.view(-1, 1)).squeeze())
    policy_loss = -(action_log_probs * advantages).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

     
def reinforce_baseline():
    policy_net = policy_network(state_dim, action_dim)
    value_net = value_network(state_dim)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    total_rewards = []

    for episode in range(num_episodes):
        state = reset()
        states, actions, rewards = [], [], []
        for t in range(2000):   
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action_probs = policy_net(state_tensor, temperature=0.995)  
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            
            next_state = step(state, action)
            reward = get_reward(next_state)   

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if state[0] >= 0.5:
                break
 
        returns = compute_returns(rewards)
        train_step(states, actions, returns, policy_net, value_net, policy_optimizer, value_optimizer)
        total_rewards.append(sum(rewards))
        if episode % 100 == 0:     
            print(f"Episode {episode}: Total Reward = {total_rewards[-1]}")

 
    plt.plot(total_rewards, color ='black')
    plt.xlabel("Episode")
    plt.ylabel("Total Return")
    plt.title("Total Return vs Episode")
    plt.show()


reinforce_baseline()
