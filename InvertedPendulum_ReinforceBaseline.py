import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class pendulum:
    def __init__(self):
        self.G = 10
        self.L = 1
        self.M = 1
        self.MAX_SPEED = 8
        self.MAX_TORQUE = 2
        self.dt = 0.05
        angle = np.random.uniform(((-5 * np.pi )/ 6), ((5 * np.pi) / 6))
        angular_velocity = np.random.uniform(-1,1)
        self.state =np.array([angle,angular_velocity], dtype=np.float64)
    def step(self, action):
        omega_double_dot = (((3*self.G)/(2*self.L))*np.sin(self.state[0]))+((3*action)/(self.M*pow(self.L,2)))
        temp = self.state[1] + (omega_double_dot*self.dt)
        #new_angular_velocity = max(min(temp, self.MAX_SPEED), (-self.MAX_SPEED))
        new_angular_velocity = np.clip(temp, -self.MAX_SPEED, self.MAX_SPEED)
        new_angle= self.state[0]+(new_angular_velocity*self.dt)
        self.state =np.array([new_angle,new_angular_velocity], dtype=np.float64)
        normalized_angle =((self.state[0]+np.pi)%(2*np.pi))-np.pi
        reward = -(pow(normalized_angle,2)+(0.1 * pow(self.state[1],2)) + 0.001*pow(action,2))
        return self.state, reward
    def reset(self):
        angle = np.random.uniform(((-5 * np.pi )/ 6), ((5 * np.pi) / 6))
        angular_velocity = np.random.uniform(-1,1)
        self.state =np.array([angle,angular_velocity], dtype=np.float64)
        return self.state


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

state_dim = 2
action_dim = 1

def generate_episode(env, policy_net):
    states, actions, rewards = [], [], []
    state = env.reset()

    for i in range(200):
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        mean, std = policy_net(state_tensor)
        action = torch.normal(mean, std).detach().numpy()[0]
        next_state, reward = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return states, actions, rewards

def compute_returns(rewards, gamma=1.0):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def train_step(states, actions, returns, policy_net, value_net, policy_optimizer, value_optimizer):
    #states = [np.array(state).flatten() for state in states]
    states_tensor = torch.tensor((states), dtype=torch.float32)
    #actions = [np.array(action).flatten() for action in actions]
    actions_tensor = torch.tensor((actions), dtype=torch.float32)
    #returns = [np.array(return1).flatten() for return1 in returns]
    returns_tensor = torch.tensor((returns), dtype=torch.float32)
    # Compute state-value estimates
    values = value_net(states_tensor).squeeze()
    advantages = returns_tensor - values.detach()

    value_loss = nn.MSELoss()(values.unsqueeze(-1), returns_tensor)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    mean, std = policy_net(states_tensor)
    action_dist = torch.distributions.Normal(mean, std)
    log_probs = action_dist.log_prob(actions_tensor).sum(dim=-1)
    policy_loss = -(log_probs * advantages).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

num_runs = 20
num_episodes = 8000
returns_over_time = np.zeros((num_runs, num_episodes))
env = pendulum()
for run in range(num_runs):
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    for episode in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy_net)
        returns = compute_returns(rewards)
        train_step(states, actions, returns, policy_net, value_net, policy_optimizer, value_optimizer)
        returns_over_time[run, episode] = sum(rewards).item()
    total_return = np.sum(returns_over_time[run])
    first_return = returns_over_time[run, 0]
    final_return = returns_over_time[run, -1]
    print(f"Run {run + 1}/{num_runs}: First Return = {first_return:.2f}, Final Return = {final_return:.2f}")

average_returns = np.mean(returns_over_time, axis=0)
std_returns = np.std(returns_over_time, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(average_returns, label="Average Return", color ='black')
plt.fill_between(range(num_episodes),
                 average_returns - std_returns,
                 average_returns + std_returns,
                 alpha=0.2, label="Standard Deviation")
plt.xlabel("Episodes")
plt.ylabel("Average Return")
plt.title("Average Return Over 20 Run(s)")
plt.legend()
plt.grid()
plt.show()



