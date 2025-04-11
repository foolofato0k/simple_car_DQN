import gym
import simple_driving
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# # Hyperparameters
# EPISODES = 2500
# LEARNING_RATE = 0.00025
# MEM_SIZE = 50000
# REPLAY_START_SIZE = 10000
# BATCH_SIZE = 32
# GAMMA = 0.99
# EPS_START = 1.0
# EPS_END = 0.01
# EPS_DECAY = 4 * MEM_SIZE
# MEM_RETAIN = 0.1
# NETWORK_UPDATE_ITERS = 5000
# FC1_DIMS = 128
# FC2_DIMS = 128

# Hyperparameters
EPISODES = 10000
LEARNING_RATE = 0.0005
MEM_SIZE = 40000
REPLAY_START_SIZE = 1000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 8 * MEM_SIZE
MEM_RETAIN = 0.1
NETWORK_UPDATE_ITERS = 1000
FC1_DIMS = 256
FC2_DIMS = 256


np.bool = np.bool_

# Network architecture
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], FC1_DIMS),
            nn.ReLU(),
            nn.Linear(FC1_DIMS, FC2_DIMS),
            nn.ReLU(),
            nn.Linear(FC2_DIMS, env.action_space.n)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)
    

# Experience Replay
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, next_state, done):
        if self.mem_count < MEM_SIZE:
            index = self.mem_count
        else:
            retain_size = int(MEM_RETAIN * MEM_SIZE)
            index = random.randint(retain_size, MEM_SIZE - 1)

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = next_state
        self.dones[index] = 1 - done
        self.mem_count += 1

    def sample(self):
        max_mem = min(self.mem_count, MEM_SIZE)
        indices = np.random.choice(max_mem, BATCH_SIZE, replace=True)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.states_[indices],
            self.dones[indices]
        )
    

# DQN Solver
class DQN_Solver:
    def __init__(self, env, train_mode=True):
        self.env = env
        self.train_mode = train_mode
        if self.train_mode:
            self.memory = ReplayBuffer(env)
        else:
            self.memory = None
        self.policy_net = Network(env)
        self.target_net = Network(env)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_count = 0

    def choose_action(self, state):
        if self.train_mode:
            if self.memory.mem_count > REPLAY_START_SIZE:
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learn_count / EPS_DECAY)
            else:
                eps = 1.0
            if random.random() < eps:
                # Non-uniform sampling based on prior knowledge
                exploration_probs = np.array([0.1, 0.2, 0.1, 0.09, 0.02, 0.09, 0.1, 0.2, 0.1])
                exploration_probs /= exploration_probs.sum()
                return np.random.choice(np.arange(9), p=exploration_probs)

            
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones)

        q_pred = self.policy_net(states)[np.arange(BATCH_SIZE), actions]
        q_next = self.target_net(next_states).max(1)[0]
        q_target = rewards + GAMMA * q_next * dones

        loss = self.policy_net.loss_fn(q_pred, q_target)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        self.learn_count += 1
        if self.learn_count % NETWORK_UPDATE_ITERS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__":
    # Training Loop
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    env = env.unwrapped
    env.action_space.seed(0)
    state, _ = env.reset(seed=0)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    agent = DQN_Solver(env)
    episode_rewards = []
    episode_history = []
    episode_batch_score = 0
    episode_reward = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.add(state, action, reward, next_state, done)

            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()

            state = next_state
            episode_batch_score += reward
            episode_reward += reward

        episode_history.append(episode)
        episode_rewards.append(episode_reward)
        episode_reward = 0

        if episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), 'q_values_latest.pth')
            print(f"[Episode {episode}] Avg Reward (last 100): {episode_batch_score / 100}")
            
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            print("Waiting for buffer to fill...")
            # Always save final Q-network at the end
        if episode == EPISODES - 1:
            torch.save(agent.policy_net.state_dict(), 'q_values_latest.pth')

    env.close()

    plt.plot(episode_history, episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on SimpleDriving-v0")
    plt.show()
