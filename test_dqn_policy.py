import torch
import gym
import simple_driving
import numpy as np

from a3_DQN import DQN_Solver  # Import your DQN network class

# --- Load environment ---
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped

state, info = env.reset()
env.observation_space

env.action_space.seed(0)  # so we get same the random sequence every time
state, info = env.reset(seed=0)  # this needs to be called once at the start before sending any actions


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Initialize policy network ---
agent = DQN_Solver(env, train_mode=False)
agent.policy_net.load_state_dict(torch.load("q_values_latest.pth"))
agent.policy_net.eval()


# --- Testing loop ---
num_episodes = 5

for episode in range(num_episodes):
    state, info = env.reset()
    if isinstance(state, tuple):  # Some Gym versions return (obs, info)
        state = state[0]
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            q_values = agent.policy_net(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        total_reward += reward
        state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()