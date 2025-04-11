import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
#from pyvirtualdisplay import Display
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from collections import defaultdict
import pickle
from IPython.display import clear_output
import torch
import random



############ PART 1 ###############
# IMPLEMENTS Q-LEARNING

def epsilon_greedy(env, state, Q, epsilon, episodes, episode, state_space_bounds, num_bins):
    """Selects an action to take based on a uniformly random sampled number."""
    discretized_state = discretize_state(state, state_space_bounds, num_bins)  # Discretize the state
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[discretized_state])  # Use the discretized state as the key
    else:
        return env.action_space.sample()
    

def simulate(env, Q, max_episode_length, epsilon, episodes, episode, state_space_bounds, num_bins):
    """Rolls out an episode of actions to be used for learning."""
    D = []
    state, info = env.reset()  # Get the initial state
    done = False
    for step in range(max_episode_length):
        discretized_state = discretize_state(state, state_space_bounds, num_bins)  # Discretize the state
        action = epsilon_greedy(env, state, Q, epsilon, episodes, episode, state_space_bounds, num_bins)  # Select action
        next_state, reward, terminated, truncated, info = env.step(action)  # Take action and get next state
        done = terminated or truncated  # Combine both conditions
        print(reward)
        D.append([discretized_state, action, reward, discretize_state(next_state, state_space_bounds, num_bins)])  # Store the data
        state = next_state  # Update state
        if done:
            break
    return D  

def discretize_state(state, state_space_bounds, num_bins):
    """
    Discretizes a continuous state into a discrete index.

    Args:
        state: The continuous state (e.g., [x, y] position).
        state_space_bounds: The bounds of the state space in each dimension.
        num_bins: The number of bins per dimension.

    Returns:
        A tuple of discrete state indices.
    """
    state_indices = []
    for i, (state_value, (low, high)) in enumerate(zip(state, state_space_bounds)):
        # Normalize the state value to be between 0 and 1
        normalized_value = (state_value - low) / (high - low)
        # Scale it to the number of bins and clip it to be in the valid range
        bin_index = int(np.clip(normalized_value * num_bins, 0, num_bins - 1))
        state_indices.append(bin_index)
    return tuple(state_indices)

def q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size, state_space_bounds, num_bins, Q):
    """Main loop of Q-learning algorithm."""
    if Q is None:
        Q = defaultdict(lambda: torch.zeros(env.action_space.n))  # Initialize Q-table if not passed
    total_reward = 0

    for episode in range(episodes):
        print(episode)
        D = simulate(env, Q, max_episode_length, epsilon, episodes, episode, state_space_bounds, num_bins)  # Simulate the episode
        for data in D:
            state, action, reward, next_state = data
            # Update Q-values using the discretized states
            Q[state][action] = (1 - step_size) * Q[state][action] + step_size * (reward + gamma * torch.max(Q[next_state]))
            total_reward += reward
        if episode % 10 == 0:
            print(f"Average total reward per episode batch since episode {episode}: {total_reward / 10}")
            save_q_table(Q, "q_table.pth")  # Save the Q-table after every 10 episodes
            total_reward = 0
    return Q

def save_q_table(Q, filename="q_table.pth"):
    """Save the Q-table to a .pth file."""
    Q_dict = {key: value.numpy() for key, value in Q.items()}  # Convert tensors to numpy arrays if needed
    torch.save(Q_dict, filename)
    print(f"Q-table saved to {filename}")

def load_q_table(filename="q_table.pth"):
    """Load the Q-table from a .pth file."""
    try:
        Q_dict = torch.load(filename, weights_only=False)
        # Convert back to a defaultdict with tensors
        Q = defaultdict(lambda: torch.zeros(env.action_space.n))
        for key, value in Q_dict.items():
            Q[key] = torch.tensor(value)  # Convert numpy arrays back to tensors if needed
        print(f"Q-table loaded from {filename}")
        return Q
    except FileNotFoundError:
        print("No saved Q-table found, initializing new Q-table.")
        return defaultdict(lambda: torch.zeros(env.action_space.n))  # Initialize Q-table

### TESTING ###

gamma = 0.7                # discount factor - determines how much to value future actions
episodes = 21            # number of episodes to play out
max_episode_length = 60    # maximum number of steps for episode roll out
epsilon = 0.9               # control how often you explore random actions versus focusing on high value state and actions
step_size = 0.1             # learning rate - controls how fast or slow to update Q-values for each iteration.

env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped

state, info = env.reset()
env.observation_space

env.action_space.seed(0)  # so we get same the random sequence every time
state, _ = env.reset(seed=0)  # this needs to be called once at the start before sending any actions

state_space_bounds = [(-40, 40), (-40, 40)]  # Define bounds for the state space (example)
num_bins = 10  # Number of bins for discretizing the state space

# Attempt to load the Q-table if it exists
Q = load_q_table("q_table.pth")

# Continue training from the loaded Q-table, or train from scratch if not found
Q = q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size, state_space_bounds, num_bins, Q)

# test your policy
state, info = env.reset()
discretized_state = discretize_state(state, state_space_bounds, num_bins)  # Discretize the initial state



steps = 0
while True and steps < 100:  # in case policy gets stuck (shouldn't happen if valid path exists and optimal policy learnt)
    ########## policy is simply taking the action with the highest Q-value for given state ##########
    action = np.argmax(Q[discretized_state])  # Use discretized state for action selection
    #################################################################################################
    state, reward, terminated, truncated, info = env.step(action)             # Hint: Check how env.step() works
    done = terminated or truncated  # Combine both conditions
    if done:
        break

env.close()