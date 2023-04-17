#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:20:29 2023

@author: thomaslyu
"""

import numpy as np
from tensorflow.keras import models, layers

# Define the FrozenLake environment
n_states = 16
n_actions = 4
game_board = np.zeros((4, 4))
game_board[0, 3] = 1
game_board[3, 0] = 2
game_board[1, 1] = -1
game_board[2, 2] = -1

# Define the hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.999
n_episodes = 10000

# Define the Q-table for table-based Q-learning
q_table = np.zeros((n_states, n_actions))

# Define the neural network for neural network-based Q-learning
nn = models.Sequential()
nn.add(layers.Dense(16, input_dim=n_states, activation='relu'))
nn.add(layers.Dense(n_actions, activation='linear'))
nn.compile(loss='mse', optimizer='adam')

def get_next_state_and_reward(state, action):
    next_state = state.copy()
    if action == 'up':
        next_state[0] = max(0, state[0] - 1)
    elif action == 'down':
        next_state[0] = min(3, state[0] + 1)
    elif action == 'left':
        next_state[1] = max(0, state[1] - 1)
    elif action == 'right':
        next_state[1] = min(3, state[1] + 1)
    reward = game_board[next_state[0], next_state[1]]
    return next_state, reward

def play_game(q_func, epsilon):
    state = np.random.randint(0, n_states)
    total_reward = 0
    done = False
    while not done:
        action = np.argmax(q_func[state, :])
        if np.random.random() < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(q_func[state, :])
        next_state, reward = get_next_state_and_reward(state, actions[action])
        q_func[state, action] += learning_rate * (reward + discount_factor * np.max(q_func[next_state, :]) - q_func[state, action])
        state = next_state
        total_reward += reward
        if reward != 0:
            done = True
    return total_reward

# Define the Q-learning algorithm
def q_learning(q_func):
    global epsilon
    rewards = []
    for i in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Choose the action using an epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_func[state, :])
            # Take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # Update the Q-values using the Bellman equation
            q_func[state, action] += learning_rate * (reward + discount_factor * np.max(q_func[next_state, :]) - q_func[state, action])
            state = next_state
        rewards.append(total_reward)
        epsilon *= epsilon_decay
    return rewards

# Train and test the table-based Q-learning algorithm
table_rewards = q_learning(q_table)
table_avg_reward = sum(table_rewards) / len(table_rewards)
print("Table-based Q-learning: Average reward over %d episodes = %.2f" % (n_episodes, table_avg_reward))

# Train and test the neural network-based Q-learning algorithm
nn_rewards = q_learning(nn.predict)
nn_avg_reward = sum(nn_rewards) / len(nn_rewards)
print("Neural network-based Q-learning: Average reward over %d episodes = %.2f" % (n_episodes, nn_avg_reward))
