import gym
import torch
#import numpy as np
from example import *
from policiyNet import PolicyNetwork
#from collections import deque
#import copy
from train_test import train, test
from utils import *
from tqdm import tqdm

# ---- check device -----#
useGPU = 0
if torch.cuda.is_available(): 
 dev = "cuda:0"
 useGPU = 1
else: 
 dev = "cpu" 
 useGPU = 0
device = torch.device(dev) 

print(device)

# ---- generate the environment ----- #
env = gym.make('CartPole-v0')
#get the observation space where:
# - 0: Cart Position
# - 1: Cart Velocity
# - 2: Pole Angle
# - 3: Pole Angular Velocity
space = env.observation_space.shape
print('Observation Shape: ', space)
#0: Push cart to the left
#1: Push cart to the right
actions = env.action_space.n
print('n. of actions: \n', actions)

usage_example = False
gym_example(usage_example, env, actions)

# ----- POLICITY NETWORK -----#

state_dim = env.observation_space.shape[0] #dimension of the input layer
fc1_size = 256 # fc1 dim
fc2_size = actions #output dim

policy_net = PolicyNetwork(state_dim=state_dim, fc1_size=fc1_size, fc2_size=fc2_size)
learning_rate = 0.0009
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

# ----- TRAIN ------ # 

#max duration of an episode
max_duration = 200
#number of total episodes
max_episodes = 1500
#discount factor
gamma = 0.99
#a list to keep track of the episode length over training time
score = []
#array of actions, i.e. 0 or 1
actions_idx = np.arange(actions)
render_interval = 100
tot_scores = []

trained_model, tot_scores = train(env=env, max_duration=max_duration, max_episodes=max_episodes, gamma=gamma, model=policy_net, score=score, actions=actions_idx, optimizer=optimizer, render_interval=render_interval)

# ---- PLOT TRAINING ---- #

#the game is considered “solved” if the agent can play an episode beyond 200 time steps
#plot losses
title = 'Plot Scores'
plot(tot_scores,title, 50)


# ---- TEST ----- #

#n. of games to play (episodes)
n_games = 100
test_render_interval = 20
test(n_games=n_games, env=env, trained_model=trained_model, actions=actions_idx, max_duration=max_duration, render_interval=test_render_interval)