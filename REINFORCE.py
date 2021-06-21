import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100_000
LEARNING_RATE = 1e-4
NUM_EPS = 10_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0').unwrapped

class PolicyNet(nn.Module):
    pass

for i_episode in range(NUM_EPS):
    env.reset()
    entropies = []
    log_probs = []
    rewards = []
    for t in count():
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()  # gpu

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            writer.add_scalar("Return", reward_sum, i_episode)
            print(f"Return: {reward_sum}")
            print()
            break

'''
entropy
log probability
argparse


'''