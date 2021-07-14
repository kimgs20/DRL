import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
import time
import math
import random
import numpy as np
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLanderContinuous-v2")

COMMENT = "REINFORCE_lunarlander"