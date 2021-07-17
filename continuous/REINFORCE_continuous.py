import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import gym
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as Animation

from disable_view_window import disable_view_window

class PolicyNet(nn.Module):

    def __init__(self, num_states ,num_actions):
        super(PolicyNet, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_act = nn.Linear(64, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_act(x)
        x = 2.0 * torch.tanh(x)  # because the action space of the Pendulum-v0 is [-2,2]
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):  # this is super important...
        R = 0
        self.optimizer.zero_grad()
        for reward, action_prob in  self.data[::-1]:
            R = reward + GAMMA * R
            loss = -torch.log(action_prob) * R  # 원래는 log안에 action을 선택할 확률 [0, 1]
                                                # loss를 minimize하는 action을 더욱 선택하도록 한다.
                                                # 각 continuous action을 선택할 확률은?
                                                # continuous value를 [0, 1] 로 mapping 하는 방법은?
            loss.backward()  # will minimize loss
        self.optimizer.step()
        self.data = []

def main():

    # REINFORCE can apply to continuous action space!!!

    env = gym.make('Pendulum-v0')
    # env = gym.make("LunarLanderContinuous-v2")
    os.makedirs(f"./{COMMENT}", exist_ok=True)
    disable_view_window()

    # store information about action and state dimensions
    NUM_STATES = env.observation_space.shape[0]  # 3
    NUM_ACTIONS = env.action_space.shape[0]      # 1

    policy_net = PolicyNet(NUM_STATES, NUM_ACTIONS).to(device)
    fig, ax = plt.subplots()
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    writer = SummaryWriter(comment=f" {COMMENT}")

    for i_episode in range(NUM_EPS):
        if i_episode % 100 == 0:
            print(f"Episode: {i_episode}")
        ims = []
        reward_sum = 0
        state = env.reset()

        for t in count():

            if i_episode > GIF_THRESH:
                im = ax.imshow(env.render(mode='rgb_array'), animated=True)
                ims.append([im])

            net_output = policy_net(torch.from_numpy(state).float())
            action_dist = Normal(net_output, torch.tensor([0.1]))  # mean, var
            action = action_dist.sample()
            action.clamp(-2.0, 2.0)
            next_state, reward, done, _ = env.step([action.item()])
            reward = reward / 100.0

            reward = torch.tensor([reward], dtype=torch.float32)
            policy_net.put_data((reward, action_prob))  # action_prob: [0, 1] for continuous action [-2, 2]

            state = next_state
            reward_sum += reward.item()

            if done is True:  # Pendulum-v0 automatically terminated in 200th step.
                if i_episode % 100 == 0:
                    print(f"Return: {reward_sum:2f}")
                    print()
                writer.add_scalar("Return", reward_sum, i_episode)
                policy_net.train_net()  # REINFORCE is update the policy at the end of the episode
                break

        if i_episode > GIF_THRESH:
            ani = Animation.ArtistAnimation(fig, ims, interval=10)
            ani.save(f'./{COMMENT}/episode_{i_episode}_return_{round(reward_sum)}.gif')

        env.close()

if __name__ == '__main__':

    # Hyperparameters
    GAMMA = 0.99
    LEARNING_RATE = 0.0005
    NUM_EPS = 10_000
    GIF_THRESH = 9_980

    COMMENT = "REINFORCE_pendulum"
    # COMMENT = "REINFORCE_lunarlander"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
