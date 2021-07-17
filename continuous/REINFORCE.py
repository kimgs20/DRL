import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import gym
from itertools import count

import matplotlib.pyplot as plt
import matplotlib.animation as Animation


class PolicyNet(nn.Module):

    def __init__(self, num_states, num_actions):
        super(PolicyNet, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_act = nn.Linear(64, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_act(x))*2  # because the action space of the Pendulum-v0 is [-2,2]
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for reward, action_prob in self.data[::-1]:
            R = reward + GAMMA * R
            loss = -torch.log(action_prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def main():

    env = gym.make('Pendulum-v0')
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

            action_prob = policy_net(torch.from_numpy(state).float())

            next_state, reward, done, _ = env.step([action_prob.item()])
            reward = reward / 100.0

            reward = torch.tensor([reward], device=device)
            policy_net.put_data((reward, action_prob))

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
    NUM_EPS = 100_000
    GIF_THRESH = 99_950
    COMMENT = "REINFORCE_pendulum"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
