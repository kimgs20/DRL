import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import gym
import random
import numpy as np
from itertools import count
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import matplotlib.animation as Animation


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x)) * 2.0  # Don't use relu on last layer!
        x = x.clip(min=-2.0, max=2.0)
        return x


class Critic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(num_states, 64)
        self.fc_action = nn.Linear(num_actions, 64)
        self.fc_q_val = nn.Linear(128, 32)
        self.fc_q_out = nn.Linear(32, 1)

    # x: state, a: action
    def forward(self, x, a):
        h_state = F.relu(self.fc_state(x))
        h_act = F.relu(self.fc_action(a))
        h_concat = torch.cat([h_state, h_act], dim=1)
        h_q = F.relu(self.fc_q_val(h_concat))
        q_value = self.fc_q_out(h_q)
        return q_value


# action exploration noise process
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def update_networks(memory, actor, actor_target, critic, critic_target, critic_optimizer, actor_optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # target networks needs only forward propagation. don't need gradient
    with torch.no_grad():
        target_q = reward_batch + GAMMA * critic_target(next_state_batch, actor_target(next_state_batch))

    critic_q = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(critic_q, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(state_batch, actor(state_batch)).mean() # gradient ascent for highest Q value
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # for softupdate
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    return actor_loss.item(), critic_loss.item()


def main():

    env = gym.make('Pendulum-v0')
    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]

    os.makedirs(f"./{COMMENT}", exist_ok=True)

    fig, ax = plt.subplots()
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Initialize Networks
    actor, actor_target = Actor(NUM_STATES, NUM_ACTIONS), Actor(NUM_STATES, NUM_ACTIONS)
    actor_target.load_state_dict(actor.state_dict())
    critic, critic_target = Critic(NUM_STATES, NUM_ACTIONS), Critic(NUM_STATES, NUM_ACTIONS)
    critic_target.load_state_dict(critic.state_dict())

    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_Q)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_MU)

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    memory = ReplayMemory(MAX_MEMORY)

    writer = SummaryWriter(comment=f" {COMMENT}")

    # main training loop
    for i_episode in range(NUM_EPS):

        print(f"Episode: {i_episode}")
        ims = []
        reward_sum = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)

        for t in count():
            if i_episode > GIF_THRESH:
                im = ax.imshow(env.render(mode='rgb_array'), animated=True)
                ims.append([im])

            action = actor(state)
            action = action.item() + ou_noise()[0]
            next_state, reward, done, _ = env.step([action])

            # change data types and add batch dimension
            action = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
            next_state = torch.from_numpy(next_state).float()
            next_state = next_state.unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            # in my opinion, push 'done' into replay buffer is awkward... (and also memory inefficient)
            memory.push(state, action, next_state, reward)

            state = next_state
            reward_sum += reward.item()

            if len(memory) > 2000:
                actor_loss, critic_loss = update_networks(memory, actor, actor_target,
                    critic, critic_target, critic_optimizer, actor_optimizer)
                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss

            if done is True:
                print(f"Return: {reward_sum:2f}")
                writer.add_scalar("Return", reward_sum, i_episode)
                writer.add_scalar("actor loss per ep", actor_loss_sum / (t + 1), i_episode)
                writer.add_scalar("critic loss per ep", critic_loss_sum / (t + 1), i_episode)
                print()
                break

        if i_episode > GIF_THRESH:
            ani = Animation.ArtistAnimation(fig, ims, interval=10)
            ani.save(f'./{COMMENT}/episode_{i_episode}_return_{round(reward_sum)}.gif')

    env.close()


if __name__ == '__main__':
    # Hyperparameters
    LR_MU       = 0.0005
    LR_Q        = 0.001
    GAMMA       = 0.99
    TAU         = 0.005  # for target network soft update
    BATCH_SIZE  = 32

    MAX_MEMORY    = 50_000
    NUM_EPS    = 2_000

    GIF_THRESH = NUM_EPS - 30

    COMMENT     = "DDPG_pendulum"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    main()
