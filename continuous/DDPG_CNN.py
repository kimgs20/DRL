import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

import os
import gym
import random
import numpy as np
from PIL import Image
from itertools import count
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import matplotlib.animation as Animation

from disable_view_window import disable_view_window


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, h, w, num_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        convh = conv2d_outsize(conv2d_outsize(conv2d_outsize(h)))
        convw = conv2d_outsize(conv2d_outsize(conv2d_outsize(w)))
        linear_input_size = convh * convw * 32

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, num_actions)

    def forward(self, x):
        # if store transition in VRAM, comment next to line
        # x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x)) * 2.0  # Don't use relu on last layer!
        x = x.clip(min=-2.0, max=2.0)
        return x


class Critic(nn.Module):
    def __init__(self, h, w, num_actions):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        convh = conv2d_outsize(conv2d_outsize(conv2d_outsize(h)))
        convw = conv2d_outsize(conv2d_outsize(conv2d_outsize(w)))
        linear_input_size = convh * convw * 32

        self.fc_state = nn.Linear(linear_input_size, 64)
        self.fc_action = nn.Linear(num_actions, 64)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    # x: state, a: action
    def forward(self, x, a):
        # if store transition in VRAM, comment next two lines
        # x = x.to(device)
        # a = a.to(device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        h_state = F.relu(self.fc_state(x))
        h_act = F.relu(self.fc_action(a))

        h_concat = torch.cat([h_state, h_act], dim=1)

        h_q = F.relu(self.fc1(h_concat))
        q_value = self.fc2(h_q)
        return q_value


def conv2d_outsize(input_len, kernel_size=5, stride=2):
    return ((input_len - kernel_size) // stride) + 1


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


def update_networks(i_episode, memory, actor, actor_target, actor_optimizer, critic, critic_target, critic_optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # target networks needs only forward propagation. don't need gradient
    with torch.no_grad():
        next_state_values[non_final_mask] = critic_target(non_final_next_states, actor_target(non_final_next_states)).squeeze(1)

    target_q = reward_batch + GAMMA * next_state_values
    critic_q = critic(state_batch, action_batch)

    critic_loss = F.smooth_l1_loss(critic_q, target_q.unsqueeze(1))

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(state_batch, actor(state_batch)).mean()  # gradient ascent for highest Q value
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # for softupdate
    if i_episode % 10 == 0:
        for i in range(10):
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    return actor_loss.item(), critic_loss.item()


def resize(img):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(50, interpolation=Image.CUBIC),
                        T.ToTensor()])
    return resize(img)


def get_pixel(env):
    img = env.render(mode='rgb_array').transpose((2, 0, 1))  # (500, 500, 3) -> (3, 500, 500)
    img = img[:, slice(120, 380), slice(120, 380)]
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    return resize(img).unsqueeze(0).to(device)  # [1, 3, 50, 50]


def main():

    env = gym.make('Pendulum-v0')
    IMG_HEIGHT, IMG_WIDTH = int(50), int(50)
    NUM_ACTIONS = env.action_space.shape[0]

    os.makedirs(f"./{COMMENT}", exist_ok=True)
    disable_view_window()

    fig, ax = plt.subplots()
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Initialize Networks
    actor, actor_target = Actor(IMG_HEIGHT, IMG_WIDTH, NUM_ACTIONS).to(device), Actor(IMG_HEIGHT, IMG_WIDTH, NUM_ACTIONS).to(device)
    actor_target.load_state_dict(actor.state_dict())
    actor_target.eval()
    critic, critic_target = Critic(IMG_HEIGHT, IMG_WIDTH, NUM_ACTIONS).to(device), Critic(IMG_HEIGHT, IMG_WIDTH, NUM_ACTIONS).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_MU)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_Q)

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
        env.reset()

        last_img = get_pixel(env)
        current_img = get_pixel(env)
        state = current_img - last_img

        for t in count():
            # if you want to see what agent actually take, uncomment next two lines
            # plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
            # plt.show()

            if i_episode > GIF_THRESH:
                im = ax.imshow(env.render(mode='rgb_array'), animated=True)
                ims.append([im])

            action = actor(state)
            if i_episode < 1000:
                action = action.item() + ou_noise()[0]
            else:
                action = action.item()
            _, reward, done, _ = env.step([action])

            reward = reward / 10.0
            reward_sum += reward

            last_img = current_img
            current_img = get_pixel(env)

            if not done:
                next_state = current_img - last_img
            else:
                next_state = None

            action = torch.tensor([action], device=device, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            # in my opinion, push 'done' into replay buffer is awkward... (and also memory inefficient)
            memory.push(state, action, next_state, reward)

            state = next_state

            if len(memory) >= 2_000:
                actor_loss, critic_loss = update_networks(i_episode, memory,
                                                          actor, actor_target, actor_optimizer,
                                                          critic, critic_target, critic_optimizer)
                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss

            if done is True:
                print(f"memory length: {len(memory)}")
                print(f"Return: {reward_sum:2f}\n")
                writer.add_scalar("Return", reward_sum, i_episode)
                writer.add_scalar("actor loss per ep", actor_loss_sum / (t + 1), i_episode)
                writer.add_scalar("critic loss per ep", critic_loss_sum / (t + 1), i_episode)
                break

        if i_episode > GIF_THRESH:
            ani = Animation.ArtistAnimation(fig, ims, interval=10)
            ani.save(f'./{COMMENT}/episode_{i_episode}_return_{round(reward_sum)}.gif')

    env.close()


if __name__ == '__main__':
    # Hyperparameters
    LR_MU = 5e-5
    LR_Q = 1e-4
    GAMMA = 0.99
    TAU = 0.005  # for target network soft update
    BATCH_SIZE = 32

    MAX_MEMORY = 30_000
    NUM_EPS = 3_000
    GIF_THRESH = NUM_EPS - 30

    COMMENT = "DDPG_pendulum_control_from_pixel"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    main()
