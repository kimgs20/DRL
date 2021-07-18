import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = x.to(device)

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env, resize):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)


def select_action(state, t, EPS_START, EPS_END, EPS_DECAY, steps_done, memory, n_actions, policy_net):
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if t == 0:
        print(f"eps: {eps_threshold:.5f}, memory: {len(memory)}")
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100_000
    TARGET_UPDATE = 10
    LEARNING_RATE = 1e-4

    NUM_EPS = 10_000
    MEMORY_CAP = 20_000

    # COMMENT = "DQN_BatchNorm"
    # COMMENT = "DQN_RMSprop"
    # COMMENT = "DQN_Adam_MSELoss"
    COMMENT = "DQN_code_refac"

    env = gym.make('CartPole-v0').unwrapped
    disable_view_window()

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    writer = SummaryWriter(comment=f" {COMMENT}")
    
    env.reset()
    init_screen = get_screen(env, resize)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_CAP)

    global steps_done
    steps_done = 0


    for i_episode in range(NUM_EPS):
        # Initialize the environment and state
        print(f"Episode: {i_episode}")
        env.reset()
        last_screen = get_screen(env, resize)
        current_screen = get_screen(env, resize)
        state = current_screen - last_screen
        reward_sum = 0

        for t in count():
            # Select and perform an action
            action = select_action(state, t, EPS_START, EPS_END, EPS_DECAY, steps_done, memory, n_actions, policy_net)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            reward_sum += reward.item()
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, resize)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer)
            if done:
                print(f"Return: {reward_sum}")
                writer.add_scalar("Return", reward_sum, i_episode)
                print()
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()

if __name__=='__main__':

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
