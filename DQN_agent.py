import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from itertools import count
import random
import os
import cv2
from datetime import datetime

from traffic3d_single_junction import Traffic3DSingleJunction
from arguments import get_args

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.head = nn.Linear(576, n_actions)
        # self.saved_log_probs = []
        # self.basic_rewards = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        print('Environment action space: {}'.format(self.n_actions))

        self.policy_net = DQN(self.n_actions)
        self.target_net = DQN(self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if args.cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = self.args.lr)

        self.buffer = ReplayBuffer(self.args.buffer_size)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'DQN')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def learn(self):
        # env.reset()
        episode_total_rewards = []
        reward_sum = 0
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            # print('Env reset')
            reward_sum = 0
            for t in count():
                # Play a frame
                # Variabalize 210, 160
                obs_tensor = self._preproc_inputs(obs)
                action = self._select_action(obs_tensor)
                # print('Action selected: {}'.format(action))
                # env.render()
                obs_new, reward, done, info = self.env.step(action)
                reward_sum += reward
                obs = obs_new
                print('Episode {}, Timestep {}, Action: {}, Reward : {}'.format(episode, t, action, reward))

                obs_next_tensor = self._preproc_inputs(obs_new)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                r_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                # save the timestep transitions into the replay buffer
                self.buffer.push(obs_tensor, action_tensor, obs_next_tensor, r_tensor)

                #print('Optimizing starts')
                if len(self.buffer) >= self.args.batch_size:
                    # print('Updating policy network')
                    self._optimize_model(self.args.batch_size)
                #print('Optimizing finishes')

                if done:
                    break

            print('[{}] Episode {} finished. Reward total: {}'.format(datetime.now(), episode, reward_sum))
            episode_total_rewards.append(reward_sum)
            np.save(self.save_path + '/episode_total_rewards_standalonglinux64.npy', episode_total_rewards)
            torch.save(self.policy_net.state_dict(), self.save_path + '/policy_network_standalonglinux64.pt')


             # Update the target network, copying all weights and biases in DQN
            if episode >= self.args.target_update_step and episode % self.args.target_update_step == 0:
                print('Updating target network')
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Learning process finished')

    def _preproc_inputs(self, input):
        x = cv2.resize(input, (100, 100))
        x = x.transpose(2, 0, 1)
        x = np.ascontiguousarray(x, dtype=np.float32) / 255
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        if self.args.cuda:
            x = x.cuda()
        return x

    def _select_action(self, state):
        # global steps_done
        sample = random.random()
        # eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
        #     math.exp(-1. * steps_done / self.args.eps_decay)
        # steps_done += 1
        if sample > self.args.random_eps:
            with torch.no_grad():
                Q_values = self.policy_net(state)
                # take the Q_value index with the largest expected return
                action_tensor = Q_values.max(1)[1].view(1, 1)
                action = action_tensor.detach().cpu().numpy().squeeze()
                return action
        else:
            return random.randrange(self.n_actions)


    def _optimize_model(self, batch_size):
        states, actions, next_states, rewards = Transition(*zip(*self.buffer.sample(batch_size)))

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        rewards = torch.cat(rewards)
        actions = torch.cat(actions)

        predicted_values = torch.gather(self.policy_net(states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_values = rewards + self.args.gamma * next_state_values

        # Compute loss
        loss = F.smooth_l1_loss(predicted_values, expected_values)

        # calculate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return

if __name__ == '__main__':
    args = get_args()
    env = Traffic3DSingleJunction()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_agent(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()
