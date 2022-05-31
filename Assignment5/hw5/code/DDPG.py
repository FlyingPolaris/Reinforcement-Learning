import argparse
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym

BATCH_SIZE = 32
LR_A = 7.5e-4
LR_C = 12e-4
GAMMA = 0.9
GAMMA_VAR = 0.995
TAU = 0.01
MEMORY_CAPACITY = 10000
env = gym.make('Pendulum-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]


class Net_A(nn.Module):
    def __init__(self, ):
        super(Net_A, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        action = x * 2
        return action


class Net_C(nn.Module):
    def __init__(self, ):
        super(Net_C, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(N_ACTIONS, 50)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fc1(s)
        y = self.fc2(a)
        q_value = self.out(F.relu(x+y))
        return q_value


class DDPG(object):
    def __init__(self):
        self.actor_eval_net, self.actor_target_net = Net_A(), Net_A()
        self.critic_eval_net, self.critic_target_net = Net_C(), Net_C()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 1))
        self.actor_optimizer = torch.optim.Adam(
            self.actor_eval_net.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_eval_net.parameters(), lr=LR_C)

        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval_net(s)[0].detach()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_params(self, t, s, isSoft=False):
        for t_param, param in zip(t.parameters(), s.parameters()):
            if isSoft:
                t_param.data.copy_(
                    t_param.data * (1.0 - TAU) + param.data * TAU)
            else:
                t_param.data.copy_(param.data)

    def learn(self):

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+N_ACTIONS])
        b_r = torch.FloatTensor(b_memory[:, -N_STATES-1:-N_STATES])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        a = self.actor_eval_net(b_s)
        q = self.critic_eval_net(b_s, a)
        a_loss = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        a_target = self.actor_target_net(b_s_)
        q_tmp = self.critic_target_net(b_s_, a_target)
        q_target = b_r + GAMMA * q_tmp
        q_eval = self.critic_eval_net(b_s, b_a)
        td_error = self.loss_func(q_target, q_eval)

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        self.update_params(self.actor_target_net, self.actor_eval_net, True)
        self.update_params(self.critic_target_net, self.critic_eval_net, True)


def train(episode_num, max_step_num):
    ddpg = DDPG()
    return_list = []
    action_low = env.action_space.low
    action_high = env.action_space.high
    var = 3.0
    for i in range(episode_num):
        s = env.reset()
        ep_r = 0
        step = 0
        while True:
            step += 1
            # env.render()
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), action_low, action_high)
            s_, r, done, info = env.step(a)
            ddpg.store_transition(s, a, (r+8)/8, s_)


            ep_r += r
            if ddpg.memory_counter > MEMORY_CAPACITY:
                var *= GAMMA_VAR
                ddpg.learn()
            if done or step == max_step_num - 1:
                if len(return_list) == 0:
                    return_list.append(ep_r)
                else:
                    return_list.append(0.95 * return_list[-1] + 0.05 * ep_r)
                print('Episode: ', i, ' Reward: %i' % (return_list[-1]))
                break
            s = s_
    np.save('return_list_'+ str(max_step_num), return_list)


def plot(episode_num, max_step_num):
    return_list = np.load('return_list_'+ str(max_step_num) +'.npy', allow_pickle=True)

    ep_range = [i + 1 for i in range(episode_num)]

    plt.figure(figsize=(7.5, 5))
    plt.plot(ep_range, return_list)
    plt.xlabel('step')
    plt.ylabel('Total Reward')
    plt.title('DDPG results')
    plt.savefig('return_'+ str(max_step_num) +'.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, default='plot')
    parser.add_argument('-s', '--max_step_num', type=int, default=800)
    parser.add_argument('-n', '--episode_num', type=int, default=250)

    args = parser.parse_args()
    action = args.action
    max_step_num = args.max_step_num
    episode_num = args.episode_num

    if action == 'train':
        train(episode_num, max_step_num)
        plot(episode_num, max_step_num)
    if action == 'plot':
        plot(episode_num, max_step_num)


if __name__ == '__main__':
    main()
