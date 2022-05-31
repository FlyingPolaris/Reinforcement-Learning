import imp
from DDQN import DDQN
from DQN import DQN
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gym
import math

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.8
GAMMA = 0.99
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('MountainCar-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(), int) else env.action_space.sample().shape


def train_net(net, episode_num):
    return_list = []
    step_list = []
    score_list = []
    for i in range(episode_num):
        s = env.reset()
        ep_r = 0
        step = 0
        score = 0
        discount = 1.0
        while True:
            step += 1
            env.render()
            a = net.choose_action(s)
            s_, r, done, info = env.step(a)

            if s_[0] <= -0.5:
                r = 100 * abs(s_[1])
            elif s_[0] > -0.5 and s_[0] <= 0.5:
                r = math.pow(2, 5*(s_[0] + 1)) + (100 * abs(s_[1])) ** 2
            else:
                r = 1000
            net.store_transition(s, a, r, s_)

            score += discount * r
            discount *= GAMMA

            ep_r += r
            if net.memory_counter > MEMORY_CAPACITY:
                net.learn()
                if done:
                    return_list.append(ep_r)
                    step_list.append(step)
                    score_list.append(score)
            if done:
                break
            s = s_
    return return_list, step_list, score_list


def train(episode_num):
    dqn = DQN()
    ddqn = DDQN()
    dqn_return_list, dqn_step_list, dqn_score_list = train_net(
        dqn, episode_num)
    ddqn_return_list, ddqn_step_list, ddqn_score_list = train_net(
        ddqn, episode_num)
    np.save('dqn_return_list', dqn_return_list)
    np.save('dqn_step_list', dqn_step_list)
    np.save('dqn_score_list', dqn_score_list)
    np.save('ddqn_return_list', ddqn_return_list)
    np.save('ddqn_step_list', ddqn_step_list)
    np.save('ddqn_score_list', ddqn_score_list)

def plot(episode_num):
    dqn_return_list = np.load('dqn_return_list.npy', allow_pickle=True)
    dqn_step_list = np.load('dqn_step_list.npy', allow_pickle=True)
    dqn_score_list = np.load('dqn_score_list.npy', allow_pickle=True)
    ddqn_return_list = np.load('ddqn_return_list.npy', allow_pickle=True)
    ddqn_step_list = np.load('ddqn_step_list.npy', allow_pickle=True)
    ddqn_score_list = np.load('ddqn_score_list.npy', allow_pickle=True)


    ep_range = [i + 1 for i in range(episode_num)]

    plt.figure(figsize=(18, 9))
    plt.plot(ep_range, dqn_step_list, label='DQN', alpha=0.6)
    plt.plot(ep_range, ddqn_step_list, label='DDQN', alpha=0.6)
    plt.xlabel('episode')
    plt.ylabel('step num')
    plt.legend()
    plt.savefig('step.png')

    plt.figure(figsize=(18, 9))
    plt.plot(ep_range, dqn_return_list, label='DQN', alpha=0.6)
    plt.plot(ep_range, ddqn_return_list, label='DDQN', alpha=0.6)
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.legend()
    plt.savefig('return.png')

    plt.figure(figsize=(18, 9))
    plt.plot(ep_range, dqn_score_list, label='DQN', alpha=0.6)
    plt.plot(ep_range, ddqn_score_list, label='DDQN', alpha=0.6)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.legend()
    plt.savefig('score.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--episode_num', type=int, default=500)
    parser.add_argument('-a', '--action', type=str, default='plot')

    args = parser.parse_args()
    episode_num = args.episode_num
    action = args.action

    if action == 'plot':
        plot(episode_num)
    elif action == 'train':
        train(episode_num)


if __name__ == '__main__':
    main()
