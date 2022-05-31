import imp
import numpy as np
from cliff_walking import CliffWalkingEnv
import sarsa
import q_learning
import argparse


def test_sarsa(gridworld, episode_num, epsilon):
    policy, Q = sarsa.sarsa(gridworld, episode_num, epsilon)
    print('----------------sarsa----------------')
    print('Q Function:')
    print(np.around(np.reshape(Q,(4,12,4)), decimals=1))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    policy_show = ['x' for x in range(48)]
    for s in range(48):
        for a in range(4):
            if policy[s][a] == 1:
                policy_show[s] = a
    policy_show[47] = 'T'
    print(np.reshape(policy_show, gridworld.shape))
    print('episode Num:')
    print(episode_num)
    print('-------------------------------------')

def test_q_learning(gridworld, episode_num, epsilon):
    policy, Q = q_learning.q_learning(gridworld, episode_num, epsilon)
    print('----------------q_learning----------------')
    print('Q Function:')
    print(np.around(np.reshape(Q,(4,12,4)), decimals=1))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    policy_show = ['x' for x in range(48)]
    for s in range(48):
        for a in range(4):
            if policy[s][a] == 1:
                policy_show[s] = a
    policy_show[47] = 'T'
    print(np.reshape(policy_show, gridworld.shape))
    print('episode Num:')
    print(episode_num)
    print('-------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='q_learning')
    parser.add_argument('-n', '--episode_num', type=int, default=1000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    args = parser.parse_args()
    method = args.method
    episode_num = args.episode_num
    epsilon = args.epsilon

    gridworld = CliffWalkingEnv()
    if method == 'sarsa':
        test_sarsa(gridworld, episode_num, epsilon)
    if method == 'q_learning':
        test_q_learning(gridworld, episode_num, epsilon)


if __name__ == '__main__':
    main()
