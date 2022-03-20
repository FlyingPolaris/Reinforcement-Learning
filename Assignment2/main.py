import numpy as np
from gridworld import GridworldEnv
import argparse
import FirstVisitMC
import EveryVisitMC
import TD


def test_first_visit_MC(gridworld, policy, episode_num):
    V = FirstVisitMC.policy_evaluation(gridworld, policy, episode_num)
    print('----------------First Visit MC----------------')
    print('Value Function:')
    print(np.around(np.reshape(V, gridworld.shape), decimals=4))
    print('episode Num:')
    print(episode_num)
    print('------------------------------------------------')


def test_every_visit_MC(gridworld, policy, episode_num):
    V = EveryVisitMC.policy_evaluation(gridworld, policy, episode_num)
    print('----------------Every Visit MC----------------')
    print('Value Function:')
    print(np.around(np.reshape(V, gridworld.shape), decimals=4))
    print('episode Num:')
    print(episode_num)
    print('------------------------------------------------')


def test_TD(gridworld, policy, episode_num):
    V = TD.policy_evaluation(gridworld, policy, episode_num)
    print('----------------TD----------------')
    print('Value Function:')
    print(np.around(np.reshape(V, gridworld.shape), decimals=4))
    print('episode Num:')
    print(episode_num)
    print('----------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='MC1')
    parser.add_argument('-n', '--episode_num', type=int, default=1000)
    args = parser.parse_args()
    method = args.method
    episode_num = args.episode_num

    gridworld = GridworldEnv([6, 6])
    policy = np.ones([gridworld.nS, gridworld.nA]) / gridworld.nA

    if method == 'MC1':
        test_first_visit_MC(gridworld, policy, episode_num)

    if method == 'MC2':
        test_every_visit_MC(gridworld, policy, episode_num)

    if method == 'TD':
        test_TD(gridworld, policy, episode_num)


if __name__ == '__main__':
    main()
