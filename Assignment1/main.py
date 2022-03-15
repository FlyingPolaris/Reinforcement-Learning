import imp
import numpy as np
from gridworld import GridworldEnv
import PolicyIteration
import ValueIteration
import argparse


def test_policy_iteration(gridworld):
    V, policy, iteration = PolicyIteration.policy_iteration(gridworld)
    print('----------------Policy Iteration----------------')
    print('Value Function:')
    print(np.reshape(V, gridworld.shape))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    print(np.reshape(np.argmax(policy, axis=1), gridworld.shape))
    print('Iteration:')
    print(iteration)
    print('------------------------------------------------')


def test_value_iteration(gridworld):
    V, policy, iteration = ValueIteration.value_iteration(gridworld)
    print('----------------Value Iteration----------------')
    print('Value Function:')
    print(np.reshape(V, gridworld.shape))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    print(np.reshape(np.argmax(policy, axis=1), gridworld.shape))
    print('Iteration:')
    print(iteration)
    print('------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='policy')
    args = parser.parse_args()
    method = args.method

    gridworld = GridworldEnv([6, 6])
    if method == 'policy':
        test_policy_iteration(gridworld)
    elif method == 'value':
        test_value_iteration(gridworld)


if __name__ == '__main__':
    main()
