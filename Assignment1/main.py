import imp
import numpy as np
from gridworld import GridworldEnv
import PolicyIteration
import ValueIteration
import argparse

def test_policy_iteration(gridworld):
    V, policy = PolicyIteration.policy_iteration(gridworld)
    print('----------------Policy Iteration----------------')
    print('Value Function:')
    print(V)
    print('Policy:')
    print(policy)
    print('------------------------------------------------')


def test_value_iteration(gridworld):
    V, policy = ValueIteration.value_iteration(gridworld)
    print('----------------Value Iteration----------------')
    print('Value Function:')
    print(V)
    print('Policy:')
    print(policy)
    print('------------------------------------------------')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--method', default='policy')
    args = parser.parse_args()
    method = args.method

    gridworld = GridworldEnv([6, 6])
    if method == 'policy':
        test_policy_iteration(gridworld)
    elif method == 'value':
        test_value_iteration(gridworld)


if __name__ == '__main__':
    main()
