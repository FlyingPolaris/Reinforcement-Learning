import numpy as np
from gridworld import GridworldEnv


def policy_evaluation(gridworld, policy, theta=0.001, gamma=0.99):
    V = np.zeros(gridworld.nS)
    iteration = 0
    while True:
        delta = 0.0
        iteration += 1
        for state in range(gridworld.nS):
            v = 0.0
            for action, pi in enumerate(policy[state]):
                for prob, next_state, reward, is_done in gridworld.P[state][action]:
                    v += pi*prob*(reward+gamma*V[next_state])
            delta = max(delta, np.abs(V[state]-v))
            V[state] = v
        if delta < theta:
            break
    return V, iteration


def policy_iteration(gridworld, theta=0.001, gamma=0.99):
    policy = np.ones([gridworld.nS, gridworld.nA]) / \
        gridworld.nA  # Init the policy randomly
    policy_stable = False
    while not policy_stable:
        policy_stable = True
        V, iteration = policy_evaluation(gridworld, policy, theta, gamma)
        for state in range(gridworld.nS):
            old_action = np.argmax(policy[state])
            action_values = np.zeros(gridworld.nA)
            for action in range(gridworld.nA):
                for prob, next_state, reward, is_done in gridworld.P[state][action]:
                    action_values[action] += prob * \
                        (reward+gamma*V[next_state])
            best_action = np.argmax(action_values)
            policy[state] = np.eye(gridworld.nA)[best_action]
            if old_action != best_action:
                policy_stable = False
    return V, policy, iteration
