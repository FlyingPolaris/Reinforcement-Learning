import numpy as np
from gridworld import GridworldEnv


def policy_evaluation(gridworld, policy, theta=0.001, gamma=0.99):
    V = np.zeros(gridworld.nS)
    while True:
        delta = 0
        for state in range(gridworld.nS):
            v = V[state]
            for action, pi in enumerate(policy[state]):
                for prob, next_state, reward, is_done in gridworld.P[state][action]:
                    V[state] += pi*prob*(reward+gamma*V[next_state])
            delta = max(delta, abs(V[state]-v))
        if delta < theta:
            break
    return V


def policy_iteration(gridworld, theta=0.001, gamma=0.99):
    policy = np.ones([gridworld.nS, gridworld.nA])/gridworld.nA
    policy_stable = True
    while policy_stable:
        V = policy_evaluation(gridworld, policy, theta, gamma)
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
    return V, policy
