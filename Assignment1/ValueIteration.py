import numpy as np
from gridworld import GridworldEnv

def value_iteration(gridworld, theta=0.001, gamma=0.99):
    V = np.zeros(gridworld.nS)
    while True:
        delta = 0.0
        for state in range(gridworld.nS):
            v = V[state]
            action_values = np.zeros(gridworld.nA)
            for action in range(gridworld.nA):
                for prob, next_state, reward, is_done in gridworld.P[state][action]:
                    action_values[action] += prob * \
                        (reward+gamma*V[next_state])
            V[state] = np.max(action_values)
            delta = max(delta, np.abs(V[state]-v))
        if delta < theta:
            break

    policy = np.zeros([gridworld.nS, gridworld.nA])
    for state in range(gridworld.nS):
        action_values = np.zeros(gridworld.nA)
        for action in range(gridworld.nA):
            for prob, next_state, reward, is_done in gridworld.P[state][action]:
                action_values[action] += prob * \
                    (reward+gamma*V[next_state])
        best_action = np.argmax(action_values)
        policy[state]=np.eye(gridworld.nA)[best_action]
    return V, policy