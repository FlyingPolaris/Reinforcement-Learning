import numpy as np

def epsilon_greedy_action(gridworld, epsilon, Q, state):
    action_list = epsilon * np.ones(gridworld.nA) / gridworld.nA
    best_action = np.argmax(Q[state])
    action_list[best_action] += 1 - epsilon
    return action_list


def sarsa(gridworld, episode_num, epsilon, alpha=0.5, gamma=0.99):
    Q = np.zeros([gridworld.nS, gridworld.nA])
    for i in range(episode_num):
        is_done = False
        state = gridworld.reset()
        action_list = epsilon_greedy_action(gridworld, epsilon, Q, state)
        action = np.random.choice(gridworld.nA, 1, p=action_list)[0]
        while not is_done:
            prob, next_state, reward, is_done = gridworld.P[state][action][0]
            action_list = epsilon_greedy_action(
                gridworld, epsilon, Q, next_state)
            next_action = np.random.choice(gridworld.nA, 1, p=action_list)[0]
            Q[state][action] = Q[state][action] + alpha * \
                (reward+gamma*Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
    policy = np.zeros([gridworld.nS, gridworld.nA])
    is_done = False
    state = gridworld.reset()
    while not is_done:
        best_action = np.argmax(Q[state])
        policy[state] = np.eye(gridworld.nA)[best_action]
        prob, state, reward, is_done = gridworld.P[state][best_action][0]
    return policy, Q
