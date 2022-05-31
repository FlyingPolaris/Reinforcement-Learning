import numpy as np

def epsilon_greedy_action(gridworld, epsilon, Q, state):
    action_list = epsilon * np.ones(gridworld.nA) / gridworld.nA
    best_action = np.argmax(Q[state])
    action_list[best_action] += 1 - epsilon
    return action_list

def q_learning(gridworld, episode_num, epsilon, alpha=0.5, gamma=0.99):
    Q = np.zeros([gridworld.nS, gridworld.nA])
    for i in range(episode_num):
        is_done = False
        state = gridworld.reset()
        while not is_done:
            action_list = epsilon_greedy_action(gridworld, epsilon, Q, state)
            action = np.random.choice(gridworld.nA, 1, p=action_list)[0]
            prob, next_state, reward, is_done = gridworld.P[state][action][0]
            best_action = np.argmax(Q[state])
            Q[state][action] = Q[state][action] + alpha * \
                (reward+gamma*Q[next_state][best_action] - Q[state][action])
            state = next_state
    policy = np.zeros([gridworld.nS, gridworld.nA])
    is_done = False
    state = gridworld.reset()
    while not is_done:
        best_action = np.argmax(Q[state])
        policy[state] = np.eye(gridworld.nA)[best_action]
        prob, state, reward, is_done = gridworld.P[state][best_action][0]
    return policy, Q