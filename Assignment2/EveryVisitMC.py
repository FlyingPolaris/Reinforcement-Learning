import numpy as np

def policy_evaluation(gridworld, policy, episode_num, gamma=0.99):
    V = np.zeros(gridworld.nS, dtype=np.float)
    N = np.zeros(gridworld.nS, dtype=np.float)
    for i in range(episode_num):
        is_done = False
        first_visit = np.zeros(gridworld.nS)
        state = np.random.randint(gridworld.nS)
        episode_reward = []
        episode_state = []
        while not is_done:
            action = np.random.choice(4, 1, p=policy[state])[0]
            prob, state, reward, is_done = gridworld.P[state][action][0]
            episode_reward.append(reward)
            episode_state.append(state)
        T = len(episode_reward)
        for t in range(T-1):
            current_state = episode_state[t]
            N[current_state] += 1
            G = 0.0
            discount = 1.0
            for j in range(t, T):
                G += discount*episode_reward[j]
                discount *= gamma
            V[current_state] = V[current_state] + (G-V[current_state])/N[current_state]
    return V
