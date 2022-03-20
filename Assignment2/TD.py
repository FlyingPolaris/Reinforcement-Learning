import numpy as np

def policy_evaluation(gridworld, policy, episode_num, alpha=0.05, gamma=0.99):
    V = np.zeros(gridworld.nS, dtype=np.float)
    for i in range(episode_num):
        is_done = False
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
            next_state = episode_state[t+1]
            V[current_state] = V[current_state] + alpha * \
                (episode_reward[t]+gamma*V[next_state]-V[current_state])
    return V