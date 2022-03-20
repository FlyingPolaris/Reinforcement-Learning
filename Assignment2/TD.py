import numpy as np

def policy_evaluation(gridworld, policy, eposide_num, alpha=0.5, gamma=0.99):
    V = np.zeros(gridworld.nS, dtype=np.float)
    for i in range(eposide_num):
        is_done = False
        state = np.random.randint(gridworld.nS)
        eposide_reward = []
        eposide_state = []
        while not is_done:
            action = np.random.choice(4, 1, p=policy[state])[0]
            prob, state, reward, is_done = gridworld.P[state][action][0]
            eposide_reward.append(reward)
            eposide_state.append(state)
        T = len(eposide_reward)
        for t in range(T-1):
            current_state = eposide_state[t]
            next_state = eposide_state[t+1]
            V[current_state] = V[current_state] + alpha * \
                (eposide_reward[t+1]+gamma*V[next_state]-V[current_state])
    return V