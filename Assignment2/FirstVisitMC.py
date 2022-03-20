import numpy as np

def policy_evaluation(gridworld, policy, eposide_num, gamma=0.99):
    V = np.zeros(gridworld.nS, dtype=np.float)
    N = np.zeros(gridworld.nS, dtype=np.float)
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
        state_visited = []
        for t in range(T):
            current_state = eposide_state[t]
            if current_state not in state_visited:
                state_visited.append(current_state)
                N[current_state] += 1
                G = 0.0
                discount = 1.0
                for j in range(t, T):
                    G += discount*eposide_reward[j]
                    discount *= gamma
                V[current_state] = V[current_state] + (G-V[current_state])/N[current_state]
    return V
