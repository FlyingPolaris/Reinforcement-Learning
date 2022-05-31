# Assignment 3

余北辰 519030910245



## 1 Experimental Purpose

- Implement Sarsa and Q-learning method separately
- Try different $\epsilon$ to investigate their impacts on performances

## 2  Experimental Background 

In this section, we present the methods of Sarsa and Q-learning separately. Some of the content in this section is referenced from Wikipedia.

### 2.1 Sarsa

Sarsa is the abbreviation of "State–action–reward–state–action", which is an on-policy learning algorithm. A Sarsa agent interacts with the environment and updates the policy based on actions taken. Q values represent the possible reward received in the next time step for taking action $a$ in state $s$, plus the discounted future reward received from the next state-action observation. The update function of Q values is shown as below:
$$
Q(s_t,a_t)=Q(s_t,a_t)+\alpha[r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
$$
The Q value for a state-action is updated by an error, adjusted by the learning rate $\alpha$. 

The pseudo-code for Sarsa is shown below:

<img src="D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407104139730.png" alt="image-20220407104139730" style="zoom:60%;" />

### 2.2 Q-learning

Q-learning is another model-free reinforcement learning algorithm to learn the value of an action in a particular state.  Different with Sarsa, it is an off-policy algorithm. The only different of Sarsa and Q-learning is how the method updates the Q values. On-policy Sarsa learns action values relative to the policy it follows, while off-policy Q-learning does it relative to the greedy policy.  

The update function of Q values is shown as below:
$$
Q(s_t,a_t)=Q(s_t,a_t)+\alpha[r_t+\gamma max_{a}\{Q(s_{t+1},a)\}-Q(s_t,a_t)]
$$
The pseudo-code for Q-learning is shown below:

<img src="D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407110239729.png" alt="image-20220407110239729" style="zoom:60%;" />

## 3 Experimental Procedure

The source code for this experiment includes `cliff_walking.py`,  `sarsa.py`,  `q_learning.py`and `main.py`.

### 3.1 cliff_walking.py

`cliff_walking.py` builds the Gridworld environment. This module references [reinforcement-learning/cliff_walking.py at master · dennybritz/reinforcement-learning (github.com)](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py).

The gridworld should be instantiated using the instruction as below:

```python
gridworld = CliffWalkingEnv()
```

The gridworld will be instantiated as below, in which state 36 is set as the start state and state 47 is set as the terminal state.

![image-20220407110858937](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407110858937.png)

The number of the states, the number of the actions and the state transfer matrix in gridworld can be obtained using the instruction as below:

```python
state_number = gridworld.nS # 48
action_number = gridworld.nA # 4
state_transfer_matrix = gridworld.P # P[state][action] = (prob, next_state, reward, is_done)
```

### 3.2 sarsa.py

`sarsa.py` is implemented based on the above pseudo-code.

First we need to define the $\epsilon$-greedy policy:

<img src="D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407111651525.png" alt="image-20220407111651525" style="zoom:50%;" />

```python
def epsilon_greedy_action(gridworld, epsilon, Q, state):
    action_list = epsilon * np.ones(gridworld.nA) / gridworld.nA
    best_action = np.argmax(Q[state])
    action_list[best_action] += 1 - epsilon
    return action_list
```

Then we can use the above $\epsilon$-greedy policy to both choose the next action and update the Q values. We implement the sarsa algorithm as below:

```python
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
```

### 3.3 q_learning.py

`q_learning.py` is implemented based on the above pseudo-code.

The defintion of $\epsilon$-greedy policy is the same as that in `sarsa.py`.

We implement the Q-learning algorithm as below, while using the $\epsilon$-greedy policy to choose the next action and greedy policy to  update the Q values：

```python
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
```

### 3.4 main.py

`main.py` instantiates the environment and test Sarsa and Q-learning separately.

```python
def test_sarsa(gridworld, episode_num, epsilon):
    policy, Q = sarsa.sarsa(gridworld, episode_num, epsilon)
    print('----------------sarsa----------------')
    print('Q Function:')
    print(np.around(np.reshape(Q,(4,12,4)), decimals=1))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    policy_show = ['x' for x in range(48)]
    for s in range(48):
        for a in range(4):
            if policy[s][a] == 1:
                policy_show[s] = a
    policy_show[47] = 'T'
    print(np.reshape(policy_show, gridworld.shape))
    print('episode Num:')
    print(episode_num)
    print('-------------------------------------')

def test_q_learning(gridworld, episode_num, epsilon):
    policy, Q = q_learning.q_learning(gridworld, episode_num, epsilon)
    print('----------------sarsa----------------')
    print('Q Function:')
    print(np.around(np.reshape(Q,(4,12,4)), decimals=1))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    policy_show = ['x' for x in range(48)]
    for s in range(48):
        for a in range(4):
            if policy[s][a] == 1:
                policy_show[s] = a
    policy_show[47] = 'T'
    print(np.reshape(policy_show, gridworld.shape))
    print('episode Num:')
    print(episode_num)
    print('-------------------------------------')
```

`test_sarsa()` and `test_q_learning()` separately use the method of Sarsa and Q-learning to finish the Cliff Walking. The Q values, the final policy and the number of episodes used are shown above.

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='q_learning')
    parser.add_argument('-n', '--episode_num', type=int, default=1000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    args = parser.parse_args()
    method = args.method
    episode_num = args.episode_num
    epsilon = args.epsilon

    gridworld = CliffWalkingEnv()
    if method == 'sarsa':
        test_sarsa(gridworld, episode_num, epsilon)
    if method == 'q_learning':
        test_q_learning(gridworld, episode_num, epsilon)
```

`main()` use parser to get the parameters entered on the command line, and depending on these parameters, the method of evaluation, the number of episodes used and the value of $\epsilon$ are chosen.

## 4 Experimental Result

#### When $\epsilon=0$:

The result of sarsa:

![image-20220407122404434](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122404434.png)

![image-20220407122430689](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122430689.png)

The result of Q-learning:

![image-20220407122315591](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122315591.png)

![image-20220407122332883](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122332883.png)

Both of the two methods choose the optimal path.

 When $\epsilon=0$, both the two methods are equal to TD(0), and the exploration is removed. That's the reason why the dangerous of cliff are not found by both the two methods.

#### When $\epsilon=0.1$:

The result of sarsa:

![image-20220407122019292](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122019292.png)

![image-20220407122043002](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122043002.png)

The result of Q-learning:

![image-20220407122221772](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122221772.png)

![image-20220407122238087](D:\Github\Reinforcement-Learning\Assignment3\Assignment 3.assets\image-20220407122238087.png)

Q-learning chooses the optimal path while Sarsa chooses the safer path. 

The reason of the difference is that when updating the Q values, on-policy Sarsa learns action values relative to the policy it follows, while off-policy Q-learning does it relative to the greedy policy.  So after the Q function is converged, the agent in Q-learning will never get into the Q value of the cliff while the agent in Sarsa has a certain probability to get into it. That is the reason why Sarsa choose to keep away from the cliff while Q-learning does not find the potential hazards.

## 5 Experimental Summary

- Systematically learn and understand Sarsa and Q-learning
- Implement and successfully run the algorithm of Sarsa and Q-learning based on pseudo-code
- Verify and Analyze the differences between the two methods based on the results