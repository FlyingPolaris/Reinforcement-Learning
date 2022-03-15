# Assignment 1

余北辰 519030910245



## 1 Experimental Purpose

- Build the Gridworld environment
- According to the idea of dynamic programming, respectively use policy iteration and value iteration to solve the shortest path problem in Gridworld 
- Compare the performance of each of the two methods

## 2 Experimental Background

In this section, we present the methods for policy iteration and value iteration separately.

### 2.1 Policy Iteration

According to the Bellman Equation, policy iteration uses alternating steps of "policy evaluation" and "policy improvement" to find a sequence of policies that improve once at a time and eventually converge to the optimal policy.

#### Policy Evaluation

Based on the current policy, traverse the entire MDP, and at each state iteratively update the value function using Bellman Expectation Equation, until the judgment condition of convergence is satisfied.

#### Policy Improvement

Based on the current value function, traverse the entire MDP, and at each state iteratively update the policy greedily. At each iteration, determine whether the policy has been stabilized. If the policy is already stable, return the existing policy and value function, otherwise repeat the policy evaluation.

The pseudo-code for the policy iteration procedure is shown below:

<img src="D:\Github\Reinforcement-Learning\Assignment1\Assignment1.assets\image-20220315085753487.png" alt="image-20220315085753487" style="zoom: 80%;" />

### 2.2 Value Iteration

Comparing with policy iteration, the process of value iteration is relatively simpler. In value iteration, the value function at each state is updated towards the optimal direction iteratively until the judgment condition of convergence is satisfied. After that, according to the obtained value function, the policy is obtained using Bellman Optimality Equation .

The pseudo-code for the value iteration procedure is shown below:

<img src="D:\Github\Reinforcement-Learning\Assignment1\Assignment1.assets\image-20220315091607926.png" alt="image-20220315091607926" style="zoom:80%;" />

## 3 Experimental Procedure

The source code for this experiment includes `gridworld.py`, `PolicyIteration.py`, `ValueIteration.py` and `main.py`.

### 3.1 gridworld.py

`gridworld.py` builds the Gridworld environment. This module references [reinforcement-learning/gridworld.py at master · dennybritz/reinforcement-learning (github.com)](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py), and is modified according to the specific scenario of this course experiment. 

The gridworld should be instantiated using the instruction as below:

```python
gridworld = GridworldEnv([6, 6])
```

The gridworld will be instantiated as below, in which state 1 and 35 are set as the terminal state.

<img src="D:\Github\Reinforcement-Learning\Assignment1\Assignment1.assets\image-20220315093539708.png" alt="image-20220315093539708" style="zoom:80%;" />

The number of the states, the number of the actions and the state transfer matrix in gridworld can be obtained using the instruction as below:

```python
state_number = gridworld.nS # 36
action_number = gridworld.nA # 4
state_transfer_matrix = gridworld.P # P[state][action] = (prob, next_state, reward, is_done)
```



### 3.2 PolicyIteration.py

`PolicyIteration.py` is implemented based on the above pseudo-code.

```python
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
```

In the policy evaluation part, first initialize the value function $V$ in each state to 0. Then using the Bellman Expectation Equation to  update $V$ once at one time, until the change of $V$ at a time is less than the predetermined threshold $\theta$. Record the time of iterations that doing policy evaluation. 

```python
def policy_iteration(gridworld, theta=0.001, gamma=0.99):
    policy = np.ones([gridworld.nS, gridworld.nA]) / gridworld.nA  # Init the policy randomly
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
```

When doing policy iteration, first of all the policy is initialized as random. Then using a flag `policy_stable` to record the change of policy in an iteration. Update the policy greedily, and stop when the policy does not change. 

### 3.3 ValueIteration.py

`ValueIteration.py` is implemented based on the above pseudo-code.

```python
def value_iteration(gridworld, theta=0.001, gamma=0.99):
    V = np.zeros(gridworld.nS)
    iteration = 0
    while True:
        delta = 0.0
        iteration += 1
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
        policy[state] = np.eye(gridworld.nA)[best_action]
    return V, policy, iteration
```

When doing value iteration, first update the value function iteratively. Record the time of iterations that update the value function. When the change of $V$ at a time is less than the predetermined threshold $\theta$, stop. After that, using Bellman Optimality Equation to get the optimal policy.

### 3.4 main.py

`main.py` instantiates the environment, and calls each module to find the shortest path.

```python
def test_policy_iteration(gridworld):
    V, policy, iteration = PolicyIteration.policy_iteration(gridworld)
    print('----------------Policy Iteration----------------')
    print('Value Function:')
    print(np.reshape(V, gridworld.shape))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    print(np.reshape(np.argmax(policy, axis=1), gridworld.shape))
    print('Iteration:')
    print(iteration)
    print('------------------------------------------------')


def test_value_iteration(gridworld):
    V, policy, iteration = ValueIteration.value_iteration(gridworld)
    print('----------------Value Iteration----------------')
    print('Value Function:')
    print(np.reshape(V, gridworld.shape))
    print('Policy:(UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3)')
    print(np.reshape(np.argmax(policy, axis=1), gridworld.shape))
    print('Iteration:')
    print(iteration)
    print('------------------------------------------------')
```

`test_policy_iteration()` and `test_value_iteration()` respectively output the results of policy iteration and value iteration : the final value function, the final policy and the number of iterations.

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='policy')
    args = parser.parse_args()
    method = args.method

    gridworld = GridworldEnv([6, 6])
    if method == 'policy':
        test_policy_iteration(gridworld)
    elif method == 'value':
        test_value_iteration(gridworld)
```

`main()` use parser to get the parameter entered on the command line, and depending on this parameter, policy or value iteration is decided.

## 4 Experimental Result

Run policy iteration:

![image-20220315111054102](D:\Github\Reinforcement-Learning\Assignment1\Assignment1.assets\image-20220315111054102.png)

Run value iteration:

![image-20220315111139564](D:\Github\Reinforcement-Learning\Assignment1\Assignment1.assets\image-20220315111139564.png)

We can see that the final value function and final policy are the same after policy iteration and value iteration. After testing, the final policy is indeed the shortest path, which means the result is accurate.

It is worth noting that the time of iteration are both 6 in policy iteration and value iteration. In policy iteration we calculate the time of iteration in policy evaluation, while in value iteration we calculate the time of iteration in updating the value function. So we can conclude that overall the two methods are equally efficient. However, in policy iteration, the policy should be updated after each time when the policy is evaluated, which make it run slower. So value iteration is more efficient in this scenario than policy iteration.

## 5 Experimental Summary

- Systematically learn and understand policy iteration and value iteration
- Implement and successfully run the algorithm of policy iteration and value iteration based on pseudo-code
- Learn the use of `numpy` and `argparse`