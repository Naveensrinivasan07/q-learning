# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given Reinforcement Learning environment using Q-Learning and comparing the state values with the First Visit Monte Carlo method.

## PROBLEM STATEMENT
For the given frozen lake environment, find the optimal policy applying the Q-Learning algorithm and compare the value functions obtained with that of First Visit Monte Carlo method. Plot graphs to analyse the difference visually.

## Q LEARNING ALGORITHM
# Step 1:
Store the number of states and actions in a variable, initialize arrays to store policy and action value function for each episode. Initialize an array to store the action value function.
# Step 2: 
Define function to choose action based on epsilon value which decides if exploration or exploitation is chosen.
# Step 3:
Create multiple learning rates and epsilon values.
# Step 4: 
Run loop for each episode, compute the action value function but in Q-Learning the maximum action value function is chosen instead of choosing the next state and next action's value. 
# Step 5:
Return the computed action value function and policy. Plot graph and compare with Monte Carlo results.
## Q LEARNING FUNCTION
### Name: NAVEEN S
### Register Number: 212222240070

## PROGRAM :
```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
      state, done=env.reset(), False
      while not done:
        action=select_action(state, Q, epsilons[e])
        next_state, reward, done, _ = env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

```





## OUTPUT:

<img width="499" height="367" alt="image" src="https://github.com/user-attachments/assets/f9bd6413-dd7e-434c-abd2-f7d49ccce16f" />

<img width="1049" height="774" alt="image" src="https://github.com/user-attachments/assets/de81891c-f7c1-4f34-bed7-12789e5d3267" />

<img width="679" height="179" alt="image" src="https://github.com/user-attachments/assets/c2cca248-241b-491c-917c-d7a5441f6ab1" />

<img width="1455" height="669" alt="image" src="https://github.com/user-attachments/assets/bce2a827-8f2d-488d-9c05-cc5a799e14eb" />

<img width="1466" height="660" alt="image" src="https://github.com/user-attachments/assets/b1db1061-74af-47cc-b4df-373c6e17a609" />



## RESULT:

Therefore, python program to find optimal policy using Q-Learning is developed and state value function obtained is compared with first visit monte carlo.
