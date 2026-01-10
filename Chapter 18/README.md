# Chapter 18: Reinforcement Learning

## Overview

Reinforcement Learning (RL) is one of the most exciting fields in Machine Learning. The breakthrough came in 2013 when DeepMind demonstrated a system that could learn to play Atari games from scratch using only raw pixels, eventually outperforming humans. This led to AlphaGo's victory against world champions in 2016-2017.

## Learning to Optimize Rewards

In RL, a software agent:
- Makes **observations** within an environment
- Takes **actions** based on those observations
- Receives **rewards** (positive or negative)
- Learns to maximize expected rewards over time

### Key Applications
- **Robotics**: Agents control robots using sensors and motors, receiving rewards for reaching goals
- **Game Playing**: Ms. Pac-Man, Go, Atari games
- **Control Systems**: Smart thermostats anticipating user needs
- **Financial Trading**: Stock market decisions with monetary gains/losses
- **Self-driving cars, recommender systems, ad placement**

## Policy Search

The **policy** is the algorithm an agent uses to determine actions. It can be:
- **Deterministic**: Always chooses the same action for a given state
- **Stochastic**: Involves randomness in action selection

### Policy Search Approaches

1. **Brute Force**: Try many parameter combinations, pick the best
2. **Genetic Algorithms**: Create generations of policies, keep the best performers, create offspring with variations
3. **Policy Gradients (PG)**: Evaluate gradients of rewards with respect to policy parameters, follow gradients toward higher rewards

## OpenAI Gym

OpenAI Gym provides simulated environments for training RL agents.

### Basic Usage

```python
import gym

# Create environment
env = gym.make("CartPole-v1")
obs = env.reset()
# obs: array([-0.01258566, -0.00156614, 0.04207708, -0.00180545])

# Render environment
env.render()

# Check available actions
env.action_space  # Discrete(2) - accelerate left (0) or right (1)

# Take action
action = 1  # accelerate right
obs, reward, done, info = env.step(action)
```

### CartPole Environment
- **Observations**: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
- **Actions**: 0 (left) or 1 (right)
- **Goal**: Keep pole upright for 200 steps

### Simple Policy Example

```python
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
```

## Neural Network Policies

Instead of hardcoding policies, use neural networks to learn optimal actions.

```python
import tensorflow as tf
from tensorflow import keras

n_inputs = 4  # env.observation_space.shape[0]
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])
```

The network outputs action probabilities. For CartPole, one output neuron gives probability p of action 0 (left); action 1 (right) has probability 1-p.

**Why random selection?** This balances exploration (trying new actions) vs exploitation (using known good actions).

## Evaluating Actions: The Credit Assignment Problem

**Challenge**: When rewards are delayed, how do we know which actions were good?

**Solution**: Evaluate actions based on the **return** - the sum of all discounted future rewards:

- Return = r₀ + γ·r₁ + γ²·r₂ + ... where γ is the discount factor (0.9-0.99)
- **Action advantage**: How much better an action is compared to other possible actions (normalized returns)

## Policy Gradients

The **REINFORCE algorithm** (1992):

1. Let the neural network play several episodes
2. At each step, compute gradients that would make chosen actions more likely (don't apply yet)
3. Compute each action's advantage
4. Multiply gradients by corresponding advantages (positive advantage → apply gradients; negative → apply opposite)
5. Take mean of gradient vectors and perform Gradient Descent

### Implementation

```python
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads
```

### Discount and Normalize Rewards

```python
def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                               for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]
```

### Training Loop

```python
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                        discount_factor)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
             for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
```

**Limitation**: Policy gradients are sample inefficient - they need many episodes to make progress.

## Markov Decision Processes (MDPs)

**Markov Chain**: Stochastic process with fixed states, randomly evolving between states with fixed probabilities depending only on current state (no memory).

**MDP**: Markov chain where:
- Agent can choose from several actions at each step
- Transition probabilities depend on chosen action
- Some transitions return rewards
- Goal: Find policy maximizing rewards over time

### Bellman Optimality Equation

The optimal state value V*(s) satisfies:

**V*(s) = max_a Σ_{s'} T(s,a,s')[R(s,a,s') + γ·V*(s')]**

Where:
- T(s,a,s') = transition probability from s to s' given action a
- R(s,a,s') = reward for transition s→s' with action a
- γ = discount factor

### Value Iteration Algorithm

**V_{k+1}(s) ← max_a Σ_{s'} T(s,a,s')[R(s,a,s') + γ·V_k(s')]**

Starting with V₀(s) = 0, iterate until convergence to optimal values.

### Q-Value Iteration

For state-action pairs, the optimal Q-Value Q*(s,a) satisfies:

**Q_{k+1}(s,a) ← Σ_{s'} T(s,a,s')[R(s,a,s') + γ·max_{a'} Q_k(s',a')]**

The optimal policy: **π*(s) = argmax_a Q*(s,a)**

```python
# Define MDP
transition_probabilities = [  # shape=[s, a, s']
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]]
rewards = [  # shape=[s, a, s']
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]

# Initialize Q-Values
Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0

# Run Q-Value Iteration
gamma = 0.90
for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp]
                * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])
```

## Temporal Difference Learning (TD Learning)

When transition probabilities and rewards are unknown, use TD Learning:

**V_{k+1}(s) ← (1-α)V_k(s) + α[r + γ·V_k(s')]**

Equivalently: **V_{k+1}(s) ← V_k(s) + α·δ_k(s,r,s')**

Where δ_k(s,r,s') = r + γ·V_k(s') - V_k(s) is the **TD error**.

## Q-Learning

Adaptation of Q-Value Iteration for unknown MDPs:

**Q(s,a) ← (1-α)Q(s,a) + α[r + γ·max_{a'} Q(s',a')]**

The algorithm watches an agent play and gradually improves Q-Value estimates.

```python
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward

def exploration_policy(state):
    return np.random.choice(possible_actions[state])

alpha0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.90   # discount factor
state = 0

for iteration in range(10000):
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state])
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state
```

Q-Learning is **off-policy**: the policy being trained differs from the exploration policy.

### Exploration Policies

**ε-greedy policy**: Act randomly with probability ε, greedily with probability 1-ε. Start with high ε (e.g., 1.0), gradually reduce to low value (e.g., 0.05).

**Exploration function**: Add curiosity bonus to Q-Values:
**Q(s,a) ← (1-α)Q(s,a) + α[r + γ·max_{a'} f(Q(s',a'), N(s',a'))]**

Where f(Q,N) = Q + κ/(1+N) encourages trying unexplored actions.

## Deep Q-Learning (DQN)

For large state spaces, use a Deep Q-Network to approximate Q-Values: **Q_θ(s,a)**

### Target Q-Value

**Q_target(s,a) = r + γ·max_{a'} Q_θ(s',a')**

Train the DQN to minimize squared error between predicted and target Q-Values.

### Implementation

```python
env = gym.make("CartPole-v0")
input_shape = [4]
n_outputs = 2

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

# Replay buffer
from collections import deque
replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

# Training step
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)
```

**Challenge**: Catastrophic forgetting - agent may forget what it learned as it explores new areas.

## DQN Variants

### 1. Fixed Q-Value Targets

Use two networks:
- **Online model**: Learns at each step, moves agent
- **Target model**: Clone of online model, used only for computing targets
- Update target model periodically (e.g., every 50 episodes)

```python
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

# In training_step(), use target model:
next_Q_values = target.predict(next_states)

# Periodically update target:
if episode % 50 == 0:
    target.set_weights(model.get_weights())
```

### 2. Double DQN

Reduces Q-Value overestimation:
- Use online model to select best actions for next states
- Use target model to estimate Q-Values of those actions

```python
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * next_best_Q_values)
    # ... rest same as before
```

### 3. Prioritized Experience Replay (PER)

Sample important experiences more frequently:
- Priority p = |δ| (TD error magnitude)
- Sampling probability P ∝ p^ζ (ζ=0.6 typical)
- Training weight w = (nP)^(-β) to compensate bias (β: 0.4→1.0)

### 4. Dueling DQN

Separate value and advantage estimation:
**Q(s,a) = V(s) + A(s,a)** where A(s,a*) = 0 for best action a*

```python
K = keras.backend
input_states = keras.layers.Input(shape=[4])
hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
state_values = keras.layers.Dense(1)(hidden2)
raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
Q_values = state_values + advantages
model = keras.Model(inputs=[input_states], outputs=[Q_values])
```

## TF-Agents Library

TF-Agents is a TensorFlow-based RL library providing:
- Many environments (OpenAI Gym, PyBullet, DM Control, Unity ML-Agents)
- Algorithms (REINFORCE, DQN, DDQN, SAC, PPO)
- Replay buffers, metrics, and utilities

### Installation

```bash
python3 -m pip install -U tf-agents
python3 -m pip install -U 'gym[atari]'
```

### TF-Agents Environments

```python
from tf_agents.environments import suite_gym

env = suite_gym.load("Breakout-v4")

# Returns TimeStep objects
time_step = env.reset()
# TimeStep(step_type, reward, discount, observation)

time_step = env.step(1)  # Fire action

# Specifications
env.observation_spec()
env.action_spec()
env.time_step_spec()
```

### Atari Preprocessing

```python
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment

max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])

tf_env = TFPyEnvironment(env)
```

**Preprocessing steps**:
- Grayscale and downsample to 84×84
- Max pooling over last 2 frames (remove flickering)
- Frame skipping (agent sees every 4th frame)
- Stack 4 frames as observation

### Training Architecture

Components:
1. **Multiple environments**: Explore in parallel
2. **Driver**: Collects trajectories using collect policy
3. **Observer**: Receives trajectories (e.g., saves to replay buffer)
4. **Replay buffer**: Stores experiences
5. **Agent**: Trains networks using sampled batches
6. **Collect policy**: Updated by agent's learned networks

### Creating the DQN

```python
from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)
```

### Creating the Agent

```python
from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,  # initial ε
    decay_steps=250000 // update_period,
    end_learning_rate=0.01)     # final ε

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000,
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99,
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()
```

### Replay Buffer and Observer

```python
from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=1000000)

replay_buffer_observer = replay_buffer.add_batch
```

### Training Metrics

```python
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
log_metrics(train_metrics)
```

### Collect Driver

```python
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)

# Initial collection with random policy
from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                         tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=20000)

final_time_step, final_policy_state = init_driver.run()
```

### Dataset

```python
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)
```

### Training Loop

```python
from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)

train_agent(10000000)
```

## Popular RL Algorithms Overview

### Actor-Critic
Combines Policy Gradients with DQN:
- **Actor** (policy net): Chooses actions
- **Critic** (DQN): Estimates Q-Values
- Actor learns from critic's estimates (faster than vanilla PG)

### A3C (Asynchronous Advantage Actor-Critic)
- Multiple agents learn in parallel
- Asynchronous updates to master network
- Estimates advantages instead of Q-Values

### A2C (Advantage Actor-Critic)
- Synchronous version of A3C
- Better GPU utilization with larger batches

### SAC (Soft Actor-Critic)
- Learns rewards + maximizes action entropy
- Encourages exploration through unpredictability
- High sample efficiency

### PPO (Proximal Policy Optimization)
- Based on A2C with clipped loss function
- Prevents excessively large weight updates
- Used in OpenAI Five (Dota 2 champion)

### Curiosity-Based Exploration
- Agent seeks surprising outcomes
- Intrinsic rewards from failed predictions
- Effective even with sparse external rewards

## Key Takeaways

1. **RL is powerful but challenging**: Training is often unstable and highly sensitive to hyperparameters and random seeds

2. **Sample efficiency varies**: Policy Gradients are sample inefficient; DQN variants and SAC are more efficient

3. **Exploration matters**: ε-greedy, curiosity bonuses, and entropy maximization help agents discover good strategies

4. **Use proven libraries**: TF-Agents provides scalable, well-tested implementations

5. **Applications are growing**: Games, robotics, resource optimization, and autonomous systems

6. **Combine techniques**: Rainbow DQN showed benefits of combining multiple improvements (fixed targets, Double DQN, Dueling DQN, PER)

## Exercises

1. Define RL and contrast with supervised/unsupervised learning
2. Identify three RL applications with their environments, agents, actions, and rewards
3. Explain discount factor and its effect on optimal policy
4. Describe how to measure RL agent performance
5. Explain credit assignment problem and solutions
6. Explain replay buffer purpose
7. Define off-policy vs on-policy algorithms
8. Implement policy gradients for LunarLander-v2
9. Use TF-Agents to train SpaceInvaders-v4 agent
10. Build RL-powered Raspberry Pi robot

---

**References**: Sutton & Barto (1992), Mnih et al. (2013, 2015), Williams (1992), Bellman (1957), and various DeepMind papers on DQN improvements.