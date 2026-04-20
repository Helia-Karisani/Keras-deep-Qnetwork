# Keras Q-Learning for CartPole

## Overview

This project implements a simple Deep Q-Learning agent using Keras and Gymnasium.

The agent is trained on the `CartPole-v1` environment, where the goal is to balance a pole on a moving cart. Instead of using a traditional Q-table, the project uses a neural network to approximate Q-values for each possible action.

## What the Project Does

1. Installs and imports Gymnasium.
2. Checks NumPy and TensorFlow versions.
3. Creates the `CartPole-v1` environment.
4. Sets seeds for reproducibility.
5. Builds a neural network to approximate Q-values.
6. Uses epsilon-greedy action selection.
7. Stores experiences in replay memory.
8. Samples mini-batches from memory.
9. Updates Q-values using the Bellman equation.
10. Trains the model through experience replay.
11. Evaluates the trained agent over 10 episodes.

## Environment

The environment used is:

- `CartPole-v1`

The state has 4 values:

- cart position
- cart velocity
- pole angle
- pole angular velocity

The action space has 2 possible actions:

- move left
- move right

The goal is to keep the pole balanced for as long as possible.

## Model Structure

### Define the Q-Learning Model

Define a neural network using Keras to approximate the Q-values.

The network takes the state as input and outputs Q-values for each action.

Architecture:

- `Input(shape=(state_size,))`
- `Dense(24, activation='relu')`
- `Dense(24, activation='relu')`
- `Dense(action_size, activation='linear')`

For CartPole:

- `state_size = 4`
- `action_size = 2`

So the model takes a 4-dimensional state and outputs 2 Q-values, one for each action.

The final layer uses `linear` activation because Q-values are continuous numeric estimates, not probabilities.

## Q-Learning Implementation

### Implementing Q-learning

This part implements the Q-learning algorithm, which involves:

- interacting with the environment
- selecting actions
- storing experiences
- updating Q-values
- training the neural network

### Epsilon-Greedy Action Selection

The notebook uses epsilon-greedy exploration.

Initial values:

- `epsilon = 1.0`
- `epsilon_min = 0.01`
- `epsilon_decay = 0.99`

The agent chooses:

- a random action with probability `epsilon`
- the best predicted action otherwise

At the beginning, the agent explores heavily. Over time, epsilon decreases, so the agent relies more on the model.

### Replay Memory

Replay memory is implemented using:

- `deque(maxlen=2000)`

Each stored experience contains:

- current state
- action
- reward
- next state
- done flag

Replay memory helps the model train on random past experiences instead of only the most recent step.

### Bellman Update

For each sampled experience, the target value is updated using the Bellman equation.

If the episode is done:

- target = reward

If the episode is not done:

- target = reward + 0.95 * max future Q-value

The discount factor is:

- `gamma = 0.95`

This makes the agent consider future rewards, not only immediate rewards.

## Training

The notebook trains the agent for:

- `episodes = 10`
- maximum `200` steps per episode
- replay batch size `64`
- training every `5` steps

Training results:

- episode 1: score 30, epsilon 1.0
- episode 2: score 20, epsilon 1.0
- episode 3: score 21, epsilon 0.98
- episode 4: score 24, epsilon 0.93
- episode 5: score 16, epsilon 0.90
- episode 6: score 10, epsilon 0.88
- episode 7: score 14, epsilon 0.85
- episode 8: score 13, epsilon 0.83
- episode 9: score 27, epsilon 0.78
- episode 10: score 9, epsilon 0.76

The training scores are unstable and do not show consistent improvement.

## Evaluation

### Evaluate performance

This loop runs 10 episodes to test the trained agent.

The agent chooses actions based on the trained model and interacts with the environment.

Evaluation results:

- episode 1: score 9
- episode 2: score 8
- episode 3: score 9
- episode 4: score 8
- episode 5: score 8
- episode 6: score 9
- episode 7: score 9
- episode 8: score 9
- episode 9: score 9
- episode 10: score 9

## Result Analysis

The final result is weak.

The trained agent only survives for about 8 to 9 steps during evaluation. For `CartPole-v1`, a strong trained agent should survive for much longer, often close to the maximum limit of 500 steps.

This means the model did not learn a strong control policy.

Likely reasons:

- only 10 training episodes were used
- replay memory had limited useful experience
- epsilon was still high during training
- the model did not train long enough
- no target network was used
- no Double DQN or stabilization method was used
- evaluation scores are close to random behavior

So this notebook demonstrates the structure of Deep Q-Learning, but the current trained agent is not successful yet.

## Why the Optimizer and Loss Function Were Chosen

### Optimizer: Adam

The model uses Adam with:

- `learning_rate = 0.001`

Adam is a practical optimizer because it adapts learning rates during training and usually works better than plain stochastic gradient descent for neural networks.

Here, Adam is used to update the Q-network weights from replay-memory mini-batches.

### Loss Function: Mean Squared Error

The model uses:

- `loss='mse'`

Mean squared error is used because the model is predicting continuous Q-values.

The goal is to make the predicted Q-value closer to the Bellman target value. Since this is a regression problem, MSE is appropriate.

## Technical Characteristics

- reinforcement learning with Gymnasium
- Deep Q-Learning style model
- neural network Q-value approximation
- epsilon-greedy exploration
- replay memory with `deque`
- mini-batch replay training
- Bellman target updates
- Keras Sequential model
- linear output layer for Q-values
- CartPole control task

## Packages Used

- `gymnasium`
- `tensorflow`
- `numpy`
- `random`
- `collections`
- `os`
- `sys`
- `warnings`

Keras components used:

- `tensorflow.keras.models.Sequential`
- `tensorflow.keras.layers.Input`
- `tensorflow.keras.layers.Dense`
- `tensorflow.keras.optimizers.Adam`

## Files

- `Keras-Qlearning.ipynb`
- `README.md`

## Summary

This project demonstrates a simple Deep Q-Learning workflow in Keras using the CartPole environment. It builds a neural network to predict Q-values, uses epsilon-greedy exploration, stores experiences in replay memory, trains with Bellman targets, and evaluates the trained agent. The implementation is useful for understanding the structure of Deep Q-Learning, but the current training result is weak because the agent only survives for about 8 to 9 steps during evaluation.
