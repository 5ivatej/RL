import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 1. Neural Network (Q-function approximator)
# --------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


# --------------------------
# 2. Agent with Experience Replay
# --------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=50000)  # replay buffer
        self.gamma = 0.99     # discount factor
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 1e-3

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            target = self.model(state_tensor).detach().clone()
            if done:
                target[action] = reward
            else:
                future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.gamma * future_q

            states.append(state_tensor)
            targets.append(target)

        states = torch.stack(states)
        targets = torch.stack(targets)

        preds = self.model(states)
        loss = self.loss_fn(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --------------------------
# 3. Training Loop
# --------------------------
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4 values (pos, vel, angle, ang.vel)
action_size = env.action_space.n            # 2 actions (left, right)
agent = DQNAgent(state_size, action_size)

episodes = 500
target_update_freq = 10

for ep in range(episodes):
    state, _ = env.reset()   # <- Gymnasium reset returns (state, info)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay()

    if ep % target_update_freq == 0:
        agent.update_target()

    print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

env.close()
print("✅ Training finished!")
