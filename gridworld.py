import numpy as np
import random

# --------------------------
# 1. ENVIRONMENT (GridWorld)
# --------------------------
class GridWorld:
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
        self.state = (0, 0)  # Start at top-left

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        if action == 0 and x > 0: x -= 1
        elif action == 1 and y < self.size - 1: y += 1
        elif action == 2 and x < self.size - 1: x += 1
        elif action == 3 and y > 0: y -= 1

        self.state = (x, y)

        # Reward system
        if self.state == self.goal:
            return self.state, 10, True  # Goal reached
        else:
            return self.state, -1, False  # Small penalty for each step


# --------------------------
# 2. Q-LEARNING AGENT
# --------------------------
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.alpha = alpha    # Learning rate
        self.gamma = gamma    # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.action_size = action_size

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploit

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        best_next = np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.alpha * (reward + self.gamma * best_next - self.q_table[x, y, action])


# --------------------------
# 3. TRAINING LOOP
# --------------------------
env = GridWorld(size=5)
agent = QLearningAgent(state_size=5, action_size=4)

episodes = 500
for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

print("✅ Training finished!")
print("Q-Table Learned:")
print(agent.q_table)

# --------------------------
# 4. TEST THE AGENT
# --------------------------
state = env.reset()
done = False
steps = []

while not done:
    action = agent.choose_action(state)
    steps.append(state)
    state, reward, done = env.step(action)

steps.append(state)
print("🏁 Path taken by agent:", steps)
