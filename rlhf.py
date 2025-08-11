import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Hyperparameters
VOCAB = ['start', 'the', 'cat', 'dog', 'is', 'happy', 'sad', 'runs', 'success', 'failure', 'end']  # Small vocabulary
VOCAB_SIZE = len(VOCAB)
STATE_DIM = 32  # Embedding dimension for state
HIDDEN_DIM = 64  # Policy network hidden size
MAX_LENGTH = 5  # Max sentence length (episode length)
GAMMA = 0.99  # Discount factor
LR = 0.001  # Learning rate
EPISODES = 1000  # Training episodes
BATCH_SIZE = 32  # For sampling episodes

# Token to index mapping
token_to_idx = {token: idx for idx, token in enumerate(VOCAB)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Reward function: Simulate human feedback
def compute_reward(sentence):
    # Positive if contains 'happy' or 'success', negative otherwise
    if 'happy' in sentence or 'success' in sentence:
        return 1.0
    else:
        return -1.0

# Policy Network: Simple MLP to predict next token logits from state embedding
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, state_dim)  # Embed tokens
        self.fc1 = nn.Linear(state_dim * MAX_LENGTH, hidden_dim)  # Flatten sequence
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # Output logits

    def forward(self, state):
        # State is list of token indices, pad to MAX_LENGTH
        state_padded = state + [0] * (MAX_LENGTH - len(state))  # Pad with 'start' index 0
        state_tensor = torch.tensor(state_padded, dtype=torch.long)
        emb = self.embedding(state_tensor).view(-1)  # Flatten embeddings
        x = F.relu(self.fc1(emb))
        logits = self.fc2(x)
        return logits

# Function to sample action from policy
def sample_action(policy, state):
    logits = policy(state)
    probs = F.softmax(logits, dim=-1)
    action = torch.multinomial(probs, 1).item()
    return action, torch.log(probs[action])  # Action and log prob

# Generate an episode
def generate_episode(policy):
    state = [token_to_idx['start']]  # Initial state
    log_probs = []
    actions = []
    rewards = [0] * MAX_LENGTH  # Rewards are 0 until end

    for t in range(MAX_LENGTH):
        action, log_prob = sample_action(policy, state)
        actions.append(action)
        log_probs.append(log_prob)
        state.append(action)  # Update state

        if idx_to_token[action] == 'end':
            break  # Early termination if 'end' is chosen

    # Compute reward at end
    sentence = [idx_to_token[idx] for idx in state[1:]]  # Exclude start
    reward = compute_reward(sentence)
    rewards[-1] = reward  # Assign to last step

    return state, actions, log_probs, rewards

# Compute discounted returns
def compute_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return returns

# Training loop
def train():
    policy = PolicyNetwork(STATE_DIM, HIDDEN_DIM, VOCAB_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    for episode in range(EPISODES):
        state, actions, log_probs, rewards = generate_episode(policy)
        returns = compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Policy gradient loss: -sum(log_prob * return)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            sentence = ' '.join([idx_to_token[idx] for idx in state[1:]])
            print(f"Episode {episode}: Sentence: '{sentence}' | Reward: {rewards[-1]}")

    return policy

# Run training
if __name__ == "__main__":
    trained_policy = train()

    # Test: Generate a sample sentence after training
    print("\nGenerating a sample sentence with trained policy:")
    state = [token_to_idx['start']]
    for _ in range(MAX_LENGTH):
        action, _ = sample_action(trained_policy, state)
        state.append(action)
        if idx_to_token[action] == 'end':
            break
    sentence = ' '.join([idx_to_token[idx] for idx in state[1:]])
    print(f"Sample: '{sentence}' | Reward: {compute_reward([idx_to_token[idx] for idx in state[1:]])}")