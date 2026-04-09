import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ─────────────────────────────────────────────
# PART A: The Neural Network (the "brain")
# ─────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    A simple neural network with 2 hidden layers.

    Think of it as a function:
        state (15 numbers)  →  Q-values (5 numbers)

    Each Q-value answers: "if I pick THIS action right now,
    how much total reward do I expect to collect?"
    """
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),  # Input layer  → Hidden layer 1 (128 neurons)
            nn.ReLU(),                   # Activation: "only pass positive signals through"
            nn.Linear(128, 64),          # Hidden layer 1 → Hidden layer 2 (64 neurons)
            nn.ReLU(),
            nn.Linear(64, action_size)   # Hidden layer 2 → Output (5 Q-values)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# PART B: The Replay Buffer (the "memory")
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores the agent's past experiences so it can learn from them later.

    Each memory is a tuple of:
        (state, action, reward, next_state, done)

    Plain English: "I was in situation X, I did action Y,
    I got reward Z, then the world looked like W."

    We keep the last 10,000 memories and sample randomly from them.
    Random sampling breaks correlations — without it, the AI would
    only learn from recent experience and forget everything else.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # Auto-drops oldest when full

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# PART C: The DQN Agent (the "student")
# ─────────────────────────────────────────────

class DQNAgent:
    """
    The full agent: has a brain (network), a memory (buffer),
    and logic for how to act and how to learn.
    """
    def __init__(self, state_size=15, action_size=5):

        self.state_size  = state_size
        self.action_size = action_size

        # Use GPU if available, otherwise CPU — makes no difference for small models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent running on: {self.device}")

        # Two networks: policy (actively learning) + target (stable reference)
        # The target network is a "frozen copy" updated slowly.
        # Without it, the AI would chase a moving target and never converge.
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        # Adam optimizer — adjusts learning rate automatically
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # Memory
        self.memory = ReplayBuffer(capacity=10000)

        # ── Exploration settings (epsilon-greedy) ──
        # At the start the agent knows nothing, so it explores randomly.
        # Over time it shifts from random exploration → using what it learned.
        self.epsilon       = 1.0    # 100% random at start
        self.epsilon_min   = 0.05   # Never go below 5% random (keeps learning)
        self.epsilon_decay = 0.995  # Multiply epsilon by this after each episode

        # ── Learning settings ──
        self.batch_size    = 64     # Learn from 64 memories at a time
        self.gamma         = 0.95   # How much to value future rewards vs immediate ones
                                    # 0 = only care about now, 1 = care about far future
        self.target_update = 10     # Copy policy → target every 10 episodes

        self.episode_count = 0

    # ── Acting ──────────────────────────────

    def act(self, state):
        """
        Choose an action given the current state.
        Uses epsilon-greedy: flip a coin — if random < epsilon, explore;
        otherwise exploit what we've learned.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)  # Random exploration

        # Convert state to tensor and ask the network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()  # Pick action with highest Q-value

    # ── Remembering ─────────────────────────

    def remember(self, state, action, reward, next_state, done):
        """Store one experience in memory."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning ────────────────────────────

    def learn(self):
        """
        The core learning step — called after every action.
        Uses the Bellman equation to update Q-value predictions.

        Plain English: "Given what actually happened vs what I predicted,
        how wrong was I? Adjust the network to be less wrong next time."
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough memories yet — skip

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors (the format PyTorch understands)
        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # What did our network predict we'd get?
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # What does the Bellman equation say we SHOULD have predicted?
        with torch.no_grad():
            next_q      = self.target_net(next_states).max(1)[0]
            target_q    = rewards + self.gamma * next_q * (1 - dones)

        # How wrong were we? (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagation — adjust the network weights to reduce the error
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Prevent huge jumps
        self.optimizer.step()

        return loss.item()

    def end_episode(self):
        """Call this at the end of each episode."""
        self.episode_count += 1

        # Decay epsilon — become less random over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically sync target network with policy network
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save the trained model to disk."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
            'episode':    self.episode_count
        }, filepath)
        print(f"Model saved → {filepath}")

    def load(self, filepath):
        """Load a previously trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon       = checkpoint['epsilon']
        self.episode_count = checkpoint['episode']
        print(f"Model loaded ← {filepath}")