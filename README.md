# 5G Intelligent Resource Allocation with Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen?logo=github-actions)

A complete Deep Reinforcement Learning project that trains a DQN agent to dynamically allocate bandwidth, power, and spectrum in a simulated 5G network — outperforming both random and fixed-rule baselines by ~8% in total reward and ~18% in fairness.

---

## What This Project Does

Traditional 5G networks use fixed rules to decide who gets bandwidth. This project replaces those rules with a trained AI agent that watches the network state every millisecond and learns the optimal allocation strategy through trial and error.

| Agent | Avg Reward | Fairness | Throughput |
|---|---|---|---|
| Random (baseline) | 58.7 | 0.767 | 0.120 |
| Fixed rule (best-signal) | 58.7 | 0.697 | 0.136 |
| **DQN (ours)** | **63.4** | **0.825** | 0.116 |

The DQN agent learns the fairness-throughput trade-off on its own — nobody hard-codes it. It sacrifices a small amount of raw throughput to serve all users more equitably, which is exactly what real network operators want.

---

## Project Structure

```
5g_rl_project/
│
├── network_env.py          # 5G network simulation (Gymnasium environment)
├── dqn_agent.py            # DQN neural network + replay buffer + agent logic
├── train.py                # Training loop — runs 500 episodes
├── plot_results.py         # Training dashboard — 5 charts
├── evaluate.py             # Head-to-head evaluation vs baselines
│
├── mlops/
│   ├── train_mlflow.py     # MLflow-instrumented training (experiment tracking)
│   ├── mlflow_evaluate.py  # Evaluation with MLflow logging
│   └── requirements_mlops.txt
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
│
├── models/                 # Saved model checkpoints (created at runtime)
│   ├── dqn_best.pth
│   └── dqn_ep500.pth
│
├── results/                # Charts and logs (created at runtime)
│   ├── training_results.png
│   ├── evaluation_results.png
│   ├── training_history.json
│   └── evaluation_summary.json
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/5g_rl_project.git
cd 5g_rl_project
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv 5g_rl_env

# Activate — Windows
5g_rl_env\Scripts\activate

# Activate — Mac / Linux
source 5g_rl_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If PyTorch fails to install with a DLL error, use the CPU-specific build:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
> ```

### 4. Verify the environment works

```bash
python test_env.py
```

Expected output:
```
Number of users:           5
State size (what AI sees): 15 numbers
Number of actions:         5
Environment is working correctly!
```

### 5. Train the agent

```bash
python train.py
```

Training runs 500 episodes (~2–4 minutes on CPU). Progress prints every 50 episodes. Models are saved to `models/`.

### 6. Plot training results

```bash
python plot_results.py
```

Saves `results/training_results.png` — reward curve, epsilon decay, fairness, throughput, loss.

### 7. Evaluate against baselines

```bash
python evaluate.py
```

Runs Random vs Fixed Rule vs DQN across 100 test episodes and saves `results/evaluation_results.png`.

---

## How It Works

### The Environment (`network_env.py`)

A simulated 5G base station serving 5 users. Each time step (1 ms slot), the environment provides:

- **State (15 numbers):** demand[5] + channel_quality[5] + current_allocation[5]
- **Action (1 of 5):** allocation strategy to apply
- **Reward:** weighted combination of throughput, Jain's fairness index, user satisfaction, and waste penalty

### The Agent (`dqn_agent.py`)

A Deep Q-Network (DQN) with:

- **Policy network:** 15 → 128 → 64 → 5 (ReLU activations)
- **Target network:** soft-updated every 10 episodes for training stability
- **Replay buffer:** stores 10,000 experiences, samples random batches of 64
- **Epsilon-greedy:** starts at ε=1.0 (pure exploration), decays to ε=0.05

### The Training Loop (`train.py`)

Runs 500 episodes. Each episode:
1. Reset the environment to random conditions
2. Agent observes state → picks action (ε-greedy) → receives reward
3. Experience stored in replay buffer
4. Agent samples a random batch → computes Bellman loss → backpropagates
5. Best model saved whenever average reward improves

---

## MLOps — Experiment Tracking with MLflow

This project includes full MLflow integration so every training run is tracked, versioned, and comparable.

### Install MLflow

```bash
pip install mlflow
```

### Run tracked training

```bash
python mlops/train_mlflow.py
```

### Launch the MLflow dashboard

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser. You will see every run with its hyperparameters, metrics, and saved artifacts side by side.

What gets tracked per run:

| Category | What is logged |
|---|---|
| Parameters | episodes, batch_size, gamma, epsilon_decay, learning_rate, num_users |
| Metrics (per episode) | reward, fairness, throughput, satisfaction, epsilon, loss |
| Artifacts | dqn_best.pth, training_results.png, evaluation_summary.json |

### Comparing runs

Change a hyperparameter (e.g. `EPISODES = 1000` or `gamma = 0.99`) and run `train_mlflow.py` again. MLflow records both runs separately so you can compare them visually in the dashboard.

---

## CI/CD with GitHub Actions

Every push to `main` automatically runs the full pipeline: install → test environment → train (short) → evaluate.

The workflow file is at `.github/workflows/ci.yml`.

To enable it: push your code to GitHub. The Actions tab will show the pipeline running on every commit.

---

## Requirements

```
numpy
matplotlib
gymnasium
torch
mlflow
```

See `requirements.txt` for pinned versions.

---

## Key Concepts

| Term | Plain English |
|---|---|
| 5G | The latest mobile network generation — faster, denser, harder to manage |
| Resource allocation | Deciding how to split bandwidth, power, spectrum among users |
| Reinforcement Learning | AI learns by trial and error — rewarded for good decisions |
| DQN | Deep Q-Network — uses a neural network to estimate "value of each action" |
| Replay buffer | Memory bank — stores past experiences so the AI can learn from them randomly |
| Epsilon-greedy | Exploration strategy — starts random, gradually shifts to using what it learned |
| Jain's Fairness Index | Score from 0–1 measuring how equally resources are distributed |

---

## Results

After 500 training episodes:

- Reward improved **34%** from first to last episode
- Fairness reached **0.918** (near-perfect equal treatment)
- DQN beat both baselines by **~8%** in total reward
- DQN beat the fixed rule by **18%** in fairness

---

## License

MIT — free to use, modify, and distribute.

---

## Author

Built as a portfolio project demonstrating Deep Reinforcement Learning applied to wireless network resource management.