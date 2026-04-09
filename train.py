import numpy as np
import os
import json
from network_env import FiveGNetworkEnv
from dqn_agent import DQNAgent

# ── Setup ────────────────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

EPISODES    = 500
PRINT_EVERY = 50   # Print a summary every N episodes

env   = FiveGNetworkEnv(num_users=5, num_resource_blocks=10)
agent = DQNAgent(state_size=15, action_size=5)

# ── Tracking ─────────────────────────────────────────────────────
history = {
    "episode":      [],
    "reward":       [],
    "avg_reward":   [],   # Smoothed over last 20 episodes
    "fairness":     [],
    "throughput":   [],
    "satisfaction": [],
    "epsilon":      [],
    "loss":         []
}

best_avg_reward = -np.inf

print("=" * 55)
print("  Training the 5G RL Agent")
print(f"  Episodes: {EPISODES}  |  Users: {env.num_users}")
print("=" * 55)

# ── Main Training Loop ───────────────────────────────────────────
for episode in range(1, EPISODES + 1):

    state, _ = env.reset()

    ep_reward     = 0
    ep_fairness   = 0
    ep_throughput = 0
    ep_satisf     = 0
    ep_loss       = []
    steps         = 0

    # ── One full episode ─────────────────────────────────────────
    while True:
        # 1. Agent picks an action
        action = agent.act(state)

        # 2. Environment responds
        next_state, reward, done, _, info = env.step(action)

        # 3. Agent stores the memory
        agent.remember(state, action, reward, next_state, done)

        # 4. Agent learns from a random batch of memories
        loss = agent.learn()
        if loss is not None:
            ep_loss.append(loss)

        # 5. Move to next state
        state       = next_state
        ep_reward  += reward
        ep_fairness   += info["fairness"]
        ep_throughput += info["throughput"]
        ep_satisf     += info["satisfaction"]
        steps += 1

        if done:
            break

    # ── End of episode bookkeeping ───────────────────────────────
    agent.end_episode()

    avg_fairness   = ep_fairness   / steps
    avg_throughput = ep_throughput / steps
    avg_satisf     = ep_satisf     / steps
    avg_loss       = np.mean(ep_loss) if ep_loss else 0.0

    history["episode"].append(episode)
    history["reward"].append(ep_reward)
    history["fairness"].append(avg_fairness)
    history["throughput"].append(avg_throughput)
    history["satisfaction"].append(avg_satisf)
    history["epsilon"].append(agent.epsilon)
    history["loss"].append(avg_loss)

    # Smoothed reward — average of last 20 episodes
    recent = history["reward"][-20:]
    avg_reward = np.mean(recent)
    history["avg_reward"].append(avg_reward)

    # ── Save best model ──────────────────────────────────────────
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        agent.save("models/dqn_best.pth")

    # ── Checkpoint every 100 episodes ───────────────────────────
    if episode % 100 == 0:
        agent.save(f"models/dqn_ep{episode}.pth")

    # ── Print progress ───────────────────────────────────────────
    if episode % PRINT_EVERY == 0 or episode == 1:
        print(f"  Ep {episode:>4}  |  "
              f"reward={ep_reward:>7.2f}  "
              f"avg20={avg_reward:>7.2f}  |  "
              f"fairness={avg_fairness:.3f}  "
              f"satisfied={avg_satisf:.0%}  |  "
              f"ε={agent.epsilon:.3f}")

# ── Save training history ────────────────────────────────────────
with open("results/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 55)
print(f"  Training complete!")
print(f"  Best avg reward : {best_avg_reward:.3f}")
print(f"  Final epsilon   : {agent.epsilon:.4f}")
print(f"  History saved   → results/training_history.json")
print("=" * 55)