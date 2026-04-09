import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("results", exist_ok=True)

# ── Load training history ─────────────────────────────────────────
with open("results/training_history.json", "r") as f:
    h = json.load(f)

episodes    = h["episode"]
rewards     = h["reward"]
avg_rewards = h["avg_reward"]
fairness    = h["fairness"]
throughput  = h["throughput"]
satisfaction = h["satisfaction"]
epsilon     = h["epsilon"]
loss        = h["loss"]

def smooth(values, window=15):
    """Rolling average to make noisy lines readable."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return out

# ── Plot ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("5G RL Agent — Training Results", fontsize=16, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])   # Reward — wide
ax2 = fig.add_subplot(gs[0, 2])    # Epsilon
ax3 = fig.add_subplot(gs[1, 0])    # Fairness
ax4 = fig.add_subplot(gs[1, 1])    # Throughput
ax5 = fig.add_subplot(gs[1, 2])    # Loss

# ── 1. Reward curve (main chart) ─────────────────────────────────
ax1.plot(episodes, rewards,     color="#b0c4de", linewidth=0.7, alpha=0.5, label="Episode reward")
ax1.plot(episodes, avg_rewards, color="#1a6faf", linewidth=2.2, label="Avg (last 20 eps)")
ax1.plot(episodes, smooth(rewards, 30), color="#e07b39", linewidth=1.5,
         linestyle="--", label="Smoothed trend")
ax1.axhline(rewards[0], color="gray", linewidth=0.8, linestyle=":", label=f"Starting reward ({rewards[0]:.1f})")
ax1.set_title("Reward over training", fontweight="bold")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total reward")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Annotate improvement
start_avg = np.mean(rewards[:20])
end_avg   = np.mean(rewards[-20:])
improvement = ((end_avg - start_avg) / abs(start_avg)) * 100
ax1.annotate(f"+{improvement:.1f}% improvement",
             xy=(episodes[-1], end_avg),
             xytext=(episodes[-1] * 0.65, end_avg * 1.05),
             fontsize=9, color="#1a6faf",
             arrowprops=dict(arrowstyle="->", color="#1a6faf", lw=1.2))

# ── 2. Epsilon decay ─────────────────────────────────────────────
ax2.plot(episodes, epsilon, color="#7b4ea0", linewidth=2)
ax2.fill_between(episodes, epsilon, alpha=0.15, color="#7b4ea0")
ax2.set_title("Exploration rate (ε)", fontweight="bold")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.annotate("Exploring\n(random)", xy=(50, 0.85), fontsize=8, color="#7b4ea0")
ax2.annotate("Exploiting\n(learned)", xy=(380, 0.12), fontsize=8, color="#7b4ea0")

# ── 3. Fairness ───────────────────────────────────────────────────
ax3.plot(episodes, fairness,        color="#c0c0c0", linewidth=0.7, alpha=0.5)
ax3.plot(episodes, smooth(fairness, 20), color="#2e8b57", linewidth=2, label="Smoothed")
ax3.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="Perfect fairness")
ax3.set_title("Network fairness", fontweight="bold")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Jain's fairness index")
ax3.set_ylim(0.4, 1.05)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── 4. Throughput ─────────────────────────────────────────────────
ax4.plot(episodes, throughput,        color="#c0c0c0", linewidth=0.7, alpha=0.5)
ax4.plot(episodes, smooth(throughput, 20), color="#cc4125", linewidth=2, label="Smoothed")
ax4.set_title("Average throughput", fontweight="bold")
ax4.set_xlabel("Episode")
ax4.set_ylabel("Normalised throughput")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── 5. Training loss ──────────────────────────────────────────────
nonzero_eps  = [e for e, l in zip(episodes, loss) if l > 0]
nonzero_loss = [l for l in loss if l > 0]
if nonzero_loss:
    ax5.plot(nonzero_eps, nonzero_loss,         color="#c0c0c0", linewidth=0.5, alpha=0.5)
    ax5.plot(nonzero_eps, smooth(nonzero_loss, 20),
             color="#e07b39", linewidth=2, label="Smoothed")
    ax5.set_title("Training loss", fontweight="bold")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("MSE loss")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

# ── Summary box ───────────────────────────────────────────────────
summary = (
    f"Training summary\n"
    f"─────────────────\n"
    f"Episodes : {len(episodes)}\n"
    f"Start reward  : {np.mean(rewards[:20]):.2f}\n"
    f"Final reward  : {np.mean(rewards[-20:]):.2f}\n"
    f"Improvement   : +{improvement:.1f}%\n"
    f"Final fairness: {np.mean(fairness[-20:]):.3f}\n"
    f"Final epsilon : {epsilon[-1]:.4f}"
)
fig.text(0.01, 0.01, summary, fontsize=8, verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

plt.savefig("results/training_results.png", dpi=150, bbox_inches="tight")
print("Chart saved → results/training_results.png")
plt.show()