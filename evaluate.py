import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from network_env import FiveGNetworkEnv
from dqn_agent import DQNAgent

os.makedirs("results", exist_ok=True)

EVAL_EPISODES = 100   # Run each agent for 100 full episodes
NUM_USERS     = 5

# ─────────────────────────────────────────────────────────────────
# The three agents we are comparing
# ─────────────────────────────────────────────────────────────────

def run_agent(agent_type, episodes=100):
    """
    Run one agent for N episodes and record everything.

    agent_type options:
        "random"  — picks a random action every step
        "fixed"   — always picks action 2 (favour best signal)
        "dqn"     — uses our trained neural network
    """
    env = FiveGNetworkEnv(num_users=NUM_USERS)

    # Set up the DQN agent if needed
    dqn = None
    if agent_type == "dqn":
        dqn = DQNAgent(state_size=15, action_size=5)
        dqn.load("models/dqn_best.pth")
        dqn.epsilon = 0.0   # No exploration during evaluation — pure learned policy

    records = {
        "rewards":      [],
        "fairness":     [],
        "throughput":   [],
        "satisfaction": [],
        "actions":      []   # Track which actions were chosen
    }

    for ep in range(episodes):
        state, _ = env.reset()

        ep_reward     = 0
        ep_fairness   = 0
        ep_throughput = 0
        ep_satisf     = 0
        ep_actions    = []
        steps         = 0

        while True:
            # Pick action based on agent type
            if agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "fixed":
                action = 2   # Always: favour best signal user
            else:
                action = dqn.act(state)

            state, reward, done, _, info = env.step(action)

            ep_reward     += reward
            ep_fairness   += info["fairness"]
            ep_throughput += info["throughput"]
            ep_satisf     += info["satisfaction"]
            ep_actions.append(action)
            steps += 1

            if done:
                break

        records["rewards"].append(ep_reward)
        records["fairness"].append(ep_fairness / steps)
        records["throughput"].append(ep_throughput / steps)
        records["satisfaction"].append(ep_satisf / steps)
        records["actions"].extend(ep_actions)

    return records


# ─────────────────────────────────────────────────────────────────
# Run all three agents
# ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Phase 5 — Head-to-Head Evaluation")
print(f"  Each agent runs {EVAL_EPISODES} episodes")
print("=" * 55)

print("\n  Running Random agent    ...", end=" ", flush=True)
r_random = run_agent("random", EVAL_EPISODES)
print("done")

print("  Running Fixed rule agent...", end=" ", flush=True)
r_fixed  = run_agent("fixed",  EVAL_EPISODES)
print("done")

print("  Running DQN agent       ...", end=" ", flush=True)
r_dqn    = run_agent("dqn",    EVAL_EPISODES)
print("done")


# ─────────────────────────────────────────────────────────────────
# Print the results table
# ─────────────────────────────────────────────────────────────────

def stats(values):
    return np.mean(values), np.std(values)

metrics = ["rewards", "fairness", "throughput", "satisfaction"]
labels  = ["Avg reward", "Fairness", "Throughput", "Satisfaction"]
agents  = {"Random": r_random, "Fixed rule": r_fixed, "DQN (ours)": r_dqn}

print("\n" + "=" * 63)
print(f"  {'Metric':<16} {'Random':>14} {'Fixed rule':>14} {'DQN (ours)':>14}")
print("  " + "-" * 59)
for m, label in zip(metrics, labels):
    row = f"  {label:<16}"
    for name, rec in agents.items():
        mean, std = stats(rec[m])
        row += f"  {mean:>6.3f} ±{std:.3f}"
    print(row)
print("=" * 63)

# Improvement of DQN over random
dqn_reward    = np.mean(r_dqn["rewards"])
random_reward = np.mean(r_random["rewards"])
fixed_reward  = np.mean(r_fixed["rewards"])
improvement_vs_random = ((dqn_reward - random_reward) / abs(random_reward)) * 100
improvement_vs_fixed  = ((dqn_reward - fixed_reward)  / abs(fixed_reward))  * 100

print(f"\n  DQN vs Random    : {improvement_vs_random:+.1f}%")
print(f"  DQN vs Fixed rule: {improvement_vs_fixed:+.1f}%")


# ─────────────────────────────────────────────────────────────────
# Save detailed results to JSON
# ─────────────────────────────────────────────────────────────────

summary = {
    "random":     {m: float(np.mean(r_random[m]))     for m in metrics},
    "fixed":      {m: float(np.mean(r_fixed[m]))      for m in metrics},
    "dqn":        {m: float(np.mean(r_dqn[m]))        for m in metrics},
    "dqn_vs_random_pct": round(improvement_vs_random, 2),
    "dqn_vs_fixed_pct":  round(improvement_vs_fixed,  2),
}
with open("results/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────
# Plot the comparison charts
# ─────────────────────────────────────────────────────────────────

colors = {"Random": "#9e9e9e", "Fixed rule": "#1d9e75", "DQN (ours)": "#534ab7"}
names  = list(agents.keys())
fig    = plt.figure(figsize=(16, 10))
fig.suptitle("Phase 5 — Agent Comparison: Random vs Fixed Rule vs DQN",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Chart 1: Reward per episode (line) ───────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, rec, ls in zip(names, [r_random, r_fixed, r_dqn], ["-", "--", "-"]):
    ax1.plot(rec["rewards"], color=colors[name], linewidth=1.2,
             linestyle=ls, alpha=0.6, label=f"{name} (raw)")

# Smoothed lines on top
def smooth(v, w=10):
    return [np.mean(v[max(0,i-w+1):i+1]) for i in range(len(v))]

for name, rec in zip(names, [r_random, r_fixed, r_dqn]):
    ax1.plot(smooth(rec["rewards"]), color=colors[name],
             linewidth=2.5, label=f"{name} (smoothed)")

ax1.set_title("Reward per episode", fontweight="bold")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total reward")
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

# ── Chart 2: Bar chart — mean reward ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
means = [np.mean(r["rewards"]) for r in [r_random, r_fixed, r_dqn]]
stds  = [np.std(r["rewards"])  for r in [r_random, r_fixed, r_dqn]]
bars  = ax2.bar(names, means, color=list(colors.values()),
                yerr=stds, capsize=5, edgecolor="white", linewidth=0.5)
for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_title("Mean reward (±std)", fontweight="bold")
ax2.set_ylabel("Avg total reward")
ax2.grid(True, alpha=0.3, axis="y")

# ── Chart 3: Fairness ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name, rec in zip(names, [r_random, r_fixed, r_dqn]):
    ax3.plot(smooth(rec["fairness"]), color=colors[name],
             linewidth=2, label=name)
ax3.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Perfect")
ax3.set_title("Fairness over episodes", fontweight="bold")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Jain's fairness index")
ax3.set_ylim(0.4, 1.05)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Chart 4: Throughput ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for name, rec in zip(names, [r_random, r_fixed, r_dqn]):
    ax4.plot(smooth(rec["throughput"]), color=colors[name],
             linewidth=2, label=name)
ax4.set_title("Throughput over episodes", fontweight="bold")
ax4.set_xlabel("Episode")
ax4.set_ylabel("Avg throughput")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Chart 5: Action distribution (DQN only) ──────────────────────
ax5 = fig.add_subplot(gs[1, 2])
action_labels = ["Equal\nsplit", "High\ndemand", "Best\nsignal",
                 "Weakest\nuser", "Balanced"]
action_colors = ["#b5d4f4", "#9fe1cb", "#cecbf6", "#f5c4b3", "#fac775"]
dqn_action_counts = [r_dqn["actions"].count(i) for i in range(5)]
total = sum(dqn_action_counts)
pcts  = [c/total*100 for c in dqn_action_counts]
bars2 = ax5.bar(action_labels, pcts, color=action_colors,
                edgecolor="white", linewidth=0.5)
for bar, pct in zip(bars2, pcts):
    if pct > 2:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
ax5.set_title("DQN action preference", fontweight="bold")
ax5.set_ylabel("% of decisions")
ax5.grid(True, alpha=0.3, axis="y")

# ── Summary box ───────────────────────────────────────────────────
summary_text = (
    f"Results summary\n"
    f"──────────────────────\n"
    f"DQN vs Random    : {improvement_vs_random:+.1f}%\n"
    f"DQN vs Fixed rule: {improvement_vs_fixed:+.1f}%\n"
    f"DQN avg reward   : {dqn_reward:.2f}\n"
    f"DQN avg fairness : {np.mean(r_dqn['fairness']):.3f}"
)
fig.text(0.01, 0.01, summary_text, fontsize=8, verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="#eeedfe", alpha=0.8))

plt.savefig("results/evaluation_results.png", dpi=150, bbox_inches="tight")
print("\n  Chart saved → results/evaluation_results.png")
print("  Summary saved → results/evaluation_summary.json")
plt.show()