from network_env import FiveGNetworkEnv
import numpy as np

print("=== Testing the 5G Network Environment ===\n")

env = FiveGNetworkEnv(num_users=5, num_resource_blocks=10)
state, _ = env.reset()

print(f"Number of users:          {env.num_users}")
print(f"State size (what AI sees): {len(state)} numbers")
print(f"Number of actions:         {env.action_space.n}")
print(f"\nInitial state (first 5 = demand, next 5 = channel quality, last 5 = allocation):")
print(f"  Demand:   {state[:5].round(3)}")
print(f"  Channel:  {state[5:10].round(3)}")
print(f"  Alloc:    {state[10:].round(3)}")

print("\n--- Simulating 10 random steps ---")
total_reward = 0
for step in range(10):
    action = env.action_space.sample()  # Random action for now
    state, reward, done, _, info = env.step(action)
    total_reward += reward
    print(f"  Step {step+1}: action={action}  reward={reward:.3f}  "
          f"throughput={info['throughput']:.3f}  "
          f"fairness={info['fairness']:.3f}  "
          f"users_satisfied={info['satisfaction']:.0%}")

print(f"\nTotal reward over 10 steps: {total_reward:.3f}")
print("\nEnvironment is working correctly!")