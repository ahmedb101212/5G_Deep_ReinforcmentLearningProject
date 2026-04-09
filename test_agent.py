from network_env import FiveGNetworkEnv
from dqn_agent import DQNAgent

print("=== Testing the DQN Agent ===\n")

env   = FiveGNetworkEnv(num_users=5)
agent = DQNAgent(state_size=15, action_size=5)

state, _ = env.reset()

print(f"Epsilon (exploration rate): {agent.epsilon}")
print(f"Memory capacity:            {agent.memory.buffer.maxlen}")
print(f"Batch size:                 {agent.batch_size}")

print("\n--- Running 5 steps with the agent ---")
for i in range(5):
    action              = agent.act(state)
    next_state, reward, done, _, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    loss = agent.learn()
    print(f"  Step {i+1}: action={action}  reward={reward:.3f}  "
          f"loss={'learning...' if loss is None else f'{loss:.4f}'}")
    state = next_state

print(f"\nMemories stored so far: {len(agent.memory)}")
print("\nAgent is working correctly!")