import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FiveGNetworkEnv(gym.Env):
    """
    A simplified 5G network environment.
    
    Imagine a single mobile tower serving multiple users.
    Every 'step' = one time slot (e.g. 1 millisecond).
    The AI decides how to split resources across users each slot.
    """

    def __init__(self, num_users=5, num_resource_blocks=10):
        super().__init__()

        self.num_users = num_users
        self.num_resource_blocks = num_resource_blocks  # Think of these as "lanes" on the highway

        # --- What the AI can SEE (the State) ---
        # For each user we track:
        #   - How much data they are demanding right now  (0.0 to 1.0)
        #   - How good their signal quality is           (0.0 to 1.0)
        #   - How many resources they currently have     (0.0 to 1.0)
        # Total numbers in the state = num_users * 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_users * 3,),
            dtype=np.float32
        )

        # --- What the AI can DO (the Action) ---
        # It picks one of several pre-set allocation strategies
        # (0 = equal split, 1 = favour highest demand, 2 = favour best signal,
        #  3 = favour poorest users, 4 = balanced mix)
        self.action_space = spaces.Discrete(5)

        # Internal bookkeeping
        self.current_step = 0
        self.max_steps = 200  # One episode = 200 time slots
        self.state = None

    def reset(self, seed=None, options=None):
        """Start a fresh episode — randomise the network conditions."""
        super().reset(seed=seed)

        # Random starting conditions for all users
        demand   = np.random.uniform(0.1, 1.0, self.num_users)   # How much data each user wants
        channel  = np.random.uniform(0.2, 1.0, self.num_users)   # Signal quality (closer = better)
        alloc    = np.ones(self.num_users) / self.num_users        # Start with equal allocation

        self.state = np.concatenate([demand, channel, alloc]).astype(np.float32)
        self.current_step = 0

        return self.state, {}

    def step(self, action):
        """
        Apply the AI's chosen action and return what happened.
        This is the core engine — one tick of the simulation.
        """
        demand  = self.state[:self.num_users]
        channel = self.state[self.num_users:self.num_users*2]

        # --- Apply the chosen allocation strategy ---
        new_alloc = self._allocate(action, demand, channel)

        # --- Calculate how well each user is served ---
        # Throughput = allocation × channel quality  (simple model of Shannon capacity)
        throughput = new_alloc * channel

        # --- Calculate the reward the AI receives ---
        reward = self._compute_reward(throughput, demand, new_alloc)

        # --- Update the world state ---
        # Demand shifts a little each step (users start/stop streaming, moving, etc.)
        demand  = np.clip(demand  + np.random.uniform(-0.1, 0.1, self.num_users), 0.1, 1.0)
        channel = np.clip(channel + np.random.uniform(-0.05, 0.05, self.num_users), 0.1, 1.0)

        self.state = np.concatenate([demand, channel, new_alloc]).astype(np.float32)
        self.current_step += 1

        done = self.current_step >= self.max_steps

        # Extra info for logging (not used by the AI, useful for us to debug)
        info = {
            "throughput": throughput.mean(),
            "fairness": self._jain_fairness(throughput),
            "satisfaction": np.mean(throughput >= demand * 0.8)
        }

        return self.state, reward, done, False, info

    def _allocate(self, action, demand, channel):
        """
        Convert an action number into an actual resource split.
        Returns an array of fractions that sum to 1.0.
        """
        scores = np.ones(self.num_users)  # Default: everyone equal

        if action == 0:
            scores = np.ones(self.num_users)           # Equal split
        elif action == 1:
            scores = demand                            # Give more to who wants more
        elif action == 2:
            scores = channel                           # Give more to who has best signal (efficient)
        elif action == 3:
            scores = 1.0 - channel + 0.1              # Give more to worst signal (fair)
        elif action == 4:
            scores = demand * channel                  # Balance demand AND quality

        # Normalise so all fractions sum to 1
        return scores / scores.sum()

    def _compute_reward(self, throughput, demand, alloc):
        """
        Score the AI's decision. Higher = better.
        We reward three things and penalise one.
        """
        # 1. Average throughput (higher = more data delivered = good)
        avg_throughput = throughput.mean()

        # 2. Fairness — are all users getting a decent service?
        #    Jain's Fairness Index: 1.0 = perfectly fair, 0 = totally unfair
        fairness = self._jain_fairness(throughput)

        # 3. Satisfaction — what fraction of users got at least 80% of what they wanted?
        satisfaction = np.mean(throughput >= demand * 0.8)

        # 4. Penalty for wasting resources (allocating to users who don't need much)
        waste_penalty = np.mean(np.maximum(alloc - demand, 0))

        # Final score — weighted combination
        reward = (0.4 * avg_throughput +
                  0.3 * fairness +
                  0.3 * satisfaction -
                  0.1 * waste_penalty)

        return float(reward)

    def _jain_fairness(self, x):
        """
        Jain's Fairness Index — a standard measure used in networking.
        Score of 1.0 means everyone gets exactly the same share.
        Score near 0 means one user hogs everything.
        """
        if x.sum() == 0:
            return 0.0
        return (x.sum() ** 2) / (len(x) * (x ** 2).sum())