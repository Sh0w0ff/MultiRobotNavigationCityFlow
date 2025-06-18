import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cityflow
import os

class CityFlowEnv(gym.Env):
    """
    A minimal CityFlow environment for reinforcement learning. Observations are the average speed of tracked vehicles,
    and actions are discrete: 0 = no operation, 1 = accelerate (placeholder).
    """
    metadata = {"render_modes": []}

    def __init__(self, config_path=None, num_agents=3, thread_num=1):
        super().__init__()
        # Determine config path
        if config_path is None:
            base_dir = os.path.dirname(__file__)
            config_path = os.path.join(base_dir, "cityflow_config", "config.json")
        print(f"Loading CityFlow config from: {config_path}")
        # Initialize CityFlow engine
        self.eng = cityflow.Engine(config_path, thread_num=thread_num)

        # Agent count and spaces
        self.num_agents = num_agents
        # Will store active vehicle IDs to track
        self.vehicle_ids = []

        # Observation: average speed (single float)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        # Action: 0 = noop, 1 = placeholder accelerate
        self.action_space = spaces.Discrete(2)
        # Observation: average speed (single float)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        # Action: 0 = noop, 1 = placeholder accelerate
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        # Reset environment and return initial observation
        self.eng.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Advance simulation (action is a placeholder)
        self.eng.next_step()
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Compute average speed of tracked vehicles
        ids = [f"vehicle_{i}" for i in range(self.num_agents)]
        speeds = []
        for vid in ids:
            try:
                info = self.eng.get_vehicle_info(vid, ["speed"])
                speeds.append(info.get("speed", 0.0))
            except Exception:
                speeds.append(0.0)
        avg_speed = np.mean(speeds) if speeds else 0.0
        return np.array([avg_speed], dtype=np.float32)

    def _get_reward(self):
        # Reward equals average speed
        return float(self._get_obs()[0])

    def close(self):
        # Clean up
        self.eng.reset()