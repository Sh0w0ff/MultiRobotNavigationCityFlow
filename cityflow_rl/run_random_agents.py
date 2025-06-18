import gym
from cityflow_env import CityFlowEnv

env = CityFlowEnv()
obs, _ = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    print(f"Step: {step}, Obs: {obs}, Reward: {reward}")

env.close()

