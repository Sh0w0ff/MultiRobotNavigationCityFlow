import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from cityflow_env import CityFlowEnv


def main():
    parser = argparse.ArgumentParser(description="Train DQN on CityFlowEnv")
    parser.add_argument("--config", type=str, default="./cityflow_config/config.json",
                        help="Path to CityFlow config.json")
    parser.add_argument("--agents", type=int, default=3,
                        help="Number of vehicles to track")
    parser.add_argument("--threads", type=int, default=1,
                        help="CityFlow engine threads")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total RL training timesteps")
    args = parser.parse_args()

    # Create vectorized environment
    env_fn = lambda: CityFlowEnv(
        config_path=args.config,
        num_agents=args.agents,
        thread_num=args.threads
    )
    env = make_vec_env(env_fn, n_envs=1)

    # Initialize DQN agent
    model = DQN(policy="MlpPolicy", env=env, verbose=1)
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    model.save("dqn_cityflow")
    print("Saved model as dqn_cityflow.zip")

    # Quick evaluation
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # handle vectorized 'done'
        if isinstance(done, (list, tuple, np.ndarray)):
            done_flag = done[0]
        else:
            done_flag = done
        if done_flag:
            obs = env.reset()
    print("Evaluation done.")

if __name__ == "__main__":
    main()