import gym
import ctr_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

if __name__ == '__main__':
    # Env and model names and paths
    env_id = "Exact-Ctr-3-Tube-Reach-v0"
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/cras_2020/cras2_2_7/learned_policy/300000_saved_model.pkl"

    # Create envs and model
    env = HERGoalEnvWrapper(gym.make(env_id))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    error = np.array([])
    achieved_goals = np.array([])
    desired_goals = np.array([])
    time_taken = np.array([])

    # Run random episodes and save sequence of actions and states to plot in matlab
    episode_reward = 0
    ep_len = 0
    obs = env.reset()

    for episode in range(100):

        for t in range(200):
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                error = np.append(error, infos.get('error'))
                break

    print("Average error: ", np.mean(error) * 1000)
