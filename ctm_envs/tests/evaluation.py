import gym
import ctm_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# Aim of this script is to run through a number of episodes, returns the error statistics
if __name__ == '__main__':
    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    # env_id = "CTR-Reach-Noisy-v0"
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/cras_exp_6/learned_policy/500000_saved_model.pkl"

    env = HERGoalEnvWrapper(gym.make(env_id))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)
    num_episodes = 100

    errors = np.array([])

    for episode in range(num_episodes):
        print('episode: ', episode)
        # Run random episodes and save sequence of actions and states to plot in matlab
        episode_reward = 0
        ep_len = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                errors = np.append(errors, infos.get('error'))
                break

        print("error mean: ", np.mean(errors) * 1000)
        print("error std: ", np.std(errors) * 1000)

    print("final error mean: ", np.mean(errors) * 1000)
    print("final error std: ", np.std(errors) * 1000)
