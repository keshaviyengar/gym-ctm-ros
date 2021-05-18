import gym
import ctm_envs

import numpy as np

import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

# @info: This script evaluates the use of action shielding. Specifically, it uses a trained agent and runs through
# values for K (the action boundary reduction rate).


def run_evaluations(env, model, num_episodes):
    set_global_seeds(np.random.randint(0,10))
    for episode in range(num_episodes):
        ep_r = 0
        ep_len = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            ep_r += reward
            ep_len += 1
            if done or infos.get('is_success', False):
                # TODO: Save some relevant data here
                break
        print("episode compete. ep_r: ", ep_r, " ep_len: ", ep_len)

if __name__ == '__main__':
    # Load env with model
    env_id = "CTR-Reach-v0"
    # Setup any required kwargs as per experiment
    kwargs = {
        'action_shielding': {'shield': False, 'K': 0},
        'normalize_obs': False,
        'goal_tolerance_parameters': {
            'inc_tol_obs': False, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0
        },
        'relative_q': True,
        'resample_joints': True,
    }
    # TODO: Create env and load model

    # Range of K [0.001, 0.35]
    k_values = np.linspace(0.001, 0.35, 10)
    for k in k_values:
        kwargs['action_shielding']['shield'] = True
        kwargs['action_shielding']['K'] = k
        # TODO: Fill out evaluations
        run_evaluations(env, model, num_episodes)


