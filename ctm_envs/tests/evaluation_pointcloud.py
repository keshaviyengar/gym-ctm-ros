import gym
import ctm_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# Aim of this script is to run through a number of episodes, record the achieved goal and error and plot in workspace to
# visualize any biased areas or are errors evenly spread.
if __name__ == '__main__':
    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/cras_exp_1/learned_policy/500000_saved_model.pkl"

    # Create envs and model
    env = HERGoalEnvWrapper(gym.make(env_id))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    errors = np.array([])
    achieved_goals = np.empty((1,3))
    episode_rewards = np.array([])

    num_episodes = 100

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
                if episode == 0:
                    achieved_goals = env.convert_obs_to_dict(obs)['achieved_goal']
                else:
                    achieved_goals = np.vstack([achieved_goals, env.convert_obs_to_dict(obs)['achieved_goal']])
                errors = np.append(errors, infos.get('error'))
                episode_rewards = np.append(episode_rewards, episode_reward)
                break

    # Save dataframe and replay in matlab
    ag_df = pd.DataFrame(data=achieved_goals,
                             columns=['ag_x', 'ag_y', 'ag_z'])
    errors_df = pd.DataFrame(data=errors, columns=['error'])
    ep_r_df = pd.DataFrame(data=episode_rewards, columns=['reward'])
    full_df = pd.concat([ag_df, errors_df, ep_r_df], axis=1, join='inner')
    full_df.to_csv('/home/keshav/eval_pointcloud_data.csv')
