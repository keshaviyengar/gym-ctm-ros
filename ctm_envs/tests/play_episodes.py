import gym
import ctm_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

if __name__ == '__main__':
    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/cras_exp_1/learned_policy/500000_saved_model.pkl"

    # Create envs and model
    env = HERGoalEnvWrapper(gym.make(env_id))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    q_joints = np.array([])
    achieved_goals = np.array([])
    desired_goals = np.array([])
    time_taken = np.array([])

    # Run random episodes and save sequence of actions and states to plot in matlab
    episode_reward = 0
    ep_len = 0
    obs = env.reset()
    env.render('save')

    q_joints = np.append(q_joints, env.convert_obs_to_dict(obs)['observation'][:9])
    achieved_goals = np.append(achieved_goals, env.convert_obs_to_dict(obs)['achieved_goal'])
    desired_goals = np.append(desired_goals, env.convert_obs_to_dict(obs)['desired_goal'])

    for t in range(200):
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)

        q_joints = np.vstack([q_joints, env.convert_obs_to_dict(obs)['observation'][:9]])
        achieved_goals = np.vstack([achieved_goals, env.convert_obs_to_dict(obs)['achieved_goal']])
        desired_goals = np.vstack([desired_goals, env.convert_obs_to_dict(obs)['desired_goal']])

        episode_reward += reward
        ep_len += 1
        env.render('save')

        if done or infos.get('is_success', False):
            print('goal tolerance: ', infos['goal_tolerance'])
            print('error: ', infos['error'] * 1000)
            break

    print("Done episode. Saving...")

    # Save dataframe and replay in matlab
    joints_df = pd.DataFrame(data=q_joints,
                             columns=['q_0', 'q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'q_7', 'q_8'])
    ag_df = pd.DataFrame(data=achieved_goals, columns=['ag_x', 'ag_y', 'ag_z'])
    dg_df = pd.DataFrame(data=desired_goals, columns=['dg_x', 'dg_y', 'dg_z'])
    full_df = pd.concat([joints_df, ag_df, dg_df], axis=1, join='inner')
    full_df.to_csv('/home/keshav/play_episode.csv')
