import gym
import ctm_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# Aim of this script is to run through a number of episodes, returns the error statistics

def evaluation(env_id, exp_id, model_path, num_episodes, output_path):
    env = HERGoalEnvWrapper(gym.make(env_id))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    goal_errors = np.empty((num_episodes), dtype=float)
    B_errors = np.empty((num_episodes), dtype=float)
    alpha_errors = np.empty((num_episodes), dtype=float)
    q_B_achieved = np.empty((num_episodes, 3), dtype=float)
    q_alpha_achieved = np.empty((num_episodes, 3), dtype=float)
    q_B_desired = np.empty((num_episodes, 3), dtype=float)
    q_alpha_desired = np.empty((num_episodes, 3), dtype=float)
    desired_goals = np.empty((num_episodes, 3), dtype=float)
    achieved_goals = np.empty((num_episodes, 3), dtype=float)
    starting_positions = np.empty((num_episodes, 3), dtype=float)
    q_B_starting = np.empty((num_episodes, 3), dtype=float)
    q_alpha_starting = np.empty((num_episodes, 3), dtype=float)

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
                goal_errors[episode] = infos.get('errors_pos')
                q_B_desired[episode, :] = infos.get('q_desired')[:3]
                q_alpha_desired[episode, :] = infos.get('q_desired')[3:]
                q_B_achieved[episode, :] = infos.get('q_achieved')[:3]
                q_alpha_achieved[episode, :] = infos.get('q_achieved')[3:]
                desired_goals[episode, :] = infos.get('desired_goal')
                achieved_goals[episode, :] = infos.get('achieved_goal')
                starting_positions[episode, :] = infos.get('starting_position')
                q_B_starting[episode, :] = infos.get('q_starting')[:3]
                q_alpha_starting[episode, :] = infos.get('q_starting')[3:]
                break

    print('mean_errors: ', np.mean(goal_errors))
    eval_df = pd.DataFrame(data=np.column_stack((desired_goals, achieved_goals, starting_positions,
                                                 q_B_desired, q_B_achieved, q_B_starting, q_alpha_desired,
                                                 q_alpha_achieved, q_alpha_starting)),
                           columns=['desired_goal_x', 'desired_goal_y', 'desired_goal_z',
                                    'achieved_goal_x', 'achieved_goal_y', 'achieved_goal_z',
                                    'starting_position_x', 'starting_position_y', 'starting_position_z',
                                    'B_desired_1', 'B_desired_2', 'B_desired_3',
                                    'B_achieved_1', 'B_achieved_2', 'B_achieved_3',
                                    'B_starting_1', 'B_starting_2', 'B_starting_3',
                                    'alpha_desired_1', 'alpha_desired_2', 'alpha_desired_3',
                                    'alpha_achieved_1', 'alpha_achieved_2', 'alpha_achieved_3',
                                    'alpha_startin_1', 'alpha_starting_2', 'alpha_starting_3',
                                    ])
    eval_df.to_csv(output_path)


if __name__ == '__main__':
    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    num_episodes = 1000
    experiment_ids = ["cras_exp_1_new", "cras_exp_2", "cras_exp_3", "cras_exp_4_new", "cras_exp_5", "cras_exp_6"]
    for exp_id in experiment_ids:
        model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/" + exp_id + "/learned_policy/500000_saved_model.pkl"
        output_path = '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/' + exp_id + '_joint_error_analysis.csv'
        evaluation(env_id, exp_id, model_path, num_episodes, output_path)

    env_id = "CTR-Reach-Noisy-v0"
    experiment_ids = ["cras_exp_8"]
    for exp_id in experiment_ids:
        model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/" + exp_id + "/learned_policy/500000_saved_model.pkl"
        output_path = '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/' + exp_id + '_joint_error_analysis.csv'
        evaluation(env_id, exp_id, model_path, num_episodes, output_path)
