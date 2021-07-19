import gym
import ctm_envs

import numpy as np

import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# @info: This script evaluates the use of action shielding. Specifically, it uses a trained agent and runs through
# values for K (the action boundary reduction rate).


# TODO: Make this a class for regular evaluations as well has for shileding
def run_evaluations(env, model, num_episodes, output_path):
    set_global_seeds(np.random.randint(0, 10))

    # Create arrays for saved stats
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
        ep_r = 0
        ep_len = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            ep_r += reward
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
        print("k: ", env.env.action_shield_K, "mean_error: ", infos.get('errors_pos'), "success: ",
              infos.get("is_success"), " ep_r: ", ep_r, " ep_len: ", ep_len)
        eval_df = pd.DataFrame(data=np.column_stack(
            (np.full_like(goal_errors, env.env.action_shield_K), desired_goals, achieved_goals, goal_errors, starting_positions,
             q_B_desired, q_B_achieved, q_B_starting, q_alpha_desired,
             q_alpha_achieved, q_alpha_starting)),
                               columns=['shield_k', 'desired_goal_x', 'desired_goal_y', 'desired_goal_z',
                                        'achieved_goal_x', 'achieved_goal_y', 'achieved_goal_z',
                                        'error_pos',
                                        'starting_position_x', 'starting_position_y', 'starting_position_z',
                                        'B_desired_1', 'B_desired_2', 'B_desired_3',
                                        'B_achieved_1', 'B_achieved_2', 'B_achieved_3',
                                        'B_starting_1', 'B_starting_2', 'B_starting_3',
                                        'alpha_desired_1', 'alpha_desired_2', 'alpha_desired_3',
                                        'alpha_achieved_1', 'alpha_achieved_2', 'alpha_achieved_3',
                                        'alpha_starting_1', 'alpha_starting_2', 'alpha_starting_3',
                                        ])
        eval_df.to_csv(output_path)


if __name__ == '__main__':
    # Load env with model
    env_id = "CTR-Reach-v0"
    models = ["/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_rel_decay_free_rot/CTR-Reach-v0_1/rl_model_500000_steps.zip"]
    # Setup any required kwargs as per experiment
    kwargs = {
        'action_shielding': {'shield': False, 'K': 0},
        'normalize_obs': False,
        'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0
        },
        'relative_q': True,
        'resample_joints': True,
        'evaluation': True,
        'constrain_alpha': False
    }

    #env = gym.make(env_id, **kwargs)
    #model_path = models[0]
    #model = HER.load(model_path, env=env)
    #num_episodes = 10000
    #run_evaluations(env, model, num_episodes,
    #                "/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_rel_decay_free_rot/evaluations.csv")

    # Range of K [0.001, 0.35]
    k_values = np.linspace(0.001, 0.35, 10)
    for k in k_values:
        print("k: ", k)
        kwargs['action_shielding']['shield'] = True
        kwargs['action_shielding']['K'] = k
        env = gym.make(env_id, **kwargs)
        model_path = models[0]
        model = HER.load(model_path, env=env)
        num_episodes = 100
        run_evaluations(env, model, num_episodes,
                        "/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_rel_decay_free_rot/shielding_evals_k_"
                        + str(k) + ".csv")
