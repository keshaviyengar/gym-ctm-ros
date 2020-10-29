import gym
import ctm_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

# Import ROS tools
import rospy


class CTMPathFollower(object):
    def __init__(self, env_id, exp_id, model_path, trajectory_type, episode_timesteps, noise_parameters):
        self.env_id = env_id
        self.exp_id = exp_id
        self.trajectory_type = trajectory_type
        # Load model and environment
        self.env = HERGoalEnvWrapper(gym.make(env_id, kwargs={'noise_parameters': noise_parameters}))
        self.model = HER.load(model_path, env=self.env)
        self.episode_timesteps = episode_timesteps

        # Setup subscriber for trajectory generator
        # self.line_trajectory_timer = rospy.Timer(rospy.Duration(0.1), self.line_trajectory_callback)
        # self.circle_trajectory_timer = rospy.Timer(rospy.Duration(0.01), self.circle_trajectory_callback)

        # Line trajectory settings
        if self.trajectory_type == "line":
            self.start_p = np.array([20, 0, 100]) / 1000
            self.finish_p = np.array([20, 40, 100]) / 1000
            self.del_p = self.finish_p - self.start_p
            self.current_goal = self.start_p

        # Circle trajectory settings
        if self.trajectory_type == "circle":
            self.offset = np.array([20, 20, 100]) / 1000
            self.radius = 20.0 / 1000
            self.thetas = np.arange(0, 2 * np.pi, np.deg2rad(5))
            self.thetas_counter = 0
            self.start_p = self.offset
            self.current_goal = self.start_p

        # Start timer
        self.prev_time = rospy.get_time()

        # Complete trajectory check
        self.shape_df = pd.DataFrame(
            columns=['episode', 'timestep', 'r1x', 'r1y', 'r1z', 'r2x', 'r2y', 'r2z', 'r3x', 'r3y', 'r3z'])
        # self.goals_df = pd.DataFrame(columns=['ag_x', 'ag_y', 'ag_z', 'dg_x', 'dg_y', 'dg_z'])
        self.traj_complete = False

        self.achieved_goals = np.array([])
        self.desired_goals = np.array([])
        self.episode_count = 0

    def line_trajectory_update(self):
        curr_time = rospy.get_time()
        delta_t = curr_time - self.prev_time
        self.prev_time = curr_time
        self.current_goal = self.current_goal + self.del_p * delta_t * 0.20

        if np.linalg.norm(self.current_goal - self.finish_p) < 0.001:
            self.traj_complete = True

        print("Distance to end: ", np.linalg.norm(self.current_goal - self.finish_p))

    def circle_trajectory_update(self):
        print('thetas_counter: ', self.thetas_counter, 'of ', self.thetas.size - 1)
        curr_time = rospy.get_time()
        delta_t = curr_time - self.prev_time
        self.prev_time = curr_time

        self.thetas_counter += 1
        if self.thetas_counter == self.thetas.size - 1:
            self.traj_complete = True
        else:
            self.current_goal = self.offset + self.radius * np.array(
                [np.cos(self.thetas[self.thetas_counter]), np.sin(self.thetas[self.thetas_counter]), 0])

    def play_episode(self, render_mode='inference'):
        self.episode_count += 1
        episode_reward = 0.0
        ep_len = 0
        # Set the observation
        obs = self.env.reset(goal=self.current_goal)

        for t in range(self.episode_timesteps):
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action space is of type Box
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, infos = self.env.step(action)

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                self.env.render(mode=render_mode)
                r_df = self.env.env.r_df
                r_df['episode'] = np.full(r_df.shape[0], self.episode_count)
                self.shape_df = pd.concat([self.shape_df, r_df], join='inner')
                break

        # Save data for the episode
        if self.achieved_goals.size == 0:
            self.achieved_goals = self.env.convert_obs_to_dict(obs)['achieved_goal']
            self.desired_goals = self.env.convert_obs_to_dict(obs)['desired_goal']
        else:
            self.achieved_goals = np.vstack([self.achieved_goals, self.env.convert_obs_to_dict(obs)['achieved_goal']])
            self.desired_goals = np.vstack([self.desired_goals, self.env.convert_obs_to_dict(obs)['desired_goal']])

        # Save shape information of the episode

    def save_data(self, name=None):
        ag_goals_df = pd.DataFrame(data=self.achieved_goals, columns=['ag_x', 'ag_y', 'ag_z'])
        dg_goals_df = pd.DataFrame(data=self.desired_goals, columns=['dg_x', 'dg_y', 'dg_z'])
        goals_df = pd.concat([ag_goals_df, dg_goals_df], axis=1, join='inner')
        if name is None:
            goals_df.to_csv(
                '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/' + self.trajectory_type + '_path_following_' + self.exp_id + '_goals.csv')
            self.shape_df.to_csv(
                '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/' + self.trajectory_type + '_path_following_' + self.exp_id + '_shape.csv')
        else:
            goals_df.to_csv(
                '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/path_following/' + self.trajectory_type + '_path_following_' + self.exp_id + '_ ' + name + '_goals.csv')
            self.shape_df.to_csv(
                '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/' + self.trajectory_type + '_path_following_' + self.exp_id + '_' + name + '_shape.csv')


def noise_sensitivity_analysis(env_id, model_path, trajectory_type, episode_timesteps, tracking_noise_intervals,
                               encoder_noise_intervals):
    # iterate through encoder intervals
    for encoder_noise in encoder_noise_intervals:
        for tracking_noise in tracking_noise_intervals:
            noise_parameters = {'rotation_std': np.deg2rad(encoder_noise),
                                'extension_std': 0.001 * np.deg2rad(encoder_noise),
                                'tracking_std': np.deg2rad(tracking_noise)}
            # Create new env with correct noise
            traj_inference = CTMPathFollower(env_id, exp_id, model_path, trajectory_type, episode_timesteps,
                                             noise_parameters)
            # Run path planning experiment and get error vector
            while not traj_inference.traj_complete:
                traj_inference.play_episode(render_mode='human')
                if trajectory_type == "circle":
                    traj_inference.circle_trajectory_update()
                elif trajectory_type == 'line':
                    traj_inference.line_trajectory_update()
            # Store as name of tracking and encoder name .csv
            traj_inference.save_data(name=str(encoder_noise) + '_' + str(tracking_noise))


if __name__ == '__main__':
    # Simple Path Following
    # env_id = "CTR-Reach-v0"
    env_id = "CTR-Reach-Noisy-v0"
    exp_id = "cras_exp_6"
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/" + exp_id + "/learned_policy/500000_saved_model.pkl"
    episode_timesteps = 20
    noise_parameters = {'rotation_std': np.deg2rad(1.0), 'extension_std': 0.001 * np.deg2rad(1.0),
                        'tracking_std': 0.0008}

    rospy.init_node("ctm_path_following")

    trajectory_type = "circle"
    traj_inference = CTMPathFollower(env_id, exp_id, model_path, trajectory_type, episode_timesteps,
                                     noise_parameters)

    while not traj_inference.traj_complete:
        traj_inference.play_episode()
        if trajectory_type == "circle":
            traj_inference.circle_trajectory_update()
        elif trajectory_type == 'line':
            traj_inference.line_trajectory_update()

    traj_inference.save_data()
