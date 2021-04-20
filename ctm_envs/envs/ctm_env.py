import gym
import numpy as np

from ctm_envs.envs.basic_obs import BasicObs
from ctm_envs.envs.trig_obs import TrigObs
from ctm_envs.envs.polar_obs import PolarObs
from ctm_envs.envs.dominant_stiffness_model import DominantStiffnessModel
from ctm_envs.envs.exact_model import ExactModel


class TubeParameters(object):
    def __init__(self, length, length_curved, outer_diameter, inner_diameter, stiffness, torsional_stiffness,
                 x_curvature, y_curvature, k):
        self.L = length
        self.L_s = length - length_curved
        self.L_c = length_curved
        # Exact model
        self.J = (np.pi * (pow(outer_diameter, 4) - pow(inner_diameter, 4))) / 32
        self.I = (np.pi * (pow(outer_diameter, 4) - pow(inner_diameter, 4))) / 64
        self.E = stiffness
        self.G = torsional_stiffness
        self.U_x = x_curvature
        self.U_y = y_curvature

        # Dominant stiffness model
        self.k = k


class GoalTolerance(object):
    def __init__(self, goal_tolerance_parameters):
        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.inc_tol_obs = self.goal_tolerance_parameters['inc_tol_obs']
        self.init_tol = self.goal_tolerance_parameters['initial_tol']
        self.final_tol = self.goal_tolerance_parameters['final_tol']
        self.N_ts = self.goal_tolerance_parameters['N_ts']
        self.function = self.goal_tolerance_parameters['function']
        valid_functions = ['constant', 'linear', 'decay']
        if self.function not in valid_functions:
            print('Not a valid function, defaulting to constant')
            self.function = 'constant'

        if self.function == 'constant':
            self.init_tol = self.final_tol

        if self.function == 'linear':
            self.a = (self.final_tol - self.init_tol) / self.N_ts
            self.b = self.init_tol

        if self.function == 'decay':
            self.a = self.init_tol
            self.r = 1 - np.power((self.final_tol / self.init_tol), 1 / self.N_ts)

        self.set_tol = self.goal_tolerance_parameters['set_tol']
        if self.set_tol == 0:
            self.current_tol = self.init_tol
        else:
            self.current_tol = self.set_tol

    def update(self, training_step):
        if self.set_tol == 0:
            if (self.function == 'linear') and (training_step <= self.N_ts):
                self.current_tol = self.linear_function(training_step)
            elif (self.function == 'decay') and (training_step <= self.N_ts):
                self.current_tol = self.decay_function(training_step)
            else:
                self.current_tol = self.final_tol
        else:
            self.current_tol = self.set_tol

    def get_tol(self):
        return self.current_tol

    def linear_function(self, training_step):
        return self.a * training_step + self.b

    def decay_function(self, training_step):
        return self.a * np.power(1 - self.r, training_step)


class CtmEnv(gym.GoalEnv):
    def __init__(self, tube_parameters, model, action_length_limit, action_rotation_limit, action_space_norm,
                 action_shielding, normalize_obs, max_episode_steps, n_substeps, goal_tolerance_parameters,
                 noise_parameters, joint_representation, relative_q, initial_q, render):

        self.num_tubes = len(tube_parameters.keys())
        # Extract tube parameters
        self.tubes = list()
        self.tube_lengths = list()
        for i in range(0, self.num_tubes):
            tube_args = tube_parameters['tube_' + str(i)]
            self.tubes.append(TubeParameters(**tube_args))
            self.tube_lengths.append(tube_args['length'])

        self.tube_lengths = self.tube_lengths

        self.action_length_limit = action_length_limit
        self.action_rotation_limit = action_rotation_limit
        self.action_space_norm = action_space_norm
        self.use_action_shield = action_shielding['shield']
        self.action_shield_K = action_shielding['K']
        self.action_shield_Beta = action_shielding['Beta']
        self.normalize_obs = normalize_obs

        if self.action_space_norm:
            n_actions = 2 * self.num_tubes
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype="float32")
        else:
            # Action space
            action_length_limit = np.full(self.num_tubes, self.action_length_limit)
            action_orientation_limit = np.full(self.num_tubes, np.deg2rad(self.action_rotation_limit))
            self.action_space = gym.spaces.Box(low=np.concatenate((-action_length_limit, -action_orientation_limit)),
                                               high=np.concatenate((action_length_limit, action_orientation_limit)),
                                               dtype="float32")

        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self.desired_q = []

        ext_tol = 0
        if model == 'dominant_stiffness':
            self.model = DominantStiffnessModel(self.tubes)
        elif model == 'exact':
            self.model = ExactModel(self.tubes)
            ext_tol = 1e-4
        else:
            print("Model unavailable")

        if render:
            from ctm_envs.envs.ctm_render import CtmRender
            print("Rendering turned on.")
            self.render_obj = CtmRender(model, self.tubes)
        else:
            self.render_obj = None

        self.r_df = None

        if joint_representation == 'basic':
            self.rep_obj = BasicObs(self.tubes, goal_tolerance_parameters, noise_parameters, initial_q, relative_q,
                                    ext_tol)
        elif joint_representation == 'trig':
            self.rep_obj = TrigObs(self.tubes, goal_tolerance_parameters, noise_parameters, initial_q, relative_q,
                                   ext_tol)
        elif joint_representation == 'polar':
            self.rep_obj = PolarObs(self.tubes, goal_tolerance_parameters, noise_parameters, initial_q, relative_q,
                                    ext_tol)
        else:
            print("Incorrect representation selected, defaulting to basic.")
            self.rep_obj = BasicObs(self.tubes, goal_tolerance_parameters, initial_q, relative_q, ext_tol)

        self.goal_tol_obj = GoalTolerance(goal_tolerance_parameters)
        self.t = 0
        self.obs_space = self.rep_obj.get_observation_space()
        self.norm_obs_space = self.rep_obj.get_normalized_observation_space()
        if self.normalize_obs:
            self.observation_space = self.norm_obs_space
        else:
            self.observation_space = self.obs_space

    def reset(self, goal=None):
        self.t = 0
        self.r_df = None
        if goal is None:
            # Resample a desired goal and its associated q joint
            self.desired_q = self.rep_obj.sample_goal()
            desired_goal = self.model.forward_kinematics(self.desired_q)
        else:
            desired_goal = goal
        achieved_goal = self.model.forward_kinematics(self.rep_obj.get_q())
        self.starting_position = achieved_goal
        self.starting_joints = self.rep_obj.get_q()
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol())
        if self.normalize_obs:
            obs = self.get_normalized_obs(obs)
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    # Shield unscaled actions based on current goal error.
    def action_shield(self, action):
        # Compute error to goal
        ag = self.model.forward_kinematics(self.rep_obj.q)
        dg = self.rep_obj.get_desired_goal()
        goal_error = np.linalg.norm(dg - ag)

        # If error less than constant K, perform shielding. Else return action
        # TODO: Get values for Beta and K
        if goal_error <=self.action_shield_K:
            # Scale the action limits down by some constant determined by current error
            shielded_action_space = gym.spaces.Box(low=self.action_shield_Beta * error * self.action_space.low,
                                                   high=self.action_shield_Beta * error * self.action_space.high)
            return np.clip(action, shielded_action_space.low, shielded_action_space.high)
        else:
            return action

    # Normalize the observation based on the the original observaion space given. This step only happens right before
    # sending end of step function. All are mapped to (-1, 1)
    # Mapping is x' = 2 * (x - xmin) / (xmax - min) - 1
    def get_normalized_obs(self, obs):
        # Normalize individually
        obs_low = self.obs_space.spaces['observation'].low
        obs_high = self.obs_space.spaces['observation'].high
        dg_low = self.obs_space.spaces['desired_goal'].low
        dg_high = self.obs_space.spaces['desired_goal'].high
        ag_low = self.obs_space.spaces['achieved_goal'].low
        ag_high = self.obs_space.spaces['achieved_goal'].high

        def normalize(x, x_min, x_max):
            return 2 * (x - x_min) / (x_max - x_min) - 1
        # New normalized observation
        norm_obs = {}
        norm_obs['observation'] = normalize(obs['observation'], obs_low, obs_high)
        norm_obs['achieved_goal'] = normalize(obs['achieved_goal'], ag_low, ag_high)
        norm_obs['desired_goal'] = normalize(obs['desired_goal'], dg_low, dg_high)
        return norm_obs

    def step(self, action):
        assert not np.all(np.isnan(action))
        if self.action_space_norm:
            # Scale actions back up if normalized for environment
            action[:self.num_tubes] = action[:self.num_tubes] * self.action_length_limit
            action[self.num_tubes:] = action[self.num_tubes:] * self.action_rotation_limit
        # Action shielding
        if self.use_action_shield:
            action = self.action_shield(action)
        for _ in range(self.n_substeps):
            self.rep_obj.set_action(action)
        # Compute FK
        achieved_goal = self.model.forward_kinematics(self.rep_obj.q)
        desired_goal = self.rep_obj.get_desired_goal()
        self.t += 1
        reward = self.compute_reward(achieved_goal, desired_goal, dict())
        done = (reward == 0) or (self.t >= self.max_episode_steps)
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol())
        if self.normalize_obs:
            obs = self.get_normalized_obs(obs)

        info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
                'errors_pos': np.linalg.norm(desired_goal - achieved_goal),
                'error': np.linalg.norm(desired_goal - achieved_goal),
                'errors_orient': 0,
                'position_tolerance': self.goal_tol_obj.get_tol(),
                'orientation_tolerance': 0}

        # Evaluation infos
        #info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
        #        'errors_pos': np.linalg.norm(desired_goal - achieved_goal),
        #        'errors_orient': 0,
        #        'position_tolerance': self.goal_tol_obj.get_tol(),
        #        'orientation_tolerance': 0,
        #        'achieved_goal': achieved_goal,
        #        'desired_goal': desired_goal, 'starting_position': self.starting_position,
        #        'q_desired': self.desired_q, 'q_achieved': self.rep_obj.get_q(), 'q_starting': self.starting_joints}

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.goal_tol_obj.get_tol()).astype(np.float32)

    def render(self, mode='human'):
        if mode == 'inference':
            import pandas as pd
            r1, r2, r3 = self.model.get_rs()
            r1_df = pd.DataFrame(data=r1, columns=['r1x', 'r1y', 'r1z'])
            r2_df = pd.DataFrame(data=r2, columns=['r2x', 'r2y', 'r2z'])
            r3_df = pd.DataFrame(data=r3, columns=['r3x', 'r3y', 'r3z'])
            t = np.empty((r1.shape[0], 1))
            t.fill(self.t)
            t_df = pd.DataFrame(data=t, columns=['timestep'])
            if self.r_df is None:
                self.r_df = pd.concat([t_df, r1_df, r2_df, r3_df], axis=1)
            else:
                r_df = pd.concat([t_df, r1_df, r2_df, r3_df], axis=1)
                self.r_df = self.r_df.append(r_df, ignore_index=True)

        if mode == 'save':
            import pandas as pd
            r1, r2, r3 = self.model.get_rs()
            r1_df = pd.DataFrame(data=r1, columns=['r1x', 'r1y', 'r1z'])
            r2_df = pd.DataFrame(data=r2, columns=['r2x', 'r2y', 'r2z'])
            r3_df = pd.DataFrame(data=r3, columns=['r3x', 'r3y', 'r3z'])
            t = np.empty((r1.shape[0], 1))
            t.fill(self.t)
            t_df = pd.DataFrame(data=t, columns=['timestep'])
            if self.r_df is None:
                self.r_df = pd.concat([t_df, r1_df, r2_df, r3_df], axis=1)
            else:
                r_df = pd.concat([t_df, r1_df, r2_df, r3_df], axis=1)
                self.r_df = self.r_df.append(r_df, ignore_index=True)
            self.r_df.to_csv('/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments/data/temp_shape.csv')

        if self.render_obj is not None:
            # TODO: Issue in pycharm, python 2 ros libaries can't be found. Run in terminal.
            self.render_obj.publish_desired_goal(self.rep_obj.get_desired_goal())
            self.render_obj.publish_achieved_goal(self.rep_obj.get_achieved_goal())

            if self.render_obj.model == 'dominant_stiffness':
                self.render_obj.publish_joints(self.rep_obj.get_q())
                self.render_obj.publish_transforms(self.model.get_r_transforms())
            elif self.render_obj.model == 'exact':
                self.render_obj.publish_joints(self.rep_obj.get_q())
                self.render_obj.publish_transforms(self.model.get_r_transforms())
            else:
                print("Incorrect model selected, no rendering")

    def close(self):
        print("Closed env.")

    def update_goal_tolerance(self, N_ts):
        self.goal_tol_obj.update(N_ts)

    def get_obs_dim(self):
        return self.rep_obj.obs_dim

    def get_goal_dim(self):
        return self.rep_obj.goal_dim
