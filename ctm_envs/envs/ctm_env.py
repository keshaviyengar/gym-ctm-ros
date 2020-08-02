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

        if self.function == 'linear':
            self.a = (self.final_tol - self.init_tol) / self.N_ts
            self.b = self.init_tol

        if self.function == 'decay':
            self.a = self.init_tol
            self.r = 1 - np.power((self.final_tol / self. init_tol), 1 / self.N_ts)

        self.current_tol = self.init_tol

    def update(self, training_step):
        if (self.function == 'linear') and (training_step <= self.N_ts):
            self.current_tol = self.linear_function(training_step)
        elif (self.function == 'decay') and (training_step <= self.N_ts):
            self.current_tol = self.decay_function(training_step)
        else:
            self.current_tol = self.final_tol

    def get_tol(self):
        return self.current_tol

    def linear_function(self, training_step):
        return self.a * training_step + self.b

    def decay_function(self, training_step):
        return self.a * np.power(1 - self.r, training_step)


class CtmEnv(gym.GoalEnv):
    def __init__(self, tube_parameters, model, action_length_limit, action_rotation_limit, max_episode_steps, n_substeps,
                 goal_tolerance_parameters, joint_representation, initial_q, render):

        self.num_tubes = len(tube_parameters.keys())
        # Extract tube parameters
        self.tubes = list()
        self.tube_lengths = list()
        for i in range(0, self.num_tubes):
            tube_args = tube_parameters['tube_' + str(i)]
            self.tubes.append(TubeParameters(**tube_args))
            self.tube_lengths.append(tube_args['length'])

        self.action_length_limit = action_length_limit
        self.action_rotation_limit = action_rotation_limit

        # Action space
        action_length_limit = np.full(self.num_tubes, self.action_length_limit)
        action_orientation_limit = np.full(self.num_tubes, np.deg2rad(self.action_rotation_limit))
        self.action_space = gym.spaces.Box(low=np.concatenate((-action_length_limit, -action_orientation_limit)),
                                           high=np.concatenate((action_length_limit, action_orientation_limit)),
                                           dtype="float32")

        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps

        if model == 'dominant_stiffness':
            self.model = DominantStiffnessModel(self.tubes)
        elif model == 'exact':
            self.model = ExactModel(self.tubes)
        else:
            print("Model unavailable")

        if render:
            from ctm_envs.envs.ctm_render import CtmRender
            print("Rendering turned on.")
            self.render_obj = CtmRender(model, self.tubes)

        if joint_representation == 'basic':
            self.rep_obj = BasicObs(self.tubes, goal_tolerance_parameters, initial_q)
        elif joint_representation == 'trig':
            self.rep_obj = TrigObs(self.tubes, goal_tolerance_parameters, initial_q)
        elif joint_representation == 'polar':
            self.rep_obj = PolarObs(self.tubes, goal_tolerance_parameters, initial_q, 1)
        else:
            print("Incorrect representation selected, defaulting to basic.")
            self.rep_obj = BasicObs(self.tubes, goal_tolerance_parameters, initial_q)

        self.goal_tol_obj = GoalTolerance(goal_tolerance_parameters)
        self.t = 0
        self.observation_space = self.rep_obj.get_observation_space()

    def reset(self):
        self.t = 0
        # Resample a desired goal and its associated q joint
        desired_q = self.rep_obj.sample_goal()
        desired_goal = self.model.forward_kinematics(desired_q)
        achieved_goal = self.model.forward_kinematics(self.rep_obj.get_q())
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol())
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        assert not np.all(np.isnan(action))
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.n_substeps):
            self.rep_obj.set_action(action)

        # Compute FK
        achieved_goal = self.model.forward_kinematics(self.rep_obj.q)
        desired_goal = self.rep_obj.get_desired_goal()
        self.t += 1
        reward = self.compute_reward(achieved_goal, desired_goal, dict())
        done = (reward == 0) or (self.t >= self.max_episode_steps)
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol())

        info = {'is_success': reward == 0, 'error': np.linalg.norm(desired_goal - achieved_goal),
                'goal_tolerance': self.goal_tol_obj.get_tol()}

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.goal_tol_obj.get_tol()).astype(np.float32)

    def render(self, mode='human'):
        if self.render_obj is not None:
            # TODO: Issue in pycharm, python 2 ros libaries can't be found. Run in terminal.
            self.render_obj.publish_desired_goal(self.rep_obj.get_desired_goal())
            self.render_obj.publish_achieved_goal(self.rep_obj.get_achieved_goal())

            if self.render_obj.model == 'dominant_stiffness':
                self.render_obj.publish_joints(self.rep_obj.get_q())
            elif self.render_obj.model == 'exact':
                self.render_obj.publish_segments(self.model.get_r())
            else:
                print("Incorrect model selected, no rendering")

    def close(self):
        print("Closed env.")

    def update_goal_tolerance(self, N_ts):
        self.goal_tol_obj.update(N_ts)
