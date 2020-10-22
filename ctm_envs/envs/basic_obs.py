import gym
import numpy as np

from ctm_envs.envs.obs_base import ObsBase

"""
This representation is the simple {beta_i, alpha_i} representation
"""


class BasicObs(ObsBase):
    def __init__(self, tube_parameters, goal_tolerance_parameters, noise_parameters, initial_q, relative_q, ext_tol):
        super().__init__(tube_parameters, goal_tolerance_parameters, noise_parameters, initial_q, relative_q, ext_tol)
        self.goal_dim = 3
        self.obs_dim = 0
        print("Basic joint representation used")

    def get_observation_space(self):
        initial_tol = self.goal_tolerance_parameters['initial_tol']
        final_tol = self.goal_tolerance_parameters['final_tol']
        rep_space = self.get_rep_space()

        if self.inc_tol_obs:
            obs_space_low = np.concatenate(
                (rep_space.low, np.array([-1, -1, -1, initial_tol])))
            obs_space_high = np.concatenate(
                (rep_space.high, np.array([1, 1, 1, final_tol])))
        else:
            obs_space_low = np.concatenate(
                (rep_space.low, np.array([-1, -1, -1])))
            obs_space_high = np.concatenate(
                (rep_space.high, np.array([1, 1, 1])))
        observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]),
                                        dtype="float32"),
            achieved_goal=gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]),
                                         dtype="float32"),
            observation=gym.spaces.Box(
                low=obs_space_low,
                high=obs_space_high,
                dtype="float32")
        ))
        self.obs_dim = obs_space_low.size
        return observation_space

    def get_rep_space(self):
        return self.q_space

    def rep2joint(self, rep):
        return rep

    def joint2rep(self, joint):
        return joint
