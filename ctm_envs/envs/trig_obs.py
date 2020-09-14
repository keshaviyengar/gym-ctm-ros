import gym
import numpy as np

from ctm_envs.envs.obs_base import ObsBase

"""
This representation is the trigonometric / cylindrical {cos(alpha)_i, sin(alpha_i), beta_i} representation
"""


class TrigObs(ObsBase):
    def __init__(self, tube_parameters, goal_tolerance_parameters, initial_q, relative_q):
        super().__init__(tube_parameters, goal_tolerance_parameters, initial_q, relative_q)
        self.goal_dim = 3
        self.obs_dim = 0
        print("Trig joint representation used")

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
        rep_low = np.array([])
        rep_high = np.array([])
        # TODO: zero tol needs to be included in model and base class
        zero_tol = 1e-4
        for tube_length in self.tube_lengths:
            rep_low = np.append(rep_low, [-1, -1, -tube_length + zero_tol])
            rep_high = np.append(rep_high, [-1, -1, -tube_length + zero_tol])

        rep_space = gym.spaces.Box(low=rep_low, high=rep_high, dtype=np.float32)
        return rep_space

    def rep2joint(self, rep):
        rep = [rep[i:i + 3] for i in range(0, len(rep), 3)]
        beta = np.empty(self.num_tubes)
        alpha = np.empty(self.num_tubes)
        for tube in range(0, self.num_tubes):
            joint = self.single_trig2joint(rep[tube])
            alpha[tube] = joint[0]
            beta[tube] = joint[1]
        return np.concatenate((beta, alpha))

    def joint2rep(self, joint):
        rep = np.array([])
        betas = joint[:self.num_tubes]
        alphas = joint[self.num_tubes:]
        for beta, alpha in zip(betas, alphas):
            trig = self.single_joint2trig(np.array([beta, alpha]))
            rep = np.append(rep, trig)
        return rep

    # Single conversion from a joint to trig representation
    @staticmethod
    def single_joint2trig(joint):
        return np.array([np.cos(joint[1]),
                         np.sin(joint[1]),
                         joint[0]])

    # Single conversion from a trig representation to joint
    @staticmethod
    def single_trig2joint(trig):
        return np.array([np.arctan2(trig[1], trig[0]), trig[2]])
