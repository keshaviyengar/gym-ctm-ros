import gym
import numpy as np

from ctm_envs.envs.obs_base import ObsBase

"""
This representation is the polar representation
d_i = {d_Re,i, d_Im,i} = {(k - B_i/L_i)cos(alpha_i), (k - B_i/L_i)sin(alpha_i)}
"""


class PolarObs(ObsBase):
    def __init__(self, tube_parameters, goal_tolerance_parameters, initial_q, k=1):
        self.k = k
        super().__init__(tube_parameters, goal_tolerance_parameters, initial_q)

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

        return observation_space

    def get_rep_space(self):
        rep_low = np.full(2*self.num_tubes, self.k-1)
        rep_high = np.full(2*self.num_tubes, self.k)

        rep_space = gym.spaces.Box(low=rep_low, high=rep_high, dtype=np.float32)
        return rep_space

    def rep2joint(self, rep):
        rep = [rep[i:i + 2] for i in range(0, len(rep), 2)]
        beta = np.empty(self.num_tubes)
        alpha = np.empty(self.num_tubes)
        for tube in range(0, self.num_tubes):
            joint = self.single_polar2joint(rep[tube], self.tube_lengths[tube])
            alpha[tube] = joint[0]
            beta[tube] = joint[1]
        return np.concatenate((beta, alpha))

    def joint2rep(self, joint):
        rep = np.array([])
        betas = joint[:self.num_tubes]
        alphas = joint[self.num_tubes:]
        for beta, alpha, L in zip(betas, alphas, self.tube_lengths):
            d_Re = (self.k - beta / L) * np.cos(alpha)
            d_Im = (self.k - beta / L) * np.sin(alpha)
            rep = np.append(rep, np.array([d_Re, d_Im]))
        return rep

    def single_polar2joint(self, polar, L):
        alpha = np.arctan2(polar[1], polar[2])
        beta = L * (self.k - np.sqrt(np.power(polar[0], 2) - np.power(polar[1], 2)))
        return np.array(alpha, beta)