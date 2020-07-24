import numpy as np
import gym

"""
This function is the base class for the joint representations of basic, cylindrical and polar.
It should implement converting joints from basic to other representations,
setting actions and returning the resultant state and
changing the goal tolerance if variable goal tolerance is used.

q: Current joint as the basic representation to limit complexty
{beta_0, ..., beta_i, alpha_0, ..., alpha_i}
"""


class ObsBase:
    def __init__(self, tube_parameters, goal_tolerance_parameters, initial_q):
        self.tubes = tube_parameters
        self.tube_lengths = [i.L for i in self.tubes]
        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.q = initial_q

        self.num_tubes = len(tube_parameters)

        self.inc_tol_obs = self.goal_tolerance_parameters['inc_tol_obs']

        # Q space
        alpha_low = np.full(self.num_tubes, -np.pi)
        alpha_high = np.full(self.num_tubes, np.pi)
        # TODO: Add zero tol to beta low
        beta_low = -np.array(self.tube_lengths)
        beta_high = np.full(self.num_tubes, 0)

        self.q_space = gym.spaces.Box(low=np.concatenate((beta_low, alpha_low)),
                                      high=np.concatenate((beta_high, alpha_high)))
        # desired, achieved goal space
        self.observation_space = self.get_observation_space()
        self.obs = None

    def set_action(self, action):
        self.q = np.clip(self.q + action, self.q_space.low, self.q_space.high)
        q_betas = self.q[:self.num_tubes]
        q_alphas = self.q[self.num_tubes:]
        for i in range(1, self.num_tubes):
            # Remember ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            q_betas[i - 1] = min(q_betas[i - 1], q_betas[i])
            q_betas[i - 1] = max(q_betas[i - 1],
                                 self.tube_lengths[i] - self.tube_lengths[i - 1] + q_betas[i])

        self.q = np.concatenate((q_betas, q_alphas))

    def sample_goal(self):
        # while loop to get constrained points, maybe switch this for a workspace later on
        sample_counter = 0
        while True:
            q_sample = self.q_space.sample()
            betas = q_sample[0:self.num_tubes]
            alphas = q_sample[self.num_tubes:]
            # Apply constraints
            valid_joint = []
            for i in range(1, self.num_tubes):
                valid_joint.append((betas[i - 1] <= betas[i]) and (
                        betas[i - 1] + self.tube_lengths[i - 1] >= self.tube_lengths[i] + betas[i]))
                # print("B", i - 1, " <= ", "B", i, " : ", q_sample[i - 1][2], " <= ", q_sample[i][2])
                # print("B", i - 1, " + L", i - 1, " <= ", "B", i, " + L", i, " : ",
                #       q_sample[i - 1][2] + self.tube_lengths[i - 1], " >= ", q_sample[i][2] + self.tube_lengths[i])
                # print("valid joint: ", valid_joint)
                # print("")
            if all(valid_joint):
                break
            sample_counter += 1
        q_constrain = np.concatenate((betas, alphas))
        return q_constrain

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance):
        rep = self.joint2rep(self.q)
        if self.inc_tol_obs:
            self.obs = {
                'desired_goal': desired_goal,
                'achieved_goal': achieved_goal,
                'observation': np.concatenate(
                    (rep, desired_goal - achieved_goal, np.array([goal_tolerance]))
                )
            }
        else:
            self.obs = {
                'desired_goal': desired_goal,
                'achieved_goal': achieved_goal,
                'observation': np.concatenate(
                    (rep, desired_goal - achieved_goal)
                )
            }
        return self.obs

    def get_q(self):
        return self.q

    def get_desired_goal(self):
        return self.obs['desired_goal']

    def get_achieved_goal(self):
        return self.obs['achieved_goal']

    def get_rep_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def rep2joint(self, rep):
        raise NotImplementedError

    def joint2rep(self, joint):
        raise NotImplementedError
