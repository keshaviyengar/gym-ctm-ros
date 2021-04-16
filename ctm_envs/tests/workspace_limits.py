import numpy as np
import gym
import ctm_envs
from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.common.env_checker import check_env

# This script uses Monte Carlo sampling to sample joint values, get end-effector points and compute the limits of the
# workspace. Based on work in Workspace Characterization for Concentric Tube Continuum Robots by Burgner-Kahrs IROS 2014

# Results for default CTR-Reach-v0 parameters
# 78700
# ag:  [ 0.00420641 -0.04430476  0.11442123]
# max x:  0.08177311355555257  max y:  0.08168310184561604  max z:  0.1652125399191837
# min x:  -0.08250110258578695  min y:  -0.0817850994404839  min z:  0.002916952694281949

# Input: HERGoalWrapped ctr_env gym environment
def workspace_limits(ctr_env):
    # Init values to minimum
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf

    # Init values to maximum
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf

    # For a number of samples
    num_samples = 1e6
    for i in range(int(num_samples)):
        # Sample from the joint states (rotation and extension)
        # Store if larger than the current maximum x,y,z and continue
        obs = ctr_env.reset()
        ag = env.convert_obs_to_dict(obs)['desired_goal']
        # Check for max workspace limits
        if ag[0] > max_x:
            max_x = ag[0]
        if ag[1] > max_y:
            max_y = ag[1]
        if ag[2] > max_z:
            max_z = ag[2]
        # Check for min workspace limits
        if ag[0] < min_x:
            min_x = ag[0]
        if ag[1] < min_y:
            min_y = ag[1]
        if ag[2] < min_z:
            min_z = ag[2]

        if i % 100 == 0:
            print(i)
            print("ag: ", ag)
            print("max x: ", max_x, " max y: ", max_y, " max z: ", max_z)
            print("min x: ", min_x, " min y: ", min_y, " min z: ", min_z)
            print("=======")

    return np.array([max_x, max_y, max_z]), np.array([min_x, min_y, min_z])


if __name__ == '__main__':
    env_id = "CTR-Reach-v0"
    env = HERGoalEnvWrapper(gym.make(env_id))
    max_limits, min_limits = workspace_limits(env)
    print(max_limits)
    print(min_limits)
