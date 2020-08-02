import gym
import numpy as np
import time
import ctm_envs


# 1. Make env with rendering.
# 2. Test dominant stiffness by publishing joints. Ensure desired goals match.
# 3. Test exact by publishing backbone.


def rendering_test(env, num_samples):
    state = env.reset()
    for i in range(0, num_samples):
        print("sample: ", i)
        a = env.action_space.sample()
        state, reward, done, info = env.step(a)
        env.render()
        time.sleep(2.0)


if __name__ == '__main__':
    dominant_stiffness_kwargs = {'model': 'dominant_stiffness', 'joint_representation': 'trig', 'render': True}
    exact_kwargs = {'model': 'exact', 'joint_representation': 'trig'}

    dominant_stiffness_env = gym.make('CTR-Reach-v0', **dominant_stiffness_kwargs)
    exact_env = gym.make('CTR-Reach-v0', **exact_kwargs)

    rendering_test(dominant_stiffness_env, 50)
    quit()

