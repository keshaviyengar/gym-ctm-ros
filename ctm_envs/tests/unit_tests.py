import gym
import ctm_envs

from gym.wrappers import FlattenDictWrapper
import numpy as np


def environment(spec, kwargs):
    env = FlattenDictWrapper(spec.make(**kwargs), ['observation', 'desired_goal', 'achieved_goal'])
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    env.close()


# Run a longer rollout on some environments
def random_rollout(spec, kwargs):
    env = FlattenDictWrapper(spec.make(**kwargs), ['observation', 'desired_goal', 'achieved_goal'])
    agent = lambda ob: env.action_space.sample()
    ob = env.reset()
    for _ in range(10):
        assert env.observation_space.contains(ob)
        a = agent(ob)
        assert env.action_space.contains(a)
        (ob, _reward, done, _info) = env.step(a)
        if done:
            break
    env.close()


def test_environment():
    spec_list = [gym.spec('CTR-Reach-v0')]
    for spec in spec_list:
        # representations = ['basic', 'trig', 'polar']
        representations = ['trig']
        model = ['exact']
        inc_tol_obs = [True, False]
        tol_function = ['constant', 'linear', 'decay']

        for i in representations:
            for j in model:
                for k in inc_tol_obs:
                    for m in tol_function:
                        goal_tolerance_args = {'inc_tol_obs': k, 'initial_tol': 0.020, 'final_tol': 0.001,
                                               'N_ts': 200000, 'function': m}
                        kwargs = {'joint_representation': i, 'model': j, 'goal_tolerance_parameters': goal_tolerance_args}
                        environment(spec, kwargs)
                        random_rollout(spec, kwargs)

if __name__ == '__main__':
    test_environment()