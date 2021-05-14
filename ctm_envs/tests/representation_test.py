import numpy as np
import gym
from gym import spaces

import ctm_envs


# Script to test various state representations.

def abs2rel(alphas, betas):
    rel_betas = np.diff(betas, prepend=betas[0])
    rel_alphas = np.diff(alphas, prepend=alphas[0])
    return rel_alphas, rel_betas


def rel2abs(rel_alphas, rel_betas):
    betas = np.concatenate([rel_betas[0], rel_betas]).cumsum()
    alphas = np.concatenate([rel_alphas[0], rel_alphas]).cumsum()
    return alphas, betas


# gamma_i = {cos(alpha_i), sin(alpha_i), beta_i}
def joint2trig(alphas, betas):
    assert np.size(betas) == np.size(alphas)
    trig = np.array([])
    for beta, alpha in zip(betas, alphas):
        trig = np.append(trig, np.array([np.cos(alpha), np.sin(alpha), beta]))
    return trig


def trig2joint(gammas):
    gammas = [gammas[i:i+3] for i in range(0, len(gammas), 3)]
    betas = np.array([])
    alphas = np.array([])
    for gamma in gammas:
        joint = np.array([np.arctan2(gamma[1], gamma[0]), gamma[2]])
        betas = np.append(betas, joint[1])
        alphas = np.append(alphas, joint[0])
    return alphas, betas

# d_i = {d_Re,i, d_Im,i} = {(k - B_i/L_i)cos(alpha_i), (k - B_i/L_i)sin(alpha_i)}
def joint2polar(alphas, betas, Lengths, k=1):
    assert np.size(betas) == np.size(alphas)
    d = np.empty((2, np.size(alphas)), np.float)
    tube = 0
    for alpha, beta, L in zip(alphas, betas, Lengths):
        d_Re = (k - beta / L) * np.cos(alpha)
        d_Im = (k - beta / L) * np.sin(alpha)
        d[:, tube] = np.array([d_Re, d_Im])
        tube += 1
    return d


def polar2joint(d, Lengths, k=1):
    assert np.size(d, 0) == 2
    d_Res = d[0, :]
    d_Ims = d[1, :]

    alphas = np.empty(np.size(d_Res), np.float)
    betas = np.empty(np.size(d_Res), np.float)
    tube = 0
    for d_Re, d_Im, L in zip(d_Res, d_Ims, Lengths):
        alpha = np.arctan2(d_Im, d_Re)
        beta = L * (k - np.sqrt(np.square(d_Im) + np.square(d_Re)))
        alphas[tube] = alpha
        betas[tube] = beta
        tube += 1
    return alphas, betas


if __name__ == '__main__':
    """
    # Need constraints to generate valid points
    num_tubes = 3
    alpha_min = -np.pi
    alpha_max = np.pi
    alpha_space = spaces.Box(low=np.full(num_tubes, alpha_min), high=np.full(num_tubes, alpha_max))
    tube_lengths = np.array([0.15, 0.10, 0.05])
    beta_min = -tube_lengths
    beta_max = np.array([0, 0, 0])
    beta_space = spaces.Box(low=beta_min, high=beta_max)

    # Sample joint space an ensure in constraints
    while True:
        alpha_sample = alpha_space.sample()
        beta_sample = beta_space.sample()

        valid_joint = []
        for i in range(1, num_tubes):
            valid_joint.append((beta_sample[i - 1] <= beta_sample[i]) and (
                        beta_sample[i - 1] + tube_lengths[i - 1] >= beta_sample[i] + tube_lengths[i]))

        if all(valid_joint):
            break

    print("polar representation:")
    print("Sampled alpha: ", alpha_sample)
    print("Sampled beta: ", beta_sample)
    d = joint2polar(alpha_sample, beta_sample, tube_lengths)
    alpha_T, beta_T = polar2joint(d, tube_lengths)
    print("Transformed alpha: ", alpha_T)
    print("Transformed beta: ", beta_T)

    print("trignometric representation:")
    print("Sampled alpha: ", alpha_sample)
    print("Sampled beta: ", beta_sample)
    trig_rep = joint2trig(alpha_sample, beta_sample)
    alpha_T, beta_t = trig2joint(trig_rep)
    print("Transformed alpha: ", alpha_T)
    print("Transformed beta: ", beta_T)

    print("relative representation:")
    print("Sampled alpha: ", alpha_sample)
    print("Sampled beta: ", beta_sample)
    alpha_rel, beta_rel = abs2rel(alpha_sample, beta_sample)
    alpha_abs, beta_abs = rel2abs(alpha_rel, beta_rel)
    print("Transformed alpha: ", alpha_abs)
    print("Transformed beta: ", beta_abs)
    """

    # Create environment for basic representation
    # TODO: Delete polar
    # TODO: Test normalization of state
    # TODO: Test action shielding (another script)
    spec_list = [gym.spec('CTR-Reach-v0')]
    representations = ['basic', 'trig']
    relative_q = [True, False]
    for spec in spec_list:
        for rep in representations:
            kwargs = {'joint_representation': rep}
            for rel_q in relative_q:
                kwargs = {'relative_q': rel_q}
                # Test representation
                test_env = spec.make(**kwargs)
                obs = test_env.reset()
                # Ground truth q
                q_ = test_env.rep_obj.get_q()
                if rep is 'trig':
                    rep_ = obs['observation'][:9]
                    q_conv = test_env.rep_obj.rep2joint(rep_)
                    if rel_q:
                        q_rel = test_env.rep_obj.qrel2abs(q_conv)
                    else:
                        q_rel = q_conv
                    # Check if converted matches ground truth
                    if not np.array_equal(q_, q_rel):
                        print("q: ", q_, "\nq_conv: ", q_conv, "\nq_rel: ", q_rel)



