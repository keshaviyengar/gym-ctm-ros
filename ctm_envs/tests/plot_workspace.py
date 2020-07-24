import matplotlib.pyplot as plt
import gym
import ctm2_envs
import numpy as np


def single_extension():
    # render the workspace, x-z, y-z, x-y
    num_q = 50
    spec_list = [gym.spec('Distal-2-Tube-Reach-v0')]
    for spec in spec_list:
        env = spec.make()
        q = np.linspace(env.q_space.low, env.q_space.high, num_q)
        rot = q[:, 0]
        ext = q[:, 1]
        points = np.empty(shape=(0, 3))

        for i in ext:
            point = env.fk.forward_kinematics([0, i])
            points = np.vstack([points, point])

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0

        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5

        plt.subplot(3, 1, 1)
        plt.plot(ys, zs, '.')
        plt.axis('equal')
        plt.xlabel('y')
        plt.ylabel('z')

        plt.subplot(3, 1, 2)
        plt.plot(xs, zs, '.')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('z')

        plt.subplot(3, 1, 3)
        plt.plot(xs, ys, '.')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def render_workspace():
    # render the workspace, x-z, y-z, x-y
    num_q = 100
    spec_list = [gym.spec('Distal-2-Tube-Reach-v0')]
    for spec in spec_list:
        env = spec.make()
        q = np.linspace(env.q_space.low, env.q_space.high, num_q)
        rot = q[:, 0]
        ext = q[:, 1]
        points = np.empty(shape=(0, 3))

        for i in rot:
            for j in ext:
                point = env.fk.forward_kinematics([i, j])
                points = np.vstack([points, point])

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0

        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5

        plt.subplot(2, 2, 4)
        plt.plot(ys, zs, '.')
        plt.axis('equal')
        plt.xlabel('y')
        plt.ylabel('z')

        plt.subplot(2, 2, 3)
        plt.plot(xs, zs, '.')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('z')

        plt.subplot(2, 2, 1)
        plt.plot(xs, ys, '.')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    #single_extension()
    render_workspace()
