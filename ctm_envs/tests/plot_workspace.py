import matplotlib.pyplot as plt
import gym
import ctm2_envs
import numpy as np


def single_extension():
    # render the workspace, x-z, y-z, x-y
    num_q = 10
    spec_list = [gym.spec('Distal-2-Tube-Reach-v0')]
    for spec in spec_list:
        env = spec.make()
        q = np.linspace(env.q_space.low, env.q_space.high, num_q)
        ext = q[:, 1::2]
        points = np.empty(shape=(0, 3))

        # Extend tube 1
        for i in ext[:, 0]:
            q = [0, i, 0, 0]
            print(q)
            point = env.fk.forward_kinematics(q)
            points = np.vstack([points, point])

        # Extend tube 2
        for i in ext[:, 1]:
            q = [0, ext[:, 0].max(), 0, i]
            print(q)
            point = env.fk.forward_kinematics(q)
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
        plt.xlabel('y')
        plt.ylabel('z')
        plt.show()


def render_workspace():
    # render the workspace, x-z, y-z, x-y
    num_q = 25
    spec_list = [gym.spec('Distal-2-Tube-Reach-v0')]
    for spec in spec_list:
        env = spec.make()
        q = np.linspace(env.q_space.low, env.q_space.high, num_q)
        ext = q[:, 1::2]
        rot = q[:, 0::2]
        points = np.empty(shape=(0, 3))

        # For every angle in rotation extend tube 1
        for i in rot[:, 1]:
            for j in ext[:, 1]:
                # For every tube 1 extension, rotation, every angle in rotation extend tube 2
                for k in rot[:, 0]:
                    for l in ext[:, 0]:
                        print("q: ", [i, j, k, l])
                        point = env.fk.forward_kinematics([i, j, k, l])
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

        plt.show()


if __name__ == '__main__':
    #single_extension()
    render_workspace()
