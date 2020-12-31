import numpy as np
import numpy.matlib
from scipy.spatial.transform import Rotation
from math import sin, cos

from ctm_envs.envs.model_base import ModelBase


class DominantStiffnessModel(ModelBase):
    def __init__(self, tube_parameters, ros=False):
        super(DominantStiffnessModel, self).__init__(tube_parameters, ros=False)
        self.k = [i.k for i in self.tubes]
        self.l_curved = [i.L_c for i in self.tubes]
        self.tube_lengths = [i.L for i in self.tubes]
        self.r1 = []
        self.r2 = []
        self.r3 = []
        self.r = []
        self.r_transforms = []

    def forward_kinematics(self, q, **kwargs):
        """
        Takes the joint states and outputs the cartesian point
        From outer tube to inner (we flip everything) because its easier to plot this way
        q = [beta_0, beta_1 ..., alpha_0, ... alpha_n]
        """
        gamma = np.flip(q[self.num_tubes:], axis=0)
        beta = q[0:self.num_tubes]
        extension_length = beta + self.tube_lengths
        distal_length = np.ediff1d(np.flip(extension_length, axis=0), to_begin=extension_length[-1])

        num_points = 10

        k = np.flip(self.k, axis=0)
        l_curved = np.flip(self.l_curved, axis=0)

        T_tube = np.matlib.identity(4)
        # array of segment points
        r = np.empty([num_points, 3, 3])
        # array of segment points transforms
        r_transforms = np.empty([num_points, 4, 4, 3])

        for i in range(0, self.num_tubes):
            l = np.linspace(0, distal_length[i], num_points)
            for count, j in enumerate(l):
                # Get segment transform
                T_trans = self.constant_curvature_transformation(gamma[i], k[i], j, T_tube, distal_length[i], l_curved[i])
                r[count, :, i] = np.array([T_trans[0, 3], T_trans[1, 3], T_trans[2, 3]])
                r_transforms[count, :, :, i] = T_trans
            # Update T_tube for tube
            T_tube = self.constant_curvature_transformation(gamma[i], k[i], distal_length[i], T_tube, distal_length[i],
                                                            l_curved[i])

        # for i in range(0, self.num_tubes):
        #     l = distal_length[i]
        #     T_tube = self.constant_curvature_transformation(gamma[i], k[i], l, T_tube, distal_length[i],
        #                                                     l_curved[i])
        #     current_pos = np.array([T_tube[0, 3],
        #                             T_tube[1, 3],
        #                             T_tube[2, 3]])
        #     r[i, :] = current_pos

            # Each point is a segment point r
            # Last point is the end effector position

        # for i in range(0, self.num_tubes):
        #     l = distal_length[i]

        #     T_curve = np.matlib.identity(4)
        #     T_straight = np.matlib.identity(4)

        #     # https://www.researchgate.net/publication/328163977
        #     try:
        #         T_curve[0, 0] = cos(gamma[i]) * cos(gamma[i]) * (cos(k[i] * l) - 1) + 1
        #         T_curve[0, 1] = sin(gamma[i]) * cos(gamma[i]) * (cos(k[i] * l) - 1)
        #         T_curve[0, 2] = -cos(gamma[i]) * sin(k[i] * l)
        #         T_curve[0, 3] = cos(gamma[i]) * (cos(k[i] * l) - 1) / k[i]
        #         T_curve[1, 0] = sin(gamma[i]) * cos(gamma[i]) * (cos(k[i] * l) - 1)
        #         T_curve[1, 1] = cos(gamma[i]) * cos(gamma[i]) * (1 - cos(k[i] * l)) + cos(k[i] * l)
        #         T_curve[1, 2] = -sin(gamma[i]) * sin(k[i] * l)
        #         T_curve[1, 3] = sin(gamma[i]) * (cos(k[i] * l) - 1) / k[i]
        #         T_curve[2, 0] = cos(gamma[i]) * sin(k[i] * l)
        #         T_curve[2, 1] = sin(gamma[i]) * sin(k[i] * l)
        #         T_curve[2, 2] = cos(k[i] * l)
        #         T_curve[2, 3] = sin(k[i] * l) / k[i]

        #         # If the extension length is greater than tha curved section, include the difference as a straight
        #         # section.
        #         if distal_length[i] > l_curved[i]:
        #             T_straight[2, 3] = distal_length[i] - l_curved[i]

        #         T_tube = T_tube * T_straight * T_curve
        #     except ValueError:
        #         print("math domain error.")
        #         print('gamma: ', gamma)
        #         print("k: ", k)
        #         print("l: ", l)
        #         print("i: ", i)

        # current_orientation = Rotation.from_dcm(T_tube[0:3, 0:3])
        # current_pos = np.array([T_tube[0, 3],
        #                         T_tube[1, 3],
        #                         np.max([T_tube[2, 3], 0])])

        assert not np.isnan(r).any()
        ee_position = r[-1,:, i]
        self.r = r
        self.r1 = r[:, :, 0]
        self.r2 = r[:, :, 1]
        self.r3 = r[:, :, 2]
        self.r_transforms = r_transforms
        return ee_position
        # return as a quaternion vector pair
        # return current_orientation, current_pos

    def constant_curvature_transformation(self, gamma, k, l, T_tube, distal_length, l_curved):
        T_curve = np.matlib.identity(4)
        T_straight = np.matlib.identity(4)

        # https://www.researchgate.net/publication/328163977
        try:
            T_curve[0, 0] = cos(gamma) * cos(gamma) * (cos(k * l) - 1) + 1
            T_curve[0, 1] = sin(gamma) * cos(gamma) * (cos(k * l) - 1)
            T_curve[0, 2] = -cos(gamma) * sin(k * l)
            T_curve[0, 3] = cos(gamma) * (cos(k * l) - 1) / k
            T_curve[1, 0] = sin(gamma) * cos(gamma) * (cos(k * l) - 1)
            T_curve[1, 1] = cos(gamma) * cos(gamma) * (1 - cos(k * l)) + cos(k * l)
            T_curve[1, 2] = -sin(gamma) * sin(k * l)
            T_curve[1, 3] = sin(gamma) * (cos(k * l) - 1) / k
            T_curve[2, 0] = cos(gamma) * sin(k * l)
            T_curve[2, 1] = sin(gamma) * sin(k * l)
            T_curve[2, 2] = cos(k * l)
            T_curve[2, 3] = sin(k * l) / k

            # If the extension length is greater than tha curved section, include the difference as a straight
            # section.
            if l > l_curved:
                T_straight[2, 3] = l - l_curved

            T_tube = T_tube * T_straight * T_curve
            return T_tube

        except ValueError:
            print("math domain error.")
            print('gamma: ', gamma)
            print("k: ", k)
            print("l: ", l)
