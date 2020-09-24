import numpy as np
from math import pi, pow
from scipy.integrate import odeint

from ctm_envs.envs.model_base import ModelBase


class ExactModel(ModelBase):
    def __init__(self, tube_parameters):
        self.r = []
        self.r_transforms = []
        super(ExactModel, self).__init__(tube_parameters, ros=False)

    # q[0:2] are extension values and q[3:5] rotation values from the base
    # in general q[..] = [betas.., alphas]
    # q0 are starting joint values
    # uz_0 is column vector of initial torsion about z axis
    # r_0 is base position
    # R_0 is initial base orientation
    def forward_kinematics(self, q, **kwargs):
        if 'q_0' not in kwargs:
            kwargs['q_0'] = np.zeros(self.num_tubes * 2)
        if 'r_0' not in kwargs:
            kwargs['r_0'] = np.array([0, 0, 0]).reshape(3, 1)
        if 'uz_0' not in kwargs:
            # initial twist
            kwargs['uz_0'] = np.zeros(self.num_tubes)

        # Separate q_beta and q_alpha
        q_beta = np.array(q[:self.num_tubes])
        q_alpha = np.array(q[self.num_tubes:])

        q_0 = kwargs['q_0']
        q_0_beta = q_0[:self.num_tubes]
        q_0_alpha = q_0[self.num_tubes:]
        uz_0 = kwargs['uz_0']
        r_0 = kwargs['r_0']
        alpha_1_0 = q_alpha[0] + q_0_alpha[0]
        R_0 = np.array(
            [[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]]) \
            .reshape(9, 1)
        alpha_0 = q_alpha.reshape(self.num_tubes, 1) + q_0_alpha.reshape(self.num_tubes, 1)

        # position of tubes' base from template (i.e., s=0)
        beta = q_beta + q_0_beta
        segments = SegmentRobot(beta, self.tubes)

        Length = np.empty(0)
        r = np.empty((0, 3))
        r_transforms = np.empty((len(segments.S), 4, 4))
        u_z = np.empty((0, self.num_tubes))
        alpha = np.empty((0, self.num_tubes))
        span = np.append([0], segments.S)
        for seg in range(0, len(segments.S)):
            # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
            y_0 = np.vstack((uz_0.reshape(self.num_tubes, 1), alpha_0, r_0, R_0)).ravel()
            s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
            s = odeint(self.ode_eq, y_0, s_span, args=(segments.U_x[:, seg], segments.U_y[:, seg],
                                                       segments.EI[:, seg], segments.GJ))
            Length = np.append(Length, s_span)
            u_z = np.vstack((u_z, s[:, 0:self.num_tubes]))
            alpha = np.vstack((alpha, s[:, self.num_tubes:2*self.num_tubes]))
            r = np.vstack((r, s[:, 2*self.num_tubes:2*self.num_tubes+3]))

            # Start with identity
            transform = np.identity(4)
            # Add rotation
            transform[:3, :3] = np.array(s[-1, -9:]).reshape(3, 3)
            # Add position
            transform[0:3, 3] = np.array(s[-1, 2*self.num_tubes:2*self.num_tubes+3]).reshape(3)
            r_transforms[seg, :, :] = transform

            # new boundary conditions for next segment
            r_0 = r[-1, :].reshape(3, 1)
            R_0 = np.array(s[-1, -9:]).reshape(9, 1)
            uz_0 = u_z[-1, :].reshape(self.num_tubes, 1)
            alpha_0 = alpha[-1, :].reshape(self.num_tubes, 1)

        tube_lengths = [i.L for i in self.tubes]
        d_tip = tube_lengths + beta
        u_z_end = np.zeros(self.num_tubes)
        tip_pos = np.zeros(self.num_tubes)
        for k in range(0, self.num_tubes):
            try:
                b = np.argmax(Length >= d_tip[k] - 1e-3)  # Find where tube curve starts
            except ValueError:
                print("Error in b calculation")
                b = 0
            try:
                u_z_end[k] = u_z[b, k]
                tip_pos[k] = b
            except IndexError:
                print("Index for u_z is zero b is NaN")
                u_z_end[k] = 0
                tip_pos[k] = 0
        self.r = r
        self.r_transforms = r_transforms
        if len(r) == 0:
            print("r is zero. Relevant debug info:")
            print("q: ", q)
        return r[-1]

    def ode_eq(self, y, s, ux_0, uy_0, ei, gj):
        # first num_tubes elements represent curvatures along z
        # second num_tubes elements represent twist angles, alpha_i
        # last 12 elements are r (positions), R (orientations)
        dydt = np.empty(2 * self.num_tubes + 12)
        ux = np.empty([self.num_tubes, 1])
        uy = np.empty([self.num_tubes, 1])

        # calculate, for each tube, tube curvature in x and y direction
        for i in range(self.num_tubes):
            ux_term = 0
            uy_term = 0
            for j in range(self.num_tubes):
                ux_term += ei[j] * ux_0[j] * np.cos(y[self.num_tubes + i] - y[self.num_tubes + j]) + ei[j] * uy_0[j] * np.sin(y[self.num_tubes + i] - y[self.num_tubes + j])
                uy_term += -ei[j] * ux_0[j] * np.sin(y[self.num_tubes + i] - y[self.num_tubes + j]) + ei[j] * uy_0[j] * np.cos(y[self.num_tubes + i] - y[self.num_tubes + j])
            ux[i] = (1 / np.sum(ei)) * ux_term
            uy[i] = (1 / np.sum(ei)) * uy_term

        # ode for twist
        for i in range(self.num_tubes):
            if ei[i] == 0:
                # ui_z
                dydt[i] = 0
                # alpha_i
                dydt[self.num_tubes + i] = 0
            else:
                # ui_z
                dydt[i] = ((ei[i]) / (gj[i])) * (ux[i] * uy_0[i] - uy[i] * ux_0[i])
                # alpha_i
                dydt[self.num_tubes + i] = y[i]

        e3 = np.array([0, 0, 1]).reshape(3, 1)
        # first num_tubes elements represent curvatures along z
        # second num_tubes elements represent twist angles, alpha_i
        # last 12 elements are r (positions), R (orientations)
        uz = y[0:self.num_tubes]
        R = np.array(y[2 * self.num_tubes + 3:]).reshape(3, 3)
        u_hat = np.array([(0, - uz[0], uy[0]), (uz[0], 0, -ux[0]), (-uy[0], ux[0], 0)])
        dr = np.dot(R, e3)
        dR = np.dot(R, u_hat).ravel()
        dydt[2 * self.num_tubes] = dr[0]
        dydt[2 * self.num_tubes + 1] = dr[1]
        dydt[2 * self.num_tubes + 2] = dr[2]

        dydt[-9:] = dR
        return dydt.ravel()

    def get_r(self):
        return self.r

    def get_r_transforms(self):
        return self.r_transforms


# Initialized with objects of class TubeParameters
class SegmentRobot(object):
    def __init__(self, base, tubes):
        n = np.size(tubes)
        stiffness = np.array([])
        curve_x = np.array([])
        curve_y = np.array([])
        L = np.array([])
        L_c = np.array([])
        I = np.array([])
        G = np.array([])
        J = np.array([])
        for tube in tubes:
            stiffness = np.append(stiffness, tube.E)
            curve_x = np.append(curve_x, tube.U_x)
            curve_y = np.append(curve_y, tube.U_y)
            L = np.append(L, tube.L)
            L_c = np.append(L_c, tube.L_c)

            I = np.append(I, tube.I)
            G = np.append(G, tube.G)
            J = np.append(J, tube.J)

        # position of tip of tubes
        d_tip = L + base
        # positions where bending starts
        d_c = d_tip - L_c
        points = np.hstack((0, base, d_c, d_tip))
        index = np.argsort(points)
        # floor is used to ensure absolute zero is used
        segment_length = 1e-5 * np.floor(1e5 * np.diff(np.sort(points)))

        e = np.zeros((n, segment_length.size))
        u_x = np.zeros((n, segment_length.size))
        u_y = np.zeros((n, segment_length.size))

        index_itr = []
        if n == 1:
            index_itr = [1, 2, 3]
        elif n == 2:
            index_itr = [1, 3, 5]
        elif n == 3:
            index_itr = [1, 3, 7]

        for i in range(n):
            # +1 because, one point is at 0
            # find where the tube begins
            a = np.where(index == i + index_itr[0])[0]
            # find where tube curve starts
            b = np.where(index == i + index_itr[1])[0]
            # Find where tube ends
            c = np.where(index == i + index_itr[2])[0]
            if segment_length[a] == 0:
                a += 1
            if segment_length[b] == 0:
                b += 2
            if (c.item() <= segment_length.size - 1) and (segment_length[c] == 0):
                c += 1

            # For the segment a to c, set the stiffness for tube i
            e[i, np.arange(a, c)] = stiffness[i]
            # For segment b to c set curvature for tube i in x direction
            u_x[i, np.arange(b, c)] = curve_x[i]
            # For segment b to c set curvature for tube i in y direction
            u_y[i, np.arange(b, c)] = curve_y[i]

        # Removing zero lengths
        length = segment_length[np.nonzero(segment_length)]
        ee = e[:, np.nonzero(segment_length)]
        uu_x = u_x[:, np.nonzero(segment_length)]
        uu_y = u_y[:, np.nonzero(segment_length)]

        # Cumulative sum to determine length from origin
        length_sum = np.cumsum(length)
        # s is the segmented abscissa (y-coordinate) of the tube after template (s=0)
        self.S = length_sum[length_sum + min(base) > 0] + min(base)
        # Truncating matrices, removing elements that correspond to the tube before template
        e_t = ee[length_sum + min(base) > np.zeros_like(ee)].reshape(n, len(self.S))
        # Each (i,j) element of above matrices correspond to jth segment of ith tube where first tube is most inner
        self.EI = (e_t.T * I).T
        self.U_x = uu_x[length_sum + min(base) > 0 * ee].reshape(n, len(self.S))
        self.U_y = uu_y[length_sum + min(base) > 0 * ee].reshape(n, len(self.S))
        self.GJ = G * J
