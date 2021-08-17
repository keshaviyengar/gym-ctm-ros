import numpy as np
from math import pi, pow
from scipy.integrate import odeint

from ctm_envs.envs.model_base import ModelBase


class CTRExactModel(ModelBase):
    def __init__(self, tube_parameters):
        self.r = []
        self.r1 = []
        self.r2 = []
        self.r3 = []
        self.r_transforms = []
        super(CTRExactModel, self).__init__(tube_parameters, ros=False)

    def forward_kinematics(self, q, **kwargs):
        q_0 = np.array([0, 0, 0, 0, 0, 0])
        # position of tubes' base from template (i.e., s=0)
        beta = q[0:3] + q_0[0:3]

        segment = Segment(self.tubes[0], self.tubes[1], self.tubes[2], beta)

        r_0_ = np.array([0, 0, 0]).reshape(3, 1)
        alpha_1_0 = q[3] + q_0[3]
        R_0_ = np.array(
            [[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]]) \
            .reshape(9, 1)
        alpha_0_ = q[3:].reshape(3, 1) + q_0[3:].reshape(3, 1)

        # initial twist
        uz_0_ = np.array([0, 0, 0])
        shape, U_z, tip = self.ctr_model(uz_0_, alpha_0_, r_0_, R_0_, segment, beta)
        return shape[-1]

    def ode_eq(self, y, s, ux_0, uy_0, ei, gj):
        dydt = np.empty([18, 1])
        ux = np.empty([3, 1])
        uy = np.empty([3, 1])
        for i in range(0, 3):
            ux[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                    (ei[0] * ux_0[0] * np.cos(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.sin(y[3 + i] - y[3 + 0]) +
                     ei[1] * ux_0[1] * np.cos(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.sin(y[3 + i] - y[3 + 1]) +
                     ei[2] * ux_0[2] * np.cos(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.sin(y[3 + i] - y[3 + 2]))
            uy[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                    (-ei[0] * ux_0[0] * np.sin(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.cos(y[3 + i] - y[3 + 0]) +
                     -ei[1] * ux_0[1] * np.sin(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.cos(y[3 + i] - y[3 + 1]) +
                     -ei[2] * ux_0[2] * np.sin(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.cos(y[3 + i] - y[3 + 2]))

        for j in range(0, 3):
            if ei[j] == 0:
                dydt[j] = 0  # ui_z
                dydt[3 + j] = 0  # alpha_i
            else:
                dydt[j] = ((ei[j]) / (gj[j])) * (ux[j] * uy_0[j] - uy[j] * ux_0[j])  # ui_z
                dydt[3 + j] = y[j]  # alpha_i

        e3 = np.array([0, 0, 1]).reshape(3, 1)
        uz = y[0:3]
        R = np.array(y[9:]).reshape(3, 3)
        u_hat = np.array([(0, - uz[0], uy[0]), (uz[0], 0, -ux[0]), (-uy[0], ux[0], 0)])
        dr = np.dot(R, e3)
        dR = np.dot(R, u_hat).ravel()

        dydt[6] = dr[0]
        dydt[7] = dr[1]
        dydt[8] = dr[2]

        for k in range(3, 12):
            dydt[6 + k] = dR[k - 3]
        return dydt.ravel()

    # CTR model
    def ctr_model(self, uz_0, alpha_0, r_0, R_0, segmentation, beta):
        Length = np.empty(0)
        r = np.empty((0, 3))
        u_z = np.empty((0, 3))
        alpha = np.empty((0, 3))
        span = np.append([0], segmentation.S)
        for seg in range(0, len(segmentation.S)):
            # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
            y_0 = np.vstack((uz_0.reshape(3, 1), alpha_0, r_0, R_0)).ravel()
            s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
            s = odeint(self.ode_eq, y_0, s_span, args=(
                segmentation.U_x[:, seg], segmentation.U_y[:, seg], segmentation.EI[:, seg], segmentation.GJ[:, seg]))
            Length = np.append(Length, s_span)
            u_z = np.vstack((u_z, s[:, (0, 1, 2)]))
            alpha = np.vstack((alpha, s[:, (3, 4, 5)]))
            r = np.vstack((r, s[:, (6, 7, 8)]))

            # new boundary conditions for next segment
            r_0 = r[-1, :].reshape(3, 1)
            R_0 = np.array(s[-1, 9:]).reshape(9, 1)
            uz_0 = u_z[-1, :].reshape(3, 1)
            alpha_0 = alpha[-1, :].reshape(3, 1)

        d_tip = np.array([self.tubes[0].L,self.tubes[1].L, self.tubes[2].L]) + beta
        u_z_end = np.array([0.0, 0.0, 0.0])
        tip_pos = np.array([0, 0, 0])
        for k in range(0, 3):
            b = np.argmax(Length >= d_tip[k] - 1e-3)  # Find where tube curve starts
            u_z_end[k] = u_z[b, k]
            tip_pos[k] = b

        return r, u_z_end, tip_pos

    def get_r(self):
        return self.r

    def get_rs(self):
        return self.r1, self.r2, self.r3

    def get_r_transforms(self):
        return self.r_transforms

class Segment:
    def __init__(self, t1, t2, t3, base):
        stiffness = np.array([t1.E, t2.E, t3.E])
        torsion = np.array([t1.G, t2.G, t3.G])
        curve_x = np.array([t1.U_x, t2.U_x, t3.U_x])
        curve_y = np.array([t1.U_y, t2.U_y, t3.U_y])

        d_tip = np.array([t1.L, t2.L, t3.L]) + base  # position of tip of the tubes
        d_c = d_tip - np.array([t1.L_c, t2.L_c, t3.L_c])  # position of the point where tube bending starts
        points = np.hstack((0, base, d_c, d_tip))
        index = np.argsort(points)
        segment_length = 1e-5 * np.floor(1e5 * np.diff(np.sort(points)))

        e = np.zeros((3, segment_length.size))
        g = np.zeros((3, segment_length.size))
        u_x = np.zeros((3, segment_length.size))
        u_y = np.zeros((3, segment_length.size))

        for i in range(0, 3):
            aa = np.where(index == i + 1)  # Find where tube begins
            a = aa[0]
            bb = np.where(index == i + 4)  # Find where tube curve starts
            b = bb[0]
            cc = np.where(index == i + 7)  # Find where tube ends
            c = cc[0]
            if segment_length[a] == 0:
                a += 1
            if segment_length[b] == 0:
                b += 1
            if segment_length[a] == 0:
                a += 1
            if c.item() <= segment_length.size - 1:
                if segment_length[c] == 0:
                    c += 1

            e[i, np.arange(a, c)] = stiffness[i]
            g[i, np.arange(a, c)] = torsion[i]
            u_x[i, np.arange(b, c)] = curve_x[i]
            u_y[i, np.arange(b, c)] = curve_y[i]

        # Getting rid of zero lengths
        length = segment_length[np.nonzero(segment_length)]
        ee = e[:, np.nonzero(segment_length)]
        gg = g[:, np.nonzero(segment_length)]
        uu_x = u_x[:, np.nonzero(segment_length)]
        uu_y = u_y[:, np.nonzero(segment_length)]

        length_sum = np.cumsum(length)
        self.S = length_sum[length_sum + min(base) > 0] + min(base)  # s is segmented abscissa of tube after template

        # Truncating matrices, removing elements that correspond to the tube before the template
        e_t = ee[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.EI = (e_t.T * np.array([t1.I, t2.I, t3.I])).T
        g_t = gg[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.GJ = (g_t.T * np.array([t1.J, t2.J, t3.J])).T
        self.U_x = uu_x[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
        self.U_y = uu_y[length_sum + min(base) > 0 * ee].reshape(3, len(self.S))
