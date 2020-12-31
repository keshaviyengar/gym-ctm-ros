import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class ModelBase:
    def __init__(self, tube_parameters, ros=False):
        self.tubes = tube_parameters
        self.num_tubes = len(tube_parameters)

        # Figures initalisation
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.5,-0.5), ylim=(-0.5,-0.5), zlim=(0,1), projection='3d')

    def forward_kinematics(self, q, **kwargs):
        raise NotImplementedError

    def show_plt(self, achieved_goal, desired_goal):
        self.fig.show()
        self.ax.clear()
        self.ax.set(xlim=(-0.05,0.05), ylim=(-0.05,0.05), zlim=(0,0.1))
        # Plot desired and achieved points
        self.ax.scatter(desired_goal[0], desired_goal[1], desired_goal[2], '-b', linewidth=2)
        self.ax.scatter(achieved_goal[0], achieved_goal[1], achieved_goal[2], '-r', linewidth=2)
        # Plot shape
        self.ax.plot(self.r1[:, 0], self.r1[:, 1], self.r1[:, 2], '-b', linewidth=2)
        self.ax.plot(self.r2[:, 0], self.r2[:, 1], self.r2[:, 2], '-r', linewidth=3)
        self.ax.plot(self.r3[:, 0], self.r3[:, 1], self.r3[:, 2], '-g', linewidth=4)
        self.fig.canvas.draw()
