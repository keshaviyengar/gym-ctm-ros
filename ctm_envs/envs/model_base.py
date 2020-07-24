import numpy as np


class ModelBase:
    def __init__(self, tube_parameters, ros=False):
        self.tubes = tube_parameters
        self.num_tubes = len(tube_parameters)

    def forward_kinematics(self, q, **kwargs):
        raise NotImplementedError
