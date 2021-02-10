from gym.envs.registration import register
import numpy as np
register(
    id='CTR-Reach-Full-Pose-v0', entry_point='ctm_envs.envs:CtmEnv',
    kwargs={
        'tube_parameters': {
            'tube_0':
                {'length': 215e-3, 'length_curved': 14.9e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
                 'k': 16.0},

            'tube_1':
                {'length': 120.2e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
                 'k': 9.0},

            'tube_2':
                {'length': 48.5e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
                 'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
                 'k': 4.0}
        },
        # 'tube_parameters': {
        #     'tube_0':
        #         {'length': 150e-3, 'length_curved': 100e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
        #          'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
        #          'k': 16.0},

        #     'tube_1':
        #         {'length': 100e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
        #          'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
        #          'k': 9.0},

        #     'tube_2':
        #         {'length': 70e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
        #          'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
        #          'k': 4.0}
        # },
        'model': 'exact',
        'action_length_limit': 0.001,
        'action_rotation_limit': 5,
        'max_episode_steps': 150,
        'n_substeps': 10,
        'pos_tolerance_parameters': {
            'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'decay'
        },
        'orient_tolerance_parameters': {
            'inc_tol_obs': True, 'initial_tol': np.deg2rad(20.0), 'final_tol': np.deg2rad(1.0),
            'N_ts': 200000, 'function': 'decay'
        },
        'pos_tolerance_parameters': {
            'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001, 'N_ts': 200000, 'function': 'decay',
        },
        'noise_parameters': {
                                  # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'pos_tracking_std': 0.0,
            'orient_tracking_std': 0.0,
        },
        'joint_representation': 'trig',
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        'initial_q': [0, 0, 0, 0, 0, 0],
        'relative_q': True,
        'render': False,
        'resample_joints': True
    },
    max_episode_steps=150.
)

register(
    id='CTR-Reach-Noisy-v0', entry_point='ctm_envs.envs:CtmEnv',
    kwargs={
        'tube_parameters': {
            'tube_0':
                {'length': 215e-3, 'length_curved': 14.9e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
                 'k': 16.0},

            'tube_1':
                {'length': 120.2e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
                 'k': 9.0},

            'tube_2':
                {'length': 48.5e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
                 'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
                 'k': 4.0}
        },
        # 'tube_parameters': {
        #     'tube_0':
        #         {'length': 150e-3, 'length_curved': 100e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
        #          'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
        #          'k': 16.0},

        #     'tube_1':
        #         {'length': 100e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
        #          'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
        #          'k': 9.0},

        #     'tube_2':
        #         {'length': 70e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
        #          'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
        #          'k': 4.0}
        # },
        'model': 'exact',
        'action_length_limit': 0.001,
        'action_rotation_limit': 5,
        'max_episode_steps': 150,
        'n_substeps': 10,
        'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'constant'
        },
        'noise_parameters': {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(1.0), 'extension_std': 0.001 * np.deg2rad(1.0), 'tracking_std': 0.0008
        },
        'joint_representation': 'trig',
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        'initial_q': [0, 0, 0, 0, 0, 0],
        'relative_q': True,
        'render': False,
        'resample_joints': True
    },
    max_episode_steps=150.
)
