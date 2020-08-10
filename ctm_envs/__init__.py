from gym.envs.registration import register

register(
    id='CTR-Reach-v0', entry_point='ctm_envs.envs:CtmEnv',
    kwargs={
        #'tube_parameters': {
        #    'tube_0':
        #        {'length': 215e-3, 'length_curved': 14.9e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
        #         'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
        #         'k': 16.0},

        #    'tube_1':
        #        {'length': 120.2e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
        #         'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
        #         'k': 9.0},

        #    'tube_2':
        #        {'length': 48.5e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
        #         'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
        #         'k': 4.0}
        'tube_parameters': {
            'tube_0':
                {'length': 150e-3, 'length_curved': 100e-3, 'inner_diameter': 1.0e-3, 'outer_diameter': 2.4e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
                 'k': 16.0},

            'tube_1':
                {'length': 100e-3, 'length_curved': 21.6e-3, 'inner_diameter': 3.0e-3, 'outer_diameter': 3.8e-3,
                 'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
                 'k': 9.0},

            'tube_2':
                {'length': 70e-3, 'length_curved': 8.8e-3, 'inner_diameter': 4.4e-3, 'outer_diameter': 5.4e-3,
                 'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
                 'k': 4.0}
        },
        'model': 'dominant_stiffness',
        'action_length_limit': 0.001,
        'action_rotation_limit': 5,
        'max_episode_steps': 150,
        'n_substeps': 10,
        'goal_tolerance_parameters': {
            'inc_tol_obs': False, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'constant'
        },
        'joint_representation': 'basic',
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        'initial_q': [0, 0, 0, 0, 0, 0],
        'relative_q': False,
        'render': False
    },
    max_episode_steps=150.
)