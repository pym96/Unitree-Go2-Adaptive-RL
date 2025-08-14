from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg(LeggedRobotCfg):
    class env:
        num_envs = 1024
        n_scan = 187
        n_priv_latent = 47
        n_proprio = 45+3
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent 
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        history_encoding = True

        include_foot_contacts = True
        contact_buf_len = 100
        num_states = 11

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        # curriculum = False
        # rough terrain only:
        # measure_heights = True
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        max_init_terrain_level = 5 # starting curriculum state
        # max_init_terrain_level = 1 # starting curriculum state
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        # terrain_proportions = [1, 0, 0, 0, 0]
        border_size = 20 # [m]
        # trimesh only:
        slope_treshold = 0.9 # slopes above this threshold will be corrected to vertical surfaces
        # todo for teacher step 6 (revise config) --jh in 1023
        teleport_robots = True
        teleport_thresh = 2.0


        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

    class commands:
        curriculum = True
        max_curriculum_x = 1.0#1.8
        max_curriculum_reverse_x = -1.0#1.8
        max_curriculum_y = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        only_forward_command = False # if true: only lin_vel_x command is used
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        # rot = [1.0, 0.0, 0.0, 0.0] #摔倒
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]    
        stiffness_noise = {'joint': 2.0}
        damping_noise = {'joint': 0.2}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5      
        decimation = 4
        use_filter = True

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        foot_name = "foot" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 1
        fix_base_link = False # fixe the base of the robot
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation

        name = "legged_robot"  # actor name
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.5, 2.5]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]

        push_robots = True
        push_interval_s = 5
        max_push_vel_xy =0.5# 2

       
        #2025.1.23 KDR_add
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_kpkd = True
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]

      

        #2025.1.23 KDR_add
        randomize_lag_timesteps = True#时滞条件随机化
        lag_timesteps = 3#3

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval_s = 8



    class rewards:
        class scales:

            powers = -2e-5
            dof_acc = -2.5e-7
            action_rate = -0.01
            action_smoothness=-0.01 #-0.01
            

            foot_clearance = -0.5#-0.5
            foot_mirror = -0.05#-0.05
            foot_slide = -0.05
            collision = -1
            
            stumble = -0.05
            # upward = 0.6#0.6
            # has_contact = 0.6#0.6
            tracking_lin_vel =2.0
            tracking_ang_vel = 1.0
            stand_nice =-0.1#-0.1
            base_height = -10.0#-10
            #lin_vel_z_up = -4.0
            lin_vel_z = -4.0
            #ang_vel_xy_up = -0.05
            ang_vel_xy = -0.1
            orientation=-0.2
            feet_contact_forces = -0.00015

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit =0.9# 0.8#0.85 percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1#0.95
        soft_torque_limit = 1#0.95
        base_height_target = 0.34#0.32
        max_contact_force = 100.  # forces above this value are penalized
        clearance_height_target = -0.22


    class costs:
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

    class cost:
        num_costs = 3

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            ang_gravity = 1.0
            ang_rpy = 10.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            # to do
            dof_pos = 0.02 # 0.05
            dof_vel = 1.5 # 0.8
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            pos_err = 0.00 # 0.02
            vel_err = 0.0 # 0.02
            height_measurements = 0.1
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)\\

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True

class Go2CfgPPO( LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
    
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False

        scan_encoder_dims = None#[128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        num_costs = 3

        teacher_act = True
        imi_flag =True
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 2 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]  
        cost_value_loss_coef = 1
        cost_viol_loss_coef = 1
    
    class depth_encoder:
        if_depth = Go2Cfg.depth.use_camera
        depth_shape = Go2Cfg.depth.resized
        buffer_len =Go2Cfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = Go2Cfg.depth.update_interval * 24

    class runner:
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        # load and resume
        # updated from load_run and chkpt
        run_name = ''
        experiment_name = 'Go2'
        policy_class_name ='ActorCriticSwAV'#ActorCriticSwAV'#'ActorCriticKmeans'#'ActorCriticSwAV_t-SNE'#'ActorCriticBarlowTwins'
        runner_class_name = 'OnPolicyNP3O'
        algorithm_class_name =  'NP3O'
        max_iterations = 20000
        num_steps_per_env = 24
        resume = True
        resume_path = ''
        load_run = -1 # -1 = last run
        checkpoint = -1 