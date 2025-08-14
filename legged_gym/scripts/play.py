from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from isaacgym import  gymapi
import numpy as np
import torch
import copy
from isaacgym.torch_utils import *
from scipy.signal import savgol_filter



import matplotlib.pyplot as plt

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task) #登记一个任务
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs,1)
    env_cfg.terrain.mesh_type ='trimesh'#'plane'
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.terrain.num_rows = 5#6
    env_cfg.terrain.num_cols =5# 10
    env_cfg.terrain.curriculum =False
    env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s=5
    env_cfg.domain_rand.max_push_vel_xy =0.1
    env_cfg.env.episode_length_s =20

    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_strength  = False
    env_cfg.domain_rand.randomize_kpkd = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.control.use_filter = True
    env_cfg.domain_rand.disturbance =True
    env_cfg.domain_rand.disturbance_range = [-300.0, 300.0]

    

    # todo
    env_cfg.terrain.border_size =20
    env_cfg.commands.ranges.lin_vel_x =[-1.0, 1.0]
    env_cfg.commands.resampling_time = 1000.
    env_cfg.commands.heading_command = False
    # env_cfg.commands.ranges.lin_vel_x =[-0.5, -0.4]
    # env_cfg.commands.ranges.lin_vel_y =[0, 0]
    # env_cfg.commands.ranges.ang_vel_yaw =[0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations() #获取最开始的观测值，都是0（四足没有地形高度时候观测值个数是48）
   
    # load policy
    train_cfg.runner.resume = True #这里确定是要resume表示用之前训练的模型来运行
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)#得到rl算法的runner
    policy = ppo_runner.get_inference_policy(device=env.device)#act_teacher
    # export policy as a jit module (used to run it from C++)
    if EXPORT_ONNX:
        path_onnx = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'onnx')
        ppo_runner.alg.actor_critic.export_to_onnx(path_onnx)
        print('Exported onnx to: ', path_onnx)
        path_pt = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'pt')
        ppo_runner.alg.actor_critic.export_to_pt(path_pt)
        print('Exported pt to: ', path_pt)
   

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 80# number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards 在打印平均episodereward前要step的数量
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64) #镜头的位置
    camera_vel = np.array([1., 1., 0.]) #镜头的速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos) #镜头的方向
    #------------------------------------------------
    # 相机相对位置
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    #定义相机的分辨率
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512
    # 机器人
    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    #相机的位置和方向跟随机器人身体运动
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)
    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    #------------------------------------------------
    img_idx = 0

    for i in range(200*int(env.max_episode_length)): #执行episode个数
        if i>=10*int(env.max_episode_length)-1:
            print("stop...")
        #---------------------------------------------------------
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        env.commands[:,0] =1
        env.commands[:,1] = 0
        env.commands[:,2] = 0
        env.commands[:,3] = 0
        #---------------------------------------------------------------


        actions = policy(obs.detach())
        # print("actions:")
        # print(actions)
        obs, _, _,rews, dones, infos = env.step(actions.detach())

        if RECORD_FRAMES: #如果记录每帧的图像
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        if MOVE_CAMERA: #如果移动相机视角
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if  (i < stop_state_log):
            logger.log_states(
                {
                    'dof_vel_FL_hip_joint': env.dof_vel[robot_index, 0].item(),
                    'dof_vel_FL_thigh_joint': env.dof_vel[robot_index, 1].item(),
                    'dof_vel_FL_calf_joint': env.dof_vel[robot_index, 2].item(),
                    'dof_vel_FR_hip_joint': env.dof_vel[robot_index, 3].item(),
                    'dof_vel_FR_thigh_joint': env.dof_vel[robot_index, 4].item(),
                    'dof_vel_FR_calf_joint': env.dof_vel[robot_index, 5].item(),
                    'dof_vel_RL_hip_joint': env.dof_vel[robot_index, 6].item(),
                    'dof_vel_RL_thigh_joint': env.dof_vel[robot_index, 7].item(),
                    'dof_vel_RL_calf_joint': env.dof_vel[robot_index, 8].item(),
                    'dof_vel_RR_hip_joint': env.dof_vel[robot_index, 9].item(),
                    'dof_vel_RR_thigh_joint': env.dof_vel[robot_index, 10].item(),
                    'dof_vel_RR_calf_joint': env.dof_vel[robot_index, 11].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states_dof_vel()
  
        if  0 < i < stop_rew_log: #记录episode每步结果
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item() #检测有多少项
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log: # 最后打印一下reward
            logger.print_rewards()

if __name__ == '__main__':
    # EXPORT_POLICY = True
    EXPORT_ONNX=True
    RECORD_FRAMES = False#True # 开启后画面会很慢
    MOVE_CAMERA = False
    args = get_args() #所有的指定的参数
    play(args)








  