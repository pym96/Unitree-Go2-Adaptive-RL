# Unitree Go2 adpative Control RL

A adpative control implementation of RL algorithm for unitree go2 quadruped robot. Ubuntu18.04 or later required.

![814-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/186bece2-4a38-41b8-bb13-f0847ca28e86)


Setup steps are below:

## Anaconda installation 

Download [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install#linux-installer)


```
conda create -n unitree_rl python==3.8 
conda activate unitree_rl
```

```
pip install torch torchaudio torchvision numpy==1.21.6 tensorboard pybullet pynput opencv-python onnx onnxruntime storage scikit-learn
```

## Issac Gym installation 

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website.

```
cd isaacgym/python
pip install -e .
```

test
```
cd examples
python 1080_balls_of_solitude.py
```

## rsl_rl installation
```
git clone https://github.com/leggedrobotics/rsl_rl.git
```

To 1.0.2 branch(For python3.8)
```
cd rsl_rl
git checkout v1.0.2
```

install
```
pip install -e .
```

## legged_gym installation
```
git clone https://github.com/leggedrobotics/legged_gym.git
```

```
cd legged_gym && pip install -e .
```


## For this project 
You should configure envs above for they are necessary prerequisites
```
git clone https://github.com/pym96/Unitree-Go2-Adaptive-RL.git
```


### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing  all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=go2 ```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy :  
```python legged_gym/scripts/play.py --task=go2```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
3. Export to onnx
   refer to legged_gym/scripts/play.py

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!

## Reference link
* **`Legged-Gym`** (built on top of NVIDIA Isaac Gym): https://leggedrobotics.github.io/legged_gym/

* **`Rsl-rl`** (A fast and simple implementation of RL algorithms, designed to run fully on GPU.):https://github.com/leggedrobotics/rsl_rl.git


* **`Unitree-rl-gymm`** ( reinforcement learning implementation based on Unitree robots) :https://github.com/unitreerobotics/unitree_rl_gym.git

## Future work
- C++ onnx RL deployment with [onnxruntime](https://github.com/microsoft/onnxruntime.git)
- 3D mapping with quadruped robot
- Terrian analysis with [elevation_mapping](https://github.com/ANYbotics/elevation_mapping.git)
- Final 3d planner with things above.....
