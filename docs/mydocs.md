# Command
- Training A1 Terrain ``pysac scripts/rlgames_train.py task=A1Terrain num_envs=2000``
- Traning A1 Terrain and Recording on wandb ``pysac scripts/rlgames_train.py task=A1Terrain num_envs=2000 wandb_activate=true``
- Continuous training A1Terrain form saved weight ``pysac scripts/rlgames_train.py task=A1Terrain num_envs=2000 checkpoint=runs/A1Terrain/nn/last_A1Terrain_ep_3000_rew_18.23488.pth wandb_activate=true`` 
## Command testing form dataset
#### Dafualt Param
#### Less linz angxy
- Test ``pysac scripts/rlgames_demo.py task=A1Terrain num_envs=64 checkpoint=runs/A1Terrain/nn/less-zlin-xyang/last_A1Terrain_ep_12700_rew_21.401018.pth``
# Config File Information
## Task Config
### env
- numEnvs: ${resolve_default:2048,${...num_envs}}
- numObservations: 188
- numActions: 12
- envSpacing: 3.
### terrain
- staticFriction: 1.0 
- dynamicFriction: 1.0  
- restitution: 0.        
- curriculum: true
- maxInitMapLevel: 0
- mapLength: 8.
- mapWidth: 8.
- numLevels: 10
- numTerrains: 20
- slopeTreshold: 0.5
### baseInitState
- pos: [0.0, 0.0, 0.42]
- rot: [1.0, 0.0, 0.0, 0.0]
- vLinear: [0.0, 0.0, 0.0]
- vAngular: [0.0, 0.0, 0.0]
### randomCommandVelocityRanges
- linear_x: [-1.0, 1.0]
- linear_y: [-1.0, 1.0]
- yaw: [-3.14, 3.14]
### control:
- stiffness: 20.0
- damping: 0.5
- actionScale: 0.25
- decimation: 4
### defaultJointAngles
- FL_hip_joint: 0.1
- RL_hip_joint: 0.1
- FR_hip_joint: -0.1
- RR_hip_joint: -0.1
- FL_thigh_joint: 0.8
- RL_thigh_joint: 1.0
- FR_thigh_joint: 0.8
- RR_thigh_joint: 1.0
- FL_calf_joint: -1.5
- RL_calf_joint: -1.5
- FR_calf_joint: -1.5
- RR_calf_joint: -1.5
### learn
- terminalReward: 0.0
- linearVelocityXYRewardScale: 1.5 # ori=1.0
- linearVelocityZRewardScale: -1.0
- angularVelocityXYRewardScale: -0.0001
- angularVelocityZRewardScale: 0.5
- orientationRewardScale: -0.
- torqueRewardScale: -0.00002 #-0.00002 
- jointAccRewardScale: -0.00000025  #-0.0005 ##ori = -0.0005 #-0.00000025
- baseHeightRewardScale: -0.0
- actionRateRewardScale: -0.01
- fallenOverRewardScale: -1.0
- hipRewardScale: -0. #25
- linearVelocityScale: 2.0
- angularVelocityScale: 0.25
- dofPositionScale: 1.0
- dofVelocityScale: 0.05
- heightMeasurementScale: 5.0
- addNoise: false
- noiseLevel: 1.0 # scales other values
- dofPositionNoise: 0.01
- dofVelocityNoise: 1.5
- linearVelocityNoise: 0.1
- angularVelocityNoise: 0.2
- gravityNoise: 0.05
- heightMeasurementNoise: 0.06
- pushInterval_s: 15
- episodeLength_s: 20
### sim
- dt: 0.005
- use_gpu_pipeline: ${eq:${...pipeline},"gpu"}
- gravity: [0.0, 0.0, -9.81]
- add_ground_plane: False
- add_distant_light: False
- use_flatcache: True
- enable_scene_query_support: False
- disable_contact_processing: True
- enable_cameras: False
- default_physics_material:
    - static_friction: 1.0
    - dynamic_friction: 1.0
    - restitution: 0.0
- physx:
    - worker_thread_count: ${....num_threads}
    - solver_type: ${....solver_type}
    - use_gpu: ${eq:${....sim_device},"gpu"}
    - solver_position_iteration_count: 4
    - solver_velocity_iteration_count: 0
    - contact_offset: 0.02
    - rest_offset: 0.0
    - bounce_threshold_velocity: 0.2
    - friction_offset_threshold: 0.04
    - friction_correlation_distance: 0.025
    - enable_sleeping: True
    - enable_stabilization: True
    - max_depenetration_velocity: 100.0
    - gpu_max_rigid_contact_count: 524288
    - gpu_max_rigid_patch_count: 163840
    - gpu_found_lost_pairs_capacity: 4194304
    - gpu_found_lost_aggregate_pairs_capacity: 33554432
    - gpu_total_aggregate_pairs_capacity: 4194304
    - gpu_max_soft_body_contacts: 1048576
    - gpu_max_particle_contacts: 1048576
    - gpu_heap_capacity: 134217728
    - gpu_temp_buffer_capacity: 33554432
    - gpu_max_num_partitions: 8
### A1:
- override_usd_defaults: False
- enable_self_collisions: True
- enable_gyroscopic_forces: False
- solver_position_iteration_count: 4
- solver_velocity_iteration_count: 0
- sleep_threshold: 0.005
- stabilization_threshold: 0.001
- density: -1
- max_depenetration_velocity: 100.0
## Train Config
### params
- seed: ${...seed}
- algo:
    - name: a2c_continuous
- model:
    - name: continuous_a2c_logstd
- network:
    - name: actor_critic
    - separate: False
    - space:
        - continuous:
            - mu_activation: None
            - sigma_activation: None
        - mu_init:
            - name: default
        - sigma_init:
            - name: const_initializer
            - val: 0
            - fixed_sigma: True

    - mlp:
        - units: [512, 256, 128]
        - activation: elu
        - d2rl: False
    - initializer:
        - name: default
    - regularizer:
        - name: None

- load_checkpoint: ${if:${...checkpoint},True,False} # flag which sets whether to load the checkpoint
- load_path: ${...checkpoint} # path to the checkpoint to load

- config:
    - name: ${resolve_default:A1Terrain,${....experiment}}
    - full_experiment_name: ${.name}
    - device: ${....rl_device}
    - device_name: ${....rl_device}
    - env_name: rlgpu
    - multi_gpu: ${....multi_gpu}
    - ppo: True
    - mixed_precision: False
    - normalize_input: True
    - normalize_value: True
    - normalize_advantage: True
    - value_bootstrap: True
    - clip_actions: False
    - num_actors: ${....task.env.numEnvs}
    - reward_shaper:
        - scale_value: 1.0
    - gamma: 0.99
    - tau: 0.95
    - e_clip: 0.2
    - entropy_coef: 0.001
    - learning_rate: 3.e-4 # overwritten by adaptive lr_schedule
    - lr_schedule: adaptive
    - kl_threshold: 0.008 # target kl for adaptive lr
    - truncate_grads: True
    - grad_norm: 1.
    - horizon_length -> 48
    จำนวนข้อมูลที่ agent interact กับ environment ก่อนนำเข้าไป train
    - minibatch_size ->  96000
    ใช้ข้อมูลทั้งหมด 96000 tuple tuple ในที่นี้จะประกอบด้วย state, action, reward, state at t + 1, sometimes done ในการ train 1 epcoh
    #if headless=true 4500[216000] or 5000[240000] (>4000)
    #else <3000[144000]
    - mini_epochs: 5
    - critic_coef: 2
    - clip_value: True
    - seq_length: 4 # only for rnn
    - bounds_loss_coef: 0.001 #0.001
    - max_epochs: ${resolve_default:80000,${....max_iterations}}
    - save_best_after: 200
    - score_to_win: 20000
    - save_frequency: 50
    - print_stats: True

# Configuration and command line arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config.
 
Common arguments for the training scripts are:

* `task=TASK` - Selects which task to use. Any of `AllegroHand`, `Ant`, `Anymal`, `AnymalTerrain`, `BallBalance`, `Cartpole`, `CartpoleCamera`, `Crazyflie`, `FactoryTaskNutBoltPick`, `FactoryTaskNutBoltPlace`, `FactoryTaskNutBoltScrew`, `FrankaCabinet`, `FrankaDeformable`, `Humanoid`, `Ingenuity`, `Quadcopter`, `ShadowHand`, `ShadowHandOpenAI_FF`, `ShadowHandOpenAI_LSTM` (these correspond to the config for each environment in the folder `omniisaacgymenvs/cfg/task`)
* `train=TRAIN` - Selects which training config to use. Will automatically default to the correct config for the environment (ie. `<TASK>PPO`).
* `num_envs=NUM_ENVS` - Selects the number of environments to use (overriding the default number of environments set in the task config).
* `seed=SEED` - Sets a seed value for randomization, and overrides the default seed in the task config
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `sim_device=SIM_DEVICE` - Device used for physics simulation. Set to `gpu` (default) to use GPU and to `cpu` for CPU.
* `device_id=DEVICE_ID` - Device ID for GPU to use for simulation and task. Defaults to `0`. This parameter will only be used if simulation runs on GPU.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and follows PyTorch-like device syntax.
* `multi_gpu=MULTI_GPU` - Whether to train using multiple GPUs. Defaults to `False`. Note that this option is only available with `rlgames_train.py`.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `enable_livestream=ENABLE_LIVESTREAM` - Whether to enable Omniverse streaming.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for. Reasonable defaults are provided for the provided environments.
* `warp=WARP` - If set to True, launch the task implemented with Warp backend (Note: not all tasks have a Warp implementation).
* `kit_app=KIT_APP` - Specifies the absolute path to the kit app file to be used.