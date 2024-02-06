from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.a1 import A1
from omniisaacgymenvs.robots.articulations.views.a1_view import A1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path

import omni.usd
from omni.isaac.core.utils.torch.rotations import *
import omni.usd
import numpy as np
import torch
import math
import os
from pathlib import Path

class A1Task(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._num_observations = 48
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        return
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._a1_translation = torch.tensor([0.0, 0.0, 0.4])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        return

    def set_up_scene(self, scene) -> None:
        self.get_a1()
        super().set_up_scene(scene)
        self._a1s = A1View(prim_paths_expr="/World/envs/.*/a1_instanceable_meshes", #prim_path name
                               name="a1view"
                               #track_contact_forces=True
                               )
        scene.add(self._a1s)
        scene.add(self._a1s._knees)
        scene.add(self._a1s._base)

        return

    def get_a1(self):
        a1 = A1(prim_path=self.default_zero_env_path + "/a1_instanceable_meshes", #prim_path name
                    name="A1",
                    #usd_path="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/urdf/a1.usd", #file name
                    #usd_path="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/test4.usd", #file name
                    #usd_path="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/new/a1.usd",
                    #usd_path="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/urdf/a1/a1.usd",
                    usd_path="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/omniverse_a1/a1_instanceable.usd",
                    #test 8 = test 5 delete drive in each joint
                    #test 7 = unlimit joint + test6
                    #test 6 = test 5
                    #test 5 = a1_instanceable_meshes + no base prim path + no thigh_shoulder prim path
                    #test 4 = a1_instanceable_meshes + no thigh_shoulder prim path
                    #test 5 = a1_instanceable_meshes + no base prim path
                    #a1_instanceble_meshes = original with instaceble code
                    translation=self._a1_translation)
        self._sim_config.apply_articulation_settings(
            "A1", 
            get_prim_at_path(a1.prim_path), 
            self._sim_config.parse_actor_config("A1")
            )

        #Configure joint properties

        joint_paths = []
        for quadrant in ["FL", "RL", "FR", "RR"]:
            for parent, joint in [("hip", "thigh"), ("thigh", "calf")]:
                joint_paths.append(f"{quadrant}_{parent}/{quadrant}_{joint}_joint")
            joint_paths.append(f"base/{quadrant}_hip_joint")

        for joint_path in joint_paths:
            set_drive(f"{a1.prim_path}/{joint_path}", "angular", "position", 0, 400, 40, 33.5) # target Position, Stiffness, Damping, Max Force

        # joint_paths = []
        # for quadrant in ["LF", "LH", "RF", "RH"]:
        #     for component, abbrev in [("HIP", "H"), ("THIGH", "K")]:
        #         joint_paths.append(f"{quadrant}_{component}/{quadrant}_{abbrev}FE")
        #     joint_paths.append(f"base/{quadrant}_HAA")
        
        # for joint_path in joint_paths:
        #     set_drive(f"{a1.prim_path}/{joint_path}", "angular", "position", 0, 400, 40, 33.5)


        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        dof_names = a1.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._a1s.get_world_poses(clone=False)
        root_velocities = self._a1s.get_velocities(clone=False)
        dof_pos = self._a1s.get_joint_positions(clone=False)
        dof_vel = self._a1s.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

        obs = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf[:] = obs
        #print(obs)
        observations = {
            self._a1s.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        # actions: (num_envs, 12) [-1, 1]
        #print(actions)
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self._a1s.count, dtype=torch.int32, device=self._device)
        self.actions[:] = actions.clone().to(self._device)
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt 
        self.current_targets[:] = tensor_clamp(current_targets, self.a1_dof_lower_limits, self.a1_dof_upper_limits)
        self._a1s.set_joint_position_targets(self.current_targets, indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._a1s.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities

        self.current_targets[env_ids] = dof_pos[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._a1s.set_joint_positions(dof_pos, indices)
        self._a1s.set_joint_velocities(dof_vel, indices)

        self._a1s.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices)
        self._a1s.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.

    def post_reset(self):
        #stage = omni.usd.get_context().get_stage()
        #ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
        #ground_prim.GetAttribute("physics:collisionEnabled").Set(False)
        #ground_prim.GetAttribute("physics:collisionEnabled").Set(True)
        self.initial_root_pos, self.initial_root_rot = self._a1s.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._a1s.get_dof_limits()
        self.a1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.a1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False)
        self.last_actions = torch.zeros(self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False)

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(self._a1s.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        #stage = omni.usd.get_context().get_stage()
        #ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
        #ground_prim.GetAttribute("physics:collisionEnabled").Set(False)
        #ground_prim.GetAttribute("physics:collisionEnabled").Set(True)
    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._a1s.get_world_poses(clone=False)
        root_velocities = self._a1s.get_velocities(clone=False)
        dof_pos = self._a1s.get_joint_positions(clone=False)
        dof_vel = self._a1s.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        rew_cosmetic = torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["cosmetic"]

        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_joint_acc  + rew_action_rate + rew_cosmetic + rew_lin_vel_z
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = self._a1s.is_base_below_threshold(threshold=0.25, ground_heights=0.0)
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()


    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over